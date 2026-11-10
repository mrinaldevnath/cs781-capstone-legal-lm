"""
summ_model.py - summarization on top of the Phase-1 language models (losses, SLED encoding, generation).

Decoder-only models B, C, D, E:  [document input][separator][summary][</s>]  - one causal sequence, the loss is on the
    summary tokens only (and </s> when the whole reference fits); at test time the prompt is [input][separator] and the
    summary is generated with the Phase-1 KV caches (models.DecoderLM.begin / advance).
Encoder-decoder model A:  the document goes through the encoder, the decoder writes the summary (cross-attention).
    With SLED (Ivgi et al., TACL 2023) the document is cut into overlapping windows of c tokens (stride (1-rho) c,
    rho/2 c context on each side); every window is encoded independently and only its middle ("effective") tokens are
    kept, so each document token is represented exactly once, with context on both sides; the decoder cross-attends to
    the concatenation of all effective tokens (fusion-in-decoder). With a document shorter than c this is plain BART.
Knowledge distillation (Hinton et al., 2015; word-level KD for sequence models, Kim & Rush, 2016): the student's
    next-token distribution on every summary token is pulled towards the teacher's (temperature T, KL scaled by T^2),
    mixed with the usual cross-entropy: loss = (1-alpha) CE + alpha T^2 KL. Teacher and student share the tokenizer.
Generation: greedy, n-gram blocking (no n-gram is generated twice, Paulus et al., 2018), </s> suppressed before a
    minimum length, PAD/BOS/MASK never generated.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from layers import Ctx
from models import build_model, ce_chunked, GenState, BOS_ID, EOS_ID, PAD_ID

MASK_ID = 3


class Holder(nn.Module):
    """A Phase-1 language model under `.lm` (the layout adapt.configure expects)."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm


def load_lm(ckpt_path, ckpt_act=True):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    cfg["ckpt"] = ckpt_act
    lm = build_model(cfg)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model"].items()}
    missing, unexpected = lm.load_state_dict(state, strict=False)
    assert not missing and not unexpected, f"checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}"
    return Holder(lm), cfg


def is_encdec(holder):
    return holder.lm.cfg["kind"] == "encdec"


# ---------------------------------------------------------------------------------------------- decoder-only
def dec_batch(prompts, targets, device):
    """Right-padded [prompt][target] sequences and the mask of target tokens."""
    L = [len(p) + len(t) for p, t in zip(prompts, targets)]
    T = max(L)
    ids = np.full((len(L), T), PAD_ID, np.int64)
    tm = np.zeros((len(L), T), bool)
    for i, (p, t) in enumerate(zip(prompts, targets)):
        ids[i, :len(p)] = p
        ids[i, len(p):len(p) + len(t)] = t
        tm[i, len(p):len(p) + len(t)] = True
    return torch.from_numpy(ids).to(device), torch.from_numpy(tm).to(device)


def dec_hidden(lm, x):
    T = x.shape[1]
    ctx = Ctx(torch.arange(T, device=x.device), ckpt=lm.ckpt and torch.is_grad_enabled(), qchunk=lm.qchunk)
    return lm.trunk(x, ctx)


def kd_chunked(hs, ws, ht, wt, T, chunk=1024):
    """Summed KL(teacher || student) of temperature-T distributions, `chunk` positions at a time."""
    def f(a, w, lt):
        ls = F.log_softmax(F.linear(a, w).float() / T, dim=-1)
        return F.kl_div(ls, lt, log_target=True, reduction="sum")
    tot = hs.new_zeros((), dtype=torch.float32)
    for i in range(0, hs.shape[0], chunk):
        with torch.no_grad():
            lt = F.log_softmax(F.linear(ht[i:i + chunk], wt).float() / T, dim=-1)
        if torch.is_grad_enabled():
            tot = tot + checkpoint(f, hs[i:i + chunk], ws, lt, use_reentrant=False)
        else:
            tot = tot + f(hs[i:i + chunk], ws, lt)
    return tot


def dec_loss(lm, ids, tm, teacher=None, kd_alpha=0.5, kd_T=2.0):
    """-> (loss averaged over target tokens, n target tokens, CE sum)."""
    x, y, m = ids[:, :-1], ids[:, 1:], tm[:, 1:]
    h = dec_hidden(lm, x)
    hs, ys = h[m], y[m]
    n = max(1, ys.numel())
    ce = ce_chunked(hs, lm.emb.weight, ys)
    if teacher is None:
        return ce / n, n, ce.detach()
    with torch.no_grad():
        ht = dec_hidden(teacher, x)[m]
    kl = kd_chunked(hs, lm.emb.weight, ht, teacher.emb.weight, kd_T)
    return ((1 - kd_alpha) * ce + kd_alpha * kd_T * kd_T * kl) / n, n, ce.detach()


# ---------------------------------------------------------------------------------------------- encoder-decoder (A) + SLED
def sled_encode(lm, docs, c=512, rho=0.5, device="cuda", group=64):
    """docs: list of 1-D int arrays. -> enc_out (B, N, d), enc_valid (B, N) (each doc token exactly once)."""
    stride = max(1, int(round(c * (1 - rho))))
    pad = (c - stride) // 2
    wins, meta = [], []
    for b, d in enumerate(docs):
        n = len(d)
        if n == 0:
            d, n = np.array([EOS_ID]), 1
        if n <= c:
            meta.append((b, len(wins), 0, 0, n))
            wins.append(d)
            continue
        for i in range(math.ceil(n / stride)):
            e0, e1 = i * stride, min(n, (i + 1) * stride)
            a = max(0, min(e0 - pad, n - c))
            meta.append((b, len(wins), a, e0, e1))
            wins.append(d[a:a + c])
    L = max(len(w) for w in wins)
    ids = np.full((len(wins), L), PAD_ID, np.int64)
    val = np.zeros((len(wins), L), bool)
    for i, w in enumerate(wins):
        ids[i, :len(w)] = w
        val[i, :len(w)] = True
    ids_t, val_t = torch.from_numpy(ids).to(device), torch.from_numpy(val).to(device)
    enc = torch.cat([lm.encode(ids_t[i:i + group], val_t[i:i + group]) for i in range(0, len(wins), group)], 0)
    per_doc = [[] for _ in docs]
    for (b, wi, a, e0, e1) in meta:
        per_doc[b].append(enc[wi, e0 - a:e1 - a])
    seqs = [torch.cat(p, 0) for p in per_doc]
    N = max(s.shape[0] for s in seqs)
    out = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    ev = torch.zeros(len(seqs), N, dtype=torch.bool, device=out.device)
    for i, s in enumerate(seqs):
        ev[i, :s.shape[0]] = True
    return out, ev


def ed_batch(targets, device):
    T = max(len(t) for t in targets)
    din = np.full((len(targets), T), PAD_ID, np.int64)
    tgt = np.full((len(targets), T), PAD_ID, np.int64)
    m = np.zeros((len(targets), T), bool)
    for i, t in enumerate(targets):
        din[i, 0] = BOS_ID
        din[i, 1:len(t)] = t[:-1]
        tgt[i, :len(t)] = t
        m[i, :len(t)] = True
    f = lambda a: torch.from_numpy(a).to(device)
    return f(din), f(tgt), f(m)


def ed_loss(lm, docs, targets, device, c=512, rho=0.5):
    enc, ev = sled_encode(lm, docs, c, rho, device)
    din, tgt, m = ed_batch(targets, device)
    T = din.shape[1]
    ctx = Ctx(torch.arange(T, device=device), ckpt=lm.ckpt and torch.is_grad_enabled(), qchunk=lm.qchunk,
              enc_out=enc, enc_valid=ev)
    x = lm.emb(din).float()
    for blk in lm.dec:
        x = blk(x, ctx)
    h = lm.norm(x)
    hs, ys = h[m], tgt[m]
    n = max(1, ys.numel())
    ce = ce_chunked(hs, lm.emb.weight, ys)
    return ce / n, n, ce.detach()


# ---------------------------------------------------------------------------------------------- generation
@torch.no_grad()
def merge_for_decode(holder, names=("w_uk", "w_uv")):
    """MLA decoding reads w_uk.weight / w_uv.weight directly (absorbed up-projections). A LoRA / LoRMA / AdaLoRA wrapper
    there is a linear map, so it is merged into an equivalent plain nn.Linear: W_eff = wrapper(I)^T (exact).
    Returns the number of merged layers. Call once before generation (the model is not trained afterwards)."""
    n = 0
    for mod in list(holder.modules()):
        for name in names:
            ch = getattr(mod, name, None)
            if ch is None or isinstance(ch, nn.Linear) or not hasattr(ch, "base"):
                continue
            base = ch.base
            dev = next(ch.parameters()).device
            ch.eval()
            I = torch.eye(base.in_features, device=dev, dtype=torch.float32)
            with torch.autocast(device_type=dev.type, enabled=False):
                W = ch(I).float().t().contiguous()                       # (out, in)
            lin = nn.Linear(base.in_features, base.out_features, bias=False).to(dev)
            lin.weight.copy_(W)
            lin.weight.requires_grad = False
            setattr(mod, name, lin)
            n += 1
    return n


@torch.no_grad()
def generate(lm, prompts, max_new, min_new=0, no_repeat=4, device="cuda", encdec=False, sled=(512, 0.5)):
    """Greedy decoding for a batch. prompts: list of 1-D int arrays. Returns list of token lists (without </s>)."""
    B = len(prompts)
    use_cuda = str(device).startswith("cuda")
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
        if encdec:
            st = GenState()
            st.enc_out, st.enc_valid = sled_encode(lm, prompts, sled[0], sled[1], device)
            st.caches = [blk.make_cache(B, max_new + 1, device, lm.cache_dtype) for blk in lm.dec]
            st.p, st.B = 0, B
            logits = lm.advance(torch.full((B,), BOS_ID, dtype=torch.long, device=device), st)
        else:
            P = max(len(p) for p in prompts)
            ids = torch.full((B, P), PAD_ID, dtype=torch.long)
            padlen = torch.zeros(B, dtype=torch.long)
            for i, p in enumerate(prompts):
                ids[i, P - len(p):] = torch.as_tensor(np.asarray(p, np.int64))
                padlen[i] = P - len(p)
            logits, st = lm.begin(ids.to(device), padlen.to(device), max_new)
        outs = [[] for _ in range(B)]
        tables = [dict() for _ in range(B)]
        done = [False] * B
        n1 = no_repeat - 1
        for t in range(max_new):
            lg = logits.float()
            lg[:, PAD_ID] = float("-inf")
            lg[:, BOS_ID] = float("-inf")
            lg[:, MASK_ID] = float("-inf")
            if t < min_new:
                lg[:, EOS_ID] = float("-inf")
            if no_repeat and t >= n1:
                rows, cols = [], []
                for i in range(B):
                    if done[i]:
                        continue
                    ban = tables[i].get(tuple(outs[i][-n1:]))
                    if ban:
                        rows.extend([i] * len(ban))
                        cols.extend(ban)
                if rows:
                    lg[torch.tensor(rows, device=lg.device), torch.tensor(cols, device=lg.device)] = float("-inf")
            nxt = lg.argmax(-1)
            nl = nxt.tolist()
            for i in range(B):
                if done[i]:
                    continue
                tok = nl[i]
                if tok == EOS_ID:
                    done[i] = True
                    continue
                if no_repeat and len(outs[i]) >= n1:
                    tables[i].setdefault(tuple(outs[i][-n1:]), set()).add(tok)
                outs[i].append(tok)
            if all(done) or t + 1 == max_new:
                break
            logits = lm.advance(nxt, st)
    return outs


def gen_many(lm, jobs, batch, device, no_repeat, encdec=False, sled=(512, 0.5), log=None):
    """jobs: list of (prompt, max_new, min_new). Batches jobs of similar prompt length and equal max_new.
    Returns list of token lists in the job order."""
    order = sorted(range(len(jobs)), key=lambda i: (jobs[i][1], len(jobs[i][0])))
    res = [None] * len(jobs)
    i = 0
    nb = 0
    while i < len(order):
        mx = jobs[order[i]][1]
        grp = [order[i]]
        i += 1
        while i < len(order) and len(grp) < batch and jobs[order[i]][1] == mx:
            grp.append(order[i])
            i += 1
        mn = min(jobs[g][2] for g in grp)
        out = generate(lm, [jobs[g][0] for g in grp], mx, mn, no_repeat, device, encdec, sled)
        for g, o in zip(grp, out):
            res[g] = o
        nb += 1
        if log and nb % 10 == 0:
            log(f"  generated {i}/{len(jobs)}")
    return res
