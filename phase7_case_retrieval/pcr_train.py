"""
pcr_train.py - one Phase-7 run: adapt a Phase-1 model into a dense case retriever, score dev/test, fuse with the
lexical stage, write predictions.

Query representation (--query):
    ctx (default)  the query is its set of CITATION PASSAGES (each <CITATION> sentence with its neighbours), each
                   encoded on its own; score(q, c) = max over passages of cos(passage, candidate). A judgment cites
                   many cases, each discussed in one passage, so the best-matching passage decides (multiple-instance
                   max pooling, as in MaxP / ColBERT-style late interaction).
    doc            the query is a long document like the candidates: mean of chunk vectors (ablation)
Training (train split only):
  * a step takes B train queries; each gets one gold candidate (positive) and H negatives
      --neg hard     H hard negatives from the lexical top-50 of that query (non-gold)       (default)
      --neg random   H random pool documents (non-gold)
      --neg inbatch  no extra negatives (in-batch only)
    every other query's documents in the batch are in-batch negatives (gold of the query itself masked out)
  * documents are long: each is represented by Kq (query) / Kd (candidate) randomly chosen chunks of L tokens,
    mean-pooled per chunk, averaged per document (hierarchical pooling)
  * loss: InfoNCE over the batch (temperature tau); with --teacher, + listwise distillation of the teacher's
    score distribution over the same batch (KL at temperature T): loss = (1-a) CE + a T^2 KL
  * 8% of train queries are held out (val) and select the checkpoint: micro-F1@7 when reranking their gold +
    lexical top-30
Inference: every document of the dev / test pools is encoded in full (first --inf_tok tokens, all chunks);
scores = document-vector cosine ("mean") and chunk MaxSim ("maxsim"). The dense score is fused with the lexical
score (z-scores, weight chosen on dev). Dev labels are used only for that choice and for reporting.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import argparse
import json
import math
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from pcr_data import BPE, K, micro_prf, topk, write_prediction
from pcr_model import Retriever, build_retriever, doc_vectors

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--cache", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model_dir", default="")
p.add_argument("--method", default="full", choices=["full", "partial", "lorma", "ia3"])
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=float, default=32)
p.add_argument("--top_k", type=int, default=4)
p.add_argument("--pool", default="mean", choices=["mean", "last"])
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--max_minutes", type=float, default=240)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--hard", type=int, default=3)
p.add_argument("--hard_top", type=int, default=50)
p.add_argument("--neg", default="hard", choices=["hard", "random", "inbatch"])
p.add_argument("--tau", type=float, default=0.05)
p.add_argument("--chunk_len", type=int, default=256)
p.add_argument("--q_chunks", type=int, default=4)
p.add_argument("--d_chunks", type=int, default=4)
p.add_argument("--eval_every", type=int, default=250)
p.add_argument("--val_pct", type=int, default=8)
p.add_argument("--inf_tok", type=int, default=12288)
p.add_argument("--inf_batch", type=int, default=96)
p.add_argument("--teacher", default="")
p.add_argument("--teacher_ckpt", default="")
p.add_argument("--kd_alpha", type=float, default=0.5)
p.add_argument("--kd_T", type=float, default=1.0)
p.add_argument("--query", default="ctx", choices=["ctx", "doc"])
p.add_argument("--ctx_len", type=int, default=160)
p.add_argument("--q_ctx", type=int, default=12)
p.add_argument("--lex_key", default="lex_nocit")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--smoke", action="store_true")
args = p.parse_args()

T0 = time.time()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG = open(os.path.join(args.out, "log.txt"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


meta = json.load(open(os.path.join(args.cache, "meta.json")))
lex = np.load(os.path.join(args.cache, "lex.npz"))
assert f"dev_{args.lex_key}" in lex.files, f"--lex_key {args.lex_key} not in lex.npz"
bpe = BPE(os.path.join(args.cache, "bpe.npz"))
TOKS = {}


def toks(doc_id):
    t = TOKS.get(doc_id)
    if t is None:
        t = bpe(doc_id)
        if len(t) == 0:
            t = np.array([2], np.int64)
        TOKS[doc_id] = t
    return t


def pick_chunks(doc_id, k, mode, L, cap=None):
    t = toks(doc_id)
    if cap:
        t = t[:cap]
    m = max(1, math.ceil(len(t) / L))
    if k is None or m <= k:
        idx = range(m)
    elif mode == "random":
        idx = sorted(rng.choice(m, size=k, replace=False).tolist())
    else:
        idx = sorted(set(np.linspace(0, m - 1, k).round().astype(int).tolist()))
    return [t[i * L:(i + 1) * L] for i in idx]


def pad(chunks):
    L = max(len(c) for c in chunks)
    ids = torch.zeros(len(chunks), L, dtype=torch.long)
    valid = torch.zeros(len(chunks), L, dtype=torch.bool)
    for i, c in enumerate(chunks):
        ids[i, :len(c)] = torch.from_numpy(np.asarray(c, np.int64))
        valid[i, :len(c)] = True
    return ids.to(dev, non_blocking=True), valid.to(dev, non_blocking=True)


def ctx_tokens(qid, k):
    t = toks(f"{qid}#c{k}")
    if len(t) > args.ctx_len:                                   # keep the middle: the marker sentence
        a = (len(t) - args.ctx_len) // 2
        t = t[a:a + args.ctx_len]
    return t


N_CTX = {q: n for s in ("train", "dev", "test") for q, n in zip(meta[s]["qids"], meta[s]["n_ctx"])}


def query_units(qids, k=None, mode="random"):
    """Chunks representing the queries + owner index: citation passages (ctx mode) or document chunks (doc mode)."""
    cs, own = [], []
    for j, q in enumerate(qids):
        if args.query == "ctx":
            n = N_CTX[q]
            idx = range(n) if (k is None or n <= k) else (sorted(rng.choice(n, size=k, replace=False).tolist())
                                                          if mode == "random" else np.linspace(0, n - 1, k).round().astype(int))
            c = [ctx_tokens(q, int(i)) for i in idx]
        else:
            c = pick_chunks(q, k, mode, args.chunk_len)
        cs += c
        own += [j] * len(c)
    return cs, own


def combine_query(Eq, owner_q, n_q, Vc):
    """(n_q, n_c) scores: max over passages (ctx) or cosine of mean document vectors (doc)."""
    if args.query == "ctx":
        sim = Eq @ Vc.T                                            # (n_units, n_c)
        out = torch.full((n_q, Vc.shape[0]), -2.0, device=sim.device, dtype=sim.dtype)
        return out.scatter_reduce(0, owner_q[:, None].expand(-1, Vc.shape[0]).contiguous(), sim, reduce="amax",
                                  include_self=True)
    return doc_vectors(Eq, owner_q, n_q) @ Vc.T


# ----------------------------------------------------------------------------- model(s)
model, (n_tr, n_all, n_wrap) = build_retriever(args.ckpt, args.method, args.pool, args.rank, args.alpha, args.top_k)
model.to(dev)
log(f"method {args.method}: trainable {n_tr / 1e6:.2f}M / {n_all / 1e6:.2f}M ({100 * n_tr / n_all:.2f}%), "
    f"wrapped {n_wrap}")
teacher = None
if args.teacher:
    tsd = torch.load(args.teacher, map_location="cpu", weights_only=False)
    ta = tsd["args"]
    teacher, _ = build_retriever(args.teacher_ckpt, ta["method"], ta["pool"], ta["rank"], ta["alpha"], ta["top_k"],
                                 saved=tsd)
    teacher.to(dev).eval()
    for q in teacher.parameters():
        q.requires_grad = False
    t_params = sum(q.numel() for q in teacher.parameters())
    log(f"teacher {args.teacher} ({ta['method']}, {t_params / 1e6:.1f}M params)")
    del tsd


@torch.no_grad()
def encode_docs(m, doc_ids, k=None, mode="even", cap=None):
    """-> chunk vectors (N, d) fp16, owner (N,) long, n tokens."""
    chunks, owner = [], []
    for j, d in enumerate(doc_ids):
        cs = pick_chunks(d, k, mode, args.chunk_len, cap)
        chunks += cs
        owner += [j] * len(cs)
    return encode_units(m, chunks, owner)


@torch.no_grad()
def encode_units(m, chunks, owner):
    m.eval()
    order = np.argsort([-len(c) for c in chunks], kind="stable")
    out = torch.zeros(len(chunks), m.backbone.d, dtype=torch.float16, device=dev)
    for s in range(0, len(order), args.inf_batch):
        b = order[s:s + args.inf_batch]
        ids, valid = pad([chunks[i] for i in b])
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out[torch.as_tensor(b, device=dev)] = m.encode_chunks(ids, valid).half()
    return out, torch.as_tensor(owner, device=dev), sum(len(c) for c in chunks)


# ----------------------------------------------------------------------------- training data
tr = meta["train"]
tr_pos = {c: j for j, c in enumerate(tr["cids"])}
gold_tr = [[tr_pos[c] for c in dict.fromkeys(g) if c in tr_pos] for g in tr["gold"]]
hardneg = lex["train_hardneg"]
is_val = np.array([zlib.crc32(f"781:{q}".encode()) % 100 < args.val_pct for q in tr["qids"]])
fit_q = [i for i in range(len(tr["qids"])) if not is_val[i] and gold_tr[i]]
val_q = [i for i in range(len(tr["qids"])) if is_val[i] and gold_tr[i]]
if args.smoke:
    fit_q, val_q = fit_q[:16], val_q[:4]
log(f"train queries: {len(fit_q)} fit, {len(val_q)} val (held out from training)")


def sample_batch():
    qi = rng.choice(fit_q, size=min(args.batch, len(fit_q)), replace=False)
    H = 0 if args.neg == "inbatch" else args.hard
    q_docs, c_docs, pos = [], [], []
    for i in qi:
        q_docs.append(tr["qids"][i])
        pos.append(len(c_docs))
        c_docs.append(tr["cids"][int(rng.choice(gold_tr[i]))])
        if H:
            if args.neg == "hard":
                cand = hardneg[i][:args.hard_top]
                cand = cand[cand >= 0]
            else:
                bad = set(gold_tr[i])
                cand = np.array([j for j in rng.choice(len(tr["cids"]), size=4 * H) if j not in bad])
            for j in rng.choice(cand, size=min(H, len(cand)), replace=False):
                c_docs.append(tr["cids"][int(j)])
    # mask: candidate j is not a negative for query i if it is gold for i (other than the sampled positive),
    # or the query itself
    gsets = [set(tr["gold"][i]) | {tr["qids"][i]} for i in qi]
    mask = torch.zeros(len(qi), len(c_docs), dtype=torch.bool)
    for a, gs in enumerate(gsets):
        for b, cd in enumerate(c_docs):
            if b != pos[a] and cd in gs:
                mask[a, b] = True
    return q_docs, c_docs, torch.tensor(pos), mask


def batch_scores(m, q_docs, c_docs, q_chunks, c_chunks):
    """Document-level cosine scores (Bq, Bc) / tau, from given chunk lists."""
    allc = q_chunks[0] + c_chunks[0]
    ids, valid = pad(allc)
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        E = m.encode_chunks(ids, valid)
    nq = len(q_chunks[0])
    owner_q = torch.as_tensor(q_chunks[1], device=dev)
    owner_c = torch.as_tensor(c_chunks[1], device=dev)
    Vc = doc_vectors(E[nq:].float(), owner_c, len(c_docs))
    return combine_query(E[:nq].float(), owner_q, len(q_docs), Vc) / args.tau, int(valid.sum())


def chunk_list(docs, k):
    cs, own = [], []
    for j, d in enumerate(docs):
        c = pick_chunks(d, k, "random", args.chunk_len)
        cs += c
        own += [j] * len(c)
    return cs, own


@torch.no_grad()
def val_f1(m):
    """micro-F1@7 of the held-out train queries when reranking gold + lexical top-30 (fixed even chunks)."""
    if not val_q:
        return 0.0, 0.0
    cand = {}
    lex_tr = lex["train_lex"]
    for i in val_q:
        top = np.argsort(-lex_tr[i])[:30].tolist()
        cand[i] = list(dict.fromkeys(gold_tr[i] + top))
    docs = sorted({tr["cids"][j] for i in val_q for j in cand[i]})
    E, own, _ = encode_docs(m, docs, k=8, mode="even")
    V = doc_vectors(E.float(), own, len(docs))
    at = {d: k for k, d in enumerate(docs)}
    qs = [tr["qids"][i] for i in val_q]
    Eq, oq, _ = encode_units(m, *query_units(qs, 32, "even"))
    pred_d, pred_l, gold = [], [], []
    for a, i in enumerate(val_q):
        cj = cand[i]
        s = combine_query(Eq[oq == a].float(), torch.zeros(int((oq == a).sum()), dtype=torch.long, device=dev), 1,
                          V[[at[tr["cids"][j]] for j in cj]])[0].cpu().numpy()
        pred_d.append([cj[j] for j in np.argsort(-s)[:K]])
        pred_l.append([cj[j] for j in np.argsort(-lex_tr[i][cj])[:K]])
        gold.append(gold_tr[i])
    return micro_prf(pred_d, gold)["f1"], micro_prf(pred_l, gold)["f1"]


# ----------------------------------------------------------------------------- train
hist, best = [], {"step": 0, "val": -1.0}
best_state = None
train_min = 0.0
tok_seen = 0
if args.steps > 0:
    model.backbone.block_ckpt = True
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    total = [args.steps]
    warm = max(1, int(0.06 * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total[0] - s) / max(1, total[0] - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    tv = time.time()
    v0, vlex = val_f1(model)
    t_val = time.time() - tv
    log(f"step 0: val F1@7 {v0:.4f} (lexical on the same candidates {vlex:.4f})")
    hist.append(dict(step=0, val=v0, val_lex=vlex))
    best = {"step": 0, "val": v0}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                  if k in {n for n, q in model.named_parameters() if q.requires_grad}}
    t_train = time.time()
    run_loss, run_kd, n_run = 0.0, 0.0, 0
    step = 0
    while step < total[0]:
        model.train()
        q_docs, c_docs, pos, mask = sample_batch()
        qc = query_units(q_docs, args.q_ctx if args.query == "ctx" else args.q_chunks)
        cc = chunk_list(c_docs, args.d_chunks)
        S, ntok = batch_scores(model, q_docs, c_docs, qc, cc)
        S = S.float().masked_fill(mask.to(dev), -1e4)
        loss = F.cross_entropy(S, pos.to(dev))
        kd = torch.zeros((), device=dev)
        if teacher is not None:
            with torch.no_grad():
                St, _ = batch_scores(teacher, q_docs, c_docs, qc, cc)
                St = St.float().masked_fill(mask.to(dev), -1e4)
            T = args.kd_T
            kd = F.kl_div(F.log_softmax(S / T, -1), F.log_softmax(St / T, -1), log_target=True,
                          reduction="batchmean") * T * T
            loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        tok_seen += ntok
        run_loss += float(loss)
        run_kd += float(kd)
        n_run += 1
        el = (time.time() - t_train) / 60
        if step == 20 and not args.smoke:          # fit the schedule to the time budget
            per = el / 20
            per_eval = t_val / 60 / args.eval_every                   # validation time per step
            total[0] = int(min(args.steps, max(40, (args.max_minutes - 2) / (per + per_eval))))
            log(f"{per * 60:.2f} s/step -> {total[0]} steps planned")
        if step % 50 == 0 or step == total[0]:
            log(f"step {step}/{total[0]} loss {run_loss / n_run:.4f} kd {run_kd / n_run:.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} {tok_seen / max(1, (time.time() - t_train)):.0f} tok/s {el:.1f} min")
            run_loss, run_kd, n_run = 0.0, 0.0, 0
        if step % args.eval_every == 0 or step == total[0] or el > args.max_minutes:
            v, _ = val_f1(model)
            hist.append(dict(step=step, val=v, minutes=round(el, 1)))
            log(f"step {step}: val F1@7 {v:.4f}")
            if v > best["val"]:
                best = {"step": step, "val": v}
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items() if k in best_state}
        if el > args.max_minutes:
            log("time budget reached")
            break
    train_min = (time.time() - t_train) / 60
    model.load_state_dict(best_state, strict=False)
    log(f"best step {best['step']} (val F1@7 {best['val']:.4f}); trained {train_min:.1f} min")
model.backbone.block_ckpt = False
peak_train = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0
if dev.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
del teacher


# ----------------------------------------------------------------------------- inference on dev / test
def zrows(S):
    """Per-query z-scores; entries <= -1e3 (the query itself, masked) stay masked and do not enter the statistics."""
    S = np.asarray(S, np.float32)
    ok = S > -1e3
    n = ok.sum(1, keepdims=True).clip(min=1)
    mu = np.where(ok, S, 0).sum(1, keepdims=True) / n
    sd = np.sqrt(np.where(ok, (S - mu) ** 2, 0).sum(1, keepdims=True) / n) + 1e-6
    return np.where(ok, (S - mu) / sd, -1e4).astype(np.float32)


@torch.no_grad()
def score_split(s, q_limit=None, c_limit=None):
    m = meta[s]
    qids, cids = m["qids"][:q_limit], m["cids"][:c_limit]
    t = time.time()
    Eq, oq, nt1 = encode_docs(model, qids, cap=args.inf_tok)
    Ec, oc, nt2 = encode_docs(model, cids, cap=args.inf_tok)
    q_mode, args.query = args.query, "ctx"
    Ex, ox, nt3 = encode_units(model, *query_units(qids))           # every citation passage of every query
    args.query = q_mode
    enc_s = time.time() - t
    Vq, Vc = doc_vectors(Eq.float(), oq, len(qids)), doc_vectors(Ec.float(), oc, len(cids))
    S_mean = (Vq @ Vc.T).cpu().numpy()
    S_max = np.zeros_like(S_mean)
    S_cdoc = np.zeros_like(S_mean)
    S_cchunk = np.zeros_like(S_mean)
    for i in range(len(qids)):
        X = Ex[ox == i]                                                # (passages, d)
        S_cdoc[i] = (X.float() @ Vc.T).max(0).values.cpu().numpy()
        sim = (X @ Ec.T).float()
        b2 = torch.full((X.shape[0], len(cids)), -2.0, device=dev)
        b2 = b2.scatter_reduce(1, oc[None].expand(X.shape[0], -1).contiguous(), sim, reduce="amax", include_self=True)
        S_cchunk[i] = b2.max(0).values.cpu().numpy()
    for i in range(len(qids)):
        Q = Eq[oq == i]                                                # (mq, d)
        sim = (Q @ Ec.T).float()                                       # (mq, Nc)
        best_ = torch.full((Q.shape[0], len(cids)), -2.0, device=dev)
        best_ = best_.scatter_reduce(1, oc[None].expand(Q.shape[0], -1).contiguous(), sim, reduce="amax", include_self=True)
        S_max[i] = best_.mean(0).cpu().numpy()
    return dict(mean=S_mean, maxsim=S_max, ctxdoc=S_cdoc, ctxchunk=S_cchunk, enc_s=enc_s,
                n_docs=len(qids) + len(cids), n_tok=nt1 + nt2 + nt3,
                qids=qids, cids=cids)


lim = (5, 80) if args.smoke else (None, None)
res = {s: score_split(s, *lim) for s in ("dev", "test")}
peak_inf = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0
gold_dev = meta["dev"]["gold"][:len(res["dev"]["qids"])]


def lex_of(s):
    r = res[s]
    return lex[f"{s}_{args.lex_key}"][:len(r["qids"]), :len(r["cids"])]


def evaluate(S, s="dev"):
    return micro_prf(topk(S, res[s]["cids"], res[s]["qids"]), gold_dev)


BETAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
table = {}
zl = {s: zrows(lex_of(s)) for s in ("dev", "test")}
KINDS = ("mean", "maxsim", "ctxdoc", "ctxchunk")
for kind in KINDS:
    zd = zrows(res["dev"][kind])
    table[f"dense_{kind}"] = evaluate(zd)
    for b in BETAS:
        table[f"lex+{b}*{kind}"] = evaluate(zl["dev"] + b * zd)
table["lexical"] = evaluate(zl["dev"])
f1 = {k: v["f1"] for k, v in table.items()}
dense_best = max((f"dense_{k}" for k in KINDS), key=lambda k: f1[k])
fused_best = max((k for k in f1 if k.startswith("lex+")), key=lambda k: f1[k])

# honest estimate of the fusion choice: 2-fold cross-fitting over dev queries (choose on one half, score the other)
half = np.array([zlib.crc32(f"781:{q}".encode()) % 2 for q in res["dev"]["qids"]])
tp = ng = 0
cand_keys = [k for k in f1 if k.startswith("lex+")]
for h in (0, 1):
    pick_i, eval_i = np.where(half == h)[0], np.where(half != h)[0]

    def f1_on(key, rows):
        kind = key.split("*")[1]
        b = float(key.split("+")[1].split("*")[0])
        S = (zl["dev"] + b * zrows(res["dev"][kind]))[rows]
        pr = topk(S, res["dev"]["cids"], [res["dev"]["qids"][i] for i in rows])
        return micro_prf(pr, [gold_dev[i] for i in rows])
    if len(pick_i) and len(eval_i):
        kbest = max(cand_keys, key=lambda k: f1_on(k, pick_i)["f1"])
        r = f1_on(kbest, eval_i)
        tp += r["tp"]
        ng += sum(len(set(gold_dev[i])) for i in eval_i)
P = tp / (K * len(gold_dev)) if gold_dev else 0
R = tp / max(ng, 1)
crossfit = 2 * P * R / (P + R) if P + R else 0.0


def scores_for(key, s):
    if key.startswith("dense_"):
        return zrows(res[s][key.split("_", 1)[1]])
    kind = key.split("*")[1]
    b = float(key.split("+")[1].split("*")[0])
    return zl[s] + b * zrows(res[s][kind])


subs = {}
for name, key in (("dense", dense_best), ("fused", fused_best)):
    S = scores_for(key, "test")
    pred = topk(S, res["test"]["cids"], res["test"]["qids"])
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)
    if not args.smoke:
        write_prediction(os.path.join(d, "prediction.json"), res["test"]["qids"], pred, res["test"]["cids"])
    else:
        json.dump(dict(zip(res["test"]["qids"], pred)), open(os.path.join(d, "prediction.json"), "w"))
    subs[name] = key
    os.system(f"cd {d} && zip -q -j ../submission_{name}.zip prediction.json")

np.savez_compressed(os.path.join(args.out, "scores.npz"),
                    **{f"{s}_{k}": res[s][k].astype(np.float16) for s in ("dev", "test") for k in KINDS})
enc_tok_s = sum(res[s]["n_tok"] for s in res) / max(1e-6, sum(res[s]["enc_s"] for s in res))
summary = dict(
    args=vars(args), trainable=n_tr, total_params=n_all, wrapped=n_wrap, train_minutes=round(train_min, 1),
    train_tok_s=round(tok_seen / max(1e-6, train_min * 60)), peak_train_gb=round(peak_train, 2),
    peak_inference_gb=round(peak_inf, 2), history=hist, best_step=best["step"], best_val=best["val"],
    encode_tok_s=round(enc_tok_s), encode_s_per_doc=round(sum(res[s]["enc_s"] for s in res) /
                                                           max(1, sum(res[s]["n_docs"] for s in res)), 4),
    dev={k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in table.items()},
    dense_choice=dense_best, fused_choice=fused_best, dev_dense=round(f1[dense_best], 4),
    dev_fused=round(f1[fused_best], 4), dev_lexical=round(f1["lexical"], 4), dev_fused_crossfit=round(crossfit, 4),
    submissions=subs, minutes_total=round((time.time() - T0) / 60, 1))
json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
log(f"DEV micro-F1@7: lexical {f1['lexical']:.4f} | dense {dense_best} {f1[dense_best]:.4f} | "
    f"fused {fused_best} {f1[fused_best]:.4f} (cross-fitted {crossfit:.4f})")
if args.model_dir and args.steps > 0:
    os.makedirs(args.model_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.out))
    torch.save({"state": model.state_dict(), "args": vars(args)}, os.path.join(args.model_dir, f"{tag}.pt"))
log(f"done in {(time.time() - T0) / 60:.1f} min")
