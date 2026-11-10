"""
tests.py - correctness tests. Runs on CPU in about a minute (tiny models), no data needed.

    python tests.py            # all tiny-model tests
    python tests.py --full     # + builds the four real models: parameter counts, one forward pass at 1024 tokens

What is checked (this is the evidence that the from-scratch attention is right):
  1  attention core == naive reference (MHA, GQA with 2 and 4 query heads per KV head, MQA, causal, sliding
     window, padding mask, two-term scores for MLA, query blocking, activation checkpointing, gradients)
  2  GQA never needs repeated K/V (grouped matmul == explicit repeat_interleave reference)
  3  Gated DeltaNet: chunked == step-by-step recurrence (outputs, final state, gradients), with/without delta term,
     lengths that are not a multiple of the chunk
  4  no future-token leakage in every model
  5  incremental KV-cache decoding == full forward, in left-padded batches of different lengths
     (covers ring-buffer wrap-around for the local layers, MLA absorbed decode, GDN state + conv tail, A cross-attn)
  6  every parameter receives a finite gradient and a tiny model can overfit a batch (loss falls) for A, B, C, D
  7  shared cross-architecture metric equals a manual computation
  8  no torch.nn attention / scaled_dot_product_attention anywhere in the model code
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import glob
import math
import time
import traceback
import argparse

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_core import attend, attend_blocks, _scores
from gated_deltanet import gdn_chunked, gdn_recurrent
from layers import Ctx
from models import build_model, BOS_ID

torch.manual_seed(0)
RESULTS = []


def check(name, fn):
    t0 = time.time()
    try:
        fn()
        RESULTS.append((name, True))
        print(f"PASS  {name}  ({time.time() - t0:.1f}s)", flush=True)
    except Exception:
        RESULTS.append((name, False))
        print(f"FAIL  {name}")
        traceback.print_exc()


def close(a, b, tol=2e-4, what=""):
    err = (a - b).abs().max().item()
    assert err < tol, f"{what} max abs err {err:.3e} > {tol}"


# ----------------------------------------------------------------------------------------- 1, 2 attention core
def naive_attention(q, k, v, causal, window, key_valid, q2=None, k2=None, scale=1.0):
    Hq, Hk = q.shape[1], k.shape[1]
    G = Hq // Hk
    kk, vv = k.repeat_interleave(G, 1), v.repeat_interleave(G, 1)
    s = q @ kk.transpose(-1, -2)
    if q2 is not None:
        k2e = k2.repeat_interleave(Hq // k2.shape[1], 1)
        s = s + q2 @ k2e.transpose(-1, -2)
    s = s * scale
    Tq, Tk = q.shape[2], k.shape[2]
    m = torch.ones(Tq, Tk, dtype=torch.bool)
    if causal:
        m = torch.tril(m)
        if window:
            m = m & ~torch.tril(torch.ones(Tq, Tk, dtype=torch.bool), -window)
    m = m[None, None].expand(q.shape[0], 1, Tq, Tk)
    if key_valid is not None:
        m = m & key_valid[:, None, None, :]
    s = s.masked_fill(~m, -1e30)
    return torch.softmax(s, -1) @ vv


def test_attention_core():
    B, T, D = 3, 23, 8
    kv = torch.ones(B, T, dtype=torch.bool)
    kv[0, :5] = False
    kv[2, :2] = False
    cases = [(4, 4, True, None, None), (4, 2, True, None, None), (4, 1, True, None, None),
             (4, 2, True, 6, None), (4, 4, False, None, None), (4, 4, True, None, kv), (4, 2, True, 6, kv),
             (4, 4, False, None, kv)]
    for Hq, Hk, causal, window, key_valid in cases:
        q = torch.randn(B, Hq, T, D, dtype=torch.double, requires_grad=True)
        k = torch.randn(B, Hk, T, D, dtype=torch.double, requires_grad=True)
        v = torch.randn(B, Hk, T, 5, dtype=torch.double, requires_grad=True)
        w = torch.randn(B, Hq, T, 5, dtype=torch.double)
        # rows whose every key is masked (left-pad queries) are garbage by design - compare real rows only
        qm = (key_valid[:, None, :, None] if (causal and key_valid is not None)
              else torch.ones(B, 1, T, 1, dtype=torch.bool)).double()
        w = w * qm
        ref = naive_attention(q, k, v, causal, window, key_valid, scale=0.37)
        gref = torch.autograd.grad((ref * w).sum(), (q, k, v))
        for qchunk, ck in [(T, False), (7, False), (7, True)]:
            o = attend_blocks(q, k, v, 0.37, causal=causal, window=window, key_valid=key_valid, qchunk=qchunk, ckpt=ck)
            close(o * qm, ref * qm, 1e-9, f"fwd Hq={Hq} Hk={Hk} causal={causal} window={window} pad={key_valid is not None} qchunk={qchunk}")
            g = torch.autograd.grad((o * w).sum(), (q, k, v))
            for a, b in zip(g, gref):
                close(a, b, 1e-8, "grad")
    # two-term scores (MLA): second key shared by all heads
    q = torch.randn(B, 4, T, D, dtype=torch.double)
    k = torch.randn(B, 4, T, D, dtype=torch.double)
    v = torch.randn(B, 4, T, 6, dtype=torch.double)
    q2 = torch.randn(B, 4, T, 3, dtype=torch.double)
    k2 = torch.randn(B, 1, T, 3, dtype=torch.double)
    ref = naive_attention(q, k, v, True, None, None, q2, k2, 0.3)
    close(attend_blocks(q, k, v, 0.3, causal=True, q2=q2, k2=k2, qchunk=5), ref, 1e-9, "two-term scores")
    # MQA with value dim != key dim and one query (decode shape)
    q = torch.randn(B, 4, 1, 12, dtype=torch.double)
    k = torch.randn(B, 1, T, 12, dtype=torch.double)
    mask = kv[:, None, None, :]
    ref = naive_attention(q, k, k, False, None, kv, scale=0.2)
    close(attend(q, k, k, mask, 0.2), ref, 1e-9, "decode shape")


def test_gqa_no_copy():
    B, T, D = 2, 9, 6
    q = torch.randn(B, 6, T, D, dtype=torch.double)
    k = torch.randn(B, 2, T, D, dtype=torch.double)
    ref = q @ k.repeat_interleave(3, 1).transpose(-1, -2)
    orig = (torch.Tensor.repeat_interleave, torch.Tensor.expand)

    def boom(*a, **kw):
        raise AssertionError("K/V were replicated to match query heads")
    torch.Tensor.repeat_interleave = boom
    torch.Tensor.expand = boom
    try:
        s = _scores(q, k)
    finally:
        torch.Tensor.repeat_interleave, torch.Tensor.expand = orig
    close(s, ref, 1e-10, "grouped scores")


# ----------------------------------------------------------------------------------------- 3 gated deltanet
def test_gdn():
    B, H, T, Dk, Dv = 2, 3, 21, 8, 6
    for delta in (True, False):
        q = torch.randn(B, H, T, Dk, dtype=torch.double)
        k = torch.randn(B, H, T, Dk, dtype=torch.double)
        q = (q / q.norm(dim=-1, keepdim=True) * Dk ** -0.5).requires_grad_()
        k = (k / k.norm(dim=-1, keepdim=True)).requires_grad_()
        v = torch.randn(B, H, T, Dv, dtype=torch.double, requires_grad=True)
        beta = torch.rand(B, H, T, dtype=torch.double).requires_grad_()
        logal = (-torch.rand(B, H, T, dtype=torch.double) * 1.5).requires_grad_()
        S0 = torch.randn(B, H, Dv, Dk, dtype=torch.double, requires_grad=True)
        w1 = torch.randn(B, H, T, Dv, dtype=torch.double)
        w2 = torch.randn(B, H, Dv, Dk, dtype=torch.double)
        o_r, S_r = gdn_recurrent(q, k, v, beta, logal, S0, delta=delta)
        g_r = torch.autograd.grad((o_r * w1).sum() + (S_r * w2).sum(), (q, k, v, beta, logal, S0))
        for chunk in (8, 64, 5):
            o_c, S_c = gdn_chunked(q, k, v, beta, logal, S0, chunk=chunk, use_delta=delta)
            close(o_c, o_r, 1e-9, f"gdn out delta={delta} chunk={chunk}")
            close(S_c, S_r, 1e-9, f"gdn state delta={delta} chunk={chunk}")
            g_c = torch.autograd.grad((o_c * w1).sum() + (S_c * w2).sum(), (q, k, v, beta, logal, S0))
            for a, b, n in zip(g_c, g_r, "q k v beta logal S0".split()):
                close(a, b, 1e-8, f"gdn grad {n} delta={delta} chunk={chunk}")
    # strong decay (long chunk, alpha ~ e^-6 per step) stays finite in fp32
    q = torch.randn(1, 1, 128, 16)
    k = torch.nn.functional.normalize(torch.randn(1, 1, 128, 16), dim=-1)
    o, S = gdn_chunked(q, k, torch.randn(1, 1, 128, 16), torch.full((1, 1, 128), 0.9),
                       torch.full((1, 1, 128), -6.0), chunk=64)
    assert torch.isfinite(o).all() and torch.isfinite(S).all()


# ----------------------------------------------------------------------------------------- tiny models
def tiny_cfg(name, **over):
    base = dict(vocab=97, d=64, n_heads=4, head_dim=16, ffn=96, rope_len=256, qchunk=8, ckpt=True,
                train_len=32, shared_prefix=24)
    if name == "a":
        base.update(kind="encdec", n_enc=2, n_dec=2)
    elif name == "b":
        base.update(kind="decoder", layout=["mha"] * 3)
    elif name == "c":
        base.update(kind="decoder", layout=["gqa", "gqa", "gqa", "mha"], n_kv_heads=2, window=8)
    elif name == "d":
        base.update(kind="decoder", layout=["gdn", "gdn", "gdn", "mla"],
                    gdn=dict(n_heads=2, head_dim=16, conv=4, chunk=8, use_delta=True),
                    mla=dict(d_latent=24, d_nope=16, d_rope=8, d_v=16))
    elif name == "d_nodelta":
        base.update(kind="decoder", layout=["gdn", "mla", "gdn", "mla"],
                    gdn=dict(n_heads=2, head_dim=16, conv=4, chunk=8, use_delta=False),
                    mla=dict(d_latent=24, d_nope=16, d_rope=8, d_v=16))
    base.update(over)
    return base


def randomize(model, std=0.3):
    """Make every projection strongly non-trivial so tests are sensitive (init std 0.02 hides errors)."""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, std)
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, 0, 0.5)


def full_logits(model, seq, enc=None):
    """Reference: one un-padded sequence through the ordinary training-style forward -> logits at every position."""
    with torch.no_grad():
        if enc is None:
            T = seq.shape[1]
            h = model.trunk(seq, Ctx(torch.arange(T)))
        else:
            h = model._decode(seq, model.encode(enc))
        return model.logits(h)[0]


def test_no_leak(name):
    cfg = tiny_cfg(name)
    model = build_model(cfg).eval()
    randomize(model)
    T, cut = 20, 11
    if cfg["kind"] == "encdec":
        enc = torch.randint(4, 97, (1, 10))
        a = torch.randint(4, 97, (1, T))
        b = a.clone()
        b[:, cut:] = torch.randint(4, 97, (1, T - cut))
        la, lb = full_logits(model, a, enc), full_logits(model, b, enc)
        close(la[:cut], lb[:cut], 1e-5, "decoder leaks future tokens")
        assert (la[cut:] - lb[cut:]).abs().max() > 1e-3
        enc2 = enc.clone()
        enc2[:, -1] = (enc2[:, -1] + 1) % 97
        assert (full_logits(model, a, enc2)[0] - la[0]).abs().max() > 1e-6, "decoder ignores the encoder"
        # encoder is bidirectional: changing the LAST prefix token changes the encoder output at position 0
        e1, e2 = model.encode(enc), model.encode(enc2)
        assert (e1[0, 0] - e2[0, 0]).abs().max() > 1e-6, "encoder is not bidirectional"
    else:
        a = torch.randint(4, 97, (1, T))
        b = a.clone()
        b[:, cut:] = torch.randint(4, 97, (1, T - cut))
        la, lb = full_logits(model, a), full_logits(model, b)
        close(la[:cut], lb[:cut], 1e-5, "future-token leakage")
        assert (la[cut:] - lb[cut:]).abs().max() > 1e-3


def test_incremental(name):
    cfg = tiny_cfg(name)
    model = build_model(cfg).eval()
    model.cache_dtype = torch.float32
    randomize(model)
    gen = 12
    lens = [20, 13, 17]
    if cfg["kind"] == "encdec":
        prompts = [torch.randint(4, 97, (L,)) for L in lens]
        tgts = [torch.randint(4, 97, (gen,)) for _ in lens]
        refs = []
        for p, t in zip(prompts, tgts):
            dec_in = torch.cat([torch.tensor([BOS_ID]), t[:-1]])[None]
            refs.append(full_logits(model, dec_in, p[None]))               # (gen, V): predicts t[0..gen-1]
        P = max(lens)
        ids = torch.zeros(len(lens), P, dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, P - len(p):] = p
        padlen = torch.tensor([P - L for L in lens])
        logits, st = model.begin(ids, padlen, gen)
        for i in range(len(lens)):
            close(logits[i], refs[i][0], 2e-4, f"{name} step 0 seq {i}")
        for j in range(gen - 1):
            logits = model.advance(torch.stack([t[j] for t in tgts]), st)
            for i in range(len(lens)):
                close(logits[i], refs[i][j + 1], 2e-4, f"{name} step {j + 1} seq {i}")
    else:
        seqs = [torch.randint(4, 97, (L + gen,)) for L in lens]
        refs = [full_logits(model, s[None]) for s in seqs]
        P = max(lens)
        ids = torch.zeros(len(lens), P, dtype=torch.long)
        for i, (s, L) in enumerate(zip(seqs, lens)):
            ids[i, P - L:] = s[:L]
        padlen = torch.tensor([P - L for L in lens])
        logits, st = model.begin(ids, padlen, gen)
        for i, L in enumerate(lens):
            close(logits[i], refs[i][L - 1], 2e-4, f"{name} prefill seq {i}")
        for j in range(gen - 1):
            tok = torch.stack([s[L + j] for s, L in zip(seqs, lens)])
            logits = model.advance(tok, st)
            for i, L in enumerate(lens):
                close(logits[i], refs[i][L + j], 2e-4, f"{name} decode step {j} seq {i}")


def test_mla_absorbed_equals_plain():
    """Direct check of the weight-absorption identity inside one MLA layer (decode path vs plain path)."""
    from attn_mla import MLA
    from layers import RotaryEmbedding
    rope = RotaryEmbedding(8, 64)
    m = MLA(32, 4, 24, 16, 8, 16, rope).eval()
    randomize(m, 0.3)
    x = torch.randn(2, 10, 32)
    with torch.no_grad():
        plain = m(x, Ctx(torch.arange(10)))
        cache = m.make_cache(2, 10, torch.device("cpu"), torch.float32)
        outs = []
        for t in range(10):
            outs.append(m(x[:, t:t + 1], Ctx(torch.full((2, 1), t), decode=True), cache))
    close(torch.cat(outs, 1), plain, 3e-4, "MLA absorbed decode vs plain")


def test_train_and_grads(name):
    cfg = tiny_cfg(name)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, betas=(0.9, 0.95), weight_decay=0.0)
    win = torch.randint(4, 97, (4, 33))
    first = last = None
    for it in range(60):
        opt.zero_grad()
        loss = model(win, (16, 20, 24)[it % 3] if cfg["kind"] == "encdec" else 0)
        loss.backward()
        if it == 0:
            first = loss.item()
            bad = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
            assert not bad, f"no/non-finite grad: {bad}"
            dead = [n for n, p in model.named_parameters() if p.grad.abs().max() == 0]
            assert not dead, f"all-zero grad: {dead}"
        opt.step()
        last = loss.item()
    assert first > math.log(97) - 0.5, f"initial loss {first} should be ~ln(V)={math.log(97):.2f}"
    assert last < 0.85 * first, f"loss did not fall: {first:.3f} -> {last:.3f}"


def test_shared_metric(name):
    cfg = tiny_cfg(name)
    model = build_model(cfg).eval()
    randomize(model, 0.1)
    win = torch.randint(4, 97, (3, 33))
    s, n = model.shared_metric(win)
    assert n == 3 * (32 - 24)
    pref = 24
    with torch.no_grad():
        if cfg["kind"] == "encdec":
            dec_in = torch.cat([torch.full((3, 1), BOS_ID), win[:, pref:31]], 1)
            lg = model.logits(model._decode(dec_in, model.encode(win[:, :pref])))
            tgt = win[:, pref:32]
        else:
            lg = model.logits(model.trunk(win[:, :32], Ctx(torch.arange(32))))[:, pref - 1:31]
            tgt = win[:, pref:32]
        ref = torch.nn.functional.cross_entropy(lg.reshape(-1, 97), tgt.reshape(-1), reduction="sum")
    close(s, ref, 1e-3, "shared metric")


def test_no_builtin_attention():
    import ast
    here = os.path.dirname(os.path.abspath(__file__))
    bad = {"scaled_dot_product_attention", "MultiheadAttention", "Transformer", "TransformerEncoderLayer",
           "TransformerDecoderLayer", "TransformerEncoder", "TransformerDecoder", "flash_attn", "xformers"}
    for f in (glob.glob(os.path.join(here, "*.py")) + glob.glob(os.path.join(here, "..", "common", "*.py"))):
        if os.path.basename(f) == "tests.py":
            continue
        tree = ast.parse(open(f, encoding="utf-8").read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.ImportFrom):
                names += [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            for n in names:
                assert n.split(".")[0] not in bad and n not in bad, f"{os.path.basename(f)} uses {n}"


def test_sampler():
    """Without-replacement sampler: epoch covers every window exactly once, ranks are disjoint, state resumes."""
    import numpy as np
    from sampler import EpochSampler

    class D:
        pass
    d = D()
    d.sources, d.weights = ["a", "b"], np.array([0.5, 0.5])
    d.bin = {"a": np.repeat(np.arange(10), 1024).astype(np.int64), "b": np.repeat(np.arange(100, 106), 1024).astype(np.int64)}
    d.bin = {k: np.concatenate([v, [v[-1]]]) for k, v in d.bin.items()}        # +1 target token -> 10 and 6 windows
    rs = [EpochSampler(d, 1, 1, r, 2) for r in (0, 1)]
    seen = {"a": [], "b": []}
    for _ in range(60):
        for s in rs:
            ep = dict(s.epoch)                                   # epoch BEFORE this draw
            v = int(s.next_batch()[0, 0])
            src = "a" if v < 100 else "b"
            seen[src].append((v, ep[src]))
    # within one epoch every window appears at most once across BOTH ranks; epoch 0 of the two ranks covers
    # exactly the windows of the first pass
    for src, n in (("a", 10), ("b", 6)):
        for e in (0, 1):
            ws = [v for v, ep in seen[src] if ep == e]
            assert len(ws) == len(set(ws)), f"source {src} epoch {e}: window repeated {sorted(ws)}"
        w0 = {v for v, ep in seen[src] if ep == 0}
        assert w0 <= (set(range(10)) if src == "a" else set(range(100, 106)))
    st = rs[0].snapshot()
    r2 = EpochSampler(d, 1, 1, 0, 2, state=st)
    assert r2.ptr == rs[0].ptr and r2.epoch == rs[0].epoch


def test_configs():
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    for k in "abcde":
        cfg = json.load(open(os.path.join(here if os.path.isdir(os.path.join(here, "configs")) else os.path.join(here, "..", "common"), "configs", f"{k}.json")))
        m = cfg["model"]
        assert m["d"] % m["n_heads"] == 0 or m["kind"] == "decoder"
        assert (m["d"] // 1) > 0 and m["head_dim"] % 2 == 0
        assert 131072 % (cfg["train"]["micro_batch"] * 2 * 1024) == 0, "micro_batch*2*1024 must divide 131072"


def test_full():
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    for k in "bcdea":
        cfg = json.load(open(os.path.join(here if os.path.isdir(os.path.join(here, "configs")) else os.path.join(here, "..", "common"), "configs", f"{k}.json")))["model"]
        model = build_model(cfg).eval()
        n = model.count_params()
        print(f"   model {k}: {n / 1e6:.2f}M parameters")
        assert 50e6 <= n <= 300e6, "outside 50M-300M"
        win = torch.randint(4, 32768, (1, 1025))
        t0 = time.time()
        with torch.no_grad():
            loss = model(win, 512).item()
        print(f"   model {k}: init loss {loss:.3f} (ln V = {math.log(32768):.3f})  forward {time.time() - t0:.0f}s")
        assert abs(loss - math.log(32768)) < 0.6
        del model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    check("attention core == naive reference (all variants, grads, blocks, checkpointing)", test_attention_core)
    check("grouped-query scores without repeated K/V", test_gqa_no_copy)
    check("Gated DeltaNet chunked == recurrent (outputs, state, gradients)", test_gdn)
    check("MLA absorbed decode == plain forward", test_mla_absorbed_equals_plain)
    for n in ("a", "b", "c", "d", "d_nodelta"):
        check(f"model {n}: no future-token leakage", lambda n=n: test_no_leak(n))
    for n in ("a", "b", "c", "d", "d_nodelta"):
        check(f"model {n}: cached decoding == full forward (left-padded batch)", lambda n=n: test_incremental(n))
    for n in ("a", "b", "c", "d"):
        check(f"model {n}: gradients reach every parameter; tiny overfit", lambda n=n: test_train_and_grads(n))
    for n in ("a", "b"):
        check(f"model {n}: shared metric", lambda n=n: test_shared_metric(n))
    check("no built-in attention used in model code", test_no_builtin_attention)
    check("without-replacement sampler (coverage, rank disjointness, resume)", test_sampler)
    check("configs a-e are consistent", test_configs)
    if args.full:
        check("real configs: 50M-300M params, init loss ~ ln V", test_full)
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} tests passed" + ("" if not bad else "   FAILED: " + "; ".join(bad)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
