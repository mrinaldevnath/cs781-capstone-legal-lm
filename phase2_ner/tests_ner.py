"""
tests_ner.py - correctness tests for the Phase-2 code (CPU, ~1 min). Run before any GPU time.

    python tests_ner.py                                              # model/PEFT/decoding tests (no data)
    python tests_ner.py --tok_root <phase-1 data> --data_dir <p2>   # + tokenizer offsets, alignment ceiling,
                                                                     #   and a 2-epoch end-to-end run on 4 docs
Checks
  1  LoRA / LoRMA+ / Adapter: identical output at init, LoRMA+ == explicit (I + s B A) W0 x, only new params train
  2  configure_trainable: full / partial / norm / lora / lorma / adapter freeze exactly what they should
  3  NERModel on tiny A/B/C/D backbones: shapes, A encoder is bidirectional, padding never changes real tokens,
     causal backbones + linear head see no future tokens, BiLSTM head sees both sides
  4  BIO decoding: viterbi never emits an invalid I-, greedy repair, decode(gold tags) == gold spans
  5  scorer: hand-made example with known macro-F1
  6  (data) tokens tile every document exactly, decode == text, alignment ceiling >= 0.99
  7  (data) end-to-end: tiny backbone trained 2 epochs on 4 docs, writes a valid submission
  8  no built-in attention in the new files
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import argparse
import ast
import glob
import json
import tempfile
import time
import traceback

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ner_data as N
import ner_eval as E
from models import build_model
from ner_model import Backbone, NERModel
from peft import LoRALinear, LoRMALinear, AdapterLinear, configure_trainable

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


def tiny_cfg(name, vocab=97):
    base = dict(vocab=vocab, d=32, n_heads=4, head_dim=8, ffn=48, rope_len=512, qchunk=8, ckpt=True)
    if name == "a":
        base.update(kind="encdec", n_enc=2, n_dec=2)
    elif name == "b":
        base.update(kind="decoder", layout=["mha"] * 2)
    elif name == "c":
        base.update(kind="decoder", layout=["gqa", "mha"], n_kv_heads=2, window=4)
    elif name == "d":
        base.update(kind="decoder", layout=["gdn", "mla"], gdn=dict(n_heads=2, head_dim=16, conv=4, chunk=8, use_delta=True),
                    mla=dict(d_latent=16, d_nope=8, d_rope=4, d_v=8))
    return base


def tiny_backbone(name, vocab=97):
    cfg = tiny_cfg(name, vocab)
    lm = build_model(cfg)
    for m in lm.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.3)
    return Backbone(cfg=cfg, state=lm.state_dict())


# ------------------------------------------------------------------------------------------------ 1
def test_wrappers():
    base = nn.Linear(12, 20, bias=False)
    x = torch.randn(3, 5, 12)
    y0 = base(x)
    for cls in (LoRALinear, LoRMALinear):
        m = cls(base, 4, 8, 0.0)
        assert torch.allclose(m(x), y0), cls.__name__
    a = AdapterLinear(base, 6)
    assert torch.allclose(a(x), y0)
    m = LoRMALinear(base, 4, 8, 0.0)
    with torch.no_grad():
        m.B.normal_()
    W0 = base.weight.detach()
    explicit = x @ ((torch.eye(20) + m.scale * m.B @ m.A) @ W0).T
    assert torch.allclose(m(x), explicit, atol=1e-5), "LoRMA+ != (I + sBA) W0 x"
    l = LoRALinear(base, 4, 8, 0.0)
    with torch.no_grad():
        l.B.normal_()
    assert torch.allclose(l(x), x @ (W0 + l.scale * l.B @ l.A).T, atol=1e-5), "LoRA != (W0 + sBA) x"


# ------------------------------------------------------------------------------------------------ 2
def test_configure():
    for name in "abcd":
        for method in ("full", "partial", "norm", "lora", "lorma", "adapter"):
            bb = tiny_backbone(name)
            x = torch.randint(4, 97, (2, 16))
            ref = bb(x).detach()
            n_tr, n_all, n_wr = configure_trainable(bb, method, r=4, alpha=8, bottleneck=4, top_k=1)
            if method in ("lora", "lorma", "adapter"):
                assert n_wr > 0
                assert torch.allclose(bb(x), ref, atol=1e-5), f"{name}/{method}: init changes the output"
            if method == "full":
                assert n_tr == n_all
            else:
                assert 0 < n_tr < n_all, (name, method, n_tr, n_all)
            assert not bb.lm.emb.weight.requires_grad or method == "full"
            model = NERModel(bb, 7, "bilstm")
            out = model(x)
            out.sum().backward()
            for n_, p in model.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None, f"{name}/{method}: no grad for {n_}"
                else:
                    assert p.grad is None, f"{name}/{method}: frozen {n_} got a grad"


# ------------------------------------------------------------------------------------------------ 3
def test_model_behaviour():
    for name in "abcd":
        bb = tiny_backbone(name).eval()
        T = 20
        x = torch.randint(4, 97, (1, T))
        for head in ("linear", "bilstm"):
            m = NERModel(bb, 7, head).eval()
            with torch.no_grad():
                o = m(x)
                assert o.shape == (1, T, 7)
                x2 = x.clone()
                x2[0, -1] = (x2[0, -1] + 1) % 97
                o2 = m(x2)
                changed_first = (o[0, 0] - o2[0, 0]).abs().max().item() > 1e-6
                if name == "a" or head == "bilstm":
                    assert changed_first, f"{name}/{head}: first token ignores the last token (no right context)"
                else:
                    assert torch.allclose(o[0, :-1], o2[0, :-1], atol=1e-5), f"{name}/{head}: future-token leakage"
                # right padding never changes real positions
                xp = torch.cat([x, torch.zeros(1, 7, dtype=torch.long)], 1)
                vp = torch.zeros(1, T + 7, dtype=torch.bool)
                vp[0, :T] = True
                op = m(xp, vp)
                assert torch.allclose(op[0, :T], o[0], atol=1e-4), f"{name}/{head}: padding changes real tokens " \
                    f"{(op[0, :T] - o[0]).abs().max():.2e}"


# ------------------------------------------------------------------------------------------------ 4
def test_decoding():
    labs = ["X", "Y"]
    rng = np.random.default_rng(0)
    for _ in range(50):
        lp = np.log(rng.dirichlet(np.ones(5), size=30))
        p = N.viterbi(lp)
        assert p[0] % 2 == 1 or p[0] == 0
        for i in range(1, 30):
            if p[i] > 0 and p[i] % 2 == 0:
                assert p[i - 1] in (p[i] - 1, p[i]), p
        g = N.greedy(lp)
        for i in range(30):
            if g[i] > 0 and g[i] % 2 == 0:
                assert i > 0 and g[i - 1] in (g[i] - 1, g[i])

    class FakeDoc:
        pass
    d = FakeDoc()
    # tokens: "Ram"(0-3) " "(3-4 ws) "Kumar, "(4-11 core 4-9) "the"(11-14)
    d.st, d.en, d.ce = np.array([0, 3, 4, 11]), np.array([3, 4, 11, 14]), np.array([3, 3, 9, 14])
    d.ids = np.zeros(4, dtype=np.int64)
    tags = N.bio_tags(d, [[0, 9, "Y"]], N.tag_ids(labs))
    assert tags.tolist() == [3, 4, 4, 0], tags
    assert N.decode_spans(d, tags, labs) == [[0, 9, "Y"]]


def test_empty_doc():
    import ner_train

    class D:
        id = "x"
        text = ""
        ids = np.zeros(0, dtype=np.int64)

        def __len__(self):
            return 0
    assert ner_train.spans_from_logp([D()], [np.zeros((0, 7), dtype=np.float32)], ["A", "B", "C"], "viterbi") == {"x": []}
    m = NERModel(tiny_backbone("b"), 7, "bilstm").eval()
    lp = ner_train.predict_logp(m, [D()], 16, 4, "center", 7, torch.device("cpu"))
    assert lp[0].shape == (0, 7)


# ------------------------------------------------------------------------------------------------ 5
def test_scorer():
    labs = ["A", "B"]
    gold = {"1": [[0, 5, "A"], [10, 15, "B"]], "2": [[3, 4, "A"]]}
    pred = {"1": [[0, 5, "A"], [10, 14, "B"]], "2": [[3, 4, "B"]]}
    r = E.score(gold, pred, labs)
    # A: tp1 fp0 fn1 -> P1 R.5 F.667 ; B: tp0 -> 0
    assert abs(r["macro_f1"] - (2 / 3) / 2) < 1e-9, r["macro_f1"]
    ea = E.error_analysis(gold, pred)["categories"]
    assert ea == {"boundary": 1, "label": 1}, ea
    E.validate_submission({"1": [[0, 5, "A"]], "2": []}, ["1", "2"], labs)


# ------------------------------------------------------------------------------------------------ 6, 7
def test_data(tok_root, data_dir):
    labs, tags = N.load_labels(os.path.join(data_dir, "labels.txt"))
    tag_of = N.tag_ids(labs)
    texts, gold = N.load_folds(data_dir)
    tok = N.load_tokenizer(tok_root)
    for mode in ("piece", "native"):
        ot = N.OffsetTokenizer(tok, mode)
        docs = [N.Doc(k, texts[k], ot, gold[k], tag_of) for k in sorted(texts)]
        for d in docs:
            assert d.st[0] == 0 and d.en[-1] == len(d.text), d.id
            assert (d.st[1:] <= d.en[:-1]).all() and (d.st[1:] >= d.st[:-1]).all(), d.id
            assert tok.decode(d.ids.tolist()) == d.text, f"decode != text for {d.id}"
        pred = {d.id: N.decode_spans(d, d.tags, labs) for d in docs}
        r = E.score(gold, pred, labs)
        print(f"      {mode}: {sum(len(d) for d in docs):,} tokens, alignment ceiling macro-F1 {r['macro_f1']:.4f}")
        if mode == "piece":
            assert r["macro_f1"] > 0.99, r["macro_f1"]


def test_e2e(tok_root, data_dir):
    import ner_train
    tok = N.load_tokenizer(tok_root)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = tiny_cfg("b", vocab=tok.vocab_size)
        lm = build_model(cfg)
        ck = os.path.join(tmp, "tiny.pt")
        torch.save({"model": lm.state_dict(), "cfg": cfg}, ck)
        texts = N.load_texts(os.path.join(data_dir, "fold1-text.json"))
        test = {k: texts[k][:3000] for k in list(texts)[:3]}
        tf = os.path.join(tmp, "test.json")
        json.dump(test, open(tf, "w"))
        out = os.path.join(tmp, "run")
        ner_train.main(["--ckpt", ck, "--tok_root", tok_root, "--data_dir", data_dir, "--test_file", tf, "--out", out,
                        "--method", "lorma", "--max_docs", "6", "--dev_frac", "0.34", "--epochs", "2", "--window", "64",
                        "--bs", "4", "--cpu", "--rank", "2", "--prefix", "8"])
        s = json.load(open(os.path.join(out, "summary.json")))
        assert "dev_viterbi" in s and os.path.exists(os.path.join(out, "submission.zip"))
        pred = json.load(open(os.path.join(out, "prediction.json")))
        assert set(pred) == set(test)


# ------------------------------------------------------------------------------------------------ 8
def test_no_builtin_attention():
    here = os.path.dirname(os.path.abspath(__file__))
    bad = {"scaled_dot_product_attention", "MultiheadAttention", "Transformer", "TransformerEncoderLayer",
           "TransformerDecoderLayer", "TransformerEncoder", "TransformerDecoder", "flash_attn", "xformers"}
    for f in (glob.glob(os.path.join(here, "*.py")) + glob.glob(os.path.join(here, "..", "common", "*.py"))):
        if os.path.basename(f).startswith("tests"):
            continue
        for node in ast.walk(ast.parse(open(f, encoding="utf-8").read())):
            names = [node.attr] if isinstance(node, ast.Attribute) else [node.id] if isinstance(node, ast.Name) else []
            for n in names:
                assert n not in bad, f"{os.path.basename(f)} uses {n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok_root", default=None)
    ap.add_argument("--data_dir", default=None)
    a = ap.parse_args()
    check("LoRA / LoRMA+ / Adapter wrappers (identity at init, exact formulas)", test_wrappers)
    check("configure_trainable for all methods on A/B/C/D (freezing, grads, identity at init)", test_configure)
    check("NERModel: bidirectional A, causal leakage, BiLSTM right context, padding invariance", test_model_behaviour)
    check("BIO tagging / viterbi / greedy / span decoding", test_decoding)
    check("empty document (no tokens) is handled in prediction", test_empty_doc)
    check("strict macro-F1 scorer + submission validator", test_scorer)
    check("no built-in attention in Phase-2 code", test_no_builtin_attention)
    if a.tok_root and a.data_dir:
        check("tokenizer offsets tile every document; alignment ceiling", lambda: test_data(a.tok_root, a.data_dir))
        check("end-to-end tiny run (train, dev eval, test submission)", lambda: test_e2e(a.tok_root, a.data_dir))
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} tests passed" + ("" if not bad else "   FAILED: " + "; ".join(bad)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
