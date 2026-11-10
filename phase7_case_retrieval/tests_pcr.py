"""CPU unit tests for Phase 7. Usage: python tests_pcr.py [tok_root]   (model tests need torch; skipped without it)"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import json
import math
import tempfile

import numpy as np

import pcr_data as D
import pcr_lex as LX


def t_records():
    a = [{"id": 1, "text": ["A b.", "C <CITATION> d."], "relevant_candidates": [2, "3"]}]
    r = list(D.records(a))
    assert r == [("1", ["A b.", "C <CITATION> d."], ["2", "3"])], r
    b = {"7": {"text": "one two"}, "8": {"text": ["x"]}}
    assert [x[0] for x in D.records(b)] == ["7", "8"] and list(D.records(b))[0][1] == ["one two"]
    c = {"9": ["s1", "s2"]}
    assert list(D.records(c)) == [("9", ["s1", "s2"], None)]
    assert D.words("The Court in <CITATION> held, (1978) 2 S.C.R. 621.") == \
        ["the", "court", "in", "citx", "held", "1978", "2", "s.c.r", "621"], D.words("The Court in <CITATION> held, (1978) 2 S.C.R. 621.")
    s = ["a", "b <CITATION>", "c", "d", "e <CITATION> f", "g", "CITATION: 1966 AIR 1053"]
    assert D.cite_context(s) == ["a", "b <CITATION>", "c", "d", "e <CITATION> f", "g"]
    assert D.cite_context(["x", "y", "z <CITATION>"], 0) == ["z <CITATION>"]
    assert D.est_year(["decided on 12 March, 1996 relying on a 1978 case", "1800 2099"]) == 1996
    assert D.est_year(["PETITIONER: X DATE OF JUDGMENT: 08/12/1965 BENCH: Y CITATOR INFO : R 1986 SC1483 (4) ACT: 1922"]) == 1965
    assert D.est_year(["X CITATION: 1966 AIR 1053 CITATOR INFO : R 1986 SC1483 (4) ACT: Indian Act 1922"]) == 1966


def t_metric():
    S = np.array([[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], float)
    cids = [f"c{j}" for j in range(10)]
    pred = D.topk(S, cids, ["c0", "q"])
    assert pred[0] == ["c1", "c2", "c3", "c4", "c5", "c6", "c7"], pred[0]        # the query itself (c0) is excluded
    assert pred[1] == ["c9", "c8", "c7", "c6", "c5", "c4", "c3"]
    m = D.micro_prf(pred, [["c1", "c9", "x"], ["c9"]])
    assert m["tp"] == 2 and abs(m["p"] - 2 / 14) < 1e-9 and abs(m["r"] - 2 / 4) < 1e-9
    with tempfile.TemporaryDirectory() as d:
        D.write_prediction(os.path.join(d, "p.json"), ["c0", "q"], pred, cids)
        assert json.load(open(os.path.join(d, "p.json")))["q"][0] == "c9"


def t_bm25():
    rng = np.random.default_rng(0)
    docs = [rng.integers(0, 30, size=rng.integers(5, 40)).astype(np.int32) for _ in range(25)]
    qs = [rng.integers(0, 40, size=rng.integers(3, 30)).astype(np.int32) for _ in range(6)]
    for n in (1, 2, 3):
        old = LX.DENSE_COLS
        for dc in (0, 5, 100000):
            LX.DENSE_COLS = dc
            S = LX.bm25_scores(LX.count_matrix(qs, n), LX.count_matrix(docs, n))
            # brute force
            grams = lambda s: [tuple(s[i:i + n]) for i in range(len(s) - n + 1)]
            G = [grams(d) for d in docs]
            N = len(docs)
            avg = np.mean([len(g) for g in G])
            df = {}
            for g in G:
                for t in set(g):
                    df[t] = df.get(t, 0) + 1
            for i, q in enumerate(qs):
                for j, g in enumerate(G):
                    sc = 0.0
                    for t in grams(q):
                        if t in df:
                            tf = g.count(t)
                            idf = math.log1p((N - df[t] + .5) / (df[t] + .5))
                            sc += idf * tf * (LX.K1 + 1) / (tf + LX.K1 * (1 - LX.B + LX.B * len(g) / avg))
                    assert abs(sc - S[i, j]) < 1e-3 * max(1, abs(sc)), (n, dc, i, j, sc, S[i, j])
        LX.DENSE_COLS = old


def t_tokenizer(tok_root):
    D._init_tok(tok_root)
    ids = D._bpe("The appellant was convicted under Section 302 IPC.\nThe High Court affirmed.")
    assert 5 < len(ids) < 60 and ids.dtype == np.uint16, ids
    txt = D._TOK.decode([int(i) for i in ids])
    assert "Section 302" in txt and "affirmed" in txt, txt


def t_models():
    import torch
    import pcr_model as PM
    here = os.path.dirname(os.path.abspath(__file__))
    torch.manual_seed(0)
    for name in "abcde":
        cfg = json.load(open(os.path.join(here if os.path.isdir(os.path.join(here, "configs")) else os.path.join(here, "..", "common"), "configs", f"{name}.json")))["model"]
        cfg.update(d=64, n_heads=4, head_dim=16, ffn=128, rope_len=256, qchunk=8)
        if "layout" in cfg:
            cfg["layout"] = cfg["layout"][-4:] if name in "cd" else cfg["layout"][:2]
        if "mla" in cfg:
            cfg["mla"] = dict(d_latent=32, d_nope=16, d_rope=8, d_v=16)
        if "gdn" in cfg:
            cfg["gdn"].update(n_heads=2, head_dim=32, chunk=8)
        if name == "c":
            cfg.update(n_kv_heads=2, window=8)
        if name == "a":
            cfg.update(n_enc=2, n_dec=2)
        ids = torch.randint(5, 500, (3, 20))
        valid = torch.ones(3, 20, dtype=torch.bool)
        valid[1, 13:] = False
        valid[2, 6:] = False
        for method in ("full", "partial", "lorma", "ia3"):
            bb = PM.Backbone(cfg=cfg, state=None)
            bb.eval()
            with torch.no_grad():
                h0 = bb(ids, valid)
            n_tr, n_all, n_w = PM.configure(bb, method, top_k=1)
            bb.eval()
            with torch.no_grad():
                h1 = bb(ids, valid)
            assert torch.allclose(h0, h1, atol=1e-5), (name, method, (h0 - h1).abs().max())   # identity at init
            assert 0 < n_tr <= n_all and (method == "full") == (n_tr == n_all), (name, method, n_tr, n_all)
            if method in ("lorma", "ia3"):
                assert n_w > 0 and n_tr < 0.1 * n_all, (name, method, n_w, n_tr, n_all)
            r = PM.Retriever(bb)
            r.eval()
            with torch.no_grad():
                e = r.encode_chunks(ids, valid)
                e1 = r.encode_chunks(ids[1:2, :13], valid[1:2, :13])                         # no padding
            assert e.shape == (3, 64) and torch.allclose(e.norm(dim=-1), torch.ones(3), atol=1e-4)
            assert torch.allclose(e[1], e1[0], atol=1e-4), (name, method, (e[1] - e1[0]).abs().max())
            # training path with block checkpointing
            r.train()
            bb.block_ckpt = True
            e = r.encode_chunks(ids, valid)
            v = PM.doc_vectors(e, torch.tensor([0, 0, 1]), 2)
            loss = -(v[0] @ v[1])
            loss.backward()
            g = [p.grad for p in r.parameters() if p.requires_grad]
            assert all(x is not None for x in g) and any(float(x.abs().sum()) > 0 for x in g), (name, method)
            bb.block_ckpt = False
        print(f"  model {name}: full/partial/lorma/ia3 OK")


if __name__ == "__main__":
    t_records(); print("records OK")
    t_metric(); print("metric OK")
    t_bm25(); print("bm25 OK")
    if len(sys.argv) > 1:
        t_tokenizer(sys.argv[1]); print("tokenizer OK")
    try:
        import torch  # noqa: F401
        have_torch = True
    except ImportError:
        have_torch = False
    if have_torch:
        t_models(); print("models OK")
    print("ALL OK")
