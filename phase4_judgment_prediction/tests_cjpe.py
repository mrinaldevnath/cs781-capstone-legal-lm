"""tests_cjpe.py - CPU unit tests for the Phase-4 data and metric code (no torch). Run: python tests_cjpe.py"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import json
import tempfile

import numpy as np

import cjpe_data as D
import cjpe_eval as E


def test_chunks():
    b = D.chunk_bounds(1100, 512)
    assert b == [(0, 76), (76, 588), (588, 1100)], b                     # end-aligned, last chunk full
    assert D.chunk_bounds(0, 512) == [(0, 0)]
    assert D.select_chunks(1100, 512, 2, "last") == [(76, 588), (588, 1100)]
    assert D.select_chunks(1100, 512, 2, "first") == [(0, 512), (512, 1024)]
    assert D.select_chunks(300, 512, 2, "first") == [(0, 300)]
    fl = D.select_chunks(5000, 512, 2, "firstlast")
    assert fl == [(0, 512), (4488, 5000)], fl
    assert D.select_chunks(700, 512, 2, "firstlast") == D.chunk_bounds(700, 512)   # short doc: whole document
    sp = D.select_chunks(512 * 20, 512, 8, "spread")
    assert len(sp) == 8 and sp[0] == (0, 512) and sp[-1] == (512 * 19, 512 * 20)
    assert all(e - s <= 512 for s, e in sp) and sp == sorted(sp)
    for how in D.SELECTS:                                                 # never more than k, never overlapping
        for n in (0, 1, 511, 512, 513, 2047, 9000):
            s = D.select_chunks(n, 512, 4, how)
            assert 1 <= len(s) <= 4, (how, n, s)
            assert all(s[i][1] <= s[i + 1][0] for i in range(len(s) - 1)), (how, n, s)


def test_batch():
    tok = np.arange(1, 1201, dtype=np.uint16)
    pieces = [D.doc_chunks(tok, 512, 2, "last"), D.doc_chunks(tok[:100], 512, 2, "last"), D.doc_chunks(tok[:0], 512, 2, "last")]
    ids, valid, doc, slot, rel = D.batch_arrays(pieces, 512)
    assert ids.shape == (4, 512) and doc.tolist() == [0, 0, 1, 2] and slot.tolist() == [0, 1, 0, 0]
    assert valid.sum(1).tolist() == [512, 512, 100, 1]
    assert ids[1, -1] == 1200 and ids[0, 0] == 1200 - 1024 + 1
    assert 0 < rel[0] < rel[1] <= 1
    mb = D.micro_batches([D.doc_chunks(tok, 512, 2, "last")] * 5, 4)
    assert mb == [[0, 1], [2, 3], [4]], mb


def test_metric():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.integers(0, 2, 200)
        p = rng.integers(0, 2, 200)
        r = E.binary_report(y, p)
        try:
            from sklearn.metrics import f1_score, roc_auc_score
            assert abs(r["macro_f1"] - f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)) < 1e-12
            s = rng.random(200).round(2)                                  # ties included
            assert abs(E.auc(y, s) - roc_auc_score(y, s)) < 1e-12
        except ImportError:
            pass
    assert E.binary_report([1, 1], [1, 1])["macro_f1"] == 0.5              # absent class -> F1 0 (sklearn zero_division=0)
    t, f = E.best_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4]))
    assert f == 1.0 and 0.2 < t <= 0.3, (t, f)


def test_files():
    with tempfile.TemporaryDirectory() as d:
        rows = [{"id": "a", "text": "x\ny", "label": 1}, {"id": "b", "text": "z", "label": 0}, {"id": "a", "text": "x \n y", "label": 1}]
        p = os.path.join(d, "dev.json")
        json.dump(rows, open(p, "w"))
        s = D.load_split(p)
        assert [x["id"] for x in s] == ["a", "b"]                         # exact duplicate removed
        json.dump(rows + [{"id": "b", "text": "z", "label": 1}], open(p, "w"))
        try:
            D.load_split(p)
            raise AssertionError("conflicting duplicate not detected")
        except ValueError:
            pass
        assert D.clean_text("ab\ncd  \n e") == "ab cd e"
        out = os.path.join(d, "prediction.json")
        D.write_prediction(["a", "b"], np.array([1, 0]), out)
        assert json.load(open(out)) == [{"id": "a", "pred": 1}, {"id": "b", "pred": 0}]


def test_cache():
    """build_cache with a stand-in tokenizer: overlap removal, offsets, labels, round trip."""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "Train")); os.makedirs(os.path.join(d, "Test"))
        tr = [{"id": f"t{i}", "text": " ".join(["w"] * (i + 1)), "label": i % 2} for i in range(6)]
        tr.append({"id": "d0", "text": "dev text", "label": 1})                         # same id as a dev case
        tr.append({"id": "t9", "text": "test\ncase", "label": 0})                       # same text as a test case
        json.dump(tr, open(os.path.join(d, "Train", "train.json"), "w"))
        json.dump([{"id": "d0", "text": "dev text", "label": 1}] * 2, open(os.path.join(d, "Train", "dev.json"), "w"))
        json.dump([{"id": "x0", "text": "test case"}], open(os.path.join(d, "Test", "test.json"), "w"))
        real = D.tokenize_all
        D.tokenize_all = lambda texts, root, procs=None: [np.array([len(w) for w in D.clean_text(t).split()], np.uint16) for t in texts]
        try:
            info = D.build_cache(d, None, os.path.join(d, "c.npz"), log=lambda s: None)
        finally:
            D.tokenize_all = real
        assert info["train"]["dropped_overlap_with_dev_test"] == 2, info
        assert info["dev"] == {**info["dev"], "rows_in_file": 2, "unique_ids": 1, "docs": 1}
        c = D.Cache(os.path.join(d, "c.npz"))
        assert c.n("train") == 6 and c.n("dev") == 1 and c.n("test") == 1
        assert c.ids("train") == [f"t{i}" for i in range(6)] and c.y("train").tolist() == [0, 1, 0, 1, 0, 1]
        assert len(c.doc("train", 3)) == 4 and c.y("test").tolist() == [-1]
        assert c.lengths("train").tolist() == [1, 2, 3, 4, 5, 6]


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
    print("ALL TESTS PASSED")
