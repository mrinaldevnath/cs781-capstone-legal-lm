"""tests_lsi.py - CPU unit tests for the Phase-5 data and metric code (no torch). Run: python tests_lsi.py"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import json
import tempfile

import numpy as np

import lsi_data as D
import lsi_eval as E


def test_chunks():
    assert D.chunk_bounds(1100, 512) == [(0, 76), (76, 588), (588, 1100)]
    assert D.select_chunks(1100, 512, 4, "spread") == D.chunk_bounds(1100, 512)          # short doc: read whole
    sp = D.select_chunks(512 * 20, 512, 4, "spread")
    assert len(sp) == 4 and sp[0] == (0, 512) and sp[-1] == (512 * 19, 512 * 20)
    assert D.select_chunks(1100, 512, 1, "first") == [(0, 512)]
    for how in D.SELECTS:
        for n in (0, 1, 511, 512, 513, 2047, 9000):
            s = D.select_chunks(n, 512, 4, how)
            assert 1 <= len(s) <= 4 and all(s[i][1] <= s[i + 1][0] for i in range(len(s) - 1)), (how, n, s)


def test_batch():
    tok = np.arange(1, 1201, dtype=np.uint16)
    pieces = [D.doc_chunks(tok, 512, 4, "spread"), D.doc_chunks(tok[:100], 512, 4, "spread"), D.doc_chunks(tok[:0], 512, 4, "spread")]
    ids, valid, pool, doc, slot, rel = D.batch_arrays(pieces, 512)
    assert ids.shape == (5, 512) and doc.tolist() == [0, 0, 0, 1, 2] and slot.tolist() == [0, 1, 2, 0, 0]
    assert pool.sum(1).tolist() == [176, 512, 512, 100, 1] and (valid == pool).all()
    ids, valid, pool, doc, slot, rel = D.batch_arrays(pieces, 512, n_prompt=20)
    assert ids.shape == (5, 532) and valid[:, :20].all() and not pool[:, :20].any() and (ids[:, :20] == 0).all()
    assert ids[2, 20] == 689 and pool.sum(1).tolist() == [176, 512, 512, 100, 1]
    assert D.micro_batches([pieces[0]] * 5, 6) == [[0, 1], [2, 3], [4]]


def test_metric():
    rng = np.random.default_rng(0)
    from sklearn.metrics import f1_score
    for _ in range(30):
        Y = (rng.random((200, 12)) < 0.2).astype(np.uint8)
        P = (rng.random((200, 12)) < 0.25)
        r = E.report(Y, P)
        assert abs(r["macro_f1"] - f1_score(Y, P, average="macro", zero_division=0)) < 1e-12
        assert abs(r["micro_f1"] - f1_score(Y, P, average="micro", zero_division=0)) < 1e-12
        assert abs(r["subset_acc"] - (Y == P).all(1).mean()) < 1e-12
    S = np.array([[0.1, 0.2, 0.05], [0.9, 0.1, 0.6]])
    d = E.decide(S, 0.5)
    assert d.tolist() == [[False, True, False], [True, False, True]]          # empty row -> its top statute
    assert E.decide(S, np.array([0.05, 0.5, 0.7]), min1=False).tolist() == [[True, False, False], [True, False, False]]
    Y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    P = np.array([[0.6, 0.1], [0.2, 0.3], [0.7, 0.35], [0.1, 0.2]])
    t = E.per_label_thresholds(Y, P, min_pos=1)
    assert 0.2 < t[0] <= 0.6 and 0.2 < t[1] <= 0.3, t
    assert E.macro_f1(Y, E.decide(P, t, min1=False)) == 1.0
    c = E.crossfit_per_label(np.tile(Y, (50, 1)), np.tile(P, (50, 1)), min1=False)
    assert c["per_label"] > 0.99, c
    b = E.frequency_buckets([1.0, 0.5, 0.0], [100, 50, 10])
    assert [x["macro_f1"] for x in b] == [1.0, 0.5, 0.0]


def test_files_and_cache():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "Train")); os.makedirs(os.path.join(d, "Test"))
        tr = [{"id": f"t{i}", "text": " ".join(["w"] * (i + 1)), "labels": [i % 3, 4]} for i in range(6)]
        tr.append({"id": "d0", "text": "dev text", "labels": [1]})                       # same id as a dev case
        tr.append({"id": "t9", "text": "test\ncase", "labels": [2]})                     # same text as a test case
        tr.append({"id": "t0", "text": "w", "labels": [4, 0, 0]})                        # exact duplicate (unsorted, repeated)
        json.dump(tr, open(os.path.join(d, "Train", "train.json"), "w"))
        json.dump([{"id": "d0", "text": "dev text", "labels": [1, 5]}], open(os.path.join(d, "Train", "dev.json"), "w"))
        json.dump([{"id": "x0", "text": "test case"}], open(os.path.join(d, "Test", "test.json"), "w"))
        real = D.tokenize_all
        D.tokenize_all = lambda texts, root, procs=None: [np.array([len(w) for w in D.clean_text(t).split()], np.uint16) for t in texts]
        try:
            info = D.build_cache(d, None, os.path.join(d, "c.npz"), log=lambda s: None)
        finally:
            D.tokenize_all = real
        assert info["train"]["dropped_overlap_with_dev_test"] == 2 and info["n_labels"] == 6, info
        c = D.Cache(os.path.join(d, "c.npz"))
        assert c.n("train") == 6 and c.labelled("train") and c.labelled("dev") and not c.labelled("test")
        assert c.labels("train", 0).tolist() == [0, 4] and c.Y("dev").tolist() == [[0, 1, 0, 0, 0, 1]]
        assert c.Y("test").sum() == 0 and len(c.doc("train", 3)) == 4
        json.dump(tr + [{"id": "t1", "text": "other", "labels": [1]}], open(os.path.join(d, "x.json"), "w"))
        try:
            D.load_split(os.path.join(d, "x.json"))
            raise AssertionError("conflicting duplicate not detected")
        except ValueError:
            pass
        out = os.path.join(d, "prediction.json")
        D.write_prediction(["a", "b"], [[3, 1, 1], [0]], out, 6)
        assert json.load(open(out)) == [{"id": "a", "pred": [1, 3]}, {"id": "b", "pred": [0]}]
        try:
            D.write_prediction(["a"], [[7]], out, 6)
            raise AssertionError("invalid id not detected")
        except AssertionError as e:
            assert "invalid" in str(e)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
    print("ALL TESTS PASSED")
