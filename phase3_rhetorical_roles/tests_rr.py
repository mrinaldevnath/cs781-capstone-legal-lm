"""tests_rr.py - checks of the torch-free parts (run: python tests_rr.py)."""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import json
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_data as D
import rr_eval as E


def test_segments():
    rng = np.random.default_rng(0)
    for _ in range(200):
        lens = rng.integers(1, 129, size=int(rng.integers(1, 400)))
        B = int(rng.integers(128, 4000))
        for segs in (D.train_segments(lens, B), D.train_segments(lens, B, rng)):
            assert segs[0][0] == 0 and segs[-1][1] == len(lens)
            for (s, e), (s2, _) in zip(segs, segs[1:]):
                assert e == s2 and e > s
            for s, e in segs:
                assert e - s == 1 or lens[s:e].sum() <= B
        segs = D.infer_segments(lens, B)
        cov = np.zeros(len(lens), int)
        for s, e in segs:
            assert e > s and (e - s == 1 or lens[s:e].sum() <= B)
            cov[s:e] += 1
        assert cov.min() >= 1 and segs[-1][1] == len(lens)
        assert all(b[0] > a[0] for a, b in zip(segs, segs[1:]))


def test_pack_and_arrays():
    rng = np.random.default_rng(1)
    for mode in ("packed", "single"):
        for npmt in (0, 20):
            sents = [rng.integers(1, 1000, size=int(rng.integers(1, 129))) for _ in range(int(rng.integers(1, 80)))]
            ids, valid, win, a, b = D.batch_arrays(sents, 512, mode, npmt)
            assert ids.shape[1] % 64 == 0 and ids.shape[1] <= 512 + 64
            for i, s in enumerate(sents):
                assert (ids[win[i], a[i]:b[i]] == s[:b[i] - a[i]]).all() and b[i] - a[i] == min(len(s), 512 - npmt)
                assert valid[win[i], a[i]:b[i]].all() and a[i] >= npmt
            assert valid[:, :npmt].all()
            assert valid.sum() == npmt * len(ids) + sum(b - a)
            if mode == "single":
                assert len(ids) == len(sents)
            else:
                assert (win[1:] >= win[:-1]).all()                    # order kept
                for w in range(len(ids)):
                    assert valid[w].sum() <= 512


def test_metric():
    """prf must equal the official scorer: sklearn macro over ALL labels, zero_division=0."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    rng = np.random.default_rng(3)
    labels = [f"L{i}" for i in range(13)]
    for _ in range(50):
        n = int(rng.integers(1, 300))
        g = [labels[i] for i in rng.integers(0, int(rng.integers(1, 13)), n)]
        p = [labels[i] for i in rng.integers(0, 13, n)]
        r = E.prf(g, p, labels)
        kw = dict(labels=labels, average="macro", zero_division=0)
        assert abs(r["macro_f1"] - f1_score(g, p, **kw)) < 1e-9
        assert abs(r["macro_p"] - precision_score(g, p, **kw)) < 1e-9
        assert abs(r["macro_r"] - recall_score(g, p, **kw)) < 1e-9
    ev = E.evaluate({"d1": ["A", "A"], "d2": ["B"]}, {"d1": ["A", "Z"], "d2": ["B"]}, ["A", "B", "C", "Z"],
                    {"d1": "CL", "d2": "IT"})
    assert abs(ev["macro_f1"] - (2 / 3 + 1) / 4) < 1e-9 and abs(ev["macro_f1_present"] - (2 / 3 + 1) / 2) < 1e-9


def test_files_and_validation():
    tmp = tempfile.mkdtemp()
    json.dump({"Fact": 0, "None": 2, "Issue": 1}, open(f"{tmp}/label_vocab.json", "w"))
    assert D.load_vocab(f"{tmp}/label_vocab.json") == ["Fact", "Issue", "None"]
    json.dump([{"id": "x y(1)", "text": ["a", "b"], "labels": ["Fact", "None"]}], open(f"{tmp}/CL_dev.json", "w"))
    json.dump({"t1": {"text": ["a", "b", "c"]}}, open(f"{tmp}/CL_test.json", "w"))
    json.dump({"t2": ["a"]}, open(f"{tmp}/IT_test.json", "w"))
    assert D.load_split(f"{tmp}/CL_dev.json")[0]["id"] == "x y(1)"
    t = D.load_split(f"{tmp}/CL_test.json") + D.load_split(f"{tmp}/IT_test.json")
    docs = [D.RRDoc(x["id"], 0, x["text"]) for x in t]
    D.validate_submission({"t1": ["Fact"] * 3, "t2": ["None"]}, docs, ["Fact", "Issue", "None"])
    for bad in ({"t1": ["Fact"] * 2, "t2": ["None"]}, {"t1": ["Fact"] * 3}, {"t1": ["Fact"] * 3, "t2": ["none"]}):
        try:
            D.validate_submission(bad, docs, ["Fact", "Issue", "None"])
            raise RuntimeError("validation should fail")
        except AssertionError:
            pass


def test_folds_and_tail():
    docs = [D.RRDoc(f"d{i}", i % 2, [[1]] * 10, np.array([0] * 9 + [5 if i % 7 == 0 else 1])) for i in range(80)]
    f = D.assign_folds(docs, 5, [5])
    assert sorted(set(f.values())) == list(range(5))
    assert all(sum(1 for d in docs if f[d.id] == k) == 16 for k in range(5))
    assert all(any(f[d.id] == k and 5 in d.y for d in docs) for k in range(5))       # every fold gets a rare doc
    assert f == D.assign_folds(list(reversed(docs)), 5, [5])                           # deterministic
    lens = np.array([50, 60, 70, 80, 90])
    assert D.tail_segment(lens, 200) == (3, 5) and D.tail_segment(lens, 50) == (4, 5) and D.tail_segment(lens, 10**6) == (0, 5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
