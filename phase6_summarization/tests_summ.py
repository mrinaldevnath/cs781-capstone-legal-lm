"""tests_summ.py - CPU unit tests for the Phase-6 data / metric code (no torch). Run: python tests_summ.py"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import json
import random
import tempfile

import numpy as np

import summ_data as SD


def test_sentences():
    s = SD.split_sentences("The appeal No. 5 of 1977 was filed. Held: S. 302 applies. The Court (Mr. X) agreed! 2. Next.")
    assert s == ["The appeal No. 5 of 1977 was filed.", "Held: S. 302 applies.", "The Court (Mr. X) agreed!", "2. Next."], s
    long = " ".join(["w"] * 200)
    assert [len(x.split()) for x in SD.split_sentences(long)] == [80, 80, 40]


def test_chunks_select():
    assert SD.chunk_bounds([100, 200, 300, 50, 700, 10], 640) == [(0, 3), (3, 4), (4, 5), (5, 6)]
    assert SD.chunk_bounds([], 640) == []
    sents = [np.arange(n, dtype=np.uint16) for n in (10, 20, 30, 40)]
    assert SD.select(sents, [0.1, 0.9, 0.5, 0.2], 55) == [1, 2]
    assert SD.select(sents, [0.1, 0.9, 0.5, 0.2], 5) == [1]                    # a too-long best sentence is cut
    assert len(SD.join(sents, [1], 5)) == 5


def test_metrics():
    rnd = random.Random(0)

    def lcs_dp(a, b):
        M = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a)):
            for j in range(len(b)):
                M[i + 1][j + 1] = M[i][j] + 1 if a[i] == b[j] else max(M[i][j + 1], M[i + 1][j])
        return M[-1][-1]
    for _ in range(200):
        a = [rnd.choice("abcde") for _ in range(rnd.randint(0, 40))]
        b = [rnd.choice("abcdef") for _ in range(rnd.randint(0, 40))]
        assert SD.lcs(a, b) == lcs_dp(a, b)
    r1, rl = SD.rouge("the cat sat on the mat", "the cat sat on a mat")
    assert abs(r1 - 2 * 5 / 12) < 1e-9 and abs(rl - 2 * 5 / 12) < 1e-9
    assert SD.cut_words("a b  c d", 2) == "a b" and SD.cut_words("a b c", 0) == "a b c"
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"])
        x, y = "The appeal is allowed and the order is set aside.", "Appeal allowed; the impugned order set aside by court."
        s = sc.score(y, x)
        r1, rl = SD.rouge(x, y)
        assert abs(s["rouge1"].fmeasure - r1) < 1e-9 and abs(s["rougeL"].fmeasure - rl) < 1e-9
        print("  rouge matches rouge_score")
    except ImportError:
        pass


def test_similarity():
    a = [np.array([5, 6, 7, 8], np.uint16), np.array([9, 10], np.uint16)]
    b = [np.array([1, 2, 3], np.uint16), np.array([5, 6, 7, 8, 11], np.uint16), np.array([9, 10, 12], np.uint16)]
    s = SD.rouge_sim(a, b, np.array([], np.int64))
    assert s.argmax(1).tolist() == [1, 2] and s[0, 0] == 0
    assert SD.val_ids([str(i) for i in range(10000)]) == SD.val_ids([str(i) for i in range(10000)])
    assert 400 < len(SD.val_ids([str(i) for i in range(10000)])) < 600


def test_build(tok_root):
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/Train"); os.makedirs(f"{d}/Test")
        rnd = random.Random(1)
        voc = "court appeal held section act order petition state the of was is high learned counsel facts".split()
        sent = lambda: " ".join(rnd.choice(voc) for _ in range(rnd.randint(5, 25))).capitalize() + "."
        tr = [{"id": f"d{i}", "document": " ".join(sent() for _ in range(rnd.randint(5, 120))),
               "summary": " ".join(sent() for _ in range(rnd.randint(1, 8)))} for i in range(120)]
        te = [{"id": f"t{i}", "document": " ".join(sent() for _ in range(rnd.randint(1, 60)))} for i in range(6)]
        tr.append({"id": "dup", "document": te[0]["document"], "summary": "x."})       # same text as a test document
        json.dump(tr, open(f"{d}/Train/train.json", "w")); json.dump(te, open(f"{d}/Test/test.json", "w"))
        info = SD.build(d, tok_root, f"{d}/c.npz", in_len=64, ch_len=96, procs=1, log=lambda s: None)
        assert info["fit"] + info["val"] == 120 and info["test"] == 6, info
        D = SD.Data(f"{d}/c.npz")
        refs = json.load(open(f"{d}/c.npz.refs.json"))
        assert set(refs) == set(D.S["val"].ids)
        for k in range(len(D.S["fit"])):
            for how in ("lead", "extract"):
                x = D.doc_input("fit", k, how, 64, use_oracle=True)
                assert 0 < len(x) <= 64
            ch = D.chunks("fit", k, 96)
            tg = D.chunk_targets("fit", k, 96, 50)
            assert len(ch) == len(tg) and all(len(c) <= 96 for c in ch) and all(len(t) <= 50 for t in tg)
            assert sum(len(c) for c in ch) == len(D.S["fit"].doc_tokens(k)) or any(len(s) > 96 for s in D.S["fit"].doc_sents(k))
            n_sum = sum(len(s) for s in D.S["fit"].sum_sents(k))
            assert sum(len(t) for t in D.chunk_targets("fit", k, 96, 10**6)) == n_sum      # every summary sentence used once
        assert len(D.doc_input("test", 0, "extract", 64)) > 0


if __name__ == "__main__":
    tok_root = sys.argv[1] if len(sys.argv) > 1 else None
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            if name == "test_build":
                if tok_root is None:
                    print("skip test_build (no tokenizer root given)")
                    continue
                f(tok_root)
            else:
                f()
            print("ok", name)
    print("ALL TESTS PASSED")
