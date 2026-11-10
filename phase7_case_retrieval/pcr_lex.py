"""
pcr_lex.py - lexical first stage: BM25 over word n-grams, n = 1..6 (Robertson & Zaragoza, 2009; n-gram BM25 as in
U-CREAT, Joshi et al., ACL Findings 2023, the strongest published IL-PCR baseline) + cheap priors.

Features for every (query, candidate) pair of a split (candidate = a document of that split's pool):
    bm25_n    n = 1..6   BM25 of the whole query against the candidate, over word n-grams
    cite_n    n = 1..3   BM25 of the query's citation-context sentences only (sentences that cite a prior case)
    ctxmax_n  n = 1..4   each citation passage of the query is scored on its own, the best passage counts
                         (a query cites many cases; each is discussed in one passage)
    rep                  1 if a reporter citation written in the query ("2 SCR 148") is the candidate's own
    citator              1 if the candidate's CITATOR INFO header lists the query's own AIR citation
    pop                  log(1 + times the candidate is a gold citation of TRAIN queries) - a prior learned from
                         train labels (leave-one-out on train itself)
    later                1 if the candidate's estimated year is later than the query's (a prior case cannot be
                         decided after the case that cites it)
Each feature is z-scored over the pool per query. A weighted sum is the lexical score; the weights are tuned by
coordinate ascent on TRAIN micro-F1@7 and only evaluated on dev (dev-tuned weights are reported for reference).
Two weight sets are tuned: `lex` (all features) and `lex_nocit` (without the CITATOR link, which reads the citing-case
list from the candidate's report header rather than the text of the judgments).

Run: python pcr_lex.py <cache_dir>  ->  <cache_dir>/lex.npz, lex_report.json
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import json
import pickle
import time

import numpy as np
import scipy.sparse as sp

from pcr_data import K, SPLITS, micro_prf, topk

NBITS = 27                       # hashed n-gram space: 2^27 columns
DENSE_COLS = 12288               # the most frequent columns are scored with a dense matmul (BLAS), the rest sparse
K1, B = 1.2, 0.75
ORDERS = (1, 2, 3, 4, 5, 6)
QTF = "binary"                   # query n-grams counted once (better than raw counts on train and dev)
CITE_ORDERS = (1, 2, 3)
CTX_ORDERS = (1, 2, 3, 4)


def ngram_cols(seq, n):
    """int32 word ids -> int64 column ids of its hashed n-grams (deterministic multiplicative hashing)."""
    L = len(seq) - n + 1
    if L <= 0:
        return np.zeros(0, np.int64)
    s = seq.astype(np.uint64)
    h = np.zeros(L, np.uint64)
    with np.errstate(over="ignore"):
        for i in range(n):
            h = h * np.uint64(1000003) + s[i:i + L] + np.uint64(1)
        h = (h * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(64 - NBITS)
    return h.astype(np.int64)


def count_matrix(seqs, n):
    indptr, idx, val = [0], [], []
    for s in seqs:
        u, c = np.unique(ngram_cols(s, n), return_counts=True)
        idx.append(u)
        val.append(c.astype(np.float32))
        indptr.append(indptr[-1] + len(u))
    idx = np.concatenate(idx) if idx else np.zeros(0, np.int64)
    val = np.concatenate(val) if val else np.zeros(0, np.float32)
    return sp.csr_matrix((val, idx.astype(np.int32), np.array(indptr, np.int64)), shape=(len(seqs), 1 << NBITS))


def _filter(M, keep):
    """CSR M with only the stored entries where keep (bool per entry) is True."""
    rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))[keep]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=M.shape[0]))]).astype(np.int64)
    return M.data[keep], M.indices[keep], indptr


def bm25_scores(Qc, Cc, qtf="count"):
    """Qc (nq x V) query counts, Cc (nc x V) candidate counts -> (nq, nc) float32 BM25 scores.
    Columns are first re-indexed to the n-grams present in the pool, ordered by document frequency; the most
    frequent DENSE_COLS are scored with a dense matmul (BLAS), the long tail with a sparse product."""
    N, nq = Cc.shape[0], Qc.shape[0]
    uc, inv = np.unique(Cc.indices, return_inverse=True)
    df = np.bincount(inv, minlength=len(uc)).astype(np.float64)
    rank = np.empty(len(uc), np.int64)
    rank[np.argsort(-df, kind="stable")] = np.arange(len(uc))
    df = df[np.argsort(-df, kind="stable")]
    idf = np.log1p((N - df + 0.5) / (df + 0.5))
    ccol = rank[inv]
    dl = np.asarray(Cc.sum(1)).ravel().astype(np.float64)
    avg = dl.mean() if N else 1.0
    rows = np.repeat(np.arange(N), np.diff(Cc.indptr))
    tf = Cc.data.astype(np.float64)
    w = (idf[ccol] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl[rows] / avg))).astype(np.float32)
    W = sp.csr_matrix((w, ccol, Cc.indptr), shape=(N, len(uc)))
    pos = np.clip(np.searchsorted(uc, Qc.indices), 0, max(len(uc) - 1, 0))
    ok = (uc[pos] == Qc.indices) if len(uc) else np.zeros(len(Qc.indices), bool)
    qd, _, qptr = _filter(Qc, ok)
    qcol = rank[pos[ok]]
    qd = qd.astype(np.float32)
    if qtf == "binary":
        qd[:] = 1.0
    elif qtf == "log":
        qd = np.log1p(qd)
    Q = sp.csr_matrix((qd, qcol, qptr), shape=(nq, len(uc)))
    D = min(DENSE_COLS, len(uc))
    out = (Q[:, :D].toarray() @ W[:, :D].toarray().T).astype(np.float32)
    if len(uc) > D:
        Qs = sp.csr_matrix(_filter(Q, Q.indices >= D), shape=Q.shape)
        Ws = sp.csr_matrix(_filter(W, W.indices >= D), shape=W.shape)
        out += (Qs @ Ws.T).toarray().astype(np.float32)
    return out


def zscore(S):
    S = S.astype(np.float32)
    mu = S.mean(1, keepdims=True)
    sd = S.std(1, keepdims=True) + 1e-6
    return (S - mu) / sd


def mask_self(S, qids, cids, val=-1e4):
    pos = {c: j for j, c in enumerate(cids)}
    for i, q in enumerate(qids):
        if q in pos:
            S[i, pos[q]] = val
    return S


def features(cache, log=print):
    meta = json.load(open(os.path.join(cache, "meta.json")))
    tr = meta["train"]
    cnt = {}
    for g in tr["gold"]:
        for c in set(g):
            cnt[c] = cnt.get(c, 0) + 1
    F = {}
    for s in SPLITS:
        t0 = time.time()
        m = meta[s]
        W = pickle.load(open(os.path.join(cache, f"words_{s}.pkl"), "rb"))
        cite = pickle.load(open(os.path.join(cache, f"cite_{s}.pkl"), "rb"))
        ctx = pickle.load(open(os.path.join(cache, f"ctx_{s}.pkl"), "rb"))
        rows = [x for cs in ctx for x in cs]
        owner = np.repeat(np.arange(len(ctx)), [len(cs) for cs in ctx])
        nq, nc = len(m["qids"]), len(m["cids"])
        feats = {}
        for n in ORDERS:
            Cc = count_matrix(W["c"], n)
            feats[f"bm25_{n}"] = zscore(bm25_scores(count_matrix(W["q"], n), Cc, QTF))
            if n in CITE_ORDERS:
                if sum(len(x) for x in cite) > 0:
                    feats[f"cite_{n}"] = zscore(bm25_scores(count_matrix(cite, n), Cc, QTF))
                else:
                    feats[f"cite_{n}"] = np.zeros_like(feats[f"bm25_{n}"])
            if n in CTX_ORDERS:
                Sx = bm25_scores(count_matrix(rows, n), Cc, QTF)
                M = np.full((nq, nc), -1e9, np.float32)
                np.maximum.at(M, owner, Sx)
                feats[f"ctxmax_{n}"] = zscore(M)
                del Sx
            log(f"  {s} n={n}: {time.time() - t0:.0f}s")
        pop = np.array([cnt.get(c, 0) for c in m["cids"]], np.float32)
        P = np.repeat(pop[None, :], len(m["qids"]), 0)
        if s == "train":                                                   # leave-one-out on train
            pos = {c: j for j, c in enumerate(m["cids"])}
            for i, g in enumerate(m["gold"]):
                for c in set(g):
                    if c in pos:
                        P[i, pos[c]] -= 1
        feats["pop"] = zscore(np.log1p(np.maximum(P, 0)))
        qy, cy = np.array(m["qyear"]), np.array(m["cyear"])
        feats["later"] = ((cy[None, :] > qy[:, None]) & (qy[:, None] > 0) & (cy[None, :] > 0)).astype(np.float32)
        for key, name in (("rep_pairs", "rep"), ("citator_pairs", "citator")):
            M = np.zeros((nq, nc), np.float32)
            for i, j in m[key]:
                M[i, j] = 1.0
            feats[name] = M
        F[s] = feats
        log(f"{s}: features in {time.time() - t0:.0f}s")
    return meta, F


def combine(feats, w):
    S = None
    for k, v in w.items():
        if v:
            S = v * feats[k] if S is None else S + v * feats[k]
    return S


def f1_of(meta_s, S):
    return micro_prf(topk(S, meta_s["cids"], meta_s["qids"]), meta_s["gold"])["f1"]


GRID = (0, 0.1, 0.25, 0.5, 1, 2, 4)
NEG_GRID = (0, -1, -2, -4, -8)


def tune(meta_s, feats, init, grid=GRID, neg_grid=NEG_GRID, passes=3, log=print):
    """Coordinate ascent on micro-F1@7."""
    w = dict(init)
    best = f1_of(meta_s, combine(feats, w))
    for p in range(passes):
        changed = False
        for k in feats:
            for v in (neg_grid if k == "later" else grid):
                if v == w.get(k, 0):
                    continue
                trial = dict(w, **{k: v})
                if not any(trial.values()):
                    continue
                f = f1_of(meta_s, combine(feats, trial))
                if f > best + 1e-6:
                    best, w, changed = f, trial, True
        log(f"  pass {p + 1}: F1 {best:.4f} {w}")
        if not changed:
            break
    return w, best


def oracle(meta_s):
    """Best possible micro-F1@7: every gold case ranked first (a query with more than 7 gold cases still gets 7)."""
    tp = sum(min(K, len(set(g))) for g in meta_s["gold"])
    P = tp / (K * len(meta_s["gold"]))
    R = tp / max(1, sum(len(set(g)) for g in meta_s["gold"]))
    return 2 * P * R / (P + R)


def recall_at(meta_s, S, n):
    S = mask_self(S.copy(), meta_s["qids"], meta_s["cids"])
    top = np.argsort(-S, 1)[:, :n]
    pos = {c: j for j, c in enumerate(meta_s["cids"])}
    hit = tot = 0
    for i, g in enumerate(meta_s["gold"]):
        gs = {pos[c] for c in g if c in pos}
        hit += len(gs & set(top[i].tolist()))
        tot += len(set(g))
    return hit / max(tot, 1)


def run(cache, log=print):
    t0 = time.time()
    meta, F = features(cache, log)
    rep = {"single_feature": {}}
    for k in F["train"]:
        if k in ("later", "rep", "citator"):
            continue
        rep["single_feature"][k] = {s: round(f1_of(meta[s], F[s][k]), 4) for s in ("train", "dev")}
    log("single features (micro-F1@7):", json.dumps(rep["single_feature"]))
    m_tr, m_dv = meta["train"], meta["dev"]
    # the date prior: how often is a gold citation 'later' than its query, vs any candidate?
    rep["oracle_f1"] = {s: round(oracle(meta[s]), 4) for s in ("train", "dev")}
    pos_tr = {c: j for j, c in enumerate(m_tr["cids"])}
    g_later = [F["train"]["later"][i, pos_tr[c]] for i, g in enumerate(m_tr["gold"]) for c in set(g) if c in pos_tr]
    rep["later_rate_gold"] = float(np.mean(g_later)) if g_later else None
    rep["later_rate_all"] = float(F["train"]["later"].mean())
    best1 = max(rep["single_feature"], key=lambda k: rep["single_feature"][k]["train"])
    init = {best1: 0.5, "later": -4}
    out, W_ = {}, {}
    for name, drop in (("lex", ()), ("lex_nocit", ("citator",))):
        sub = {s: {k: v for k, v in F[s].items() if k not in drop} for s in SPLITS}
        log(f"tuning weights on TRAIN ({name})")
        w, _ = tune(m_tr, sub["train"], init, grid=GRID, neg_grid=NEG_GRID, log=log)
        W_[name] = w
        rep[f"{name}_weights"] = w
        rep[f"{name}_f1"] = {s: round(f1_of(meta[s], combine(sub[s], w)), 4) for s in ("train", "dev")}
        for s in SPLITS:
            out[f"{s}_{name}"] = mask_self(combine(sub[s], w).astype(np.float32), meta[s]["qids"], meta[s]["cids"])
    rep["weights"], rep["fused"] = W_["lex"], rep["lex_f1"]
    log("weights tuned on DEV (reference only, not used)")
    w_dev, f_dv = tune(m_dv, F["dev"], init, grid=GRID, neg_grid=NEG_GRID, log=log)
    rep["dev_tuned_reference"] = dict(weights=w_dev, dev=round(f_dv, 4))
    for s in SPLITS:
        for k, v in F[s].items():
            out[f"{s}__{k}"] = v.astype(np.float16)
    for s in ("train", "dev"):
        rep[f"{s}_recall@"] = {n: round(recall_at(meta[s], out[f"{s}_lex"], n), 4) for n in (7, 20, 50, 100, 200)}
        pr = micro_prf(topk(out[f"{s}_lex"], meta[s]["cids"], meta[s]["qids"]), meta[s]["gold"])
        rep[f"{s}_prf"] = {k: round(v, 4) for k, v in pr.items()}
    # hard negatives for the neural models: the top-100 lexical candidates of each train query that are not gold
    S = out["train_lex"]
    order = np.argsort(-S, 1)[:, :150]
    pos = {c: j for j, c in enumerate(m_tr["cids"])}
    hn = np.full((len(m_tr["qids"]), 100), -1, np.int32)
    for i, g in enumerate(m_tr["gold"]):
        bad = {pos[c] for c in g if c in pos}
        row = [j for j in order[i] if j not in bad][:100]
        hn[i, :len(row)] = row
    out["train_hardneg"] = hn
    np.savez(os.path.join(cache, "lex.npz"), **out)
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(rep, open(os.path.join(cache, "lex_report.json"), "w"), indent=1)
    log(json.dumps({k: v for k, v in rep.items() if k != "single_feature"}, indent=1))
    return rep


if __name__ == "__main__":
    run(sys.argv[1])
