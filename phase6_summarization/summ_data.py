"""
summ_data.py - Phase-6 data pipeline (no torch): sentences, tokenization, held-out split, the three published ways of
fitting a long judgment into a short-context model, and the evaluation metrics.

Split.  train.json -> 'fit' (95%) + 'val' (5%, fixed seed, never trained on; selects checkpoints, length, decoding).
        test.json is only predicted.

Input strategies (all from the long-document summarization literature):
  lead     truncation: the first IN tokens of the judgment                         (baseline in every paper below)
  extract  extract-then-abstract (Pilault et al., EMNLP 2020): a sentence classifier picks the important sentences;
           its training labels are the two document sentences with the highest ROUGE against each reference-summary
           sentence ("oracle"). The language model is trained on the oracle extracts and tested on the classifier's
           extracts (the paper's best setting). Sentences are kept in document order up to IN tokens.
  chunk    divide-and-conquer (Gidiotis & Tsoumakas 2020, DANCER; Shukla et al., AACL 2022, on IN-Abs): the judgment is
           cut into consecutive chunks of <= CH tokens without breaking sentences; every reference-summary sentence is
           assigned to the chunk that contains its most similar document sentence; each chunk is trained to produce the
           summary sentences assigned to it. At test time every chunk is summarised with an equal share of the length
           budget and the chunk summaries are concatenated in order.
  sled     SLED (Ivgi et al., TACL 2023), encoder-decoder only: overlapping chunks of c tokens are encoded independently,
           the middle (effective) part of each is kept and the decoder cross-attends to all of them (fusion-in-decoder).
           Implemented in summ_model.py; here only the token stream is prepared.

Similarity = ROUGE-1/ROUGE-2 F on token-id sets (the 100 most frequent ids are ignored), computed with sparse matrix
products so a 1,000-sentence judgment is matched in milliseconds.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import json
import re
import time
import zlib

import numpy as np

_WS = re.compile(r"\s+")
# sentence boundary: . ? ! followed by space and a capital / digit / bracket / quote. Abbreviations ("Ors.", "No.") are
# kept together with a short list; very long "sentences" are cut into <= MAX_SENT_WORDS-word pieces.
_SENT = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9(\[\"'])")
_ABBR = re.compile(r"(?:\b(?:No|Nos|Ors|Anr|Art|Arts|Sec|Secs|Ss|S|Mr|Mrs|Ms|Dr|Hon|Vol|Cl|Co|Ltd|Pvt|Corpn|Govt|"
                   r"Dept|Ex|vs|Vs|v|viz|i\.e|e\.g|etc|cf|para|paras|p|pp|Rs|St|Jt|Inc|Bros|Smt|Sri|Shri|Kumari|Ch|"
                   r"Cr|Crl|Civ|Misc|App|Appl|Spl|Regd|Asstt|Addl|Distt|Dist|Supdt)\.|\b[A-Z]\.)$")
_ENUM = re.compile(r"\(?[0-9ivxIVXa-zA-Z]{1,4}[.)]")              # "2." / "(iv)" / "a." paragraph numbers
MAX_SENT_WORDS = 80
SEP_TEXT = " ### Summary of the judgment: "
STOP_TOP = 100


def clean(t):
    return _WS.sub(" ", (t or "").replace(" ", " ")).strip()


def split_sentences(text):
    parts, cur = [], ""
    for p in _SENT.split(clean(text)):
        cur = f"{cur} {p}" if cur else p
        if _ABBR.search(cur.split()[-1] if cur.split() else "") or _ENUM.fullmatch(cur):
            continue                                             # "No. 5" / "S. 302": not a sentence end
        parts.append(cur)
        cur = ""
    if cur:
        parts.append(cur)
    out = []
    for p in parts:
        w = p.split()
        for k in range(0, len(w), MAX_SENT_WORDS):
            out.append(" ".join(w[k:k + MAX_SENT_WORDS]))
    return [s for s in out if s]


def val_ids(ids, frac=0.05, seed=781):
    """Deterministic held-out set: the documents whose crc32(seed:id) falls in the lowest `frac`."""
    return {i for i in ids if (zlib.crc32(f"{seed}:{i}".encode()) % 100000) < frac * 100000}


def load_json(path):
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):                                      # {"id": "document"} form
        d = [{"id": k, "document": v} for k, v in d.items()]
    out, seen = [], set()
    for r in d:
        i = str(r["id"])
        if i in seen:
            continue
        seen.add(i)
        out.append({"id": i, "document": r.get("document") or r.get("text") or "", "summary": r.get("summary")})
    return out


def find_file(data_dir, split):
    hits = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f == f"{split}.json" or (f.startswith(split) and f.endswith(".json")):
                hits.append(os.path.join(root, f))
    hits.sort(key=lambda p: (os.path.basename(p) != f"{split}.json", len(p)))
    return hits[0] if hits else None


# ---------------------------------------------------------------------------------------------- tokenization
_TOK = None


def _init(tok_root):
    global _TOK
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ner_data import load_tokenizer, OffsetTokenizer
    _TOK = OffsetTokenizer(load_tokenizer(tok_root), "native")


def _enc_sents(sents):
    """Each sentence tokenized with a trailing space (as inside running text) -> list of uint16 arrays."""
    out = []
    for k, s in enumerate(sents):
        ids = _TOK.encode(s + (" " if k < len(sents) - 1 else ""))[0]
        out.append(np.asarray(ids, dtype=np.uint16))
    return out


def tokenize_docs(list_of_sentence_lists, tok_root, procs=None):
    import multiprocessing as mp
    procs = procs or max(1, min(8, os.cpu_count() or 1))
    if procs == 1:
        _init(tok_root)
        return [_enc_sents(s) for s in list_of_sentence_lists]
    with mp.get_context("fork").Pool(procs, initializer=_init, initargs=(tok_root,)) as pool:
        return pool.map(_enc_sents, list_of_sentence_lists, chunksize=8)


def encode_text(text, tok_root):
    if _TOK is None:
        _init(tok_root)
    return [int(i) for i in _TOK.encode(text)[0]]


class Split:
    """Ragged storage: doc k has sentences so[k]..so[k+1]-1, sentence j has tokens to[j]..to[j+1]-1 (same for summaries)."""

    def __init__(self, ids, doc_sents, sum_sents=None):
        self.ids = list(ids)
        self.d_tok, self.d_to, self.d_so = self._pack(doc_sents)
        self.has_sum = sum_sents is not None
        if self.has_sum:
            self.s_tok, self.s_to, self.s_so = self._pack(sum_sents)

    @staticmethod
    def _pack(docs):
        sl = [len(s) for d in docs for s in d]
        tok = np.concatenate([s for d in docs for s in d]) if sl and sum(sl) else np.zeros(0, np.uint16)
        to = np.concatenate([[0], np.cumsum(sl)]).astype(np.int64)
        so = np.concatenate([[0], np.cumsum([len(d) for d in docs])]).astype(np.int64)
        return tok.astype(np.uint16), to, so

    def __len__(self):
        return len(self.ids)

    def doc_sents(self, k):
        return [self.d_tok[self.d_to[j]:self.d_to[j + 1]] for j in range(self.d_so[k], self.d_so[k + 1])]

    def sum_sents(self, k):
        return [self.s_tok[self.s_to[j]:self.s_to[j + 1]] for j in range(self.s_so[k], self.s_so[k + 1])]

    def doc_tokens(self, k):
        return self.d_tok[self.d_to[self.d_so[k]]:self.d_to[self.d_so[k + 1]]]

    def sum_tokens(self, k):
        return self.s_tok[self.s_to[self.s_so[k]]:self.s_to[self.s_so[k + 1]]]

    def save(self, prefix, z):
        z[f"{prefix}_ids"] = np.array(self.ids)
        for n in ("d_tok", "d_to", "d_so") + (("s_tok", "s_to", "s_so") if self.has_sum else ()):
            z[f"{prefix}_{n}"] = getattr(self, n)

    @classmethod
    def load(cls, prefix, z):
        o = cls.__new__(cls)
        o.ids = [str(i) for i in z[f"{prefix}_ids"]]
        for n in ("d_tok", "d_to", "d_so"):
            setattr(o, n, z[f"{prefix}_{n}"])
        o.has_sum = f"{prefix}_s_tok" in z
        if o.has_sum:
            for n in ("s_tok", "s_to", "s_so"):
                setattr(o, n, z[f"{prefix}_{n}"])
        return o


# ---------------------------------------------------------------------------------------------- similarity (ROUGE on id sets)
def _bigrams(a):
    a = a.astype(np.int64)
    return a[:-1] * 32768 + a[1:] if len(a) > 1 else np.zeros(0, np.int64)


def set_matrix(sents, stop, bigram=False, dim=1 << 20):
    """Binary CSR (n_sents, dim) of token ids (or hashed bigrams) per sentence; stop ids removed for unigrams."""
    from scipy.sparse import csr_matrix
    rows, cols = [], []
    for r, s in enumerate(sents):
        if bigram:
            u = np.unique(_bigrams(s) % dim)
        else:
            u = np.unique(s.astype(np.int64))
            u = u[~np.isin(u, stop)]
        rows.append(np.full(len(u), r))
        cols.append(u)
    rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    m = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(sents), dim))
    return m


def rouge_sim(sum_sents, doc_sents, stop):
    """(n_sum, n_doc) mean of ROUGE-1 and ROUGE-2 F (set form) between every summary and document sentence."""
    out = np.zeros((len(sum_sents), len(doc_sents)), np.float32)
    if not len(sum_sents) or not len(doc_sents):
        return out
    for big in (False, True):
        A, B = set_matrix(sum_sents, stop, big), set_matrix(doc_sents, stop, big)
        inter = (A @ B.T).toarray()
        na, nb = np.asarray(A.sum(1)).ravel(), np.asarray(B.sum(1)).ravel()
        out += 2 * inter / np.maximum(na[:, None] + nb[None, :], 1)
    return out / 2


def stop_ids(split, n=STOP_TOP):
    c = np.bincount(split.d_tok.astype(np.int64), minlength=32768)
    return np.argsort(-c)[:n]


# ---------------------------------------------------------------------------------------------- oracle + chunks
def oracle(split, k, stop, top=2, max_sum_tok=None):
    """Pilault et al.: the `top` most similar document sentences of every summary sentence (within the first
    max_sum_tok summary tokens). Returns (sorted doc-sentence indices, best doc sentence of every summary sentence)."""
    ds, ss = split.doc_sents(k), split.sum_sents(k)
    if max_sum_tok is not None:
        keep, n = [], 0
        for s in ss:
            if n >= max_sum_tok:
                break
            keep.append(s)
            n += len(s)
        ss_o = keep
    else:
        ss_o = ss
    sim = rouge_sim(ss, ds, stop)
    best = sim.argmax(1) if sim.size else np.zeros(len(ss), np.int64)
    pick = set()
    if sim.size:
        for i in range(len(ss_o)):
            pick.update(np.argsort(-sim[i])[:top].tolist())
    return sorted(pick), best


def chunk_bounds(sent_lens, CH):
    """Consecutive chunks of whole sentences with <= CH tokens (a longer sentence is its own chunk)."""
    out, a, n = [], 0, 0
    for j, L in enumerate(sent_lens):
        if n and n + L > CH:
            out.append((a, j))
            a, n = j, 0
        n += L
    if len(sent_lens):
        out.append((a, len(sent_lens)))
    return out


# ---------------------------------------------------------------------------------------------- extractor
def sentence_features(doc_sents, stop, dim=1 << 18):
    """SummaRuNNer-style features: content (hashed ids), position (relative decile + absolute index bin), length bin,
    salience (cosine with the document) bin. Returns a CSR matrix (n_sents, dim + 40)."""
    from scipy.sparse import csr_matrix, hstack
    n = len(doc_sents)
    if n == 0:
        return csr_matrix((0, dim + 40), dtype=np.float32)
    rows, cols, vals = [], [], []
    for r, s in enumerate(doc_sents):
        u = np.unique(s.astype(np.int64))
        u = u[~np.isin(u, stop)]
        rows.append(np.full(len(u), r)); cols.append(u % dim); vals.append(np.full(len(u), 1.0 / np.sqrt(max(1, len(u)))))
    C = csr_matrix((np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))), shape=(n, dim))
    docv = np.asarray(C.sum(0)).ravel()
    sal = C @ docv / (np.linalg.norm(docv) + 1e-9)
    sal_rank = np.argsort(np.argsort(-sal)) / max(1, n - 1)                        # 0 = most salient
    F = np.zeros((n, 40), np.float32)
    idx = np.arange(n)
    F[idx, np.minimum((idx / max(1, n) * 10).astype(int), 9)] = 1                 # relative decile 0-9
    ab = np.digitize(idx, [1, 2, 3, 5, 10, 20, 50, 100, 200])                      # absolute index 10-19
    F[idx, 10 + ab] = 1
    ln = np.array([len(s) for s in doc_sents])
    F[idx, 20 + np.digitize(ln, [5, 10, 20, 40, 80, 120, 160, 240, 400])] = 1      # length 20-29
    F[idx, 30 + np.minimum((sal_rank * 10).astype(int), 9)] = 1                    # salience rank decile 30-39
    return hstack([C, csr_matrix(F)]).tocsr()


def train_extractor(split, docs, labels, stop, log=print, seed=0):
    from scipy.sparse import vstack
    from sklearn.linear_model import SGDClassifier
    X = vstack([sentence_features(split.doc_sents(k), stop) for k in docs]).tocsr()
    y = np.concatenate([labels[k] for k in docs])
    clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=8, tol=None, random_state=seed, class_weight="balanced")
    clf.fit(X, y)
    log(f"extractor: {X.shape[0]} sentences, {y.mean():.3f} positive")
    return clf


def select(doc_sents, scores, budget):
    """Highest-scoring sentences until `budget` tokens, returned in document order (a too-long sentence is cut)."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    pick, n = [], 0
    for j in order:
        L = len(doc_sents[j])
        if n + L > budget:
            if n == 0:
                pick.append(j); n = budget
            continue
        pick.append(j)
        n += L
        if n >= budget - 4:
            break
    return sorted(pick)


def join(sents, idx, budget):
    if not idx:
        return np.zeros(0, np.uint16)
    return np.concatenate([sents[j] for j in idx])[:budget]


# ---------------------------------------------------------------------------------------------- build everything once
def build(data_dir, tok_root, out, in_len=448, ch_len=640, procs=None, log=print, max_docs=None):
    t0 = time.time()
    tr = load_json(find_file(data_dir, "train"))
    tp = find_file(data_dir, "test")
    te = load_json(tp) if tp else []
    if max_docs:
        tr, te = tr[:max_docs], te[:max(4, max_docs // 4)]
    te_text = {clean(x["document"]) for x in te}
    n0 = len(tr)
    tr = [x for x in tr if clean(x["document"]) not in te_text and x["summary"]]
    vs = val_ids([x["id"] for x in tr])
    fit = [x for x in tr if x["id"] not in vs]
    val = [x for x in tr if x["id"] in vs]
    log(f"train.json {n0} -> fit {len(fit)} + val {len(val)} (dropped {n0 - len(tr)} overlapping test / empty); test {len(te)}")
    S = {}
    for name, rows in (("fit", fit), ("val", val), ("test", te)):
        if not rows:
            continue
        ds = tokenize_docs([split_sentences(x["document"]) or [clean(x["document"])[:2000] or "."] for x in rows], tok_root, procs)
        ss = tokenize_docs([split_sentences(x["summary"]) for x in rows], tok_root, procs) if rows[0]["summary"] is not None else None
        S[name] = Split([x["id"] for x in rows], ds, ss)
        log(f"{name}: {len(rows)} docs, {len(S[name].d_tok):,} doc tokens, {len(S[name].d_so) and S[name].d_so[-1]:,} sentences"
            + (f", {len(S[name].s_tok):,} summary tokens" if ss is not None else "") + f" ({time.time() - t0:.0f}s)")
    stop = stop_ids(S["fit"])
    # oracle labels (fit + val; val only for reporting) and chunk assignments
    lab, ext_or, ch_t = {}, {}, {}
    for name in ("fit", "val"):
        sp = S[name]
        lab[name], ext_or[name], ch_t[name] = [], [], []
        for k in range(len(sp)):
            ds = sp.doc_sents(k)
            pick, best = oracle(sp, k, stop, top=2, max_sum_tok=640)
            y = np.zeros(len(ds), np.int8)
            y[pick] = 1
            lab[name].append(y)
            ext_or[name].append(np.array(pick, np.int32))
            cb = chunk_bounds([len(s) for s in ds], ch_len)
            owner = np.zeros(len(ds), np.int32)
            for c, (a, b) in enumerate(cb):
                owner[a:b] = c
            ch_t[name].append(owner[best].astype(np.int32) if len(best) else np.zeros(0, np.int32))
        log(f"oracle {name}: {np.mean([y.mean() for y in lab[name]]):.3f} of sentences selected ({time.time() - t0:.0f}s)")
    clf = train_extractor(S["fit"], range(len(S["fit"])), lab["fit"], stop, log)
    scores = {}
    for name, sp in S.items():
        scores[name] = [clf.decision_function(sentence_features(sp.doc_sents(k), stop)).astype(np.float32) for k in range(len(sp))]
    z = {"info": json.dumps(dict(fit=len(S["fit"]), val=len(S.get("val", [])), test=len(S.get("test", [])),
                                 in_len=in_len, ch_len=ch_len, seconds=round(time.time() - t0)))}
    for name, sp in S.items():
        sp.save(name, z)
        z[f"{name}_score"] = np.concatenate(scores[name]) if scores[name] else np.zeros(0, np.float32)
        if name in lab:
            z[f"{name}_oracle"] = np.concatenate(ext_or[name]) if ext_or[name] else np.zeros(0, np.int32)
            z[f"{name}_oracle_off"] = np.concatenate([[0], np.cumsum([len(e) for e in ext_or[name]])]).astype(np.int64)
            z[f"{name}_chunk_of_sum"] = np.concatenate(ch_t[name]) if ch_t[name] else np.zeros(0, np.int32)
    z["stop"] = stop
    np.savez(out, **z)
    json.dump({x["id"]: clean(x["summary"]) for x in val}, open(out + ".refs.json", "w"))
    log(f"cache {out} written ({time.time() - t0:.0f}s)")
    return json.loads(z["info"])


class Data:
    """Everything a run needs: the three splits and the example builders for every input strategy."""

    def __init__(self, path):
        z = dict(np.load(path, allow_pickle=False))
        self.info = json.loads(str(z["info"]))
        self.S = {n: Split.load(n, z) for n in ("fit", "val", "test") if f"{n}_ids" in z}
        self.score, self.oracle, self.chunk_of_sum = {}, {}, {}
        for n, sp in self.S.items():
            off = sp.d_so
            sc = z[f"{n}_score"]
            self.score[n] = [sc[off[k]:off[k + 1]] for k in range(len(sp))]
            if f"{n}_oracle" in z:
                o, oo = z[f"{n}_oracle"], z[f"{n}_oracle_off"]
                self.oracle[n] = [o[oo[k]:oo[k + 1]].tolist() for k in range(len(sp))]
                c, so = z[f"{n}_chunk_of_sum"], sp.s_so
                self.chunk_of_sum[n] = [c[so[k]:so[k + 1]] for k in range(len(sp))]
        self.stop = z["stop"]

    # ---- one (input, target) per document: lead / extract
    def doc_input(self, split, k, how, in_len, use_oracle=False):
        sp = self.S[split]
        if how == "lead":
            return sp.doc_tokens(k)[:in_len]
        ds = sp.doc_sents(k)
        if how == "extract":
            if use_oracle and split in self.oracle:
                idx = self.oracle[split][k]
                if sum(len(ds[j]) for j in idx) < in_len:          # fill the remaining budget with the classifier's picks
                    sc = np.asarray(self.score[split][k]).copy()
                    sc[idx] = 1e9
                    idx = select(ds, sc, in_len)
                else:
                    sc = np.full(len(ds), -1e9, np.float32)
                    sc[idx] = np.asarray(self.score[split][k])[idx]
                    idx = select(ds, sc, in_len)
            else:
                idx = select(ds, self.score[split][k], in_len)
            return join(ds, idx, in_len)
        if how == "sled":
            return sp.doc_tokens(k)[:in_len]                       # SLED reads the first in_len tokens (chunked in the model)
        raise ValueError(how)

    def target(self, split, k, out_len):
        return self.S[split].sum_tokens(k)[:out_len]

    def full_target_len(self, split, k):
        sp = self.S[split]
        return int(sp.s_to[sp.s_so[k + 1]] - sp.s_to[sp.s_so[k]])

    # ---- chunk strategy: many (chunk, chunk-summary) pairs per document
    def chunks(self, split, k, ch_len):
        ds = self.S[split].doc_sents(k)
        return [np.concatenate(ds[a:b])[:ch_len] for a, b in chunk_bounds([len(s) for s in ds], ch_len)]

    def chunk_targets(self, split, k, ch_len, out_len):
        ss = self.S[split].sum_sents(k)
        n_ch = len(chunk_bounds([len(s) for s in self.S[split].doc_sents(k)], ch_len))
        own = self.chunk_of_sum[split][k]
        tg = [[] for _ in range(n_ch)]
        for s, c in zip(ss, own):
            tg[min(int(c), n_ch - 1)].append(s)
        return [np.concatenate(t)[:out_len] if t else np.zeros(0, np.uint16) for t in tg]


# ---------------------------------------------------------------------------------------------- metrics
_W = re.compile(r"[a-z0-9]+")


def words(t):
    return _W.findall(t.lower())


def lcs(a, b):
    """Length of the longest common subsequence (bit-parallel, Hyyro 2004)."""
    if not a or not b:
        return 0
    M = {}
    for i, x in enumerate(a):
        M[x] = M.get(x, 0) | (1 << i)
    full = (1 << len(a)) - 1
    V = full
    for y in b:
        U = V & M.get(y, 0)
        V = ((V + U) | (V - U)) & full
    return len(a) - bin(V).count("1")


def rouge(cand, ref):
    """ROUGE-1 and ROUGE-L F1 (lower-cased alphanumeric tokens, no stemming)."""
    from collections import Counter
    c, r = words(cand), words(ref)
    if not c or not r:
        return 0.0, 0.0
    o = sum((Counter(c) & Counter(r)).values())
    f = lambda m: 0.0 if m == 0 else 2 * m / (len(c) + len(r))
    return f(o), f(lcs(c, r))


def cut_words(t, n):
    w = t.split()
    return " ".join(w[:n]) if n else " ".join(w)
