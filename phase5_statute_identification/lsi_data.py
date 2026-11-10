"""
lsi_data.py - Phase-5 Legal Statute Identification data handling (no torch).

    load_split        read train.json / dev.json / test.json (a list of {id, text, labels: [statute ids]});
                      exact duplicates (same id and same text) removed, conflicting duplicates are an error
    clean_text        line breaks -> spaces, whitespace collapsed (the <ENTITY> / <SECTION> / <ACT> masks are kept)
    build_cache       tokenize every split once (Phase-1 tokenizer, several processes) -> one .npz:
                      flat uint16 token array + per-document offsets, flat label array + label offsets, ids
                      training documents that also appear in dev/test (same id or same text) are dropped
    Cache             access to the cached documents (tokens of document i, its label set, multi-hot matrix)
    select_chunks     which 512-token chunks of a document the model reads (first / last / firstlast / spread)
    batch_arrays      chunks of several documents -> padded id matrix (+ prompt slots) + document index + position
    micro_batches     groups of documents that fit one forward pass (memory)
    write_prediction  official format: [{"id": ..., "pred": [statute ids]}, ...]

The statute mentions themselves are masked in the text (<SECTION> of the <ACT>), so the model has to infer the statutes
from the facts and arguments. Documents have 1-87,000 words (median ~430, mean ~1,100): about half fit one 512-token
chunk, 78% fit four.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import hashlib
import json
import re

import numpy as np

SPLITS = ("train", "dev", "test")
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------------------------- files
def clean_text(t):
    return _WS.sub(" ", t.replace("\n", " ")).strip()


def text_hash(t):
    return hashlib.md5(clean_text(t).encode("utf-8")).hexdigest()


def find_split(data_dir, split):
    """train.json / dev.json / test.json anywhere under data_dir (the HF repo keeps them in Train/ and Test/)."""
    for root, _, files in os.walk(data_dir):
        if f"{split}.json" in files:
            return os.path.join(root, f"{split}.json")
    return None


def _labels(x):
    v = x.get("labels", x.get("label"))
    if v is None:
        return None
    v = [v] if isinstance(v, (int, str)) else v
    return sorted({int(i) for i in v})


def load_split(path):
    """-> list of {id, text, labels (sorted unique ints) or None}, file order, exact duplicates removed."""
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        d = [dict(id=k, **(v if isinstance(v, dict) else {"text": v})) for k, v in d.items()]
    out, seen = [], {}
    for x in d:
        i = str(x["id"])
        lab = _labels(x)
        if i in seen:
            prev = out[seen[i]]
            if prev["labels"] != lab or text_hash(prev["text"]) != text_hash(x["text"]):
                raise ValueError(f"id {i} occurs twice with different text or labels in {path}")
            continue
        seen[i] = len(out)
        out.append({"id": i, "text": x["text"], "labels": lab})
    return out


def file_stats(path):
    d = json.load(open(path, encoding="utf-8"))
    n = len(d)
    u = len({str(x["id"]) for x in d}) if isinstance(d, list) else n
    return n, u


# ---------------------------------------------------------------------------------------------- tokenization
_TOK = None


def _init_worker(tok_root):
    global _TOK
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ner_data import load_tokenizer, OffsetTokenizer
    _TOK = OffsetTokenizer(load_tokenizer(tok_root), "native")     # exactly the Phase-1 tokenization


def _encode(text):
    t = clean_text(text)
    ids = _TOK.encode(t)[0] if t else np.zeros(0, dtype=np.int64)
    return np.asarray(ids, dtype=np.uint16)


def tokenize_all(texts, tok_root, procs=None):
    import multiprocessing as mp
    procs = procs or max(1, min(8, os.cpu_count() or 1))
    if procs == 1:
        _init_worker(tok_root)
        return [_encode(t) for t in texts]
    with mp.get_context("fork").Pool(procs, initializer=_init_worker, initargs=(tok_root,)) as pool:
        return pool.map(_encode, texts, chunksize=16)


def build_cache(data_dir, tok_root, out, procs=None, log=print):
    """Tokenize every available split once; drop training documents that also appear in dev/test."""
    import time
    splits, info = {}, {}
    for sp in SPLITS:
        p = find_split(data_dir, sp)
        if p is None:
            log(f"{sp}: not found under {data_dir}")
            continue
        n_rows, n_ids = file_stats(p)
        splits[sp] = load_split(p)
        info[sp] = dict(rows_in_file=n_rows, unique_ids=n_ids, docs=len(splits[sp]))
    assert "train" in splits and "dev" in splits, "train.json and dev.json are required"
    held = {x["id"] for sp in ("dev", "test") for x in splits.get(sp, [])}
    held_h = {text_hash(x["text"]) for sp in ("dev", "test") for x in splits.get(sp, [])}
    keep = [x for x in splits["train"] if x["id"] not in held and text_hash(x["text"]) not in held_h]
    info["train"]["dropped_overlap_with_dev_test"] = len(splits["train"]) - len(keep)
    splits["train"] = keep
    n_labels = 1 + max(l for sp in ("train", "dev") for x in splits[sp] for l in x["labels"])
    info["n_labels"] = n_labels
    arrays = {}
    for sp, docs in splits.items():
        t0 = time.time()
        toks = tokenize_all([x["text"] for x in docs], tok_root, procs)
        lens = np.array([len(t) for t in toks], dtype=np.int64)
        labs = [x["labels"] or [] for x in docs]
        arrays[f"{sp}_tok"] = np.concatenate(toks) if toks else np.zeros(0, np.uint16)
        arrays[f"{sp}_off"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        arrays[f"{sp}_ids"] = np.array([x["id"] for x in docs])
        arrays[f"{sp}_lab"] = np.array([l for L in labs for l in L], dtype=np.int64)
        arrays[f"{sp}_loff"] = np.concatenate([[0], np.cumsum([len(L) for L in labs])]).astype(np.int64)
        arrays[f"{sp}_has_labels"] = np.array([x["labels"] is not None for x in docs])
        arrays[f"{sp}_words"] = np.array([len(x["text"].split()) for x in docs], dtype=np.int64)
        nl = np.array([len(L) for L in labs])
        info[sp].update(tokens=int(lens.sum()), tok_median=int(np.median(lens)) if len(lens) else 0,
                        tok_p90=int(np.percentile(lens, 90)) if len(lens) else 0,
                        tok_max=int(lens.max()) if len(lens) else 0,
                        labelled=bool(arrays[f"{sp}_has_labels"].all()) if len(docs) else False,
                        labels_per_doc=float(nl.mean()) if len(nl) else 0.0, seconds=round(time.time() - t0))
        log(f"{sp}: {json.dumps(info[sp])}")
    np.savez(out, info=json.dumps(info), **arrays)
    return info


class Cache:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.info = json.loads(str(z["info"]))
        self.n_labels = int(self.info["n_labels"])
        self.splits = {}
        for sp in SPLITS:
            if f"{sp}_tok" in z.files:
                self.splits[sp] = dict(tok=z[f"{sp}_tok"], off=z[f"{sp}_off"], ids=[str(i) for i in z[f"{sp}_ids"]],
                                       lab=z[f"{sp}_lab"], loff=z[f"{sp}_loff"], has=z[f"{sp}_has_labels"],
                                       words=z[f"{sp}_words"])

    def has(self, sp):
        return sp in self.splits

    def n(self, sp):
        return len(self.splits[sp]["ids"]) if sp in self.splits else 0

    def doc(self, sp, i):
        s = self.splits[sp]
        return s["tok"][s["off"][i]:s["off"][i + 1]]

    def ids(self, sp):
        return self.splits[sp]["ids"]

    def labels(self, sp, i):
        s = self.splits[sp]
        return s["lab"][s["loff"][i]:s["loff"][i + 1]]

    def labelled(self, sp):
        return sp in self.splits and bool(self.splits[sp]["has"].all())

    def Y(self, sp, idx=None):
        """Multi-hot label matrix (n, n_labels) uint8."""
        idx = np.arange(self.n(sp)) if idx is None else idx
        Y = np.zeros((len(idx), self.n_labels), dtype=np.uint8)
        for r, i in enumerate(idx):
            Y[r, self.labels(sp, i)] = 1
        return Y

    def lengths(self, sp):
        o = self.splits[sp]["off"]
        return o[1:] - o[:-1]


# ---------------------------------------------------------------------------------------------- chunks
SELECTS = ("last", "first", "firstlast", "spread")


def chunk_bounds(n_tok, chunk):
    """Non-overlapping chunks aligned to the END of the document (the last chunk is always full; the first may be
    short). -> list of (start, end), in document order."""
    if n_tok <= 0:
        return [(0, 0)]
    out, e = [], n_tok
    while e > 0:
        s = max(0, e - chunk)
        out.append((s, e))
        e = s
    return out[::-1]


def select_chunks(n_tok, chunk, k, how):
    """Which chunks the model reads -> list of (start, end) in document order (at most k). A document of at most
    k chunks is always read whole (except 'first', which is start-aligned).
        last       the final k chunks
        first      the first k chunks (start-aligned)
        firstlast  first ceil(k/2) + last floor(k/2)
        spread     k chunks evenly spaced over the whole document (whole-document coverage at a fixed cost)"""
    if how == "first":
        b = [(s, min(n_tok, s + chunk)) for s in range(0, max(n_tok, 1), chunk)]
        return b[:k]
    b = chunk_bounds(n_tok, chunk)
    if len(b) <= k:
        return b
    if how == "last":
        return b[-k:]
    if how == "firstlast":
        nf = (k + 1) // 2
        head = [(s, min(n_tok, s + chunk)) for s in range(0, nf * chunk, chunk)]
        tail = b[-(k - nf):] if k - nf else []
        head = [h for h in head if not tail or h[1] <= tail[0][0]]
        return head + tail
    if how == "spread":
        idx = np.unique(np.round(np.linspace(0, len(b) - 1, k)).astype(int))
        return [b[i] for i in idx]
    raise ValueError(how)


def doc_chunks(tok, chunk, k, how):
    """-> (list of token arrays, relative position of each chunk's centre in [0, 1])."""
    n = len(tok)
    sel = select_chunks(n, chunk, k, how)
    pieces = [tok[s:e] for s, e in sel]
    rel = [((s + e) / 2) / max(n, 1) for s, e in sel]
    return pieces, rel


def batch_arrays(docs_pieces, chunk, n_prompt=0):
    """docs_pieces: list over documents of (pieces, rel). ->
        ids (n_chunks, P+L) int64 (0-padded), valid (n_chunks, P+L) bool (attention mask; prompt slots are valid),
        pool (n_chunks, P+L) bool (the chunk's own tokens: what is averaged), doc (n_chunks,), slot (n_chunks,)
        index of the chunk inside its document, rel (n_chunks,) float32. P = n_prompt placeholder slots (id 0) that
        prompt / prefix tuning fill with learned vectors; L = longest chunk (<= chunk)."""
    flat = [(di, p, r) for di, (ps, rs) in enumerate(docs_pieces) for p, r in zip(ps, rs)]
    L = max(1, max(len(p) for _, p, _ in flat))
    P = n_prompt
    ids = np.zeros((len(flat), P + L), dtype=np.int64)
    valid = np.zeros((len(flat), P + L), dtype=bool)
    valid[:, :P] = True
    for j, (_, p, _) in enumerate(flat):
        n = max(1, len(p))                                  # an empty document still gets one (padding) token
        ids[j, P:P + len(p)] = p
        valid[j, P:P + n] = True
    pool = valid.copy()
    pool[:, :P] = False
    doc = np.array([di for di, _, _ in flat], dtype=np.int64)
    slot = np.concatenate([np.arange(len(ps)) for ps, _ in docs_pieces]).astype(np.int64)
    rel = np.array([r for _, _, r in flat], dtype=np.float32)
    return ids, valid, pool, doc, slot, rel


def micro_batches(pieces, max_chunks):
    """Consecutive groups of documents with at most max_chunks chunks in total (at least one document each)."""
    out, cur, n = [], [], 0
    for j, (ps, _) in enumerate(pieces):
        k = max(1, len(ps))
        if cur and n + k > max_chunks:
            out.append(cur)
            cur, n = [], 0
        cur.append(j)
        n += k
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------------------------------------- submission
def write_prediction(ids, label_sets, path, n_labels):
    """Official format: a JSON list of {"id": <case id>, "pred": [statute ids]} - integers, valid, unique, sorted;
    one entry per test document."""
    assert len(ids) == len(label_sets) == len(set(ids)), "one prediction per unique test id"
    rows = []
    for i, s in zip(ids, label_sets):
        p = sorted({int(x) for x in s})
        assert all(0 <= x < n_labels for x in p), f"invalid statute id in {i}: {p}"
        rows.append({"id": i, "pred": p})
    json.dump(rows, open(path, "w", encoding="utf-8"))
    return rows
