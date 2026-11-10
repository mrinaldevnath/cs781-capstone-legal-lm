"""
rr_data.py - Phase-3 Rhetorical Role data handling (no torch).

    load_split / load_vocab   read the official json files (list-of-docs or dict-of-docs)
    RRDoc                     one document: sentence token ids (Phase-1 tokenization), labels, domain
    build_cache / load_cache  tokenize every split once per notebook (pickle) so all jobs reuse it
    train_segments            contiguous sentence ranges under a token budget (random offset = augmentation)
    infer_segments            overlapping ranges; every sentence is later predicted by its most central range
    pack                      sentences of one range -> backbone windows (sentences are never split)
    batch_arrays              ids / valid / pooling indices for one range (what the model consumes)

Hierarchy: a document has 19-1816 sentences (5k-77k tokens), far more than one 512-token window, so sentences
are packed into windows, pooled to one vector each, and a sentence-level head reads a whole range of them.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import json
import pickle

import numpy as np

DOMAINS = ("CL", "IT")
SPLITS = ("train", "dev", "test")


# ---------------------------------------------------------------------------------------------- files
def load_vocab(path):
    """label_vocab.json is {label: index}; returns labels ordered by index."""
    v = json.load(open(path, encoding="utf-8"))
    if isinstance(v, dict):
        return [k for k, _ in sorted(v.items(), key=lambda kv: kv[1])]
    return list(v)


def load_split(path):
    """-> list of dicts {id, text: [sentences], labels: [...] or None}, in file order.
    Train/dev files are lists of {id, text, labels}; test files are {id: {...}} or {id: [sentences]}."""
    d = json.load(open(path, encoding="utf-8"))
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            v = v if isinstance(v, dict) else {"text": v}
            out.append({"id": str(k), "text": list(v["text"]), "labels": v.get("labels")})
    else:
        for v in d:
            out.append({"id": str(v["id"]), "text": list(v["text"]), "labels": v.get("labels")})
    for x in out:
        if x["labels"] is not None:
            assert len(x["labels"]) == len(x["text"]), f"label/sentence count mismatch in {x['id']}"
    return out


def split_path(data_dir, dom, split):
    return os.path.join(data_dir, f"{dom}_{split}.json")


# ---------------------------------------------------------------------------------------------- documents
class RRDoc:
    __slots__ = ("id", "dom", "sents", "y", "n")

    def __init__(self, did, dom, sents, y=None):
        self.id, self.dom, self.sents, self.y = did, dom, sents, y
        self.n = len(sents)

    def lens(self):
        return np.array([len(s) for s in self.sents], dtype=np.int64)


def make_tokenizer(tok_root):
    """Exact Phase-1 tokenization (pretok 'native'): sentences need no character alignment, so we keep the
    token distribution the backbones were pretrained on."""
    from ner_data import load_tokenizer, OffsetTokenizer
    return OffsetTokenizer(load_tokenizer(tok_root), "native")


def tokenize(otok, s, max_tok, fallback):
    ids = otok.encode(s)[0][:max_tok] if s.strip() else np.zeros(0, dtype=np.int64)
    return ids if len(ids) else fallback


def build_cache(data_dir, tok_root, out, max_tok=128):
    """Tokenize every available split once. Cache: {(dom, split): [ {id, sents, labels} ]} + vocab."""
    otok = make_tokenizer(tok_root)
    fallback = otok.encode(".")[0][:1]
    cache = {"labels": load_vocab(os.path.join(data_dir, "label_vocab.json")), "max_tok": max_tok, "docs": {}}
    for dom in DOMAINS:
        for sp in SPLITS:
            p = split_path(data_dir, dom, sp)
            if not os.path.exists(p):
                continue
            cache["docs"][(dom, sp)] = [{"id": x["id"], "labels": x["labels"],
                                         "sents": [tokenize(otok, s, max_tok, fallback) for s in x["text"]]}
                                        for x in load_split(p)]
    with open(out, "wb") as f:
        pickle.dump(cache, f)
    return cache


def load_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def docs_from_cache(cache, dom, split, max_docs=None, max_sents=None):
    lab_ix = {l: i for i, l in enumerate(cache["labels"])}
    out = []
    for x in cache["docs"].get((dom, split), [])[:max_docs]:
        sents = x["sents"][:max_sents]
        y = None if x["labels"] is None else np.array([lab_ix[l] for l in x["labels"][:max_sents]], dtype=np.int64)
        out.append(RRDoc(x["id"], DOMAINS.index(dom), sents, y))
    return out


def limit_docs(docs, max_docs=None, max_sents=None):
    """Debug helper: first max_docs documents, each cut to max_sents sentences."""
    return [RRDoc(d.id, d.dom, d.sents[:max_sents], None if d.y is None else d.y[:max_sents]) for d in docs[:max_docs]]


def assign_folds(docs, n_folds, rare_ids=()):
    """Deterministic document-level folds. Documents that contain a rare label (Dissent, PrecedentOverruled) are
    dealt round-robin first, so every fold gets some; the other documents follow (sorted by domain and id)."""
    has_rare = lambda d: d.y is not None and bool(np.isin(d.y, list(rare_ids)).any())
    order = sorted([d for d in docs if has_rare(d)], key=lambda d: (d.dom, d.id)) + \
        sorted([d for d in docs if not has_rare(d)], key=lambda d: (d.dom, d.id))
    return {d.id: i % n_folds for i, d in enumerate(order)}


# ---------------------------------------------------------------------------------------------- segments
def _greedy(lens, start, budget):
    """End (exclusive) of the longest range starting at `start` whose token total fits `budget` (>= 1 sentence)."""
    e, tot = start, 0
    while e < len(lens) and (e == start or tot + lens[e] <= budget):
        tot += lens[e]
        e += 1
    return e


def train_segments(lens, budget, rng=None):
    """Non-overlapping ranges covering all sentences. With rng the first range is cut at a random token
    offset, so range borders move between epochs."""
    out, s = [], 0
    if rng is not None and len(lens):
        e = _greedy(lens, 0, int(rng.integers(1, budget + 1)))
        out.append((0, e))
        s = e
    while s < len(lens):
        e = _greedy(lens, s, budget)
        out.append((s, e))
        s = e
    return out


def tail_segment(lens, budget):
    """The range that ends at the last sentence and fits the budget (the end of the document)."""
    s, tot = len(lens), 0
    while s > 0 and (s == len(lens) or tot + lens[s - 1] <= budget):
        s -= 1
        tot += lens[s]
    return s, len(lens)


def infer_segments(lens, budget):
    """Overlapping ranges (next range starts about half a budget later) covering all sentences."""
    n = len(lens)
    out, s = [], 0
    while True:
        e = _greedy(lens, s, budget)
        out.append((s, e))
        if e >= n:
            return out
        nxt = _greedy(lens, s, budget // 2)          # about half of the range, at least one sentence
        s = max(s + 1, min(nxt, e))


def centrality(s, e):
    i = np.arange(s, e)
    return np.minimum(i - s, e - 1 - i).astype(np.float64)


# ---------------------------------------------------------------------------------------------- packing
def pack(lens, capacity, mode="packed"):
    """Group consecutive sentences into windows of <= capacity tokens.
    mode 'packed': greedy, several sentences per window (local token context);
    mode 'single': one sentence per window (sentence-only encoding).
    -> list of lists of sentence positions (0-based within the range)."""
    if mode == "single":
        return [[i] for i in range(len(lens))]
    wins, cur, tot = [], [], 0
    for i, L in enumerate(lens):
        L = min(int(L), capacity)
        if cur and tot + L > capacity:
            wins.append(cur)
            cur, tot = [], 0
        cur.append(i)
        tot += L
    if cur:
        wins.append(cur)
    return wins


def batch_arrays(sents, window=512, mode="packed", n_prompt=0, round_to=64):
    """Arrays for one range of sentences.
    ids (n_win, L) int64, valid (n_win, L) bool, and per sentence: win index, start a, end b (token positions
    inside its window, b exclusive). With prompt tuning the first n_prompt positions of every window are
    placeholders (id 0, valid) that the model replaces by learned vectors."""
    cap = window - n_prompt
    lens = [min(len(s), cap) for s in sents]
    wins = pack(lens, cap, mode)
    used = [sum(lens[i] for i in w) for w in wins]
    L = n_prompt + max(used)
    L = int(np.ceil(L / round_to) * round_to)
    ids = np.zeros((len(wins), L), dtype=np.int64)
    valid = np.zeros((len(wins), L), dtype=bool)
    win = np.zeros(len(sents), dtype=np.int64)
    a = np.zeros(len(sents), dtype=np.int64)
    b = np.zeros(len(sents), dtype=np.int64)
    for wi, w in enumerate(wins):
        p = n_prompt
        valid[wi, :n_prompt] = True
        for i in w:
            t = sents[i][:lens[i]]
            ids[wi, p:p + len(t)] = t
            valid[wi, p:p + len(t)] = True
            win[i], a[i], b[i] = wi, p, p + len(t)
            p += len(t)
    return ids, valid, win, a, b


def rel_positions(n, s, e):
    """Relative position i / (n - 1) of sentences s..e-1 in a document of n sentences."""
    return np.arange(s, e, dtype=np.float32) / max(1, n - 1)


# ---------------------------------------------------------------------------------------------- output
def validate_submission(pred, docs, labels):
    """Official checks: every test id, nothing extra, one valid label per sentence, order preserved."""
    ids = {d.id for d in docs}
    assert set(pred) == ids, f"id mismatch: missing {sorted(ids - set(pred))[:3]} extra {sorted(set(pred) - ids)[:3]}"
    ok = set(labels)
    for d in docs:
        p = pred[d.id]
        assert len(p) == d.n, f"{d.id}: {len(p)} predictions for {d.n} sentences"
        bad = [x for x in p if x not in ok]
        assert not bad, f"{d.id}: unknown labels {bad[:3]}"


if __name__ == "__main__":                 # python rr_data.py DATA_DIR TOK_ROOT OUT.pkl
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    c = build_cache(sys.argv[1], sys.argv[2], sys.argv[3])
    for k, v in c["docs"].items():
        print(k, len(v), "docs", sum(len(x["sents"]) for x in v), "sentences")
