"""
ner_data.py - Phase-2 L-NER data handling (no torch): loading, offset-exact tokenization, BIO tags,
windows, span decoding, dev split.

Everything here works on RAW document text and exact character offsets (end exclusive, as in the gold
files: text[start:end] == entity_text for all 8,393 training entities).

Tokenization with character offsets
-----------------------------------
The Phase-1 tokenizer pretokenizes text into units r"\\S+\\s*" (a word + its trailing whitespace) and runs
byte-level BPE inside each unit. A unit glues punctuation to words ("Court, ", "(Appellant) "), so a gold
span that stops before the comma does not end on a token boundary: on the training data only 61% of gold
spans start AND end on unit boundaries.

pretok="piece" (default) - constrained BPE: every unit is cut further into pieces at letter / digit /
    punctuation edges (r"[^\\W\\d_]+|\\d+|_+|[^\\w\\s]"), the unit's trailing whitespace stays on its last piece,
    and BPE runs inside each piece with the end-of-unit marker only on the last piece. The result is exactly
    what the Phase-1 BPE would produce if it chose to split at those edges ("Court" + ", "), so the token
    distribution stays close to pretraining while 99.46% of gold spans become token-aligned (the rest are
    gold quirks such as "High Court" inside "High Courts").
pretok="native" - the Phase-1 tokenization unchanged (ablation).

Each token carries: char start, char end (incl. trailing whitespace) and core end (end without trailing
whitespace). Pure-whitespace tokens have core_end == start. Offsets come from the bytes of every token
(tok._id_bytes) mapped back to characters, so they are exact for any Unicode text.
"""
import bisect
import json
import os
import re
import sys

import numpy as np

UNIT_RE = re.compile(r"\S+\s*|\s+")
CORE_PIECE_RE = re.compile(r"[^\W\d_]+|\d+|_+|[^\w\s]")
MAX_PIECE_CHARS = 96
IGNORE = -100


# ---------------------------------------------------------------------------------------------- labels
def load_labels(path):
    labs = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    tags = ["O"] + [p + l for l in labs for p in ("B-", "I-")]
    return labs, tags


def tag_ids(labs):
    """label -> (B id, I id). Tag layout: O=0, B-L(k)=1+2k, I-L(k)=2+2k."""
    return {l: (1 + 2 * k, 2 + 2 * k) for k, l in enumerate(labs)}


# ---------------------------------------------------------------------------------------------- data
def load_folds(data_dir, folds=(1, 2)):
    texts, gold = {}, {}
    for f in folds:
        T = json.load(open(os.path.join(data_dir, f"fold{f}-text.json"), encoding="utf-8"))
        gp = os.path.join(data_dir, f"fold{f}-gold.json")
        G = json.load(open(gp, encoding="utf-8")) if os.path.exists(gp) else {}
        for k, t in T.items():
            texts[str(k)] = t
            gold[str(k)] = [[int(e[0]), int(e[1]), e[2]] for e in G.get(k, [])]
    return texts, gold


def load_texts(path):
    return {str(k): v for k, v in json.load(open(path, encoding="utf-8")).items()}


def dev_split(gold_by_fold, dev_frac, seed):
    """Document-level split: dev_frac of the documents of EVERY fold (deterministic)."""
    dev = []
    for ids in gold_by_fold:
        ids = sorted(ids)
        n = int(round(dev_frac * len(ids)))
        perm = np.random.default_rng(seed).permutation(len(ids))
        dev += [ids[i] for i in perm[:n]]
    return set(dev)


# ---------------------------------------------------------------------------------------------- tokenizer
def load_tokenizer(tok_root):
    """The Phase-1 tokenizer (module + json live in the Phase-1 data folder)."""
    sys.path.insert(0, tok_root)
    try:
        import legal_bpe as L
    except ImportError:
        import legal_bpe_v2 as L
    tdir = os.path.join(tok_root, "tokenizer")
    cands = sorted(f for f in os.listdir(tdir) if f.endswith(".json"))
    pick = cands[-1]
    mp = os.path.join(tok_root, "manifest.json")
    if os.path.exists(mp):                              # exactly the file Phase 1 used (p1_loader.load_tokenizer)
        want = f"legal_bpe_{json.load(open(mp))['vocab_size']}.json"
        if want in cands:
            pick = want
    return L.LegalBPE.load(os.path.join(tdir, pick))


class OffsetTokenizer:
    """Wraps LegalBPE: encode(text) -> ids, start, end, core_end (numpy int arrays, char offsets)."""

    def __init__(self, tok, pretok="piece"):
        assert pretok in ("piece", "native")
        self.tok, self.pretok = tok, pretok
        self.eow = "</w>"
        self.byte_syms = [f"<0x{b:02x}>" for b in range(256)]
        self.cache = {}

    def _bpe(self, s, eow):
        """BPE ids of string s; eow=True appends the end-of-unit marker (as Phase-1 does for every unit)."""
        key = (s, eow)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        syms = [self.byte_syms[b] for b in s.encode("utf-8", "surrogatepass")] + ([self.eow] if eow else [])
        ranks = self.tok.merge_ranks
        while True:
            best_r, best_i = None, -1
            for i in range(len(syms) - 1):
                r = ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_r is None or r < best_r):
                    best_r, best_i = r, i
            if best_r is None:
                break
            syms[best_i:best_i + 2] = [syms[best_i] + syms[best_i + 1]]
        ids = []
        for sym in syms:
            if sym == self.eow:
                continue
            i = self.tok.stoi.get(sym)
            if i is None:                     # cannot happen for a well-formed vocab; fall back to bytes
                ids.extend(self.tok.stoi[bs] for bs in re.findall(r"<0x[0-9a-f]{2}>", sym))
            else:
                ids.append(i)
        self.cache[key] = ids
        return ids

    def _pieces(self, text):
        """-> list of (char_offset, piece_string, eow)."""
        out = []
        for m in UNIT_RE.finditer(text):
            u, u0 = m.group(), m.start()
            core = u.rstrip()
            if self.pretok == "native" or not core:
                for k in range(0, len(u), MAX_PIECE_CHARS):
                    out.append((u0 + k, u[k:k + MAX_PIECE_CHARS], True))
                continue
            ps = [(p.start(), p.end()) for p in CORE_PIECE_RE.finditer(core)]
            # CORE_PIECE_RE covers every non-space char of the core, so pieces tile it exactly
            for j, (a, b) in enumerate(ps):
                last = j == len(ps) - 1
                if last:
                    b = len(u)                # trailing whitespace stays on the last piece
                for k in range(a, b, MAX_PIECE_CHARS):
                    e = min(b, k + MAX_PIECE_CHARS)
                    out.append((u0 + k, u[k:e], last and e == b))
        return out

    def encode(self, text):
        ids, st, en, ce = [], [], [], []
        idb = self.tok._id_bytes
        for p0, piece, eow in self._pieces(text):
            pid = self._bpe(piece, eow)
            starts = [0]                                   # byte offset where each char starts (+ total)
            for ch in piece:
                starts.append(starts[-1] + len(ch.encode("utf-8", "surrogatepass")))
            bpos = 0
            for i in pid:
                bs, be = bpos, bpos + len(idb[i])
                bpos = be
                cs = bisect.bisect_right(starts, bs) - 1  # char containing the first byte
                cend = bisect.bisect_left(starts, be)      # first char boundary at/after the last byte
                core_len = len(piece[cs:cend].rstrip())
                ids.append(i)
                st.append(p0 + cs)
                en.append(p0 + cend)
                ce.append(p0 + cs + core_len)
            assert bpos == starts[-1], f"byte mismatch in piece {piece!r}"
        a = lambda x: np.asarray(x, dtype=np.int64)
        return a(ids), a(st), a(en), a(ce)


# ---------------------------------------------------------------------------------------------- documents
class Doc:
    __slots__ = ("id", "text", "ids", "st", "en", "ce", "tags", "gold")

    def __init__(self, did, text, otok, gold=None, tag_of=None):
        self.id, self.text, self.gold = did, text, gold
        self.ids, self.st, self.en, self.ce = otok.encode(text)
        self.tags = bio_tags(self, gold, tag_of) if (gold is not None and tag_of is not None) else None

    def __len__(self):
        return len(self.ids)


def bio_tags(doc, gold, tag_of):
    """Token tags from char spans. A word token belongs to an entity if its core overlaps the span;
    a pure-whitespace token belongs to it only if it lies strictly inside the span."""
    n = len(doc.ids)
    tags = np.zeros(n, dtype=np.int64)
    st, en, ce = doc.st, doc.en, doc.ce
    is_ws = ce <= st
    for s, e, lab in sorted(gold, key=lambda x: (x[0], x[1])):
        lo = np.searchsorted(en, s, side="right")          # first token ending after s
        hi = np.searchsorted(st, e, side="left")           # tokens starting before e
        idx = [i for i in range(lo, hi)
               if (not is_ws[i] and st[i] < e and ce[i] > s) or (is_ws[i] and st[i] > s and en[i] < e)]
        if not idx:
            continue
        b, ii = tag_of[lab]
        tags[idx[0]] = b
        tags[idx[1:]] = ii
    return tags


def decode_spans(doc, tags, labs):
    """BIO tag ids (len n) -> [[start, end, label]] in char offsets (entity = first..last word token).
    Any non-O tag opens an entity; following I tags of the same label extend it."""
    out = []
    n = len(tags)
    is_ws = doc.ce <= doc.st
    i = 0
    while i < n:
        t = int(tags[i])
        if t == 0:
            i += 1
            continue
        k = (t - 1) // 2
        j = i + 1
        while j < n and int(tags[j]) == 2 + 2 * k:
            j += 1
        words = [p for p in range(i, j) if not is_ws[p]]
        if words:
            out.append([int(doc.st[words[0]]), int(doc.ce[words[-1]]), labs[k]])
        i = j
    return out


# ---------------------------------------------------------------------------------------------- windows
def window_starts(n, W, stride, offset=0):
    """Start positions covering [0, n) with windows of W tokens (the last window ends exactly at n)."""
    if n <= W:
        return [0]
    s = list(range(offset, n - W + 1, stride))
    if not s or s[0] != 0:
        s = [0] + s
    if s[-1] != n - W:
        s.append(n - W)
    return sorted(set(s))


def make_windows(docs, W, stride, rng=None):
    """-> list of (doc_index, start). With rng, a random offset per document (augmentation)."""
    out = []
    for di, d in enumerate(docs):
        off = int(rng.integers(0, stride)) if rng is not None else 0
        out += [(di, s) for s in window_starts(len(d), W, stride, off)]
    return out


def prefix_len(doc, P):
    """Number of header tokens prepended to every window of this doc (0 if prefix off or the doc is shorter)."""
    return P if (P and len(doc) >= P) else 0


def window_arrays(doc, s, W, P=0):
    """ids, tags, valid for one window, right-padded to a fixed length. With P > 0 the layout is
    [first P tokens of the document (the header: parties, court, date) | window of W tokens]; the prefix carries
    no loss (tag IGNORE) and the predictions of a window are read from positions P.. (see prefix_len)."""
    n = len(doc)
    e = min(n, s + W)
    P = prefix_len(doc, P)
    L = W + (P if P else 0)
    ids = np.zeros(L, dtype=np.int64)
    tags = np.full(L, IGNORE, dtype=np.int64)
    valid = np.zeros(L, dtype=bool)
    if P:
        ids[:P] = doc.ids[:P]
        valid[:P] = True
    ids[P:P + e - s] = doc.ids[s:e]
    valid[P:P + e - s] = True
    if doc.tags is not None:
        tags[P:P + e - s] = doc.tags[s:e]
    return ids, tags, valid


def centrality(W, mode):
    """Score of each window position for choosing which window predicts a token (higher = better)."""
    j = np.arange(W)
    return j.astype(np.float64) if mode == "left" else np.minimum(j, W - 1 - j).astype(np.float64)


# ---------------------------------------------------------------------------------------------- decoding
def allowed_transitions(n_tags):
    """(from, to) bool: I-X may only follow B-X or I-X. Index 0 = O."""
    A = np.ones((n_tags, n_tags), dtype=bool)
    for to in range(2, n_tags, 2):                       # I tags
        A[:, to] = False
        A[to - 1, to] = True                             # B-X -> I-X
        A[to, to] = True                                 # I-X -> I-X
    start = np.ones(n_tags, dtype=bool)
    start[2::2] = False
    return A, start


def viterbi(logp, A=None, start=None):
    """logp (n, T) log-probabilities -> best BIO-valid tag sequence (numpy)."""
    n, T = logp.shape
    if A is None:
        A, start = allowed_transitions(T)
    trans = np.where(A, 0.0, -1e9)
    score = logp[0] + np.where(start, 0.0, -1e9)
    back = np.zeros((n, T), dtype=np.int32)
    for i in range(1, n):
        cand = score[:, None] + trans                    # (from, to)
        back[i] = cand.argmax(0)
        score = cand.max(0) + logp[i]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(score.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def greedy(logp):
    """argmax + repair: an I-X that does not continue X becomes B-X."""
    t = logp.argmax(1).astype(np.int64)
    for i in range(len(t)):
        if t[i] > 0 and t[i] % 2 == 0:
            prev = t[i - 1] if i else 0
            if prev not in (t[i] - 1, t[i]):
                t[i] -= 1
    return t
