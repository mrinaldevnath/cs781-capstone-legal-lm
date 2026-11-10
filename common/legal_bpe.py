# legal_bpe_v2.py
r"""
CS781 Phase 1 - legal BPE tokenizer + text rules (library; imported by every other script).

Contents
  1. clean_v2 / keep_doc       frozen text cleaning: the shipped tokenizer was trained on its output
  2. repair layer              repair_text / clean_for_training / repair_self_test: production-only
                               repairs applied BEFORE clean_v2 (see the comment above that section)
  3. exact_key / hash_select / is_val_index
                               dedup key, RNG-free subsampling, frozen validation predicate
  4. LegalBPE / BPETrainer     byte-level BPE (Assignment-1 lineage), JSON tokenizer file

Tokenizer design (why each choice)
  * Byte-level symbols <0xNN> + end-of-unit marker </w>: every string is encodable, so there is
    no <unk> and decode(encode(x)) == x exactly (checked on raw text, 0 failures).
  * Units are r"\S+\s*|\s+" (word + trailing whitespace): whitespace is part of the token, so
    newlines/indentation round-trip. Units longer than MAX_UNIT_CHARS are split on CHARACTER
    boundaries in units(), used by both training and encoding (so they cannot disagree).
  * Greedy merge of the most frequent adjacent pair; ties broken by (highest count, then
    lexicographically smallest pair); minimum pair frequency 2. Deterministic, so the same
    sample always gives the same file. Prefix property: the first (V-260) merges of a larger run
    ARE the V-token tokenizer (truncate()).
  * Specials <pad>=0 <s>=1 </s>=2 <mask>=3: the minimal set shared by decoder-only and BART-style
    models. No <unk>. Base alphabet = 4 + 256 = 260 symbols.
  * Encoding raises KeyError on a symbol that is not in the vocab instead of dropping it.
  * Training counts units per DOCUMENT (jsonl), exactly as encoding sees them.
"""

import hashlib
import heapq
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------- vocab
SPECIALS = ["<pad>", "<s>", "</s>", "<mask>"]
PAD_ID, BOS_ID, EOS_ID, MASK_ID = 0, 1, 2, 3
EOW = "</w>"
BYTE_SYMS = [f"<0x{b:02x}>" for b in range(256)]
BASE_ITOS = SPECIALS + BYTE_SYMS
N_BASE = len(BASE_ITOS)                       # 260

UNIT_RE = re.compile(r"\S+\s*|\s+")
MAX_UNIT_CHARS = 96
_SYMBOL_RE = re.compile(r"<0x[0-9a-f]{2}>|</w>")


def units(text):
    """Pretokenization shared by training AND encoding (must match)."""
    out = []
    for u in UNIT_RE.findall(text):
        if len(u) <= MAX_UNIT_CHARS:
            out.append(u)
        else:  # pathological runs (e.g. 5,000 underscores): split on char boundaries
            out.extend(u[k:k + MAX_UNIT_CHARS] for k in range(0, len(u), MAX_UNIT_CHARS))
    return out


# ----------------------------------------------------------------------------- cleaning
_ZW_RE = re.compile(r"[\u200b\u2060\ufeff\u00ad]")          # ZWSP, word joiner, BOM, soft hyphen
_SURR_RE = re.compile(r"[\ud800-\udfff]")                    # lone surrogates from bad JSON escapes
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")      # C0/C1 controls except \t \n
_SPACE_RE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_MULTI_NL = re.compile(r"\n{3,}")
_WS_RE = re.compile(r"\s+")
_BOILER_RE = re.compile(
    r"^\s*(?:page\s*:?\s*\d+.*|\d{1,4}\s*|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*)$", re.I)

_LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")
_NONLATIN_RE = re.compile(
    r"[\u0370-\u06ff\u0900-\u0fff\u1000-\u109f\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")
_DEVA_RE = re.compile(r"[\u0900-\u097f]")


def clean_v2(t):
    """Minimal, lossless-where-it-matters cleanup for LM pretraining text.
    Keeps punctuation, case, numbers, citations and all printable Unicode."""
    t = unicodedata.normalize("NFC", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    t = _SURR_RE.sub("", t)
    t = _ZW_RE.sub("", t)
    t = _CTRL_RE.sub("", t)
    lines = []
    for ln in t.split("\n"):
        if _BOILER_RE.match(ln):          # page-number-only / date-only lines (v1 rule, kept)
            continue
        lines.append(_SPACE_RE.sub(" ", ln).strip())
    t = "\n".join(lines)
    t = _MULTI_NL.sub("\n\n", t)
    return t.strip()


def script_counts(t, probe=20000):
    """(latin letters, non-latin letters, devanagari chars) on a prefix probe."""
    p = t[:probe]
    return len(_LATIN_RE.findall(p)), len(_NONLATIN_RE.findall(p)), len(_DEVA_RE.findall(p))


def nonlatin_frac(t):
    lat, non, _ = script_counts(t)
    return non / max(lat + non, 1)


def keep_doc(t, min_words=40, max_words=200_000, max_nonlatin=0.10):
    """Document filter applied to CLEANED text. Returns (keep, n_words, reason)."""
    nw = len(t.split())
    if nw < min_words:
        return False, nw, "short"
    if nw > max_words:
        return False, nw, "long"
    if nonlatin_frac(t) > max_nonlatin:
        return False, nw, "script"
    return True, nw, "ok"


# ----------------------------------------------------------------------------- repair layer
# clean_v2 above is FROZEN: the shipped 32k tokenizer was trained on its output. It deletes soft
# hyphens and C1 control characters and it keeps U+FFFD. A raw-character audit of the full Train
# files (evidence/data_analysis_report.json) showed those characters usually carry meaning or are
# damage that can be undone, so the production pipeline repairs them FIRST:
#   * U+00AD soft hyphen is used as a real hyphen in SC ("NON<SHY>REPORTABLE")      -> "-"
#   * UTF-8 decoded as cp1252/latin-1 ("a-circumflex euro oe" = left double quote)   -> original char
#   * lone cp1252 bytes left as C1 controls in Acts (0x97 em dash, 0x85 ellipsis ...) -> cp1252 char
#   * UTF-8 punctuation with the high bits stripped in SC: "b", U+FFFD, "&" = ellipsis -> original char
#   * U+F0B7 (Symbol-font bullet) -> U+2022; bidi embedding marks -> deleted; remaining U+FFFD -> deleted
# Documents that are mostly debris (raw RTF, zipped Office files, random bytes; seen in Tribunals)
# are dropped by is_garbage(). Repairs change well under 0.01% of characters, so the tokenizer
# (trained without them) is unaffected in practice; see README "Known limitations".
_CONT = {}                                       # char -> byte value, for UTF-8 continuation bytes 0x80-0xBF
for _b in range(0x80, 0xC0):
    _CONT[chr(_b)] = _b                          # latin-1 view (C1 controls, U+00A0-U+00BF)
    try:
        _CONT[bytes([_b]).decode("cp1252")] = _b  # cp1252 view (euro, quotes, dashes, bullet, ...)
    except UnicodeDecodeError:
        pass
_CC = "[" + "".join(re.escape(c) for c in sorted(_CONT)) + "]"
_MOJI3_RE = re.compile("\xe2(" + _CC + ")(" + _CC + ")")        # E2 xx xx: general punctuation, rupee, euro
_MOJI2_RE = re.compile("([\xc2\xc3])(" + _CC + ")")             # C2/C3 xx: Latin-1 range characters
_B7 = {"&": "\U00002026", "\x13": "\U00002013", "\x14": "\U00002014", "\x18": "\U00002018",
       "\x19": "\U00002019", "\x1c": "\U0000201c", "\x1d": "\U0000201d", '"': "\U00002022"}
_B7_RE = re.compile("b\U0000fffd([" + re.escape("".join(_B7)) + "])")
_C1_RE = re.compile("[\x80-\x9f]")
_BIDI_RE = re.compile("[\U0000200e\U0000200f\U0000202a-\U0000202e\U00002066-\U00002069]")
# debris characters: U+FFFD, C0 controls (except tab/LF/FF/CR), private use, Samaritan..Arabic-ext
# (the latter is how OLE/Word binary headers decode)
_JUNK_RE = re.compile("[\U0000fffd\x00-\x08\x0b\x0e-\x1f\U0000e000-\U0000f8ff\U00000800-\U0000089f]")
_FFFD_RE = re.compile("\U0000fffd")
_BIN_MARKERS = ("{\\rtf", "PK\x03\x04", "[Content_Types].xml", "mimetypeapplication/")


def _moji3(m):
    try:
        c = bytes([0xE2, _CONT[m.group(1)], _CONT[m.group(2)]]).decode("utf-8")
    except UnicodeDecodeError:
        return m.group(0)
    return c if "\U00002000" <= c <= "\U000020ff" else m.group(0)


def _moji2(m):
    try:
        c = bytes([ord(m.group(1)), _CONT[m.group(2)]]).decode("utf-8")
    except UnicodeDecodeError:
        return m.group(0)
    return c if c >= "\xa0" else m.group(0)


def _c1(m):
    try:
        return bytes([ord(m.group(0))]).decode("cp1252")
    except UnicodeDecodeError:                   # 0x81 0x8D 0x8F 0x90 0x9D are undefined in cp1252
        return ""


def repair_text(t):
    """Repairs applied before clean_v2. Returns (text, number_of_debris_characters_left)."""
    if "\xe2" in t:
        t = _MOJI3_RE.sub(_moji3, t)
    if "\xc2" in t or "\xc3" in t:
        t = _MOJI2_RE.sub(_moji2, t)
    if "\U0000fffd" in t:
        t = _B7_RE.sub(lambda m: _B7[m.group(1)], t)
    t = _C1_RE.sub(_c1, t)
    t = t.replace("\xad", "-").replace("\U0000f0b7", "\U00002022")
    t = _BIDI_RE.sub("", t)
    return t, len(_JUNK_RE.findall(t))


def is_garbage(t, junk, junk_max=0.01):
    """Mostly-debris document (more than junk_max of its characters) or an embedded RTF/zip/Office file."""
    if junk > junk_max * max(len(t), 1):
        return True
    head = t[:3000]
    return any(m in head for m in _BIN_MARKERS)


def clean_for_training(raw, min_words=40, max_words=200_000, max_nonlatin=0.10, junk_max=0.01):
    """The full production cleaning of one raw document.
    repair_text -> garbage filter -> delete leftover U+FFFD -> clean_v2 -> keep_doc.
    Returns (cleaned_text, n_words) if kept, else (None, reason) with reason in
    {"garbage", "short", "long", "script"}."""
    t, junk = repair_text(raw)
    if is_garbage(t, junk, junk_max):
        return None, "garbage"
    t = clean_v2(_FFFD_RE.sub("", t))
    ok, nw, why = keep_doc(t, min_words, max_words, max_nonlatin)
    return (t, nw) if ok else (None, why)


REPAIR_TESTS = [   # (raw input, expected output of repair_text + delete U+FFFD + clean_v2)
    ("NON\xadREPORTABLE appeal", "NON-REPORTABLE appeal"),
    ("Section 3\x97Definitions\x85 the court\x92s",
     "Section 3\U00002014Definitions\U00002026 the court\U00002019s"),
    ("\xe2\U000020ac\U00000153quoted\xe2\U000020ac\x9d and \xe2\x80\x9cq\xe2\x80\x9d",
     "\U0000201cquoted\U0000201d and \U0000201cq\U0000201d"),
    ("Premji Nathu b\U0000fffd& Appellant", "Premji Nathu \U00002026 Appellant"),
    ("\U0000f0b7 item one", "\U00002022 item one"),
    ("Rs.\xc2\xa0500 caf\xc3\xa9", "Rs. 500 caf\xe9"),
    # text that must pass through untouched: rupee sign, curly quotes, Devanagari
    ("plain \U000020b9 5,00,000/- \U0000201ctext\U0000201d \U00000928\U0000094d\U0000092f\U0000093e\U0000092f",
     "plain \U000020b9 5,00,000/- \U0000201ctext\U0000201d \U00000928\U0000094d\U0000092f\U0000093e\U0000092f"),
    ("\U00000645\U0000202bx\U0000202c y\U0000fffdz", "\U00000645x yz"),
]


def repair_self_test():
    """Raises AssertionError if the repair layer does not behave as documented (run at every start-up)."""
    for raw, want in REPAIR_TESTS:
        got = clean_v2(_FFFD_RE.sub("", repair_text(raw)[0]))
        assert got == want, f"repair self-test failed: {raw!r} -> {got!r}, expected {want!r}"


def _h64(s):
    return int.from_bytes(hashlib.md5(s.encode("utf-8", "surrogatepass")).digest()[:8], "little")


def exact_key(t):
    """Exact duplicate key: md5 of whitespace-normalised full text (case kept)."""
    return _h64(_WS_RE.sub(" ", t).strip())


def prefix_key_v1(t):
    """v1 near-dup key (first 1200 chars, lowercased) - kept only to measure its false positives."""
    return _h64(_WS_RE.sub(" ", t[:1200].lower()))


def hash_select(tag, i, rate):
    """Deterministic, RNG-free Bernoulli(rate) selection of record i."""
    if rate >= 1.0:
        return True
    h = int.from_bytes(hashlib.md5(f"{tag}:{i}".encode()).digest()[:4], "little")
    return h < rate * 4294967296.0


def is_val_index(i):
    """Frozen ~1% validation predicate (record index within its source file)."""
    return i % 100 == 0


# ----------------------------------------------------------------------------- tokenizer
class LegalBPE:
    def __init__(self, merges, itos=None):
        self.merges = [list(m) for m in merges]
        self.merge_ranks = {}
        for r, (a, b) in enumerate(self.merges):
            self.merge_ranks.setdefault((a, b), r)
        self.itos = list(itos) if itos is not None else BASE_ITOS + [a + b for a, b in self.merges]
        self.stoi = {}
        for i, tok in enumerate(self.itos):
            self.stoi.setdefault(tok, i)          # duplicate strings: first id wins
        self._id_bytes = [self._tok_bytes(t) for t in self.itos]

    @staticmethod
    def _tok_bytes(tok):
        if tok in SPECIALS:
            return b""
        parts = _SYMBOL_RE.findall(tok)
        if "".join(parts) != tok:
            raise ValueError(f"malformed token {tok!r}")
        return bytes(int(m[3:5], 16) for m in parts if m != EOW)

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode_word(self, word):
        tokens = [BYTE_SYMS[b] for b in word.encode("utf-8", "surrogatepass")] + [EOW]
        ranks = self.merge_ranks
        while True:
            best_rank, best_i = None, -1
            for i in range(len(tokens) - 1):
                r = ranks.get((tokens[i], tokens[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            tokens[best_i:best_i + 2] = [tokens[best_i] + tokens[best_i + 1]]
        return tokens

    def encode_unit_ids(self, unit, cache=None):
        if cache is not None:
            hit = cache.get(unit)
            if hit is not None:
                return hit
        ids = []
        for s in self.encode_word(unit):
            if s == EOW:                          # bare end marker carries no bytes
                continue
            i = self.stoi.get(s)
            if i is None:
                raise KeyError(f"symbol {s!r} from unit {unit!r} not in vocab")
            ids.append(i)
        if cache is not None:
            cache[unit] = ids
        return ids

    def encode_to_ids(self, text, cache=None):
        ids = []
        for u in units(text):
            ids.extend(self.encode_unit_ids(u, cache))
        return ids

    def decode_bytes(self, ids):
        return b"".join(self._id_bytes[i] for i in ids)

    def decode(self, ids):
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def truncate(self, target_vocab):
        """Smaller tokenizer = first (target - 260) merges (greedy-BPE prefix property)."""
        k = target_vocab - N_BASE
        if not 0 <= k <= len(self.merges):
            raise ValueError(f"cannot truncate {self.vocab_size} -> {target_vocab}")
        return LegalBPE(self.merges[:k])

    def save(self, path, meta=None):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"format": "legal_bpe_v2", "specials": SPECIALS, "merges": self.merges,
                       "itos": self.itos, "vocab_size": len(self.itos), "meta": meta or {}},
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("specials") != SPECIALS:
            raise ValueError(f"{path}: not a v2 tokenizer (specials {d.get('specials')})")
        return cls(d["merges"], d["itos"])


# ----------------------------------------------------------------------------- trainer
def _merge_scan(syms, a, b, merged):
    out, i, n = [], 0, len(syms)
    while i < n:
        if i + 1 < n and syms[i] == a and syms[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(syms[i])
            i += 1
    return out


class BPETrainer:
    """Incremental BPE (same merges as the A-1 naive trainer): exact pair counts,
    pair -> unit-type index, lazy max-heap; each merge revisits only affected units."""

    def __init__(self, vocab_size, min_pair_freq=2, log_every=60):
        self.vocab_size = vocab_size
        self.min_pair_freq = min_pair_freq
        self.log_every = log_every
        self.merges, self.gain_curve = [], []

    def train(self, unit_freqs):
        t0 = time.time()
        syms, freq = [], []
        for u, f in unit_freqs.items():
            syms.append([BYTE_SYMS[b] for b in u.encode("utf-8", "surrogatepass")] + [EOW])
            freq.append(f)
        pc, idx = Counter(), defaultdict(set)
        for i, s in enumerate(syms):
            f = freq[i]
            for j in range(len(s) - 1):
                p = (s[j], s[j + 1])
                pc[p] += f
                idx[p].add(i)
        heap = [(-c, p[0], p[1]) for p, c in pc.items() if c >= self.min_pair_freq]
        heapq.heapify(heap)
        print(f"[bpe] {len(syms):,} unit types, {sum(freq):,} units ({time.time()-t0:.0f}s)", flush=True)

        max_merges = self.vocab_size - N_BASE
        last = time.time()
        while len(self.merges) < max_merges and heap:
            negc, a, b = heapq.heappop(heap)
            c = pc.get((a, b), 0)
            if c != -negc or c < self.min_pair_freq:
                continue
            merged = a + b
            self.merges.append([a, b])
            self.gain_curve.append(c)
            for i in sorted(idx.pop((a, b), set())):
                s, f = syms[i], freq[i]
                touched = set()
                for j in range(len(s) - 1):
                    p = (s[j], s[j + 1])
                    pc[p] -= f
                    touched.add(p)
                ns = _merge_scan(s, a, b, merged)
                syms[i] = ns
                for j in range(len(ns) - 1):
                    p = (ns[j], ns[j + 1])
                    pc[p] += f
                    idx[p].add(i)
                    touched.add(p)
                for p in sorted(touched):
                    if pc[p] >= self.min_pair_freq:
                        heapq.heappush(heap, (-pc[p], p[0], p[1]))
            if time.time() - last > self.log_every:
                print(f"[bpe] {len(self.merges):,}/{max_merges:,} merges, last count {c:,}", flush=True)
                last = time.time()
        print(f"[bpe] done: {len(self.merges):,} merges in {(time.time()-t0)/60:.1f} min", flush=True)
        return LegalBPE(self.merges)


def count_units_jsonl(paths, field="text"):
    """Unit frequencies over jsonl files, one document per line."""
    total, buf = Counter(), []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                buf.extend(units(json.loads(line)[field]))
                if len(buf) >= 2_000_000:
                    total.update(buf)
                    buf = []
    total.update(buf)
    return total
