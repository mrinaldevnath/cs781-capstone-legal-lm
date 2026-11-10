#!/usr/bin/env python
"""
CS781 Phase 1 - production preprocessing + tokenization.   OFFICIAL TRAIN DATA ONLY.

  raw Train JSON
    -> clean_for_training  (repair mojibake -> drop debris docs -> clean_v2 -> length/script filter)
    -> exact dedup         (within a source; first occurrence wins, across train/val)
    -> split               (record index % 100 == 0 -> val; else hash-subsampled into the train pool)
    -> 32k byte-level BPE  (the frozen tokenizer from inspect_p1.py; never retrained here)
    -> uint16 memmaps      (one stream per source and split; </s> after every document)

Every design choice and the evidence for it: README.md.
All text rules live in legal_bpe_v2.py; this file only orchestrates and enforces invariants.

Output ($OUT, default ./cs781_data_v2):
  manifest.json                config, tokenizer md5, per-source counts / drops / tokens, default mix
  tokenizer/legal_bpe_32768.json, legal_bpe_v2.py, p1_loader.py, prep_p1.py   (provenance copies)
  train/{acts,sc,hc,trib}.bin  uint16 token ids; </s> (id 2) after EVERY document; no <s> stored
  train/{src}.idx.npy          int64 token offsets, length n_docs+1; doc k = bin[idx[k]:idx[k+1]] (incl. </s>)
  train/{src}.docid.npy        int64 original record index of every kept document (traceability)
  val/...                      same layout, held-out documents
  dev_prompts.jsonl            {src, i, prompt (512 words), reference (next 128 words)} from val docs
Mixing, block length and packing are decided at LOAD time (p1_loader.py), never here.

Run (a CPU session is enough; no GPU quota is used):
    pip install ijson huggingface_hub numpy
    export HF_TOKEN=...                       # or LOCAL_DATA_DIR=<folder holding the four train_*.json>
    TOK_PATH=./p1_inspect/bpe_v2_32768.json python prep_p1.py 2>&1 | tee prep_log.txt
Resumable: a finished source is skipped on re-run (delete $OUT/{train,val}/<src>.done.json to redo it).
Env: TOK_PATH, OUT, LOCAL_DATA_DIR, NPROC (worker processes), SMOKE=1 (test hook: any vocab size, tiny batches).
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import hashlib
import json
import multiprocessing as mp
import shutil
import threading
import time
from collections import Counter

import numpy as np

import legal_bpe as L

# ============================================================================ config
TOK_PATH = os.environ.get("TOK_PATH", "./p1_inspect/bpe_v2_32768.json")
OUT = os.environ.get("OUT", "./cs781_data_v2")
LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR")
SMOKE = os.environ.get("SMOKE") == "1"           # test hook only: skips the 32768 check, uses tiny batches
HF_REPO = "Exploration-Lab/CS-781-Capstone"

VOCAB = 32768                                    # fixed decision (README section 5)
SOURCES = ["acts", "sc", "trib", "hc"]           # processing order: small first = early feedback
SRC_FILES = {"acts": "train_acts.json", "sc": "train_supreme_court.json",
             "hc": "train_high_courts.json", "trib": "train_tribunals.json"}

# Train pool = fraction of the (non-val) records kept, chosen by an RNG-free hash of the record index.
# Expected cleaned tokens at 32k: acts ~137M, sc ~236M, hc ~0.34*1.79B ~ 610M, trib ~0.28*1.17B ~ 330M.
POOL_RATE = {"acts": 1.0, "sc": 1.0, "hc": 0.34, "trib": 0.28}
DEFAULT_MIX = {"sc": 0.40, "hc": 0.35, "trib": 0.15, "acts": 0.10}   # recorded in the manifest, applied at load time
PLANNED_TOKENS = {"10h_with_overhead": 1.085e9, "11h_no_overhead": 1.26e9}   # from bench_t4.py, 32k vocab

MIN_WORDS, MAX_WORDS, MAX_NONLATIN = 40, 200_000, 0.10
JUNK_MAX = 0.01                  # drop a document if more than 1% of its characters are debris
DEV_PROMPTS_PER_SRC = 250
NPROC = int(os.environ.get("NPROC", max(1, (os.cpu_count() or 2) - 1)))   # one core is left for the JSON reader
BATCH_CHARS = 4_000_000          # raw characters per worker task
MAX_INFLIGHT = 4 * NPROC         # bounded queue: the reader never runs far ahead of the workers
CACHE_MAX = 1_500_000            # per-worker unit->ids cache; cleared when it grows past this

if SMOKE:
    BATCH_CHARS = 50_000


# ============================================================================ io
def data_path(fname):
    if LOCAL_DATA_DIR:
        return os.path.join(LOCAL_DATA_DIR, fname)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, f"Phase_1_Pretraining/Train/{fname}",
                           repo_type="dataset", token=os.environ.get("HF_TOKEN"))


def stream_items(path):
    """Top-level items of a JSON array, values of a {id: record} object, or jsonl lines - streamed."""
    import ijson
    p = path.replace("\\", "/")
    if "/Test/" in p or os.path.basename(p).lower().startswith("phase1- test"):
        raise RuntimeError(f"refusing to read test data: {path}")
    with open(path, "rb") as f:
        h = f.read(4096).lstrip(b"\xef\xbb\xbf").lstrip()[:1]
        f.seek(0)
        if h == b"[":
            yield from ijson.items(f, "item")
            return
        is_jsonl = False
        try:
            json.loads(f.readline(2_000_000).decode("utf-8-sig"))
            is_jsonl = bool(f.readline(2_000_000).strip())
        except (ValueError, UnicodeDecodeError):
            pass
        f.seek(0)
        if is_jsonl:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        else:
            for _, v in ijson.kvitems(f, ""):
                yield v


def rec_text(rec):
    if isinstance(rec, str):
        return rec
    if isinstance(rec, dict):
        v = rec.get("text")
        if isinstance(v, str):
            return v
    return None


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _h32(s):
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "little")


# ============================================================================ workers
_TOK, _CACHE = None, None


def _init_worker(tok_path):
    global _TOK, _CACHE
    _TOK, _CACHE = L.LegalBPE.load(tok_path), {}


def process_batch(items):
    """items: [(record_index, split, raw_text)] -> [(i, split, reason, n_words, dedup_key, uint16_ids, dev_pair)]"""
    global _CACHE
    if len(_CACHE) > CACHE_MAX:
        _CACHE = {}
    V = _TOK.vocab_size
    out = []
    for i, split, raw in items:
        t, x = L.clean_for_training(raw, MIN_WORDS, MAX_WORDS, MAX_NONLATIN, JUNK_MAX)
        if t is None:
            out.append((i, split, x, 0, None, None, None))
            continue
        nw = x
        ids = _TOK.encode_to_ids(t, _CACHE)
        arr = np.asarray(ids, dtype=np.int64)
        # hard invariants: ids fit the vocab, text never produces a special id, exact lossless round trip
        if arr.size == 0 or arr.min() < len(L.SPECIALS) or arr.max() >= V:
            raise AssertionError(f"bad ids in record {i}: min={arr.min()} max={arr.max()} V={V}")
        if _TOK.decode_bytes(ids) != t.encode("utf-8", "surrogatepass"):
            raise AssertionError(f"round-trip failure in record {i}")
        dev = None
        if split == "val" and nw >= 640:             # 512-word prompt + 128-word reference
            w = t.split()
            dev = (" ".join(w[:512]), " ".join(w[512:640]))
        out.append((i, split, "ok", nw, L.exact_key(t), arr.astype(np.uint16), dev))
    return out


# ============================================================================ tokenizer gate
# Strings that must round-trip and must never yield a special id (ASCII source: unicode via escapes).
SPOT = ["AIR 1973 SC 1461", "(2017) 10 SCC 1", "Section 302 IPC read with Section 34",
        "Art. 21 of the Constitution", "Cr.P.C.", "\U000020b9 5,00,000/-", "Rs. 5,00,000/-",
        "\U0000201cthe appellant\U00002019s counsel\U0000201d",
        "\U00000928\U0000094d\U0000092f\U0000093e\U0000092f\U0000093e\U00000932\U0000092f",   # Devanagari
        "_" * 300, "literal </s> <s> <pad> <mask> in text", "W.P.(C) No. 1234/2019",
        "Hon'ble Mr. Justice", "  double  spaces\n\nand newlines\t"]


def tokenizer_gate(tok):
    """Fail fast, before any data is read, if the tokenizer or the repair layer is not what we expect."""
    rep = {"path": TOK_PATH, "md5": md5_file(TOK_PATH), "vocab_size": tok.vocab_size}
    if tok.itos[:len(L.SPECIALS)] != L.SPECIALS:
        raise SystemExit(f"special tokens wrong: {tok.itos[:4]}")
    if not SMOKE and tok.vocab_size != VOCAB:
        raise SystemExit(f"{TOK_PATH} has vocab {tok.vocab_size}, expected {VOCAB}")
    L.repair_self_test()
    spot = []
    for s in SPOT:
        ids = tok.encode_to_ids(s, {})
        if tok.decode_bytes(ids) != s.encode("utf-8"):
            raise SystemExit(f"round trip failed on {s!r}")
        if min(ids) < len(L.SPECIALS):
            raise SystemExit(f"text produced a special id: {s!r} -> {ids}")
        spot.append({"text": s[:40], "n": len(ids),
                     "pieces": [tok.decode_bytes([k]).decode("utf-8", "replace") for k in ids][:20]})
    big = " ".join(SPOT) * 50
    if tok.encode_to_ids(big, {}) != tok.encode_to_ids(big, None):
        raise SystemExit("encoding is not deterministic / cache-dependent")
    longest = sorted(range(len(L.BASE_ITOS), tok.vocab_size), key=lambda k: -len(tok._id_bytes[k]))[:50]
    rep["spot"] = spot
    rep["longest_tokens"] = [tok._id_bytes[k].decode("utf-8", "replace") for k in longest]
    print("[gate] tokenizer OK:", rep["md5"], f"V={tok.vocab_size}", flush=True)
    for s in spot:
        print(f"   {s['n']:3d} | {' | '.join(s['pieces'])}", flush=True)
    print("   longest tokens:", [x[:30] for x in rep["longest_tokens"][:15]], flush=True)
    return rep


# ============================================================================ per source
class _Writer:
    """Append-only token stream + offsets for one (split, source)."""

    def __init__(self, split, src):
        self.dir = f"{OUT}/{split}"
        os.makedirs(self.dir, exist_ok=True)
        self.base = f"{self.dir}/{src}"
        self.f = open(self.base + ".bin.tmp", "wb")
        self.offsets, self.docids, self.n = [0], [], 0
        self.eos = np.array([L.EOS_ID], dtype=np.uint16).tobytes()

    def add(self, i, arr):
        self.f.write(arr.tobytes())
        self.f.write(self.eos)
        self.n += arr.size + 1
        self.offsets.append(self.n)
        self.docids.append(i)

    def close(self, stats):
        self.f.close()
        os.replace(self.base + ".bin.tmp", self.base + ".bin")
        np.save(self.base + ".idx.npy", np.asarray(self.offsets, dtype=np.int64))
        np.save(self.base + ".docid.npy", np.asarray(self.docids, dtype=np.int64))
        assert os.path.getsize(self.base + ".bin") == 2 * self.n
        json.dump(stats, open(self.base + ".done.json", "w"), indent=1)   # written last = "source finished" marker


def run_source(src, pool):
    done = {sp: f"{OUT}/{sp}/{src}.done.json" for sp in ("train", "val")}
    dev_path = f"{OUT}/dev/{src}.jsonl"
    if all(os.path.exists(p) for p in done.values()) and os.path.exists(dev_path):
        print(f"[{src}] cached", flush=True)
        return {sp: json.load(open(p)) for sp, p in done.items()}, \
               [json.loads(l) for l in open(dev_path, encoding="utf-8")]

    t0 = time.time()
    path = data_path(SRC_FILES[src])
    rstats = Counter()                       # touched only by the reader generator
    sem = threading.BoundedSemaphore(MAX_INFLIGHT)

    def tasks():
        batch, nch = [], 0
        for i, rec in enumerate(stream_items(path)):
            rstats["records"] += 1
            raw = rec_text(rec)
            if raw is None:
                rstats["no_text"] += 1
                continue
            if not raw.strip():
                rstats["empty"] += 1
                continue
            if L.is_val_index(i):
                split = "val"
            elif L.hash_select(f"pool:{src}", i, POOL_RATE[src]):
                split = "train"
            else:
                rstats["not_in_pool"] += 1
                continue
            batch.append((i, split, raw))
            nch += len(raw)
            if nch >= BATCH_CHARS:
                sem.acquire()
                yield batch
                batch, nch = [], 0
        if batch:
            sem.acquire()
            yield batch

    W = {sp: _Writer(sp, src) for sp in ("train", "val")}
    S = {sp: Counter() for sp in ("train", "val")}
    seen, dev, last = set(), [], time.time()
    for res in pool.imap(process_batch, tasks()):       # imap is ordered => deterministic output
        sem.release()
        for i, split, why, nw, key, arr, dv in res:
            st = S[split]
            st["selected"] += 1
            if why != "ok":
                st[f"drop_{why}"] += 1
                continue
            if key in seen:                  # first occurrence (lowest record index) wins, across splits
                st["drop_dup"] += 1
                continue
            seen.add(key)
            W[split].add(i, arr)
            st["docs"] += 1
            st["words"] += nw
            if dv is not None:
                dev.append((_h32(f"dev:{src}:{i}"), i, dv))
        if time.time() - last > 60:
            last = time.time()
            print(f"[{src}] {rstats['records']:,} records read | train {W['train'].n/1e6:,.0f}M tok "
                  f"| val {W['val'].n/1e6:,.1f}M tok | {(last-t0)/60:.1f} min", flush=True)

    out = {}
    for sp in ("train", "val"):
        st = dict(S[sp])
        st["tokens"] = W[sp].n               # includes one </s> per document
        st["tokens_per_word"] = round((W[sp].n - st.get("docs", 0)) / max(st.get("words", 1), 1), 4)
        st["reader"] = dict(rstats)
        st["pool_rate"] = POOL_RATE[src] if sp == "train" else "i%100==0"
        st["minutes"] = round((time.time() - t0) / 60, 1)
        W[sp].close(st)
        out[sp] = st
    dev.sort()
    rows = [{"src": src, "i": i, "prompt": p, "reference": r} for _, i, (p, r) in dev[:DEV_PROMPTS_PER_SRC]]
    os.makedirs(f"{OUT}/dev", exist_ok=True)
    with open(dev_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tr = out["train"]
    drops = {k: v for k, v in tr.items() if k.startswith("drop_")}
    print(f"[{src}] done {tr['minutes']} min | train docs {tr.get('docs',0):,} tokens {tr['tokens']/1e6:,.1f}M "
          f"| val docs {out['val'].get('docs',0):,} tokens {out['val']['tokens']/1e6:,.2f}M | drops {drops}", flush=True)
    g = tr.get("drop_garbage", 0) / max(tr.get("selected", 1), 1)
    if g > 0.03:
        print(f"[{src}] WARNING: {g:.1%} of documents dropped as garbage - inspect before training", flush=True)
    return out, rows


# ============================================================================ main
def verify_outputs(V):
    """Re-open every memmap: sizes, offsets, </s> placement, id range."""
    for sp in ("train", "val"):
        for s in SOURCES:
            b = np.memmap(f"{OUT}/{sp}/{s}.bin", dtype=np.uint16, mode="r")
            idx = np.load(f"{OUT}/{sp}/{s}.idx.npy")
            assert idx[0] == 0 and idx[-1] == b.size and np.all(np.diff(idx) >= 2), (sp, s)
            if b.size:
                assert int(b.max()) < V, (sp, s)
                assert np.all(b[idx[1:] - 1] == L.EOS_ID), (sp, s)             # </s> ends every document
                assert int((b == L.EOS_ID).sum()) == idx.size - 1, (sp, s)     # and appears nowhere else
                assert int((b < len(L.SPECIALS)).sum()) == idx.size - 1, (sp, s)
    print("[verify] memmaps OK", flush=True)


def main():
    T0 = time.time()
    import ijson
    print(f"ijson backend: {ijson.backend}   workers: {NPROC}   (want yajl2_c; 'python' is ~10x slower)", flush=True)
    os.makedirs(f"{OUT}/tokenizer", exist_ok=True)
    tok = L.LegalBPE.load(TOK_PATH)
    gate = tokenizer_gate(tok)
    shutil.copy(TOK_PATH, f"{OUT}/tokenizer/legal_bpe_{tok.vocab_size}.json")
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in ("legal_bpe_v2.py", "p1_loader.py", "prep_p1.py"):
        if os.path.exists(os.path.join(here, fn)):
            shutil.copy(os.path.join(here, fn), f"{OUT}/{fn}")

    stats, dev_rows = {}, []
    with mp.Pool(NPROC, initializer=_init_worker, initargs=(TOK_PATH,)) as pool:
        for s in SOURCES:
            stats[s], rows = run_source(s, pool)
            dev_rows += rows
    with open(f"{OUT}/dev_prompts.jsonl", "w", encoding="utf-8") as f:
        for r in dev_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    verify_outputs(tok.vocab_size)

    train_tok = {s: stats[s]["train"]["tokens"] for s in SOURCES}
    epochs = {k: {s: round(DEFAULT_MIX[s] * T / max(train_tok[s], 1), 2) for s in SOURCES}
              for k, T in PLANNED_TOKENS.items()}
    manifest = {
        "format": "cs781_p1_v2", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dtype": "uint16", "eos_id": L.EOS_ID, "specials": L.SPECIALS, "vocab_size": tok.vocab_size,
        "layout": "doc k = bin[idx[k]:idx[k+1]], ends with </s>; no <s> stored",
        "tokenizer": gate, "sources": SOURCES, "files": SRC_FILES,
        "config": {"pool_rate": POOL_RATE, "val_rule": "record_index % 100 == 0",
                   "min_words": MIN_WORDS, "max_words": MAX_WORDS, "max_nonlatin": MAX_NONLATIN,
                   "junk_max": JUNK_MAX,
                   "dedup": "exact md5 of whitespace-normalised cleaned text, within source, "
                            "first occurrence wins across train/val",
                   "cleaning": "legal_bpe_v2.clean_for_training: repair_text -> garbage filter -> "
                               "delete U+FFFD -> clean_v2 -> keep_doc"},
        "default_mix": DEFAULT_MIX, "planned_tokens": PLANNED_TOKENS,
        "epochs_at_default_mix": epochs, "stats": stats,
        "totals": {"train_tokens": sum(train_tok.values()),
                   "val_tokens": sum(stats[s]["val"]["tokens"] for s in SOURCES),
                   "dev_prompts": len(dev_rows)},
        "minutes": round((time.time() - T0) / 60, 1),
    }
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1, ensure_ascii=False)

    print("\n==== PREP SUMMARY ====")
    print(f"{'src':5s} {'train docs':>11s} {'train tok':>10s} {'val docs':>9s} {'val tok':>8s} {'tok/word':>8s}  drops(train)")
    for s in SOURCES:
        tr, va = stats[s]["train"], stats[s]["val"]
        d = {k[5:]: v for k, v in tr.items() if k.startswith("drop_")}
        print(f"{s:5s} {tr.get('docs',0):11,d} {tr['tokens']/1e6:9.1f}M {va.get('docs',0):9,d} "
              f"{va['tokens']/1e6:7.2f}M {tr['tokens_per_word']:8.3f}  {d}")
    print(f"total train {manifest['totals']['train_tokens']/1e9:.3f}B tokens "
          f"({2*manifest['totals']['train_tokens']/1e9:.2f} GB) | val {manifest['totals']['val_tokens']/1e6:.1f}M "
          f"| dev prompts {len(dev_rows)}")
    print(f"epochs at default mix {DEFAULT_MIX}: {epochs}")
    print(f"done in {manifest['minutes']} min -> {OUT}")


if __name__ == "__main__":
    main()
