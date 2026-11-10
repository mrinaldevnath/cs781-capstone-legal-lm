#!/usr/bin/env python
"""
CS781 Phase 1 - measurement stage. OFFICIAL TRAIN DATA ONLY: never opens any Test/ file.

Measures only what changes a pending decision (README sections 3-5):
  A  schema / malformed records                    -> text-field extraction, cleaning
  B  exact vs v1-prefix duplicates, cross-source   -> dedup method
  C  script / non-ASCII / mojibake profile         -> language-filter threshold, encoding fixes
  D  cleaned size per source                       -> per-source pool sizes (data budget)
  E  BPE 16k / 32k / 48k on held-out val docs:     -> vocabulary choice
     tokens/word, chars/token, raw round-trip, under-trained tokens,
     512+128-word window lengths (guard for block length 1024)
  F  unigram source classifier validated on held-out val docs, then applied to the
     Phase 2-7 TRAIN texts (the data our Phase-1 model must serve) -> mix evidence
     (disable with DOWNSTREAM=0)

It ALSO TRAINS THE TOKENIZER (stages 1-2): one BPE at 49,152 merges on a hash-selected sample of
cleaned, non-validation, de-duplicated training text, truncated to 32,768 and 16,384. The file
p1_inspect/bpe_v2_32768.json is the tokenizer that prep_p1.py consumes. The sample is cleaned
with clean_v2 only (not the later repair layer) - see README "Known limitations".

Run (any machine with the Phase-1 files cached from Hugging Face; 18 min on 12 cores):
    pip install ijson huggingface_hub numpy
    export HF_TOKEN=...                       # or LOCAL_DATA_DIR=<folder holding the four train_*.json>
    python inspect_p1.py 2>&1 | tee inspect_log.txt
Needs legal_bpe_v2.py in the same folder. Resumable: finished stages are cached in $WORK (./p1_inspect).
DOWNSTREAM=0 skips stage F's downloads (enough to rebuild the identical tokenizer).
SMOKE=1 is a test hook (tiny vocabularies / samples).
The summary is printed after "==== SUMMARY".
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import json
import multiprocessing as mp
import random
import re
import time
from collections import Counter

import numpy as np

import legal_bpe as L

# ----------------------------------------------------------------------------- config
WORK = os.environ.get("WORK", "./p1_inspect")
LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR")        # optional: folder with the 4 train_*.json
RUN_DOWNSTREAM = os.environ.get("DOWNSTREAM", "1") == "1"
SMOKE = os.environ.get("SMOKE") == "1"
HF_REPO = "Exploration-Lab/CS-781-Capstone"

SOURCES = ["acts", "sc", "hc", "trib"]
SRC_FILES = {"acts": "train_acts.json", "sc": "train_supreme_court.json",
             "hc": "train_high_courts.json", "trib": "train_tribunals.json"}
RAW_MB = {"acts": 729, "sc": 1320, "hc": 10700, "trib": 6480}
APPROX_DOCS = {"acts": 19_000, "sc": 56_000, "hc": 2_400_000, "trib": 860_000}  # rough (v1 run); sets sampling rates only
TOK_SAMPLE_MB = {"acts": 25, "sc": 45, "hc": 60, "trib": 50}                     # same ~180 MB budget as v1
PROBE_DOCS = 5000            # docs/source for the script / length / cleaning profile
VAL_SAMPLE_DOCS = 1500       # held-out val docs/source for fertility + classifier validation
VOCABS = [16384, 32768, 49152]
MAX_NONLATIN = 0.10          # default doc filter; the profile below shows what other thresholds would drop
FLOP_D, FLOP_LAYERS, FLOP_CTX = 768, 12, 1024   # ILLUSTRATIVE ~110M body for the FLOPs/word table only
DOWN_MAX_MB = 1500           # skip downstream TRAIN files larger than this (download cost)
DOWN_DOCS_PER_FILE = 600
NPROC = min(12, os.cpu_count() or 4)

if SMOKE:
    VOCABS = [300, 400, 500]
    APPROX_DOCS = {k: 1 for k in SOURCES}
    TOK_SAMPLE_MB = {k: 0.3 for k in SOURCES}
    PROBE_DOCS, VAL_SAMPLE_DOCS = 10**9, 200

V_CLS = 32768 if 32768 in VOCABS else VOCABS[-1]   # tokenizer used by the source classifier
PROBE_RATE = {s: min(1.0, PROBE_DOCS / APPROX_DOCS[s]) for s in SOURCES}
TOK_RATE = {s: 1.0 if SMOKE else min(1.0, 1.3 * TOK_SAMPLE_MB[s] / RAW_MB[s]) for s in SOURCES}
_NONASCII_RE = re.compile(r"[^\x00-\x7f]")
_WORD_RE = re.compile(r"\S+")
os.makedirs(WORK, exist_ok=True)


def tok_path(v):
    return f"{WORK}/bpe_v2_{v}.json"


# ----------------------------------------------------------------------------- io
def data_path(fname):
    if LOCAL_DATA_DIR:
        return os.path.join(LOCAL_DATA_DIR, fname)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, f"Phase_1_Pretraining/Train/{fname}",
                           repo_type="dataset", token=os.environ.get("HF_TOKEN"))


def stream_items(path):
    """Top-level items of a JSON array, values of a JSON object, or jsonl lines - streamed."""
    import ijson
    assert "/Test/" not in path.replace("\\", "/"), f"refusing to read test data: {path}"
    with open(path, "rb") as f:
        head = f.read(4096)
        f.seek(0)
        h = head.lstrip(b"\xef\xbb\xbf").lstrip()[:1]
        if h == b"[":
            yield from ijson.items(f, "item")
            return
        is_jsonl = False
        try:                                            # jsonl: first line is a complete object
            json.loads(f.readline(2_000_000).decode("utf-8-sig"))   # capped: a 1-line giant object fails here
            is_jsonl = bool(f.readline(2_000_000).strip())
        except (ValueError, UnicodeDecodeError):
            pass
        f.seek(0)
        if is_jsonl:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        else:                                           # one object: {id: record, ...}
            for _, v in ijson.kvitems(f, ""):
                yield v


def rec_text(rec):
    if isinstance(rec, str):
        return rec
    if isinstance(rec, dict):
        for k in ("text", "body", "content", "judgment", "document"):
            v = rec.get(k)
            if isinstance(v, str):
                return v
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                return "\n".join(v)
    return None


# ----------------------------------------------------------------------------- stage 1
def _profile(raw, P):
    P["n"] += 1
    P["raw_chars"] += len(raw)
    lat, non, deva = L.script_counts(raw)
    P["nonlatin"].append(non / max(lat + non, 1))
    P["deva_docs"] += deva > 0
    P["mojibake_docs"] += ("\u00e2\u20ac" in raw) or ("\u00c3" in raw)
    t = L.clean_v2(raw)
    ok, nw, why = L.keep_doc(t, max_nonlatin=MAX_NONLATIN)
    P["drop"][why] += 1
    P["words"].append(nw)
    if ok:
        P["kept_chars"] += len(t)
        na = _NONASCII_RE.findall(t)
        P["nonascii_chars"] += len(na)
        P["nonascii"].update(na)


def stage1(src):
    out_json = f"{WORK}/stage1_{src}.json"
    if os.path.exists(out_json):
        print(f"[s1] {src}: cached", flush=True)
        return json.load(open(out_json))
    t0 = time.time()
    path = data_path(SRC_FILES[src])
    rnd = random.Random(1234)
    keysets = Counter()
    n_rec = n_empty = n_nofield = 0
    raw_lens = []
    exact_seen, prefix_seen = set(), set()
    exact_dup = prefix_dup = prefix_only = 0
    P = dict(n=0, raw_chars=0, kept_chars=0, nonascii_chars=0, deva_docs=0, mojibake_docs=0,
             drop=Counter(), nonlatin=[], words=[], nonascii=Counter())
    val_res, n_val = [], 0
    tok_bytes = tok_docs = 0
    budget = TOK_SAMPLE_MB[src] * 1e6
    with open(f"{WORK}/tok_sample_{src}.jsonl", "w", encoding="utf-8") as tf:
        for i, rec in enumerate(stream_items(path)):
            n_rec += 1
            if isinstance(rec, dict):
                ks = tuple(sorted(rec.keys()))
                if ks in keysets or len(keysets) < 30:
                    keysets[ks] += 1
                else:
                    keysets[("<other>",)] += 1
            else:
                keysets[(f"<{type(rec).__name__}>",)] += 1
            raw = rec_text(rec)
            if raw is None:
                n_nofield += 1
                continue
            if not raw.strip():
                n_empty += 1
                continue
            raw_lens.append(len(raw))
            ek, pk = L.exact_key(raw), L.prefix_key_v1(raw)
            dup = ek in exact_seen
            if dup:
                exact_dup += 1
            else:
                exact_seen.add(ek)
            if pk in prefix_seen:
                prefix_dup += 1
                prefix_only += not dup
            else:
                prefix_seen.add(pk)
            if dup:
                continue
            if L.is_val_index(i):                      # held out: never in tokenizer sample / probe
                n_val += 1
                if len(val_res) < VAL_SAMPLE_DOCS:
                    val_res.append((i, raw))
                else:
                    j = rnd.randrange(n_val)
                    if j < VAL_SAMPLE_DOCS:
                        val_res[j] = (i, raw)
                continue
            if L.hash_select(f"probe:{src}", i, PROBE_RATE[src]):
                _profile(raw, P)
            if tok_bytes < budget and L.hash_select(f"tok:{src}", i, TOK_RATE[src]):
                t = L.clean_v2(raw)
                ok, _, _ = L.keep_doc(t, max_nonlatin=MAX_NONLATIN)
                if ok:
                    tf.write(json.dumps({"i": i, "text": t}, ensure_ascii=False) + "\n")
                    tok_bytes += len(t.encode("utf-8", "surrogatepass"))
                    tok_docs += 1
            if n_rec % 250_000 == 0:
                print(f"[s1] {src}: {n_rec:,} records ({time.time()-t0:.0f}s)", flush=True)

    with open(f"{WORK}/val_raw_{src}.jsonl", "w", encoding="utf-8") as vf:
        for i, raw in sorted(val_res):
            vf.write(json.dumps({"i": i, "text": raw}, ensure_ascii=False) + "\n")
    np.save(f"{WORK}/exact_keys_{src}.npy", np.fromiter(exact_seen, dtype=np.uint64, count=len(exact_seen)))

    lens = np.array(raw_lens, dtype=np.int64)
    nl = np.array(P["nonlatin"]) if P["nonlatin"] else np.zeros(1)
    w = np.array(P["words"]) if P["words"] else np.zeros(1)
    ne = max(len(lens), 1)
    pn = max(P["n"], 1)
    res = {
        "src": src, "records": n_rec, "no_text_field": n_nofield, "empty_text": n_empty,
        "keysets": [[list(k), c] for k, c in keysets.most_common(6)],
        "nonempty": len(lens), "raw_chars_total": int(lens.sum()),
        "raw_chars_pct": {str(p): int(np.percentile(lens, p)) for p in (1, 5, 50, 95, 99, 99.9)} if len(lens) else {},
        "exact_dup_rate": exact_dup / ne, "v1_prefix_dup_rate": prefix_dup / ne,
        "v1_prefix_only_rate": prefix_only / ne,
        "val_docs_total": n_val, "val_sampled": len(val_res),
        "tok_sample_docs": tok_docs, "tok_sample_mb": round(tok_bytes / 1e6, 1),
        "probe": {
            "docs": P["n"], "drop": dict(P["drop"]),
            "char_retention": P["kept_chars"] / max(P["raw_chars"], 1),
            "nonlatin_gt": {str(th): round(float((nl > th).mean()), 4) for th in (0.01, 0.05, 0.10, 0.20, 0.50)},
            "docs_with_devanagari": P["deva_docs"] / pn,
            "mojibake_docs": P["mojibake_docs"] / pn,
            "words_pct": {str(p): int(np.percentile(w, p)) for p in (5, 50, 95, 99)},
            "lt40_words": round(float((w < 40).mean()), 4), "ge640_words": round(float((w >= 640).mean()), 4),
            "nonascii_per_1k_chars": 1000 * P["nonascii_chars"] / max(P["kept_chars"], 1),
            "top_nonascii": [[f"U+{ord(c):04X}", c, round(1e6 * n / max(P["kept_chars"], 1), 1)]
                             for c, n in P["nonascii"].most_common(25)],
        },
        "seconds": round(time.time() - t0),
    }
    json.dump(res, open(out_json, "w"), indent=1, ensure_ascii=False)
    print(f"[s1] {src}: done in {res['seconds']}s", flush=True)
    return res


# ----------------------------------------------------------------------------- stage 2
def stage2_tokenizer():
    if all(os.path.exists(tok_path(v)) for v in VOCABS):
        print("[s2] tokenizers cached", flush=True)
        return
    t0 = time.time()
    uf = L.count_units_jsonl([f"{WORK}/tok_sample_{s}.jsonl" for s in SOURCES])
    tok = L.BPETrainer(max(VOCABS)).train(uf)
    if tok.vocab_size < max(VOCABS):
        raise SystemExit(f"tokenizer sample too small: only {tok.vocab_size} < {max(VOCABS)} reachable")
    for v in VOCABS:
        t = tok if v == tok.vocab_size else tok.truncate(v)
        t.save(tok_path(v), meta={"trained_on": f"{WORK}/tok_sample_*.jsonl", "max_nonlatin": MAX_NONLATIN})
    print(f"[s2] tokenizers saved ({(time.time()-t0)/60:.1f} min); trained vocab {tok.vocab_size}", flush=True)


# ----------------------------------------------------------------------------- stage 3
def eval_job(args):
    """Fertility / round-trip / window lengths / token counts on held-out val docs."""
    src, v = args
    tok = L.LegalBPE.load(tok_path(v))
    cache = {}
    st = dict(src=src, vocab=v, docs=0, words=0, chars=0, bytes=0, tokens=0, rt_checked=0, rt_fail=0)
    counts = np.zeros(tok.vocab_size, dtype=np.int64)
    plen, clen, prompt_ids = [], [], []
    with open(f"{WORK}/val_raw_{src}.jsonl", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)["text"]
            if st["rt_checked"] < 300:                       # round trip on RAW, uncleaned text
                st["rt_checked"] += 1
                st["rt_fail"] += tok.decode_bytes(tok.encode_to_ids(raw, cache)) != raw.encode("utf-8", "surrogatepass")
            t = L.clean_v2(raw)
            ok, nw, _ = L.keep_doc(t, max_nonlatin=MAX_NONLATIN)
            if not ok:
                continue
            ids = tok.encode_to_ids(t, cache)
            st["docs"] += 1
            st["words"] += nw
            st["chars"] += len(t)
            st["bytes"] += len(t.encode("utf-8", "surrogatepass"))
            st["tokens"] += len(ids)
            counts += np.bincount(np.asarray(ids, dtype=np.int64), minlength=tok.vocab_size)
            ws = [m.end() for _, m in zip(range(640), _WORD_RE.finditer(t))]
            if len(ws) == 640:                               # 512-word prompt + 128-word continuation
                p = tok.encode_to_ids(t[:ws[511]], cache)
                c = tok.encode_to_ids(t[ws[511]:ws[639]], cache)
                plen.append(len(p))
                clen.append(len(c))
                if v == V_CLS:
                    prompt_ids.append(np.asarray(p, dtype=np.int32))
    return st, counts, np.array(plen), np.array(clen), prompt_ids


def unigram_job(src):
    """Per-source unigram counts (32k tokenizer) from the tokenizer-training sample."""
    tok = L.LegalBPE.load(tok_path(V_CLS))
    cache = {}
    counts = np.zeros(tok.vocab_size, dtype=np.int64)
    with open(f"{WORK}/tok_sample_{src}.jsonl", encoding="utf-8") as f:
        for line in f:
            ids = tok.encode_to_ids(json.loads(line)["text"], cache)
            counts += np.bincount(np.asarray(ids, dtype=np.int64), minlength=tok.vocab_size)
    return src, counts


# ----------------------------------------------------------------------------- stage 4: downstream TRAIN texts
_SKIP_KEYS = {"id", "label", "labels", "pred", "relevant_candidates", "gold"}


def flatten_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(p for p in (flatten_text(e) for e in x) if p)
    if isinstance(x, dict):
        return "\n".join(p for k, p in ((k, flatten_text(v)) for k, v in x.items())
                         if p and k not in _SKIP_KEYS)
    return ""


def downstream_texts():
    from huggingface_hub import HfApi, hf_hub_download
    api, token = HfApi(), os.environ.get("HF_TOKEN")
    tops = sorted(e.path for e in api.list_repo_tree(HF_REPO, repo_type="dataset", token=token)
                  if e.path.startswith("Phase_") and not e.path.startswith("Phase_1"))
    out = {}
    for d in tops:
        files = [e for e in api.list_repo_tree(HF_REPO, path_in_repo=d, recursive=True,
                                               repo_type="dataset", token=token)
                 if getattr(e, "size", None) is not None and e.path.endswith(".json")
                 and "/Train/" in e.path and "/Test/" not in e.path]
        texts = []
        for fe in files:
            if fe.size > DOWN_MAX_MB * 1e6:
                print(f"[s4] skip {fe.path} ({fe.size/1e6:.0f} MB > {DOWN_MAX_MB})", flush=True)
                continue
            try:
                p = hf_hub_download(HF_REPO, fe.path, repo_type="dataset", token=token)
                n = 0
                for item in stream_items(p):
                    t = flatten_text(item)
                    if len(t) >= 1500:
                        texts.append(t)
                        n += 1
                        if n >= DOWN_DOCS_PER_FILE:
                            break
                print(f"[s4] {fe.path}: {n} docs", flush=True)
            except Exception as ex:                          # schema surprises must not kill the run
                print(f"[s4] {fe.path}: FAILED {type(ex).__name__}: {ex}", flush=True)
        out[d] = texts
    return out


def first_words(t, n=512):
    ws = [m.end() for _, m in zip(range(n), _WORD_RE.finditer(t))]
    return t[:ws[-1]] if ws else t


# ----------------------------------------------------------------------------- main
def main():
    T0 = time.time()
    try:
        import ijson
        print(f"ijson backend: {ijson.backend}  (want yajl2_c; 'python' is ~10x slower)", flush=True)
    except ImportError:
        raise SystemExit("pip install ijson")

    # stage 1: one process per source (HC dominates wall time)
    with mp.Pool(len(SOURCES)) as pool:
        s1 = {r["src"]: r for r in pool.map(stage1, SOURCES)}

    keys = {s: np.load(f"{WORK}/exact_keys_{s}.npy") for s in SOURCES}
    cross = {f"{a}&{b}": int(np.intersect1d(keys[a], keys[b], assume_unique=True).size)
             for k, a in enumerate(SOURCES) for b in SOURCES[k + 1:]}

    stage2_tokenizer()

    # stage 3: fertility etc. + per-source unigram counts, in parallel
    with mp.Pool(NPROC) as pool:
        ev = pool.map(eval_job, [(s, v) for v in VOCABS for s in SOURCES])
        uni = dict(pool.map(unigram_job, SOURCES))

    fert, guard, under = {}, {}, {}
    pooled = {v: np.zeros(v, dtype=np.int64) for v in VOCABS}
    plens = {v: [] for v in VOCABS}
    clens = {v: [] for v in VOCABS}
    val_prompts = {}
    for st, counts, pl, cl, pids in ev:
        v, s = st["vocab"], st["src"]
        fert[(s, v)] = st
        pooled[v][:len(counts)] += counts
        plens[v].append(pl)
        clens[v].append(cl)
        if v == V_CLS:
            val_prompts[s] = pids
    for v in VOCABS:
        tot = np.concatenate([a + b for a, b in zip(plens[v], clens[v])]) if plens[v] else np.zeros(1)
        guard[v] = {p: int(np.percentile(tot, p)) for p in (50, 99)} | {"max": int(tot.max()),
                    "over_1024": round(float((tot > 1024).mean()), 4), "n": int(tot.size)}
        M = pooled[v].sum()
        exp_1b = pooled[v][len(L.BASE_ITOS):] / max(M, 1) * 1e9       # merged tokens only
        under[v] = {"sample_tokens": int(M), "unseen_in_sample": int((pooled[v][len(L.BASE_ITOS):] == 0).sum()),
                    "lt1000_per_1B": int((exp_1b < 1000).sum()), "lt100_per_1B": int((exp_1b < 100).sum())}

    # source classifier: add-1 smoothed unigram LMs, validated on held-out 512-word val prompts
    logp = np.stack([np.log((uni[s] + 1) / (uni[s].sum() + len(uni[s]))) for s in SOURCES])

    def classify(ids):
        return int(np.argmax(logp[:, ids].sum(axis=1)))

    confusion = {s: Counter(SOURCES[classify(x)] for x in val_prompts.get(s, [])) for s in SOURCES}

    down = {}
    if RUN_DOWNSTREAM:
        tok32 = L.LegalBPE.load(tok_path(V_CLS))
        cache = {}
        for phase, texts in downstream_texts().items():
            if not texts:
                down[phase] = {"docs": 0}
                continue
            labs = Counter(SOURCES[classify(np.asarray(tok32.encode_to_ids(first_words(t), cache), dtype=np.int64))]
                           for t in texts)
            down[phase] = {"docs": len(texts)} | {s: round(labs[s] / len(texts), 3) for s in SOURCES}

    # ------------------------------------------------------------------- report
    summary = {"stage1": s1, "cross_source_exact_dups": cross,
               "fertility": {f"{s}|{v}": fert[(s, v)] for s, v in fert},
               "window_guard": guard, "undertrained": under,
               "classifier_confusion": {s: dict(c) for s, c in confusion.items()},
               "downstream_source_shares": down, "minutes": round((time.time() - T0) / 60, 1)}
    json.dump(summary, open(f"{WORK}/summary.json", "w"), indent=1, ensure_ascii=False, default=str)

    print("\n==== SUMMARY ====")
    print("A/B. records, schema, duplicates")
    for s in SOURCES:
        r = s1[s]
        print(f"  {s:5s} rec={r['records']:,} nofield={r['no_text_field']:,} empty={r['empty_text']:,} "
              f"exact_dup={r['exact_dup_rate']:.2%} v1prefix_dup={r['v1_prefix_dup_rate']:.2%} "
              f"(prefix-only={r['v1_prefix_only_rate']:.2%}) raw_chars={r['raw_chars_total']/1e9:.2f}G "
              f"pct={r['raw_chars_pct']}")
        print(f"        keysets={r['keysets'][:3]}")
    print(f"  cross-source exact dups: {cross}")

    print("C. cleaning / script profile (probe docs, non-val)")
    for s in SOURCES:
        p = s1[s]["probe"]
        print(f"  {s:5s} n={p['docs']} drop={p['drop']} retention={p['char_retention']:.3f} "
              f"nonlatin>thr={p['nonlatin_gt']} deva_docs={p['docs_with_devanagari']:.2%} "
              f"mojibake={p['mojibake_docs']:.2%} words={p['words_pct']} <40w={p['lt40_words']:.2%} "
              f">=640w={p['ge640_words']:.2%} nonascii/1k={p['nonascii_per_1k_chars']:.2f}")
        print(f"        top non-ASCII (per 1M chars): {p['top_nonascii'][:15]}")

    print("D/E. fertility on held-out val docs (cleaned) | est. cleaned tokens per source")
    print(f"  {'src':5s} {'vocab':>6s} {'tok/word':>8s} {'char/tok':>8s} {'byte/tok':>8s} {'rt_fail':>8s} {'est_tokens':>11s}")
    for v in VOCABS:
        for s in SOURCES:
            st, r = fert[(s, v)], s1[s]
            cpt = st["chars"] / max(st["tokens"], 1)
            est = r["raw_chars_total"] * (1 - r["exact_dup_rate"]) * r["probe"]["char_retention"] / max(cpt, 1e-9)
            print(f"  {s:5s} {v:6d} {st['words'] and st['tokens']/st['words']:8.3f} {cpt:8.2f} "
                  f"{st['bytes']/max(st['tokens'],1):8.2f} {st['rt_fail']:4d}/{st['rt_checked']:<3d} {est/1e6:9.0f}M")
    nb = 12 * FLOP_LAYERS * FLOP_D ** 2
    print(f"  FLOPs/word, ILLUSTRATIVE body d={FLOP_D} L={FLOP_LAYERS} ctx={FLOP_CTX} (benchmark tokens/s replaces this):")
    for v in VOCABS:
        tpw = np.mean([fert[(s, v)]["tokens"] / max(fert[(s, v)]["words"], 1) for s in SOURCES])
        fpt = 6 * nb + 6 * FLOP_D * v + 12 * FLOP_LAYERS * FLOP_CTX * FLOP_D
        print(f"    {v:6d}: tok/word(avg over sources)={tpw:.3f}  FLOPs/token={fpt/1e6:.0f}M  FLOPs/word={tpw*fpt/1e6:.0f}M")
    print(f"  512+128-word window tokens (val docs): {guard}")
    print(f"  under-trained merged tokens (projected occurrences in a 1B-token run): {under}")

    print("F. source classifier on held-out 512-word val prompts (true -> predicted)")
    for s in SOURCES:
        n = sum(confusion[s].values())
        print(f"  {s:5s} n={n:4d} acc={confusion[s][s]/max(n,1):.2%} {dict(confusion[s])}")
    print("  Phase 2-7 TRAIN texts, share classified as each source:")
    for ph, d in down.items():
        print(f"    {ph}: {d}")
    print(f"\ntotal {summary['minutes']} min | full details: {WORK}/summary.json")


if __name__ == "__main__":
    main()
