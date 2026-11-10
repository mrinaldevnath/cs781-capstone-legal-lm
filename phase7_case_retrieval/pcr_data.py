"""
pcr_data.py - Phase-7 Prior Case Retrieval data: loading, cleaning, tokenization, metric.

Splits (each query split has its own candidate pool and is only ever scored against it):
    train.json + train_candidates.json      labels used for training and for tuning lexical weights
    dev.json   + dev_candidates.json        labels used only for selection / evaluation
    test.json  + test_candidates.json       no labels; only predicted

`build(data_root, tok_root, out)` writes one cache directory:
    meta.json          per split: query ids, pool ids, gold lists (train/dev), per-doc estimated year
    words_<split>.pkl  lexical word-id sequences of queries / pool docs (shared word vocabulary)
    cite_<split>.pkl   word-id sequences of the citation-context sentences of each query (sentences with
                       a <CITATION> marker and their neighbours) - the parts of a judgment that refer to prior cases
    ctx_<split>.pkl    the same contexts kept SEPARATE: one word-id sequence per citation passage of each query
                       (a query cites many cases, each discussed in its own passage)
    meta[split]["n_ctx"]          number of citation passages of each query (their BPE ids: "<qid>#c<k>")
    meta[split]["rep_pairs"]      (query, candidate) index pairs where a reporter citation written in the query
                                  ("2 SCR 148", "A.I.R. 1956 S.C. 96", "(1978) 1 SCC 248") is the candidate's own
                                  reporter citation from its header
    meta[split]["citator_pairs"]  pairs where the candidate's 'CITATOR INFO' header (later cases that cite it) lists
                                  the query's own AIR citation
    bpe.npz            Phase-1 BPE ids of every distinct document (first MAX_TOK tokens), for the neural models
The query itself is never a candidate for itself (masked everywhere).
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import json
import pickle
import re
import time
from multiprocessing import Pool

import numpy as np

K = 7                     # every query gets exactly 7 candidates
MAX_TOK = 16384           # BPE tokens kept per document for the neural models
SPLITS = ("train", "dev", "test")

# ----------------------------------------------------------------------------- loading
def _read_json(path):
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head in "[{":
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
        return [json.loads(l) for l in f if l.strip()]          # JSON lines


def _text_of(r):
    for k in ("text", "document", "content", "sentences", "body"):
        if k in r:
            t = r[k]
            return [str(s) for s in t] if isinstance(t, (list, tuple)) else [str(t)]
    raise KeyError(f"no text field in record with keys {list(r)[:8]}")


def _id_of(r):
    for k in ("id", "query_id", "case_id", "doc_id", "candidate_id"):
        if k in r:
            return str(r[k])
    raise KeyError(f"no id field in record with keys {list(r)[:8]}")


def _gold_of(r):
    for k in ("relevant_candidates", "relevant", "labels", "gold"):
        if k in r:
            return [str(x) for x in r[k]]
    return None


def records(obj):
    """Yield (id, sentences, gold-or-None) from a list of dicts, a dict id->record, or a dict id->text."""
    if isinstance(obj, dict):
        for key in ("data", "queries", "candidates", "documents"):
            if key in obj and isinstance(obj[key], (list, dict)) and len(obj) <= 3:
                yield from records(obj[key])
                return
        for k, v in obj.items():
            if isinstance(v, dict):
                yield (str(v.get("id", k)), _text_of(v), _gold_of(v))
            else:
                yield (str(k), [str(s) for s in v] if isinstance(v, list) else [str(v)], None)
    else:
        for r in obj:
            yield (_id_of(r), _text_of(r), _gold_of(r))


def find_files(root):
    """{split: (queries_path, candidates_path)} searched recursively under root."""
    out = {}
    allf = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.endswith((".json", ".jsonl"))]
    for s in SPLITS:
        q = sorted([p for p in allf if re.fullmatch(fr"{s}\.jsonl?", os.path.basename(p).lower())], key=len)
        c = sorted([p for p in allf if re.fullmatch(fr"{s}_candidates?\.jsonl?", os.path.basename(p).lower())], key=len)
        if q and c:
            out[s] = (q[0], c[0])
    return out


# ----------------------------------------------------------------------------- cleaning
CITE_RE = re.compile(r"<\s*CITATION\s*>")
WORD_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
YEAR_RE = re.compile(r"\b(19[4-9]\d|20[0-2]\d)\b")


def words(text):
    """Lower-cased alphanumeric words; the citation marker becomes the single word 'citx'."""
    return WORD_RE.findall(CITE_RE.sub(" citx ", text).lower())


def cite_context(sents, width=1):
    """Sentences that contain a citation marker, with `width` neighbours on each side, in order."""
    hit = [i for i, s in enumerate(sents) if CITE_RE.search(s)]
    keep = sorted({j for i in hit for j in range(max(0, i - width), min(len(sents), i + width + 1))})
    return [sents[j] for j in keep]


DATE_RE = re.compile(r"DATE OF JUDGMENT\s*:?\s*\d{1,2}[/.-]\d{1,2}[/.-](\d{4})")


def est_year(sents):
    """Year of a judgment: the 'DATE OF JUDGMENT: dd/mm/yyyy' header when present (Supreme Court reports), else the
    latest plausible year mentioned (a judgment cannot mention a later year than its own date, so this is a lower
    bound of its date, usually equal to it). The 'CITATOR INFO' header lists LATER cases and is skipped. 0 = unknown."""
    head = " ".join(sents[:3])
    m = DATE_RE.search(head)
    if m:
        return int(m.group(1))
    text = re.sub(r"CITATOR INFO\s*:.*?(?=ACT:|HEADNOTE:|JUDGMENT:|$)", " ", " ".join(sents), count=1, flags=re.S)
    ys = [int(y) for y in YEAR_RE.findall(text)]
    return max(ys) if ys else 0


def contexts(sents, width=1):
    """One passage per citation marker: the marker sentence and `width` neighbours on each side (deduplicated).
    Queries without a marker get their first 3 sentences."""
    hit = [i for i, s in enumerate(sents) if CITE_RE.search(s)]
    out = list(dict.fromkeys(" ".join(sents[max(0, i - width):i + width + 1]) for i in hit))
    return out or [" ".join(sents[:3])]


# reporter citations: the candidate's own (header) and those written in a query's text
HEAD_AIR = re.compile(r"CITATION:\s*(\d{4})\s+AIR\s+(\d+)")
HEAD_SCR = re.compile(r"(\d{4})\s+SCR\s*(?:Supl\.\s*)?\((\d+)\)\s*(\d+)")
HEAD_SCC = re.compile(r"(\d{4})\s+SCC\s*\((\d+)\)\s*(\d+)")
CITATOR = re.compile(r"CITATOR INFO\s*:(.*?)(?:ACT:|HEADNOTE:|JUDGMENT:|$)", re.S)
CIT_ENTRY = re.compile(r"(\d{4})\s*SC\s*(\d+)")
Q_SCR = re.compile(r"(\d+)\s*S\.?\s?C\.?\s?R\.?\s*(\d+)")
Q_SCC = re.compile(r"(\d+)\s*S\.?\s?C\.?\s?C\.?\s*(\d+)")
Q_AIR = re.compile(r"A\.?\s?I\.?\s?R\.?\s*[-(]*\s*(\d{4})\)?\s*S\.?\s?C\.?\s*(\d+)")


def own_reporter_keys(text):
    h, k = text[:1500], set()
    m = HEAD_AIR.search(h)
    if m:
        k.add(("AIR", m.group(1), m.group(2)))
    k |= {("SCR", v, p) for _, v, p in HEAD_SCR.findall(h)}
    k |= {("SCC", v, p) for _, v, p in HEAD_SCC.findall(h)}
    return k


def cited_reporter_keys(text):
    return ({("SCR", v, p) for v, p in Q_SCR.findall(text)} | {("SCC", v, p) for v, p in Q_SCC.findall(text)} |
            {("AIR", y, p) for y, p in Q_AIR.findall(text)})


def citator_keys(text):
    m = CITATOR.search(text[:6000])
    return set(CIT_ENTRY.findall(m.group(1))) if m else set()


def own_air(text):
    m = HEAD_AIR.search(text[:1500])
    return (m.group(1), m.group(2)) if m else None


def citation_links(Q, C):
    """-> (reporter-match pairs, citator pairs) as [query index, candidate index] lists (the query itself excluded)."""
    qtext = [" ".join(q[1]) for q in Q]
    ctext = [" ".join(c[1]) for c in C]
    inv = {}
    for j, t in enumerate(ctext):
        for k in own_reporter_keys(t):
            inv.setdefault(k, set()).add(j)
    cit = {}
    for j, t in enumerate(ctext):
        for k in citator_keys(t):
            cit.setdefault(k, set()).add(j)
    rep, ctr = set(), set()
    for i, t in enumerate(qtext):
        for k in cited_reporter_keys(t):
            rep |= {(i, j) for j in inv.get(k, ())}
        oa = own_air(t)
        if oa:
            ctr |= {(i, j) for j in cit.get(oa, ())}
    rep = sorted((i, j) for i, j in rep if C[j][0] != Q[i][0])
    ctr = sorted((i, j) for i, j in ctr if C[j][0] != Q[i][0])
    return [list(x) for x in rep], [list(x) for x in ctr]


# ----------------------------------------------------------------------------- tokenization (BPE) workers
_TOK = None


def _init_tok(tok_root):
    global _TOK, _CACHE
    sys.path.insert(0, tok_root)
    from p1_loader import load_tokenizer
    _TOK, _CACHE = load_tokenizer(tok_root), {}


def _bpe(text):
    ids, n = [], 0
    for para in text.split("\n"):
        ids.extend(_TOK.encode_to_ids(para + "\n", _CACHE))
        if len(ids) >= MAX_TOK:
            break
    return np.asarray(ids[:MAX_TOK], dtype=np.uint16)


# ----------------------------------------------------------------------------- build
def build(data_root, tok_root, out, workers=4, bpe=True, log=print):
    os.makedirs(out, exist_ok=True)
    files = find_files(data_root)
    log("data files:", files)
    assert {"train", "dev", "test"} <= set(files), f"missing splits; found {sorted(files)}"
    t0 = time.time()
    vocab = {}
    meta, texts = {}, {}
    for s, (qp, cp) in files.items():
        Q = list(records(_read_json(qp)))
        C = list(records(_read_json(cp)))
        seen = set()
        C = [c for c in C if not (c[0] in seen or seen.add(c[0]))]
        qids = [q[0] for q in Q]
        cids = [c[0] for c in C]
        gold = [q[2] for q in Q] if all(q[2] is not None for q in Q) else None
        wq = [[vocab.setdefault(w, len(vocab)) for w in words(" ".join(q[1]))] for q in Q]
        wc = [[vocab.setdefault(w, len(vocab)) for w in words(" ".join(c[1]))] for c in C]
        cq = [[vocab.setdefault(w, len(vocab)) for w in words(" ".join(cite_context(q[1])))] for q in Q]
        ctx = [contexts(q[1]) for q in Q]
        wx = [[np.asarray([vocab.setdefault(w, len(vocab)) for w in words(x)], np.int32) for x in cs] for cs in ctx]
        rep_pairs, cit_pairs = citation_links(Q, C)
        pool_set = set(cids)
        info = dict(qids=qids, cids=cids, gold=gold,
                    qyear=[est_year(q[1]) for q in Q], cyear=[est_year(c[1]) for c in C],
                    n_cite_sents=[sum(bool(CITE_RE.search(x)) for x in q[1]) for q in Q],
                    qlen_words=[len(x) for x in wq], clen_words=[len(x) for x in wc],
                    n_ctx=[len(cs) for cs in ctx], rep_pairs=rep_pairs, citator_pairs=cit_pairs)
        if gold is not None:
            info["gold_missing_from_pool"] = sum(g not in pool_set for gl in gold for g in gl)
            info["gold_is_self"] = sum(q in gl for q, gl in zip(qids, gold))
        meta[s] = info
        with open(os.path.join(out, f"words_{s}.pkl"), "wb") as f:
            pickle.dump(dict(q=[np.asarray(x, np.int32) for x in wq], c=[np.asarray(x, np.int32) for x in wc]), f)
        with open(os.path.join(out, f"cite_{s}.pkl"), "wb") as f:
            pickle.dump([np.asarray(x, np.int32) for x in cq], f)
        with open(os.path.join(out, f"ctx_{s}.pkl"), "wb") as f:
            pickle.dump(wx, f)
        for q, cs in zip(qids, ctx):
            for k, x in enumerate(cs):
                texts.setdefault(f"{q}#c{k}", x)
        for i, q in zip(qids, Q):
            texts.setdefault(i, "\n".join(q[1]))
        for i, c in zip(cids, C):
            texts.setdefault(i, "\n".join(c[1]))
        log(f"{s}: {len(qids)} queries, {len(cids)} candidates, gold {'yes' if gold else 'no'} "
            f"({time.time() - t0:.0f}s)")
    meta["n_words"] = len(vocab)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    if bpe:
        ids = sorted(texts)
        t1 = time.time()
        with Pool(workers, initializer=_init_tok, initargs=(tok_root,)) as p:
            toks = p.map(_bpe, [texts[i] for i in ids], chunksize=16)
        lens = np.array([len(t) for t in toks], np.int64)
        np.savez(os.path.join(out, "bpe.npz"), ids=np.array(ids), lens=lens,
                 flat=np.concatenate(toks) if toks else np.zeros(0, np.uint16))
        log(f"BPE: {len(ids)} distinct documents, median {int(np.median(lens))} tokens "
            f"(cap {MAX_TOK}), {time.time() - t1:.0f}s")
    return stats(meta)


def stats(meta):
    out = {}
    for s in SPLITS:
        m = meta[s]
        d = dict(queries=len(m["qids"]), pool=len(m["cids"]),
                 query_words_median=int(np.median(m["qlen_words"])), cand_words_median=int(np.median(m["clen_words"])),
                 queries_with_citation_marker=int(sum(x > 0 for x in m["n_cite_sents"])),
                 citation_passages_median=float(np.median(m["n_ctx"])),
                 reporter_links=len(m["rep_pairs"]), citator_links=len(m["citator_pairs"]))
        if m["gold"] is not None:
            g = [len(x) for x in m["gold"]]
            d.update(gold_mean=round(float(np.mean(g)), 2), gold_median=float(np.median(g)), gold_max=max(g),
                     gold_zero=sum(x == 0 for x in g), gold_missing_from_pool=m["gold_missing_from_pool"],
                     gold_is_self=m["gold_is_self"])
            g_sets = [set(x) for x in m["gold"]]
            for key in ("rep_pairs", "citator_pairs"):
                ok = sum(m["cids"][j] in g_sets[i] for i, j in m[key])
                d[key.replace("_pairs", "_precision")] = round(ok / max(1, len(m[key])), 3)
                d[key.replace("_pairs", "_recall")] = round(ok / max(1, sum(len(x) for x in g_sets)), 3)
        out[s] = d
    return out


class BPE:
    """Random access to the cached BPE ids by document id."""

    def __init__(self, path):
        z = np.load(path)
        self.flat = z["flat"]
        self.lens = z["lens"]
        self.off = np.concatenate([[0], np.cumsum(self.lens)])
        self.index = {str(i): k for k, i in enumerate(z["ids"])}

    def __call__(self, doc_id):
        k = self.index[doc_id]
        return self.flat[self.off[k]:self.off[k + 1]].astype(np.int64)


# ----------------------------------------------------------------------------- metric / output
def topk(scores, cids, qids=None, k=K):
    """scores (nq, nc) -> list of top-k candidate id lists (the query itself is never returned)."""
    S = np.array(scores, dtype=np.float64, copy=True)
    if qids is not None:
        pos = {c: j for j, c in enumerate(cids)}
        for i, q in enumerate(qids):
            if q in pos:
                S[i, pos[q]] = -np.inf
    part = np.argpartition(-S, kth=min(k, S.shape[1] - 1), axis=1)[:, :k]
    order = np.take_along_axis(part, np.argsort(-np.take_along_axis(S, part, 1), axis=1), 1)
    return [[cids[j] for j in row] for row in order]


def micro_prf(pred, gold, k=K):
    tp = sum(len(set(p[:k]) & set(g)) for p, g in zip(pred, gold))
    npred = k * len(pred)
    ngold = sum(len(set(g)) for g in gold)
    P = tp / npred if npred else 0.0
    R = tp / ngold if ngold else 0.0
    return dict(f1=2 * P * R / (P + R) if P + R else 0.0, p=P, r=R, tp=tp)


def write_prediction(path, qids, pred, cids):
    pool = set(cids)
    out = {}
    for q, p in zip(qids, pred):
        assert len(p) == K and len(set(p)) == K and all(c in pool for c in p) and q not in p, q
        out[q] = list(p)
    json.dump(out, open(path, "w"), indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(build(sys.argv[1], sys.argv[2], sys.argv[3], bpe=len(sys.argv) < 5), indent=1))
