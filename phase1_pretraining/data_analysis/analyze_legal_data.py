#!/usr/bin/env python3

import os
import re
import json
import hashlib
import unicodedata
from collections import Counter, defaultdict

import numpy as np
from huggingface_hub import hf_hub_download

# ============================================================
# CONFIG
# ============================================================

HF_REPO = "Exploration-Lab/CS-781-Capstone"

SOURCES = {
    "acts": "train_acts.json",
    "sc": "train_supreme_court.json",
    "hc": "train_high_courts.json",
    "trib": "train_tribunals.json",
}

OUT_DIR = "legal_data_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# Number of representative documents saved per source
SAMPLE_DOCS = 100

# Never use Test data
assert all("/Test/" not in x for x in SOURCES.values())


# ============================================================
# DOWNLOAD / CACHE
# ============================================================


def get_file(filename):
    """
    HF download. If already cached, huggingface_hub simply
    returns the cached path.
    """
    return hf_hub_download(
        repo_id=HF_REPO,
        filename=f"Phase_1_Pretraining/Train/{filename}",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )


# ============================================================
# STREAM JSON / JSONL
# ============================================================


def stream_records(path):
    """
    Supports:
      - JSON array
      - JSONL
      - JSON object containing records

    Never loads the entire dataset into memory.
    """

    import ijson

    normalized = path.replace("\\", "/")

    if "/Test/" in normalized or "\\Test\\" in normalized:
        raise RuntimeError(f"REFUSING TO READ TEST DATA: {path}")

    with open(path, "rb") as f:

        head = f.read(4096)
        f.seek(0)

        first = head.lstrip(b"\xef\xbb\xbf").lstrip()[:1]

        # JSON array
        if first == b"[":
            yield from ijson.items(f, "item")
            return

        # Try JSONL
        first_line = f.readline()
        f.seek(0)

        try:
            json.loads(first_line.decode("utf-8-sig"))
            is_jsonl = True
        except Exception:
            is_jsonl = False

        if is_jsonl:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
            return

        # Otherwise assume object: {id: record, ...}
        for _, value in ijson.kvitems(f, ""):
            yield value


# ============================================================
# TEXT EXTRACTION
# ============================================================

TEXT_KEYS = [
    "text",
    "body",
    "content",
    "judgment",
    "document",
]


def extract_text(record):

    if isinstance(record, str):
        return record

    if not isinstance(record, dict):
        return None

    for key in TEXT_KEYS:
        value = record.get(key)

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            strings = [x for x in value if isinstance(x, str)]
            if strings:
                return "\n".join(strings)

    return None


# ============================================================
# TEXT STATISTICS
# ============================================================

WORD_RE = re.compile(r"\S+")

SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
}


def script_counts(text):

    counts = Counter()

    for ch in text:
        cp = ord(ch)

        for name, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[name] += 1
                break

    return counts


def unicode_profile(text):

    counter = Counter()

    for ch in text:

        if ord(ch) > 127:
            counter[f"U+{ord(ch):04X}"] += 1

    return counter


def suspicious_chars(text):

    chars = {
        "U+FFFD": "\ufffd",  # replacement character
        "U+F0B7": "\uf0b7",  # private-use bullet
        "U+00A0": "\u00a0",  # NBSP
        "U+200B": "\u200b",  # zero-width space
        "U+200C": "\u200c",
        "U+200D": "\u200d",
        "U+FEFF": "\ufeff",  # BOM / zero-width no-break
        "U+00AD": "\u00ad",  # soft hyphen
    }

    return {name: text.count(ch) for name, ch in chars.items()}


def word_count(text):
    return len(WORD_RE.findall(text))


# ============================================================
# REPRESENTATIVE SAMPLING
# ============================================================


def should_sample(index, total):
    """
    Deterministic sampling across the corpus.

    Roughly uniform by document position.
    """
    if total <= SAMPLE_DOCS:
        return True

    step = total / SAMPLE_DOCS

    target = int(index / step)

    return abs(index - target * step) < 1


# ============================================================
# ANALYZE ONE SOURCE
# ============================================================


def analyze_source(source, filename):

    print(f"\n{'=' * 70}")
    print(f"SOURCE: {source}")
    print(f"FILE:   {filename}")
    print(f"{'=' * 70}")

    path = get_file(filename)

    print("Path:", path)
    print("Size:", round(os.path.getsize(path) / (1024**3), 2), "GB")

    total = 0
    empty = 0
    no_text = 0

    total_chars = 0
    total_words = 0

    lengths = []

    exact_hashes = set()
    exact_duplicates = 0

    unicode_counter = Counter()
    script_counter = Counter()
    suspicious_counter = Counter()

    mojibake_docs = 0

    samples = []

    # Metadata
    keysets = Counter()

    # Specific suspicious examples
    suspicious_examples = defaultdict(list)

    for i, record in enumerate(stream_records(path)):

        total += 1

        if isinstance(record, dict):
            keysets[tuple(sorted(record.keys()))] += 1

        text = extract_text(record)

        if text is None:
            no_text += 1
            continue

        if not text.strip():
            empty += 1
            continue

        # -------------------------------
        # basic stats
        # -------------------------------

        n_chars = len(text)
        n_words = word_count(text)

        total_chars += n_chars
        total_words += n_words

        lengths.append(n_chars)

        # -------------------------------
        # exact duplicate
        # -------------------------------

        h = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).digest()

        if h in exact_hashes:
            exact_duplicates += 1
        else:
            exact_hashes.add(h)

        # -------------------------------
        # unicode
        # -------------------------------

        unicode_counter.update(unicode_profile(text))

        # -------------------------------
        # scripts
        # -------------------------------

        scripts = script_counts(text)

        for s, count in scripts.items():
            script_counter[s] += count

        # -------------------------------
        # suspicious chars
        # -------------------------------

        susp = suspicious_chars(text)

        for name, count in susp.items():

            if count:
                suspicious_counter[name] += count

                if len(suspicious_examples[name]) < 10:
                    suspicious_examples[name].append(
                        {
                            "index": i,
                            "count": count,
                            "snippet": text[:1000],
                        }
                    )

        # -------------------------------
        # mojibake indicators
        # -------------------------------

        mojibake_patterns = [
            "â€",
            "â€™",
            "â€œ",
            "â€",
            "Ã©",
            "Ã¨",
            "Ã¼",
            "Â ",
            "b�",
        ]

        if any(x in text for x in mojibake_patterns):
            mojibake_docs += 1

        # -------------------------------
        # representative samples
        # -------------------------------

        if should_sample(i, max(total, 1)):

            samples.append(
                {
                    "index": i,
                    "chars": n_chars,
                    "words": n_words,
                    "scripts": dict(scripts),
                    "suspicious": {k: v for k, v in susp.items() if v},
                    "text": text[:12000],
                }
            )

        if total % 250000 == 0:
            print(f"{source}: {total:,} records | " f"{total_chars / 1e9:.2f}B chars")

    # ========================================================
    # FINAL STATS
    # ========================================================

    lengths_np = np.array(lengths, dtype=np.int64)

    result = {
        "source": source,
        "filename": filename,
        "records": total,
        "no_text_field": no_text,
        "empty_text": empty,
        "total_chars": total_chars,
        "total_words": total_words,
        "mean_chars": (float(lengths_np.mean()) if len(lengths_np) else 0),
        "median_chars": (float(np.median(lengths_np)) if len(lengths_np) else 0),
        "percentile_chars": (
            {str(p): float(np.percentile(lengths_np, p)) for p in [1, 5, 25, 50, 75, 95, 99, 99.9]}
            if len(lengths_np)
            else {}
        ),
        "duplicate_rate": (exact_duplicates / max(len(lengths), 1)),
        "exact_duplicates": exact_duplicates,
        "script_character_counts": dict(script_counter),
        "suspicious_character_counts": dict(suspicious_counter),
        "unicode_top_50": [[code, count] for code, count in unicode_counter.most_common(50)],
        "mojibake_documents": mojibake_docs,
        "keysets": [
            {
                "keys": list(keys),
                "count": count,
            }
            for keys, count in keysets.most_common(20)
        ],
        "suspicious_examples": dict(suspicious_examples),
        "sample_count": len(samples),
    }

    # Save samples separately
    sample_path = os.path.join(OUT_DIR, f"{source}_samples.jsonl")

    with open(sample_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("\nRESULT")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\nSamples saved:", sample_path)

    return result


# ============================================================
# CROSS SOURCE DUPLICATES
# ============================================================


def cross_source_duplicates(source_files):

    print("\nChecking cross-source exact duplicates...")

    hashes = {}

    for source, path in source_files.items():

        print("Hashing:", source)

        hs = set()

        for record in stream_records(path):

            text = extract_text(record)

            if not text:
                continue

            h = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).digest()

            hs.add(h)

        hashes[source] = hs

    result = {}

    sources = list(hashes)

    for i in range(len(sources)):

        for j in range(i + 1, len(sources)):

            a = sources[i]
            b = sources[j]

            result[f"{a}&{b}"] = len(hashes[a] & hashes[b])

    return result


# ============================================================
# MAIN
# ============================================================


def main():

    print("\nLEGAL DATA INSPECTION")
    print("TRAINING DATA ONLY")
    print("NO TEST DATA WILL BE READ")
    print()

    source_paths = {}

    for source, filename in SOURCES.items():

        path = get_file(filename)

        source_paths[source] = path

    all_results = {}

    for source, filename in SOURCES.items():

        all_results[source] = analyze_source(source, filename)

    # cross = cross_source_duplicates(source_paths)
    cross = {}

    report = {
        "sources": all_results,
        "cross_source_exact_duplicates": cross,
    }

    report_path = os.path.join(OUT_DIR, "data_analysis_report.json")

    with open(report_path, "w", encoding="utf-8") as f:

        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print("Report:")
    print(report_path)

    print("\nSamples per source:")
    for source in SOURCES:
        print(os.path.join(OUT_DIR, f"{source}_samples.jsonl"))


if __name__ == "__main__":
    main()
