"""
ner_post.py - document-level consistency post-processing (no torch, no GPU, works on finished runs).

Observation (training data): inside one document a string that is an entity is annotated with the SAME label at
almost every occurrence (oracle string propagation: P 0.79-0.99, R 0.90-1.00 per label). A token tagger with a 512-token
window cannot know that "Custodian" is the respondent of THIS case; but if it finds the string at a few places, the
other occurrences can be filled in and conflicting labels voted away.

For every distinct string predicted in a document:
    votes   = number of predicted spans with this exact text, per label; majority label L (ties -> more characters of evidence = first)
    occ     = all word-bounded occurrences of the string in the text
    accept  = votes[L] >= min_votes  and  votes[L] / occ >= frac  and  len(string) >= min_len
accepted strings: every predicted span of that text gets label L, and every other occurrence that does not overlap a
different predicted span is added with label L. Longer strings are processed first.

    # tune (min_votes, frac) on the dev documents of a finished run, write a new submission with the best setting
    python ner_post.py --run runs/a_e10_nopre --data_dir p2data --test_file p2data/fold3-text.json
    # fixed setting
    python ner_post.py --run runs/a_e10_nopre --data_dir p2data --test_file ... --min_votes 2 --frac 0.3
Outputs in --run: prediction_post.json, submission_post.zip, post_summary.json
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ner_data as N
import ner_eval as E


def occurrences(text, s):
    return [(m.start(), m.end()) for m in re.finditer(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text)]


def propagate(text, spans, min_votes=2, frac=0.3, min_len=3, labels=None):
    """spans: [[start, end, label], ...] -> new span list (same format, sorted)."""
    cur = {(int(a), int(b)): l for a, b, l in spans}
    by_str = defaultdict(Counter)
    for (a, b), l in cur.items():
        by_str[text[a:b]][l] += 1
    cov = bytearray(len(text))
    for (a, b) in cur:
        for i in range(a, b):
            cov[i] = 1
    for s in sorted(by_str, key=len, reverse=True):
        if len(s) < min_len or not s.strip():
            continue
        lab, v = by_str[s].most_common(1)[0]
        if labels is not None and lab not in labels:
            continue
        occ = occurrences(text, s)
        if not occ or v < min_votes or v / len(occ) < frac:
            continue
        for (a, b) in occ:
            if (a, b) in cur:
                cur[(a, b)] = lab                      # vote away conflicting labels
            elif not any(cov[a:b]):
                cur[(a, b)] = lab                      # fill in a missed occurrence
                for i in range(a, b):
                    cov[i] = 1
    return [[a, b, l] for (a, b), l in sorted(cur.items())]


def apply_all(texts, pred, **kw):
    return {k: propagate(texts[k], pred[k], **kw) for k in pred}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder with dev_pred.json and prediction.json")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--test_file", default=None)
    ap.add_argument("--min_votes", type=int, default=None)
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument("--min_len", type=int, default=3)
    a = ap.parse_args()
    labs, _ = N.load_labels(os.path.join(a.data_dir, "labels.txt"))
    texts, gold = N.load_folds(a.data_dir)
    dev = json.load(open(os.path.join(a.run, "dev_pred.json")))
    g = {k: gold[k] for k in dev}
    base = E.score(g, dev, labs)
    print(f"dev before: macro-F1 {base['macro_f1']:.4f}  micro {base['micro_f1']:.4f}")
    grid = [(mv, fr) for mv in (1, 2, 3) for fr in (0.0, 0.15, 0.3, 0.5, 0.7)]
    if a.min_votes is not None and a.frac is not None:
        grid = [(a.min_votes, a.frac)]
    rows = []
    for mv, fr in grid:
        r = E.score(g, apply_all(texts, dev, min_votes=mv, frac=fr, min_len=a.min_len), labs)
        rows.append((r["macro_f1"], mv, fr, r))
        print(f"  min_votes {mv} frac {fr:.2f}: macro-F1 {r['macro_f1']:.4f}  micro {r['micro_f1']:.4f}")
    rows.sort(key=lambda x: -x[0])
    best_f1, mv, fr, r = rows[0]
    print(f"\nbest: min_votes {mv} frac {fr}  macro-F1 {best_f1:.4f} (before {base['macro_f1']:.4f})")
    print(f"{'label':10s} {'before':>7s} {'after':>7s}")
    for l in labs:
        print(f"{l:10s} {base['per_label'][l]['f1']:7.3f} {r['per_label'][l]['f1']:7.3f}")
    summ = dict(before=base["macro_f1"], best=best_f1, min_votes=mv, frac=fr,
                grid=[(x[1], x[2], x[0]) for x in rows],
                per_label_before={l: base["per_label"][l]["f1"] for l in labs},
                per_label_after={l: r["per_label"][l]["f1"] for l in labs})
    if a.test_file:
        tt = N.load_texts(a.test_file)
        pred = json.load(open(os.path.join(a.run, "prediction.json")))
        new = apply_all(tt, pred, min_votes=mv, frac=fr, min_len=a.min_len)
        E.validate_submission(new, list(tt), labs)
        pj = os.path.join(a.run, "prediction_post.json")
        json.dump(new, open(pj, "w"))
        zp = os.path.join(a.run, "submission_post.zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pj, arcname="prediction.json")
        n0, n1 = sum(map(len, pred.values())), sum(map(len, new.values()))
        print(f"\ntest: {n0} -> {n1} spans; wrote {zp}")
        summ.update(test_spans_before=n0, test_spans_after=n1)
    json.dump(summ, open(os.path.join(a.run, "post_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
