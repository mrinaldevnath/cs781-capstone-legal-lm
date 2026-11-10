"""
ner_eval.py - strict span-level scoring for L-NER (no torch) + submission validation + error analysis.

Strict: a prediction is a true positive only if (doc, start, end, label) matches a gold entity exactly.
Per label: P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2PR/(P+R) (0 when undefined).
Macro = unweighted mean over the official label list (the competition's primary metric is strict macro-F1).

    python ner_eval.py --gold fold2-gold.json --pred prediction.json --labels labels.txt
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import argparse
import json
from collections import Counter, defaultdict


def _key_set(spans_by_doc):
    out = set()
    for d, spans in spans_by_doc.items():
        for e in spans:
            out.add((str(d), int(e[0]), int(e[1]), e[2]))
    return out


def score(gold, pred, labels, docs=None):
    """gold/pred: {doc_id: [[s, e, label, ...], ...]}. docs: restrict to these ids (default: gold's)."""
    docs = set(map(str, docs if docs is not None else gold.keys()))
    G = {k for k in _key_set(gold) if k[0] in docs}
    P = {k for k in _key_set(pred) if k[0] in docs}
    tp = Counter(k[3] for k in G & P)
    ng = Counter(k[3] for k in G)
    npred = Counter(k[3] for k in P)
    per = {}
    for l in labels:
        p = tp[l] / npred[l] if npred[l] else 0.0
        r = tp[l] / ng[l] if ng[l] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[l] = dict(p=p, r=r, f1=f, tp=tp[l], gold=ng[l], pred=npred[l])
    n = len(labels)
    res = dict(macro_p=sum(v["p"] for v in per.values()) / n,
               macro_r=sum(v["r"] for v in per.values()) / n,
               macro_f1=sum(v["f1"] for v in per.values()) / n)
    TP, NG, NP = sum(tp.values()), len(G), len(P)
    mp, mr = (TP / NP if NP else 0.0), (TP / NG if NG else 0.0)
    res.update(micro_p=mp, micro_r=mr, micro_f1=2 * mp * mr / (mp + mr) if mp + mr else 0.0, per_label=per)
    return res


def error_analysis(gold, pred, docs=None):
    """Classify every false positive / false negative:
       label   - exact span found, wrong label
       boundary- same label, overlapping span, different offsets
       both    - overlapping span, different label and offsets
       missed  - gold entity with no overlapping prediction
       spurious- prediction overlapping no gold entity
    plus a confusion counter for label errors (gold -> predicted)."""
    docs = set(map(str, docs if docs is not None else gold.keys()))
    cats = Counter()
    conf = Counter()
    bound_by_label = Counter()
    for d in docs:
        g = [(int(e[0]), int(e[1]), e[2]) for e in gold.get(d, [])]
        p = [(int(e[0]), int(e[1]), e[2]) for e in pred.get(d, [])]
        gs, ps = set(g), set(p)
        for (s, e, l) in g:
            if (s, e, l) in ps:
                continue
            ov = [(a, b, m) for (a, b, m) in p if a < e and b > s]
            if not ov:
                cats["missed"] += 1
            elif any(a == s and b == e for a, b, m in ov):
                cats["label"] += 1
                conf[(l, [m for a, b, m in ov if a == s and b == e][0])] += 1
            elif any(m == l for a, b, m in ov):
                cats["boundary"] += 1
                bound_by_label[l] += 1
            else:
                cats["both"] += 1
        for (a, b, m) in p:
            if (a, b, m) in gs:
                continue
            if not any(s < b and e > a for (s, e, l) in g):
                cats["spurious"] += 1
    return dict(categories=dict(cats), label_confusions={f"{a}->{b}": c for (a, b), c in conf.most_common(20)},
                boundary_errors_by_label=dict(bound_by_label))


def validate_submission(pred, test_ids, labels):
    """Raises AssertionError on any format problem the evaluator would reject."""
    ids = set(map(str, test_ids))
    assert set(pred) == ids, f"ids differ: missing {len(ids - set(pred))}, unknown {len(set(pred) - ids)}"
    lab = set(labels)
    for d, spans in pred.items():
        assert isinstance(spans, list), d
        for e in spans:
            assert isinstance(e, list) and len(e) in (3, 4), (d, e)
            assert isinstance(e[0], int) and isinstance(e[1], int) and 0 <= e[0] < e[1], (d, e)
            assert e[2] in lab, (d, e)
    return True


def format_report(res, labels):
    lines = [f"macro F1 {res['macro_f1']:.4f}  P {res['macro_p']:.4f}  R {res['macro_r']:.4f}   "
             f"micro F1 {res['micro_f1']:.4f}"]
    for l in labels:
        v = res["per_label"][l]
        lines.append(f"  {l:10s} F1 {v['f1']:.3f}  P {v['p']:.3f}  R {v['r']:.3f}  gold {v['gold']:5d}  pred {v['pred']:5d}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--labels", required=True)
    a = ap.parse_args()
    labels = [l.strip() for l in open(a.labels) if l.strip()]
    gold, pred = json.load(open(a.gold)), json.load(open(a.pred))
    res = score(gold, pred, labels, docs=[d for d in gold if d in pred])
    print(format_report(res, labels))
    print(json.dumps(error_analysis(gold, pred, docs=[d for d in gold if d in pred]), indent=1))


if __name__ == "__main__":
    main()
