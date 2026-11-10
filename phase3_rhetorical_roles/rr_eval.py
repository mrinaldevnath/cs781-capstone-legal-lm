"""
rr_eval.py - Phase-3 metric and error analysis (no torch).

Official metric (evaluate_phase3.py): sklearn f1_score(y_true, y_pred, labels=<all 13 labels>, average="macro",
zero_division=0) over the sentences of CL and IT POOLED. Every label counts, so a label missing from the gold of
the evaluated documents scores 0 whatever we predict (on our dev set Dissent and PrecedentOverruled are absent:
the dev ceiling is 11/13 = 0.846). `prf` reproduces it exactly (checked against sklearn in tests_rr.py).
Also reported: the macro-F1 over the labels present in the gold (easier to read) and per-domain scores.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
from collections import Counter

import numpy as np


def prf(gold, pred, labels):
    """gold, pred: flat label sequences; labels: the label set averaged over (official: all 13)."""
    assert len(gold) == len(pred)
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    per = {}
    for l in labels:
        P = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        R = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        per[l] = dict(p=P, r=R, f1=F, support=tp[l] + fn[l], pred=tp[l] + fp[l])
    n = max(1, len(labels))
    return dict(macro_f1=sum(v["f1"] for v in per.values()) / n,
                macro_p=sum(v["p"] for v in per.values()) / n,
                macro_r=sum(v["r"] for v in per.values()) / n,
                acc=sum(tp.values()) / max(1, len(gold)), per_label=per)


def evaluate(gold_by_doc, pred_by_doc, labels, dom_by_doc=None):
    """Official pooled macro-F1 over all labels + present-labels variant + per-domain macro-F1 (all labels)."""
    ids = sorted(gold_by_doc)
    g = [x for k in ids for x in gold_by_doc[k]]
    p = [x for k in ids for x in pred_by_doc[k]]
    r = prf(g, p, labels)
    present = [l for l in labels if l in set(g)]
    r["macro_f1_present"] = prf(g, p, present)["macro_f1"]
    r["n_present"] = len(present)
    if dom_by_doc:
        for dom in sorted(set(dom_by_doc.values())):
            k2 = [k for k in ids if dom_by_doc[k] == dom]
            r[f"macro_f1_{dom}"] = prf([x for k in k2 for x in gold_by_doc[k]], [x for k in k2 for x in pred_by_doc[k]],
                                       labels)["macro_f1"]
    return r


def confusion(gold, pred, labels):
    ix = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for g, p in zip(gold, pred):
        M[ix[g], ix[p]] += 1
    return M


def top_confusions(gold, pred, k=12):
    c = Counter((g, p) for g, p in zip(gold, pred) if g != p)
    return [f"{g}->{p}:{n}" for (g, p), n in c.most_common(k)]


def format_report(r):
    rows = [f"macro-F1 {r['macro_f1']:.4f} (P {r['macro_p']:.4f} R {r['macro_r']:.4f})  acc {r['acc']:.4f}  "
            f"present-labels({r['n_present']}) {r['macro_f1_present']:.4f}" +
            "".join(f"  {k[9:]} {r[k]:.4f}" for k in ("macro_f1_CL", "macro_f1_IT") if k in r)]
    for l, v in sorted(r["per_label"].items(), key=lambda kv: -kv[1]["support"]):
        rows.append(f"  {l:24s} F1 {v['f1']:.3f}  P {v['p']:.3f}  R {v['r']:.3f}  gold {v['support']:5d}  pred {v['pred']:5d}")
    return "\n".join(rows)
