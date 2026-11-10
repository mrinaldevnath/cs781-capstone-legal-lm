"""
cjpe_eval.py - Phase-4 metrics (no torch, no sklearn needed; matches sklearn's f1_score(average="macro") on {0, 1}).

    binary_report(y, pred)          macro-F1 (primary), macro precision / recall, per-class P/R/F1, accuracy, confusion
    best_threshold(y, p1)           decision threshold on P(label = 1) that maximises macro-F1 (calibration on dev)
    auc(y, p1)                      ROC-AUC (threshold-free ranking quality)
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import numpy as np


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def binary_report(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    out = {"n": int(len(y)), "acc": float((y == pred).mean()) if len(y) else 0.0, "per_class": {}}
    for c in (0, 1):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p, r, f = _prf(tp, fp, fn)
        out["per_class"][str(c)] = {"p": p, "r": r, "f1": f, "support": int((y == c).sum())}
    pc = out["per_class"]
    out["macro_f1"] = (pc["0"]["f1"] + pc["1"]["f1"]) / 2
    out["macro_p"] = (pc["0"]["p"] + pc["1"]["p"]) / 2
    out["macro_r"] = (pc["0"]["r"] + pc["1"]["r"]) / 2
    out["confusion"] = {"tn": int(((y == 0) & (pred == 0)).sum()), "fp": int(((y == 0) & (pred == 1)).sum()),
                        "fn": int(((y == 1) & (pred == 0)).sum()), "tp": int(((y == 1) & (pred == 1)).sum())}
    out["pred_1_share"] = float((pred == 1).mean()) if len(pred) else 0.0
    return out


def macro_f1(y, pred):
    return binary_report(y, pred)["macro_f1"]


def best_threshold(y, p1, grid=None):
    """-> (threshold, macro-F1). Ties go to the threshold closest to 0.5."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    best = (0.5, -1.0)
    for t in grid:
        f = macro_f1(y, (np.asarray(p1) >= t).astype(int))
        if f > best[1] + 1e-12 or (abs(f - best[1]) <= 1e-12 and abs(t - 0.5) < abs(best[0] - 0.5)):
            best = (float(t), f)
    return best


def auc(y, p1):
    y, p1 = np.asarray(y).astype(int), np.asarray(p1, dtype=np.float64)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return float("nan")
    order = np.argsort(p1, kind="mergesort")
    ranks = np.empty(len(p1))
    s = p1[order]
    i = 0
    while i < len(s):                                     # average ranks for ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def format_report(r):
    pc = r["per_class"]
    c = r["confusion"]
    return (f"macro-F1 {r['macro_f1']:.4f}  P {r['macro_p']:.4f}  R {r['macro_r']:.4f}  acc {r['acc']:.4f}  | "
            f"class 0: P {pc['0']['p']:.3f} R {pc['0']['r']:.3f} F1 {pc['0']['f1']:.3f} | "
            f"class 1: P {pc['1']['p']:.3f} R {pc['1']['r']:.3f} F1 {pc['1']['f1']:.3f} | "
            f"tn {c['tn']} fp {c['fp']} fn {c['fn']} tp {c['tp']} | predicted 1: {100 * r['pred_1_share']:.1f}%")
