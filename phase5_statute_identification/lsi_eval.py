"""
lsi_eval.py - Phase-5 multi-label metrics and decision rules (no torch; macro/micro F1 equal sklearn's on {0,1}
indicator matrices with zero_division=0).

    report(Y, pred)                  macro-F1 (primary; over all statute columns), micro-F1, subset accuracy,
                                     macro precision / recall, per-label F1 and support, predicted labels per document
    decide(P, thr, min1=True)        scores -> label sets: P >= thr (scalar = global, vector = per label); with min1
                                     every document gets at least its highest-scoring statute
    best_global(Y, P)                the single threshold that maximises macro-F1
    per_label_thresholds(Y, P)       one threshold per statute (each maximises that statute's F1)
    crossfit_per_label(Y, P)         honest estimate of per-label thresholds: fit on one half of dev, score the other
    frequency_buckets(...)           macro-F1 over frequent / middle / rare statutes (by training frequency)
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import numpy as np

GRID = np.round(np.arange(0.02, 0.951, 0.01), 2)


def _counts(Y, pred):
    Y, pred = Y.astype(bool), pred.astype(bool)
    tp = (Y & pred).sum(0).astype(np.float64)
    fp = (~Y & pred).sum(0).astype(np.float64)
    fn = (Y & ~pred).sum(0).astype(np.float64)
    return tp, fp, fn


def _f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)


def macro_f1(Y, pred):
    return float(_f1(*_counts(Y, pred)).mean())


def report(Y, pred):
    tp, fp, fn = _counts(Y, pred)
    f1 = _f1(tp, fp, fn)
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-12), 0.0)
    T, FP, FN = tp.sum(), fp.sum(), fn.sum()
    return {"n": int(len(Y)), "macro_f1": float(f1.mean()), "macro_p": float(p.mean()), "macro_r": float(r.mean()),
            "micro_f1": float(2 * T / max(2 * T + FP + FN, 1e-12)),
            "micro_p": float(T / max(T + FP, 1e-12)), "micro_r": float(T / max(T + FN, 1e-12)),
            "subset_acc": float((Y.astype(bool) == pred.astype(bool)).all(1).mean()) if len(Y) else 0.0,
            "pred_per_doc": float(pred.sum(1).mean()) if len(Y) else 0.0,
            "gold_per_doc": float(Y.sum(1).mean()) if len(Y) else 0.0,
            "empty_pred": int((pred.sum(1) == 0).sum()),
            "per_label_f1": f1.tolist(), "per_label_p": p.tolist(), "per_label_r": r.tolist(),
            "support": Y.sum(0).astype(int).tolist()}


def decide(P, thr, min1=True):
    pred = P >= np.asarray(thr)[None] if np.ndim(thr) else P >= thr
    if min1 and len(P):
        none = ~pred.any(1)
        pred[none, P[none].argmax(1)] = True
    return pred


def best_global(Y, P, grid=GRID, min1=True):
    """-> (threshold, macro-F1). Ties go to the higher threshold (fewer labels)."""
    best = (0.5, -1.0)
    for t in grid:
        f = macro_f1(Y, decide(P, t, min1))
        if f > best[1] + 1e-12 or (abs(f - best[1]) <= 1e-12 and t > best[0]):
            best = (float(t), f)
    return best


def per_label_thresholds(Y, P, grid=GRID, min_pos=5, fallback=None):
    """One threshold per statute maximising its own F1 (independent per column, so it is fast: one pass per grid
    value). Statutes with fewer than min_pos positives keep the fallback (default: the best global threshold)."""
    if fallback is None:
        fallback = best_global(Y, P, grid, min1=False)[0]
    Yb = Y.astype(bool)
    best_f = np.full(Y.shape[1], -1.0)
    best_t = np.full(Y.shape[1], fallback, dtype=np.float64)
    for t in grid:
        f = _f1(*_counts(Yb, P >= t))
        upd = f > best_f + 1e-12
        best_f[upd], best_t[upd] = f[upd], t
    best_t[Yb.sum(0) < min_pos] = fallback
    return best_t


def crossfit_per_label(Y, P, seed=0, folds=2, min1=True):
    """Honest macro-F1 of per-label thresholds: fit on (folds-1)/folds of dev, score the held-out part; also the same
    for a global threshold. -> dict(global=..., per_label=...) (means over folds)."""
    rng = np.random.default_rng(seed)
    part = rng.permutation(len(Y)) % folds
    g, pl = [], []
    for k in range(folds):
        tr, te = part != k, part == k
        tg = best_global(Y[tr], P[tr], min1=min1)[0]
        tv = per_label_thresholds(Y[tr], P[tr], fallback=tg)
        g.append(macro_f1(Y[te], decide(P[te], tg, min1)))
        pl.append(macro_f1(Y[te], decide(P[te], tv, min1)))
    return {"global": float(np.mean(g)), "per_label": float(np.mean(pl))}


def frequency_buckets(per_label_f1, train_freq, n_buckets=3):
    """Macro-F1 over statutes grouped by training frequency (most frequent third first)."""
    order = np.argsort(-np.asarray(train_freq))
    groups = np.array_split(order, n_buckets)
    f = np.asarray(per_label_f1)
    return [dict(labels=g.tolist(), freq_min=int(np.min(np.asarray(train_freq)[g])),
                 freq_max=int(np.max(np.asarray(train_freq)[g])), macro_f1=float(f[g].mean())) for g in groups]


def format_report(r):
    return (f"macro-F1 {r['macro_f1']:.4f} (P {r['macro_p']:.4f} R {r['macro_r']:.4f}) | micro-F1 {r['micro_f1']:.4f} | "
            f"subset acc {r['subset_acc']:.4f} | labels/doc predicted {r['pred_per_doc']:.2f} gold {r['gold_per_doc']:.2f}")
