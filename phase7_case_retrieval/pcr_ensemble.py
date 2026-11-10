"""
pcr_ensemble.py - final Phase-7 system: lexical score + the averaged dense scores of several fine-tuned Phase-1 models.

Usage: python pcr_ensemble.py <folder with bundle_p7_*.zip> <out dir>

For every run in the bundles: per-query z-scores of its four dense scores (dev and test, scores.npz). The run's dense
kind is the one with the best dev micro-F1@7 alone. Ensembles:
    single   : one run fused with the lexical score (as in the runs themselves)
    all      : mean of the z-scored dense scores of ALL trained runs (no selection), fused with lex
    greedy   : forward selection of runs on dev (add the run that most improves fused F1, stop when none helps)
Fusion: S = z(lex) + beta * z(mean dense), beta chosen on dev. Every choice made on dev is also scored by 2-fold
cross-fitting over dev queries (choose on one half, score the other), so the reported number is not inflated by the
choice itself. Test predictions are written for each ensemble.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import glob
import io
import json
import zipfile
import zlib

import numpy as np

from pcr_data import K, micro_prf, topk, write_prediction

BETAS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
KINDS = ("mean", "maxsim", "ctxdoc", "ctxchunk")


def zrows(S):
    S = np.asarray(S, np.float32)
    ok = S > -1e3
    n = ok.sum(1, keepdims=True).clip(min=1)
    mu = np.where(ok, S, 0).sum(1, keepdims=True) / n
    sd = np.sqrt(np.where(ok, (S - mu) ** 2, 0).sum(1, keepdims=True) / n) + 1e-6
    return np.where(ok, (S - mu) / sd, -1e4).astype(np.float32)


class _Zip:
    def __init__(self, path):
        self.z, self.name = zipfile.ZipFile(path), os.path.basename(path)

    def names(self):
        return self.z.namelist()

    def read(self, n):
        return self.z.read(n)


class _Dir:
    def __init__(self, path):
        self.root, self.name = path, os.path.basename(path)

    def names(self):
        return [os.path.relpath(os.path.join(d, f), self.root).replace(os.sep, "/")
                for d, _, fs in os.walk(self.root) for f in fs]

    def read(self, n):
        return open(os.path.join(self.root, n), "rb").read()


def sources(folder):
    """Bundles as zips (bundle_p7_*.zip) or already extracted (a folder holding lex_scores.npz)."""
    out = [_Zip(p) for p in sorted(glob.glob(os.path.join(folder, "**", "bundle_p7_*.zip"), recursive=True))]
    seen = set()
    for p in sorted(glob.glob(os.path.join(folder, "**", "lex_scores.npz"), recursive=True)):
        d = os.path.dirname(p)
        if d not in seen:
            seen.add(d)
            out.append(_Dir(d))
    return out


def load(folder):
    meta, lex, runs = None, None, {}
    for src in sources(folder):
        names = set(src.names())
        if meta is None and "meta.json" in names:
            meta = json.loads(src.read("meta.json"))
            lex = dict(np.load(io.BytesIO(src.read("lex_scores.npz"))))
        for n in sorted(names):
            if n.endswith("/scores.npz") and n.count("/") == 1:
                tag = n.split("/")[0]
                if tag in runs:
                    continue
                sm = json.loads(src.read(f"{tag}/summary.json")) if f"{tag}/summary.json" in names else {}
                runs[tag] = dict(scores=dict(np.load(io.BytesIO(src.read(n)))), summary=sm, bundle=src.name)
    assert meta is not None, f"no bundle found under {folder}"
    print(f"{len(runs)} runs found:", ", ".join(sorted(runs)))
    return meta, lex, runs


def main(folder, out, lex_key="lex_nocit"):
    os.makedirs(out, exist_ok=True)
    meta, lex, runs = load(folder)
    dv, ts = meta["dev"], meta["test"]
    gold = dv["gold"]
    half = np.array([zlib.crc32(f"781:{q}".encode()) % 2 for q in dv["qids"]])

    def f1(S, rows=None):
        rows = np.arange(len(gold)) if rows is None else rows
        return micro_prf(topk(S[rows], dv["cids"], [dv["qids"][i] for i in rows]), [gold[i] for i in rows])

    zl = {"dev": zrows(lex[f"dev_{lex_key}"]), "test": zrows(lex[f"test_{lex_key}"])}
    lines = [f"lexical ({lex_key}) dev F1 {f1(zl['dev'])['f1']:.4f}"]
    Z = {}
    for tag, r in sorted(runs.items()):
        best = max(KINDS, key=lambda k: f1(zrows(r["scores"][f"dev_{k}"]))["f1"])
        Z[tag] = {s: zrows(r["scores"][f"{s}_{best}"].astype(np.float32)) for s in ("dev", "test")}
        Z[tag]["kind"] = best
    trained = [t for t in Z if not t.startswith("Z")]

    def fuse(tags, beta, s="dev"):
        D = np.mean([Z[t][s] for t in tags], 0)
        return zl[s] + beta * zrows(D)

    def best_beta(tags, rows=None):
        return max(BETAS, key=lambda b: f1(fuse(tags, b), rows)["f1"])

    def greedy(rows=None):
        chosen, cur = [], f1(zl["dev"], rows)["f1"]
        while True:
            cand = [(max(f1(fuse(chosen + [t], b), rows)["f1"] for b in BETAS), t) for t in trained if t not in chosen]
            if not cand:
                break
            v, t = max(cand)
            if v <= cur + 1e-9:
                break
            chosen, cur = chosen + [t], v
        return chosen

    def crossfit(procedure):
        """procedure(rows) -> (tags, beta) chosen on rows; returns F1 over dev scoring each half with the other's choice."""
        tp = ng = 0
        for h in (0, 1):
            pick, ev = np.where(half == h)[0], np.where(half != h)[0]
            tags, beta = procedure(pick)
            tp += f1(fuse(tags, beta), ev)["tp"]
            ng += sum(len(set(gold[i])) for i in ev)
        P, R = tp / (K * len(gold)), tp / ng
        return 2 * P * R / (P + R)

    results = {}
    for tag in sorted(Z):
        b = best_beta([tag])
        results[f"single_{tag}"] = dict(tags=[tag], beta=b, dev=f1(fuse([tag], b))["f1"],
                                       crossfit=crossfit(lambda rows, t=tag: ([t], best_beta([t], rows))))
    b = best_beta(trained)
    results["all_trained"] = dict(tags=trained, beta=b, dev=f1(fuse(trained, b))["f1"],
                                  crossfit=crossfit(lambda rows: (trained, best_beta(trained, rows))))
    g = greedy()
    if g:
        b = best_beta(g)
        results["greedy"] = dict(tags=g, beta=b, dev=f1(fuse(g, b))["f1"],
                                 crossfit=crossfit(lambda rows: ((lambda c: (c or trained, best_beta(c or trained, rows)))(greedy(rows)))))
    for name, r in results.items():
        pr = f1(fuse(r["tags"], r["beta"]))
        r.update(p=pr["p"], r=pr["r"])
    order = sorted(results, key=lambda k: -results[k]["crossfit"])
    lines.append(f"{'system':22s} | dev F1 | cross-fitted | beta | runs")
    for k in order:
        r = results[k]
        lines.append(f"{k:22s} | {r['dev']:.4f} | {r['crossfit']:.4f}       | {r['beta']:<4} | "
                     + ", ".join(f"{t}({Z[t]['kind']})" for t in r["tags"]))
    for name in ("all_trained", "greedy"):
        if name in results:
            r = results[name]
            S = fuse(r["tags"], r["beta"], "test")
            pred = topk(S, ts["cids"], ts["qids"])
            d = os.path.join(out, f"{name}_{lex_key}")
            os.makedirs(d, exist_ok=True)
            write_prediction(os.path.join(d, "prediction.json"), ts["qids"], pred, ts["cids"])
            with zipfile.ZipFile(os.path.join(out, f"ENS_{name}{'' if lex_key == 'lex_nocit' else '_citator'}.zip"), "w",
                                 zipfile.ZIP_DEFLATED) as zf:
                zf.write(os.path.join(d, "prediction.json"), arcname="prediction.json")
    json.dump(results, open(os.path.join(out, f"ensemble_{lex_key}.json"), "w"), indent=1)
    rep = "\n".join(lines)
    open(os.path.join(out, f"ENSEMBLE_REPORT_{lex_key}.txt"), "w").write(rep)
    print(rep)
    return results


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *sys.argv[3:])