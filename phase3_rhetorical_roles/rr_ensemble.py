"""
rr_ensemble.py - average the sentence-level probabilities of several runs, argmax, write submission.zip.
Sentence-level outputs make any runs combinable (different backbones / tokenizations / methods).

    python rr_ensemble.py --runs runs/b_lorma runs/d_lorma ... --data_dir p3data --out runs/ens_x [--split dev|test|both]

"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import argparse
import json
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_data as D
import rr_eval as E


def load_probs(path):
    f = np.load(path)
    ids = [str(x) for x in f["ids"]]
    return {k: f[f"p{i}"].astype(np.float32) for i, k in enumerate(ids)}


def average(paths):
    ps = [load_probs(p) for p in paths]
    keys = list(ps[0])
    for q in ps[1:]:
        assert set(q) == set(keys), "runs cover different documents"
    return {k: np.mean([q[k] for q in ps], axis=0) for k in keys}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="both", choices=["dev", "test", "both"])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    labels = D.load_vocab(os.path.join(a.data_dir, "label_vocab.json"))
    summary = dict(runs=a.runs)

    if a.split in ("dev", "both"):
        paths = [os.path.join(r, "dev_prob.npz") for r in a.runs if os.path.exists(os.path.join(r, "dev_prob.npz"))]
        if paths:
            P = average(paths)
            gold, dom = {}, {}
            for dm in D.DOMAINS:
                for x in D.load_split(D.split_path(a.data_dir, dm, "dev")):
                    gold[x["id"]], dom[x["id"]] = x["labels"], dm
            pred = {k: [labels[i] for i in P[k].argmax(1)] for k in gold}
            r = E.evaluate(gold, pred, labels, dom)
            summary["dev"] = r
            summary["dev_members"] = len(paths)
            json.dump(pred, open(os.path.join(a.out, "dev_pred.json"), "w"))
            print(f"DEV ensemble of {len(paths)}:\n" + E.format_report(r))

    if a.split in ("test", "both"):
        paths = [os.path.join(r, "test_prob.npz") for r in a.runs]
        P = average(paths)
        docs = [D.RRDoc(x["id"], D.DOMAINS.index(dm), x["text"])
                for dm in D.DOMAINS for x in D.load_split(D.split_path(a.data_dir, dm, "test"))]
        pred = {d.id: [labels[i] for i in P[d.id].argmax(1)] for d in docs}
        D.validate_submission(pred, docs, labels)
        pj = os.path.join(a.out, "prediction.json")
        json.dump(pred, open(pj, "w", encoding="utf-8"))
        with zipfile.ZipFile(os.path.join(a.out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pj, arcname="prediction.json")
        summary["test_members"] = len(paths)
        summary["test_label_counts"] = {l: sum(v.count(l) for v in pred.values()) for l in labels}
        print(f"TEST ensemble of {len(paths)} -> {a.out}/submission.zip")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
