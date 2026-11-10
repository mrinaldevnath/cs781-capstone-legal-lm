"""
ner_ensemble.py - combine finished runs by averaging their per-token tag probabilities, then BIO-Viterbi decode.

Works because every run tokenizes the same documents with the same Phase-1 tokenizer (same --pretok), so the
token positions line up exactly across architectures.

    python ner_ensemble.py --runs /kaggle/input/r1/ner_a_full /kaggle/input/r2/ner_d_lorma ... \
        --tok_root ... --data_dir ... --test_file .../fold3-text.json --out /kaggle/working/ens
Dev scoring is done only on documents present in every run's dev_logp.npz (same --seed / --dev_frac).
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import argparse
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ner_data as N
import ner_eval as E
from ner_train import spans_from_logp, write_submission


def mean_logp(paths, ids):
    out = {}
    for k in ids:
        ps = [np.exp(np.load(p)[k].astype(np.float32)) for p in paths]
        out[k] = np.log(np.mean(ps, 0) + 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tok_root", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--test_file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pretok", default="piece")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    labs, _ = N.load_labels(os.path.join(a.data_dir, "labels.txt"))
    ot = N.OffsetTokenizer(N.load_tokenizer(a.tok_root), a.pretok)
    rep = {}
    devp = [os.path.join(r, "dev_logp.npz") for r in a.runs]
    if all(os.path.exists(p) for p in devp):
        common = set.intersection(*[set(np.load(p).files) for p in devp])
        if common:
            texts, gold = N.load_folds(a.data_dir)
            docs = [N.Doc(k, texts[k], ot) for k in sorted(common)]
            lp = mean_logp(devp, [d.id for d in docs])
            pred = spans_from_logp(docs, [lp[d.id] for d in docs], labs, "viterbi")
            r = E.score({d.id: gold[d.id] for d in docs}, pred, labs)
            print("ENSEMBLE DEV\n" + E.format_report(r, labs))
            for run, p in zip(a.runs, devp):
                one = spans_from_logp(docs, [np.load(p)[d.id].astype(np.float32) for d in docs], labs, "viterbi")
                print(f"  single {os.path.basename(run.rstrip('/'))}: macro-F1 {E.score({d.id: gold[d.id] for d in docs}, one, labs)['macro_f1']:.4f}")
            json.dump(pred, open(os.path.join(a.out, "dev_pred.json"), "w"))
            rep["dev"] = r
    if a.test_file:
        tt = N.load_texts(a.test_file)
        docs = [N.Doc(k, tt[k], ot) for k in tt]
        lp = mean_logp([os.path.join(r, "test_logp.npz") for r in a.runs], [d.id for d in docs])
        pred = spans_from_logp(docs, [lp[d.id] for d in docs], labs, "viterbi")
        print("wrote", write_submission(pred, [d.id for d in docs], labs, a.out))
    json.dump(dict(runs=a.runs, **rep), open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
