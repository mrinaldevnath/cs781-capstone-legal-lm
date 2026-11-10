"""
summ_coarse.py - Summ^N stage 1 (Zhang et al., ACL 2022): run a saved chunk-and-join model (e.g. D08) over every
document of the chosen splits and store its chunk summaries (token ids) as the input of the stage-2 model.

    python summ_coarse.py --saved models/D08_full_chunk.pt --base_ckpt <Phase-1 D final.pt> --tok_root ... --cache ... \
        --out_dir coarse --shard 0 --n_shards 2

Each chunk gets an equal share of `--budget` tokens (at least `--floor`, at most the model's chunk output length), so
the concatenated stage-1 summary of a document is about `--budget` tokens and fits the stage-2 input.
Writes <out_dir>/coarse_<split>_<shard>.json = {doc id: [token ids]}.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import argparse
import json
import math
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summ_data as SD
import summ_model as SM
from summ_train import load_saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved", required=True)
    ap.add_argument("--base_ckpt", default=None)
    ap.add_argument("--tok_root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", default="fit,val,test")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--budget", type=int, default=576)
    ap.add_argument("--floor", type=int, default=24)
    ap.add_argument("--gen_batch", type=int, default=48)
    ap.add_argument("--max_docs", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    holder, sargs = load_saved(a.saved, device, a.base_ckpt)
    assert sargs["input"] == "chunk", "stage 1 must be a chunk-and-join model"
    SM.merge_for_decode(holder)
    D = SD.Data(a.cache)
    sep = np.array(SD.encode_text(SD.SEP_TEXT, a.tok_root), np.int64)
    ch_len, ch_out, no_rep = sargs["ch_len"], sargs["ch_out"], sargs["no_repeat"]
    for split in a.splits.split(","):
        if split not in D.S:
            continue
        sp = D.S[split]
        n = len(sp) if not a.max_docs else min(len(sp), a.max_docs)
        docs = [k for k in range(n) if k % a.n_shards == a.shard]
        jobs, owner = [], []
        for k in docs:
            chs = D.chunks(split, k, ch_len)
            mx = int(min(ch_out, max(a.floor, math.ceil(a.budget / len(chs)))))
            for c in chs:
                jobs.append((np.concatenate([c, sep]), mx, mx // 2))
                owner.append(k)
        t0 = time.time()
        outs = SM.gen_many(holder.lm, jobs, a.gen_batch, device, no_rep, False, (512, 0.5),
                           lambda s: print(f"[{split}] {s}", flush=True))
        res = {sp.ids[k]: [] for k in docs}
        for k, o in zip(owner, outs):
            res[sp.ids[k]].extend(int(t) for t in o)
        json.dump(res, open(os.path.join(a.out_dir, f"coarse_{split}_{a.shard}.json"), "w"))
        L = [len(v) for v in res.values()]
        print(f"{split} shard {a.shard}: {len(docs)} docs, {len(jobs)} chunks, {np.mean(L):.0f} tokens per document, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
