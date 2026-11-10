#!/usr/bin/env python
"""
CS781 Phase 1 - training-throughput benchmark on Kaggle 2x T4 (DDP, fp16 AMP, seq 1024).

The model is a REPRESENTATIVE ~110M-param GPT-style decoder (d=768, 12 layers, 12 heads,
tied embeddings) used only to measure tokens/s. It is not an architecture decision.
Token *values* do not affect speed; vocabulary *size* does (embedding + softmax), so the
run measures the main vocab for MEASURE_SECONDS and the other candidate vocabs briefly.

Kaggle: Accelerator = "GPU T4 x2". Upload this file (and optionally the tokenizer json), then in a cell:
    !TOKENIZER=/kaggle/input/<dataset>/bpe_v2_32768.json torchrun --standalone --nproc_per_node=2 bench_t4.py
Env (all optional): TOKENIZER, VOCABS="32768,16384,49152", MEASURE_SECONDS=360, EXTRA_SECONDS=120,
                    WARMUP_SECONDS=90, ACCUM=4, COMPILE=0, DATA_BIN=<uint16 token file>
~17 min total (~0.3 GPU-h). The results are printed after "==== BENCH RESULTS".
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import contextlib
import gc
import json
import math
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

SMOKE = os.environ.get("SMOKE") == "1"
SEQ = int(os.environ.get("SEQ", 1024))
D = int(os.environ.get("D", 768))
NL = int(os.environ.get("LAYERS", 12))
NH = int(os.environ.get("HEADS", 12))
MEASURE = float(os.environ.get("MEASURE_SECONDS", 360))
EXTRA = float(os.environ.get("EXTRA_SECONDS", 120))
WARMUP = float(os.environ.get("WARMUP_SECONDS", 90))
ACCUM = int(os.environ.get("ACCUM", 4))
COMPILE = os.environ.get("COMPILE", "0") == "1"
PEAK_FP16 = 65e12            # T4 fp16 tensor-core peak (NVIDIA spec), for MFU only
OVERHEAD = 0.95              # allowance for eval/checkpointing in a real run
CANDIDATE_B = [16, 12, 10, 8, 6, 4, 2, 1]
if SMOKE:
    SEQ, D, NL, NH, MEASURE, EXTRA, WARMUP, CANDIDATE_B = 64, 64, 2, 2, 5, 3, 2, [2]


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh = nh
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.nh, C // self.nh).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.V = V
        self.wte = nn.Embedding(V, D)
        self.wpe = nn.Embedding(SEQ, D)
        self.blocks = nn.ModuleList(Block(D, NH) for _ in range(NL))
        self.ln_f = nn.LayerNorm(D)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, tgt):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = F.linear(self.ln_f(x), self.wte.weight)          # tied output head
        return F.cross_entropy(logits.view(-1, self.V), tgt.reshape(-1))


# ----------------------------------------------------------------------------- setup
def setup():
    if "RANK" in os.environ:
        cuda = torch.cuda.is_available() and not SMOKE
        dist.init_process_group("nccl" if cuda else "gloo")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available() and not SMOKE:
        dev = torch.device(f"cuda:{local}")
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")
    return rank, world, local, dev


class Batches:
    def __init__(self, V, dev, rank):
        self.V, self.dev = V, dev
        self.rng = np.random.default_rng(1000 + rank)
        p = os.environ.get("DATA_BIN")
        self.data = np.memmap(p, dtype=np.uint16, mode="r") if p and os.path.exists(p) else None

    def get(self, B):
        if self.data is not None:
            ix = self.rng.integers(0, len(self.data) - SEQ - 1, B)
            x = torch.from_numpy(np.stack([self.data[i:i + SEQ + 1] for i in ix]).astype(np.int64)) % self.V
            x = x.to(self.dev, non_blocking=True)
        else:
            x = torch.randint(0, self.V, (B, SEQ + 1), device=self.dev)
        return x[:, :-1], x[:, 1:]


def allreduce(x, op, dev, world):
    t = torch.tensor([x], dtype=torch.float64, device=dev)
    if world > 1:
        dist.all_reduce(t, op=op)
    return t.item()


def stop_flag(cond, dev, world):
    """Rank 0 decides; everyone follows (avoids DDP hangs from per-rank clocks)."""
    t = torch.tensor([int(cond)], device=dev)
    if world > 1:
        dist.broadcast(t, 0)
    return bool(t.item())


# ----------------------------------------------------------------------------- one vocab
def bench(V, seconds, rank, world, local, dev):
    cuda = dev.type == "cuda"
    model = GPT(V).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if cuda else contextlib.nullcontext
    data = Batches(V, dev, rank)

    def raw_step(m, B):
        x, y = data.get(B)
        with amp():
            loss = m(x, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    raw_step(model, 1)                  # allocates AdamW states before probing
    chosen = CANDIDATE_B[-1]
    for B in CANDIDATE_B:               # largest micro-batch that fits, probed WITHOUT DDP
        ok = True
        try:
            raw_step(model, B)
            if cuda:
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            ok = False
        opt.zero_grad(set_to_none=True)
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        if ok:
            chosen = B
            break
    B = int(allreduce(chosen, dist.ReduceOp.MIN if world > 1 else None, dev, world))

    run_model = torch.compile(model) if COMPILE else model
    ddp = nn.parallel.DistributedDataParallel(run_model, device_ids=[local] if cuda else None) if world > 1 else run_model

    def step():
        for m in range(ACCUM):
            x, y = data.get(B)
            sync_ctx = ddp.no_sync() if (world > 1 and m < ACCUM - 1) else contextlib.nullcontext()
            with sync_ctx:
                with amp():
                    loss = ddp(x, y) / ACCUM
                scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    t = time.time()
    n = 0
    while True:                                   # warm-up (includes compile, cuDNN/cuBLAS autotune)
        step()
        n += 1
        if stop_flag(n >= 5 and time.time() - t >= (WARMUP if seconds == MEASURE else min(WARMUP, 45)), dev, world):
            break
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    if world > 1:
        dist.barrier()

    t0, steps = time.time(), 0
    while True:
        step()
        steps += 1
        if stop_flag(time.time() - t0 >= seconds, dev, world):
            break
    if cuda:
        torch.cuda.synchronize()
    elapsed = allreduce(time.time() - t0, dist.ReduceOp.MAX if world > 1 else None, dev, world)
    peak = allreduce(torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0,
                     dist.ReduceOp.MAX if world > 1 else None, dev, world)

    tokens = steps * ACCUM * B * SEQ * world
    tps = tokens / elapsed
    flops_tok = 6 * (12 * NL * D * D + V * D) + 12 * NL * SEQ * D
    res = {"vocab": V, "params_M": round(n_params / 1e6, 1), "micro_batch_per_gpu": B, "accum": ACCUM,
           "global_batch_tokens": B * ACCUM * SEQ * world, "steps": steps, "seconds": round(elapsed, 1),
           "tokens_per_s": round(tps), "tokens_per_s_per_gpu": round(tps / world),
           "step_time_s": round(elapsed / steps, 3), "peak_mem_GiB": round(peak, 2),
           "mfu": round(tps * flops_tok / (world * PEAK_FP16), 3),
           "proj_10h_B": round(tps * 36000 / 1e9, 3), "proj_11h_B": round(tps * 39600 / 1e9, 3),
           "proj_10h_B_with_5pct_overhead": round(tps * 36000 * OVERHEAD / 1e9, 3)}

    del ddp, run_model, model, opt
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    if world > 1:
        dist.barrier()
    return res


def main():
    rank, world, local, dev = setup()
    vocabs = [int(v) for v in os.environ.get("VOCABS", "32768,16384,49152").split(",")]
    tp = os.environ.get("TOKENIZER")
    if tp:
        vocabs[0] = json.load(open(tp))["vocab_size"]
        vocabs = [vocabs[0]] + [v for v in vocabs[1:] if v != vocabs[0]]
    if SMOKE:
        vocabs = [300, 500]
    if rank == 0:
        print(f"torch {torch.__version__} | world {world} | device {dev} | "
              f"GPUs {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if dev.type == 'cuda' else '-'}")
        print(f"model d={D} layers={NL} heads={NH} seq={SEQ} accum={ACCUM} compile={COMPILE} | "
              f"data={'DATA_BIN' if os.environ.get('DATA_BIN') else 'random ids (speed is value-independent)'}", flush=True)
    results = []
    for k, V in enumerate(vocabs):
        r = bench(V, MEASURE if k == 0 else EXTRA, rank, world, local, dev)
        results.append(r)
        if rank == 0:
            print(f"[bench] vocab {V}: {r['tokens_per_s']:,} tok/s", flush=True)
    if rank == 0:
        print("\n==== BENCH RESULTS ====")
        for r in results:
            print(json.dumps(r))
        main_r = results[0]
        print(f"MAIN vocab {main_r['vocab']}: {main_r['tokens_per_s']:,} tokens/s effective "
              f"({main_r['tokens_per_s_per_gpu']:,}/GPU, MFU {main_r['mfu']:.0%}) -> "
              f"10h = {main_r['proj_10h_B']:.2f}B tokens, 11h = {main_r['proj_11h_B']:.2f}B "
              f"(10h with 5% eval/ckpt overhead = {main_r['proj_10h_B_with_5pct_overhead']:.2f}B)")
        json.dump(results, open("bench_results.json", "w"), indent=1)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
