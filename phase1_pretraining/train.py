"""
train.py - Phase-1 pretraining for all four architectures (A: enc-dec, B: GPT/MHA, C: local-GQA+global-MHA,
D: GDN+MLA). Same recipe, same data order for every model; only configs/*.json differ.

    # real run (2 GPUs):
    torchrun --standalone --nproc_per_node=2 train.py --config ../common/configs/b.json --data_root /kaggle/input/cs781-data-v2 \
             --out /kaggle/working/run_b            # stops by itself after --max_hours (default 11)
    # resume after a session restart (attach the old output as an input dataset):
    ... --resume /kaggle/input/<old-output>/run_b/latest.pt
    # 7-minute throughput/memory probe (no checkpoints written):
    torchrun --standalone --nproc_per_node=2 train.py --config ../common/configs/b.json --data_root ... --probe 4,8,16
    # 200-step learning check:
    torchrun --standalone --nproc_per_node=2 train.py --config ../common/configs/b.json --data_root ... --smoke

Recipe: AdamW(0.9, 0.95, eps 1e-8), wd 0.1 on matrices only, clip 1.0, peak LR 6e-4, 300 warm-up steps,
warmup-stable-decay (linear to 0 over the last 20% of the run), global batch 131,072 tokens, fp16 autocast
+ GradScaler. The run length is min(token budget, wall-clock): every 20 steps the end of the run is
re-estimated from the measured step time so the LR decay always finishes before --max_hours (or --stop_at).
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import contextlib
import datetime
import json
import math
import queue
import threading
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model, WINDOW
from p1_loader import P1Data
from sampler import EpochSampler, Prefetch

T_START = time.time()               # process start: --max_hours is measured from here

A_SPLITS = (512, 640, 768)          # model A: encoder prefix length per micro-batch (decoder gets the rest)


# ------------------------------------------------------------------------------------------ utilities
def parse_time(s):
    """'2026-10-06 19:30' (IST assumed unless 'UTC' given) -> unix seconds."""
    s = s.strip()
    tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    if s.endswith("UTC"):
        tz, s = datetime.timezone.utc, s[:-3].strip()
    elif s.endswith("IST"):
        s = s[:-3].strip()
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=tz).timestamp()


class Logger:
    def __init__(self, out, rank):
        self.rank, self.out = rank, out
        if rank == 0:
            os.makedirs(out, exist_ok=True)
            self.txt = open(os.path.join(out, "train_log.txt"), "a")
            self.jsonl = open(os.path.join(out, "metrics.jsonl"), "a")

    def say(self, msg):
        if self.rank == 0:
            line = f"[{datetime.datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}"
            print(line, flush=True)
            self.txt.write(line + "\n")
            self.txt.flush()

    def metric(self, d):
        if self.rank == 0:
            self.jsonl.write(json.dumps(d) + "\n")
            self.jsonl.flush()


def make_optimizer(model, tc):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim != 2]
    groups = [{"params": decay, "weight_decay": tc["wd"]}, {"params": no_decay, "weight_decay": 0.0}]
    kw = dict(lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]), eps=1e-8)
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, fused=True, **kw)
        except Exception:
            pass
    return torch.optim.AdamW(groups, **kw)


def lr_at(step, total, tc):
    if step < tc["warmup"]:
        return tc["lr"] * (step + 1) / tc["warmup"]
    ds = int((1.0 - tc["decay_frac"]) * total)
    if step < ds:
        return tc["lr"]
    return tc["lr"] * max(0.0, (total - step) / max(1, total - ds))


@torch.no_grad()
def evaluate(model_raw, val, n_windows, rank, world, device, use_cuda, mb=16):
    """Shared cross-architecture metric (loss on tokens 768..1023 given 0..767) per source."""
    res = {}
    for src in val.sources:
        w = torch.from_numpy(val.val_windows(src, n_windows, WINDOW))[rank::world]
        tot = torch.zeros(2, dtype=torch.float64, device=device)
        for i in range(0, w.shape[0], mb):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                s, n = model_raw.shared_metric(w[i:i + mb].to(device))
            tot[0] += float(s)
            tot[1] += n
        if world > 1:
            dist.all_reduce(tot)
        res[src] = (tot[0] / tot[1]).item()
    res["avg"] = float(np.mean([res[s] for s in val.sources]))
    res["mix"] = float(sum(wt * res[s] for s, wt in zip(val.sources, val.weights)))
    return res


def save_ckpt(path, model_raw, opt, scaler, state, cfg, model_only=False):
    obj = {"model": model_raw.state_dict(), "cfg": cfg, "state": state}
    if not model_only:
        obj.update(opt=opt.state_dict(), scaler=scaler.state_dict())
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", default="/kaggle/input/cs781-data-v2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max_hours", type=float, default=11.0,
                    help="wall-clock limit of THIS session, measured from process start. The run length is planned so the "
                         "LR decay, final eval and final.pt finish before it (Kaggle sessions end at 12 h).")
    ap.add_argument("--stop_at", default=None, help='optional absolute stop, e.g. "2026-10-06 19:30" (IST)')
    ap.add_argument("--init_from", default=None, help="continual pretraining: load WEIGHTS ONLY from a .pt (fresh optimizer/schedule)")
    ap.add_argument("--data_state", default=None, help="continue the data stream from the sampler position stored in this .pt")
    ap.add_argument("--smoke_steps", type=int, default=200)
    ap.add_argument("--no_ckpt", action="store_true", help="store attention activations instead of recomputing them in backward (faster, more memory)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--probe", default=None, help="comma list of micro-batch sizes to benchmark, e.g. 4,8,16")
    ap.add_argument("--micro_batch", type=int, default=None)
    ap.add_argument("--budget_tokens", type=float, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--hours", type=float, default=10.0, help="only used to print the projected budget in --probe")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    mc, tc = cfg["model"], cfg["train"]
    if args.micro_batch:
        tc["micro_batch"] = args.micro_batch
    if args.budget_tokens:
        tc["budget_tokens"] = args.budget_tokens
    if args.compile:
        tc["compile"] = True
    if args.no_ckpt:
        mc["ckpt"] = False
    if args.seed is not None:
        tc["seed"] = args.seed
    if args.lr is not None:
        tc["lr"] = args.lr
    if args.warmup is not None:
        tc["warmup"] = args.warmup
    if args.smoke:
        tc.update(warmup=20, eval_every=100, eval_windows=16, ckpt_minutes=1e9, budget_tokens=args.smoke_steps * tc["global_batch_tokens"])

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local)
    device = torch.device("cuda", local) if use_cuda else torch.device("cpu")
    if ddp:
        dist.init_process_group("nccl" if use_cuda else "gloo")
    out = args.out or os.path.join("/kaggle/working" if os.path.isdir("/kaggle/working") else ".", "run_" + cfg["name"])
    if args.smoke or args.probe:
        out = out + ("_smoke" if args.smoke else "_probe")
    log = Logger(out, rank)
    log.say(f"config {args.config}  world={world}  device={device}  out={out}")

    # ---- model
    torch.manual_seed(tc["seed"])
    model_raw = build_model(mc).to(device)
    n_params = model_raw.count_params()
    log.say(f"model {cfg['name']}: {n_params / 1e6:.2f}M parameters")
    if tc.get("compile") and mc["kind"] == "encdec":
        # model A passes a varying Python int (split) into forward; DDPOptimizer's graph splitting
        # cannot handle int graph inputs, so turn it off (grads are then all-reduced after backward).
        import torch._dynamo as _dyn
        _dyn.config.optimize_ddp = False
        _dyn.config.cache_size_limit = max(getattr(_dyn.config, "cache_size_limit", 8), 16)
    model = torch.compile(model_raw) if tc.get("compile") else model_raw
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if use_cuda else None)
    opt = make_optimizer(model_raw, tc)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    is_a = mc["kind"] == "encdec"
    mb = tc["micro_batch"]
    seqs_global = tc["global_batch_tokens"] // WINDOW                      # 128 sequences
    accum = max(1, seqs_global // (mb * world))
    assert accum * mb * world == seqs_global, f"micro_batch*world must divide {seqs_global}"
    train = P1Data(args.data_root, "train", tc.get("mix"))
    val = P1Data(args.data_root, "val", tc.get("mix"))
    # "all data seen" cap: stop once EVERY source has been passed over at least once
    cap_tokens = max(train.bin[s].size / w for s, w in zip(train.sources, train.weights))
    budget = min(tc["budget_tokens"], cap_tokens)
    max_steps = int(math.ceil(budget / tc["global_batch_tokens"]))
    total = max_steps
    log.say(f"mix {dict(zip(train.sources, np.round(train.weights, 3)))}  all-data-seen cap {cap_tokens / 1e9:.2f}B tokens; "
            f"budget {budget / 1e9:.2f}B; epochs at budget: {train.epochs(budget)}")
    data_state = None
    if args.data_state:
        data_state = torch.load(args.data_state, map_location="cpu")["state"].get("data")
        log.say(f"data stream continues from {args.data_state}: {'ok' if data_state else 'NO SAMPLER STATE FOUND'}")

    # ---- resume
    step, best, elapsed0 = 0, float("inf"), 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model_raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best, elapsed0 = ck["state"]["step"], ck["state"].get("best", best), ck["state"].get("elapsed", 0.0)
        total = min(total, ck["state"].get("total", total))
        data_state = ck["state"].get("data", data_state)
        log.say(f"resumed from {args.resume} at step {step}")
        del ck
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu")
        model_raw.load_state_dict(ck["model"])
        log.say(f"continual pretraining: weights loaded from {args.init_from}; fresh optimizer and schedule")
        del ck

    # ---- probe mode: throughput + memory per micro-batch size, then exit
    if args.probe:
        for m in [int(x) for x in args.probe.split(",")]:
            if seqs_global % (m * world):
                continue
            acc = seqs_global // (m * world)
            pf = Prefetch(EpochSampler(train, m, tc["seed"], rank, world, WINDOW), local)
            torch.cuda.reset_peak_memory_stats() if use_cuda else None
            try:
                t0 = None
                for it in range(8):
                    if it == 3:
                        if use_cuda:
                            torch.cuda.synchronize()
                        t0 = time.time()
                    opt.zero_grad(set_to_none=True)
                    for i in range(acc):
                        sync = contextlib.nullcontext() if (i == acc - 1 or not ddp) else model.no_sync()
                        with sync:
                            with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                                loss = model(pf.get(device), A_SPLITS[i % 3] if is_a else 0)
                            scaler.scale(loss / acc).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model_raw.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                if use_cuda:
                    torch.cuda.synchronize()
                dt = (time.time() - t0) / 5
                tps = tc["global_batch_tokens"] / dt
                mem = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0
                log.say(f"PROBE micro_batch={m} accum={acc}: {dt:.2f} s/step  {tps:,.0f} window-tok/s  "
                        f"peak mem {mem:.1f} GiB  -> {tps * 3600 * args.hours * 0.97 / 1e9:.2f}B tokens in {args.hours} h")
            except torch.cuda.OutOfMemoryError:
                log.say(f"PROBE micro_batch={m}: OUT OF MEMORY")
                opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache() if use_cuda else None
        return

    # ---- train
    stop_ts = T_START + args.max_hours * 3600
    if args.stop_at:
        stop_ts = min(stop_ts, parse_time(args.stop_at))
    sampler = EpochSampler(train, mb, tc["seed"], rank, world, WINDOW, state=data_state)
    pf = Prefetch(sampler, local)
    sup_per_step = sum(model_raw.supervised_tokens(mb * world, A_SPLITS[i % 3] if is_a else 0) for i in range(accum))
    log.say(f"micro_batch={mb} accum={accum} global_batch={tc['global_batch_tokens']} tokens "
            f"({sup_per_step} supervised)  max_steps={max_steps}  max_hours={args.max_hours}")
    t_start = time.time()
    t_ckpt = time.time()
    win_t, win_s = time.time(), step
    run_loss = torch.zeros((), device=device)
    first_loss = None
    last_log = time.time()
    done = False
    hist = {}
    while step < total and not done:
        lr = lr_at(step, total, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device)
        for i in range(accum):
            split = A_SPLITS[(step * accum + i) % 3] if is_a else 0
            sync = contextlib.nullcontext() if (i == accum - 1 or not ddp) else model.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                    loss = model(pf.get(device), split)
                scaler.scale(loss / accum).backward()
            step_loss += loss.detach() / accum
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model_raw.parameters(), tc["clip"])
        scaler.step(opt)
        scaler.update()
        step += 1
        run_loss += step_loss

        # ---- every 20 steps: agree (across ranks) on the run length given the wall clock
        if step % 20 == 0:
            now = time.time()
            new_total = total
            if step - win_s >= 20:
                per_step = (now - win_t) / (step - win_s) * 1.03
                remaining = stop_ts - 300 - now
                cand = max(step + 1, min(max_steps, step + int(remaining / per_step)))
                ds = int((1.0 - tc["decay_frac"]) * total)
                new_total = cand if step < ds else min(total, cand)
            tt = torch.tensor([new_total, 1 if now > stop_ts else 0], device=device, dtype=torch.long)
            if ddp:
                dist.broadcast(tt, 0)
            if tt[0].item() != total:
                log.say(f"run length re-estimated: total_steps {total} -> {int(tt[0])}")
            total, done = int(tt[0]), bool(tt[1])
            if step - win_s >= 100:
                win_t, win_s = now, step

        if step % 10 == 0:
            lv = (run_loss / 10).item()
            run_loss.zero_()
            if first_loss is None:
                first_loss = lv
            dt = (time.time() - last_log) / 10
            last_log = time.time()
            tps = tc["global_batch_tokens"] / dt
            mem = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0
            gnv = float(gn)
            log.say(f"step {step}/{total} loss {lv:.4f} lr {lr:.2e} gnorm {gnv:.2f} {tps:,.0f} tok/s "
                    f"{dt:.2f}s/step mem {mem:.1f}G scale {scaler.get_scale():.0f} "
                    f"elapsed {(time.time() - t_start + elapsed0) / 3600:.2f}h ep {sampler.epochs_from(pf.last_snap)}")
            log.metric(dict(step=step, loss=lv, lr=lr, epochs=sampler.epochs_from(pf.last_snap), gnorm=gnv, tok_s=tps, scale=scaler.get_scale(),
                            tokens=step * tc["global_batch_tokens"],
                            supervised_tokens=step * sup_per_step, time=time.time()))
            if not math.isfinite(lv):
                log.say("WARNING: non-finite loss on this rank (GradScaler skips such steps)")

        last = step >= total or done
        if step % tc["eval_every"] == 0 or last:
            t0 = time.time()
            r = evaluate(model_raw, val, tc["eval_windows"], rank, world, device, use_cuda)
            log.say("EVAL step %d  " % step + "  ".join(f"{k}={v:.4f}" for k, v in r.items()) +
                    f"  ppl_avg={math.exp(r['avg']):.1f}  ({time.time() - t0:.0f}s)")
            log.metric(dict(step=step, eval=r))
            if r["avg"] < best and rank == 0:
                best = r["avg"]
                save_ckpt(os.path.join(out, "best.pt"), model_raw, opt, scaler, {}, mc, model_only=True)
        if rank == 0 and (last or (time.time() - t_ckpt) / 60 >= tc["ckpt_minutes"]) and not args.smoke:
            st = dict(step=step, total=total, best=best, elapsed=time.time() - t_start + elapsed0, data=pf.last_snap)
            save_ckpt(os.path.join(out, "latest.pt"), model_raw, opt, scaler, st, mc)
            t_ckpt = time.time()
            log.say(f"checkpoint saved (step {step})")

    if rank == 0:
        save_ckpt(os.path.join(out, "final.pt"), model_raw, opt, scaler, dict(step=step, total=total, data=pf.last_snap), mc, model_only=True)
        json.dump(cfg, open(os.path.join(out, "config_used.json"), "w"), indent=1)
        open(os.path.join(out, "DONE"), "w").write(f"step {step}\n")
        log.say(f"DONE at step {step} (planned {total}); first logged loss {first_loss}; final.pt written")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
