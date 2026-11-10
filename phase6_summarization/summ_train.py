"""
summ_train.py - one Phase-6 experiment: fine-tune a Phase-1 model for abstractive summarization, pick the checkpoint on
the held-out 'val' split (5% of train.json, never trained on), generate val + test summaries, choose the summary length
on val by BERTScore F1 (roberta-large, the bert-score default for English), write the submission.

    python summ_train.py --ckpt final.pt --tok_root <cs781-legal-data> --cache p6_cache.npz --method lorma --lr 2e-4 \
        --input extract --out runs/D02 --model_dir models

Inputs (--input): lead | extract | chunk | sled  (see summ_data.py; sled = model A only)
Methods (--method): full, lora, lorma, adalora, adapter, qlora, ... (adapt.py)
Distillation: --teacher models/<run>.pt (a saved run of a larger model, same input strategy) --kd_alpha --kd_T
Outputs in --out: summary.json, val_gen.json, test_gen.json (full generations), prediction.json, submission.zip, log.txt;
the trainable weights (and the arguments needed to rebuild the model) in --model_dir/<run>.pt.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import argparse
import json
import math
import time
import zipfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summ_data as SD
import summ_model as SM
from adapt import configure, controller, DEFAULT_LR

LR_DEFAULT = {"full": 5e-5, "lorma": 2e-4, "lora": 2e-4, "adalora": 5e-4, "adapter": 1e-4, "qlora": 2e-4}


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tok_root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--method", default="lorma")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--target_rank", type=int, default=8)
    ap.add_argument("--bottleneck", type=int, default=64)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--input", default="extract", choices=["lead", "extract", "chunk", "sled", "stage2"])
    ap.add_argument("--coarse_dir", default=None, help="stage2: folder with coarse_{fit,val,test}.json (stage-1 chunk summaries)")
    ap.add_argument("--in_len", type=int, default=448)
    ap.add_argument("--out_len", type=int, default=576)
    ap.add_argument("--ch_len", type=int, default=640)
    ap.add_argument("--ch_out", type=int, default=384)
    ap.add_argument("--chunk_budget", type=int, default=640, help="total summary tokens shared equally by the chunks")
    ap.add_argument("--sled_in", type=int, default=3072)
    ap.add_argument("--sled_c", type=int, default=512)
    ap.add_argument("--sled_rho", type=float, default=0.5)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max_minutes", type=float, default=600, help="training time limit (generation comes after)")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--teacher_ckpt", default=None, help="Phase-1 checkpoint the teacher was trained from (this notebook)")
    ap.add_argument("--kd_alpha", type=float, default=0.5)
    ap.add_argument("--kd_T", type=float, default=2.0)
    ap.add_argument("--gen_batch", type=int, default=32)
    ap.add_argument("--min_new", type=int, default=256)
    ap.add_argument("--no_repeat", type=int, default=4)
    ap.add_argument("--words", default="200,275,350,425,0", help="summary lengths (words) tried on val; 0 = untruncated")
    ap.add_argument("--no_bertscore", action="store_true")
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--max_docs", type=int, default=0, help="smoke test: this many train/val documents")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--val_docs", type=int, default=0, help="limit val documents for generation (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


class Log:
    def __init__(self, path):
        self.f = open(path, "a")

    def say(self, s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def trainable_state(holder):
    return {n: p.detach().cpu().clone() for n, p in holder.named_parameters() if p.requires_grad}


def build_model(a, ckpt, ckpt_act=True):
    holder, cfg = SM.load_lm(ckpt, ckpt_act)
    n_tr, n_all, wrapped = configure(holder, a["method"], a["rank"], a["alpha"], 0.05, a["bottleneck"], 4, 20)
    return holder, cfg, n_tr, n_all, wrapped


def load_saved(path, device, base_ckpt=None):
    """Rebuild a saved run (base checkpoint + method + trained weights) for inference / as a teacher.
    base_ckpt: the Phase-1 checkpoint of that model in THIS notebook (the saved path may differ)."""
    sv = torch.load(path, map_location="cpu", weights_only=False)
    base = base_ckpt if base_ckpt else sv["args"]["ckpt"]
    holder, cfg, *_ = build_model(sv["args"], base, ckpt_act=False)
    missing = set(sv["state"]) - {n for n, _ in holder.named_parameters()}
    assert not missing, f"saved parameters not in the model: {sorted(missing)[:5]}"
    params = dict(holder.named_parameters())
    with torch.no_grad():
        for n, v in sv["state"].items():
            params[n].copy_(v)
    for p in holder.parameters():
        p.requires_grad = False
    return holder.to(device).eval(), sv["args"]


def examples(D, split, a, sep, train):
    """-> list of (prompt, target, doc index). Train: oracle extracts; val/test: classifier extracts."""
    sp = D.S[split]
    n = len(sp) if not a.max_docs else min(len(sp), a.max_docs)
    out = []
    for k in range(n):
        if a.input == "chunk":
            chs = D.chunks(split, k, a.ch_len)
            tgs = D.chunk_targets(split, k, a.ch_len, a.ch_out) if sp.has_sum else [None] * len(chs)
            for c, t in zip(chs, tgs):
                if t is not None and len(t) == 0 and train:
                    continue                                            # chunks with no summary sentence: not trained on
                tt = None if t is None else np.concatenate([t, [SM.EOS_ID]]) if len(t) < a.ch_out else t
                out.append((np.concatenate([c, sep]), tt, k))
            continue
        if a.input == "sled":
            p = D.doc_input(split, k, "sled", a.sled_in)
        elif a.input == "stage2":
            p = np.concatenate([coarse_input(D, split, k, a.in_len), sep])
        else:
            p = np.concatenate([D.doc_input(split, k, a.input, a.in_len, use_oracle=train), sep])
        t = None
        if sp.has_sum:
            t = D.target(split, k, a.out_len)
            if D.full_target_len(split, k) < a.out_len:
                t = np.concatenate([t, [SM.EOS_ID]])
        out.append((p, t, k))
    return out


def coarse_input(D, split, k, in_len):
    """Summ^N stage 2: the stage-1 chunk summaries of document k (token ids), cut to in_len tokens."""
    ids = D.coarse[split].get(D.S[split].ids[k]) or [SM.EOS_ID]
    return np.asarray(ids[:in_len], np.int64)


def batches(ex, micro, rng):
    """Micro-batches of similar length (less padding), in random order."""
    idx = sorted(range(len(ex)), key=lambda i: len(ex[i][0]) + len(ex[i][1]) + rng.random())
    mb = [idx[i:i + micro] for i in range(0, len(idx), micro)]
    rng.shuffle(mb)
    return mb


def loss_of(holder, ex, ids, a, device, teacher=None):
    lm = holder.lm
    if a.input == "sled":
        return SM.ed_loss(lm, [ex[i][0] for i in ids], [ex[i][1] for i in ids], device, a.sled_c, a.sled_rho)
    x, tm = SM.dec_batch([ex[i][0] for i in ids], [ex[i][1] for i in ids], device)
    return SM.dec_loss(lm, x, tm, None if teacher is None else teacher.lm, a.kd_alpha, a.kd_T)


@torch.no_grad()
def val_loss(holder, ex, a, device):
    holder.eval()
    tot, n = 0.0, 0
    for i in range(0, len(ex), a.micro):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, k, ce = loss_of(holder, ex, list(range(i, min(len(ex), i + a.micro))), a, device)
        tot += float(ce)
        n += k
    holder.train()
    return tot / max(1, n)


def summaries(holder, D, split, a, sep, device, tok, log):
    """Generate one summary per document of `split` (chunk input: concatenated chunk summaries)."""
    sp = D.S[split]
    n = len(sp) if not (a.val_docs and split == "val") else min(len(sp), a.val_docs)
    if a.max_docs:
        n = min(n, a.max_docs)
    encdec = a.input == "sled"
    jobs, owner = [], []
    for k in range(n):
        if a.input == "chunk":
            chs = D.chunks(split, k, a.ch_len)
            mx = int(min(a.ch_out, max(48, math.ceil(a.chunk_budget / len(chs)))))
            for c in chs:
                jobs.append((np.concatenate([c, sep]), mx, mx // 2))
                owner.append(k)
        elif encdec:
            jobs.append((D.doc_input(split, k, "sled", a.sled_in), a.out_len, a.min_new))
            owner.append(k)
        elif a.input == "stage2":
            jobs.append((np.concatenate([coarse_input(D, split, k, a.in_len), sep]), a.out_len, a.min_new))
            owner.append(k)
        else:
            jobs.append((np.concatenate([D.doc_input(split, k, a.input, a.in_len), sep]), a.out_len, a.min_new))
            owner.append(k)
    t0 = time.time()
    outs = SM.gen_many(holder.lm, jobs, a.gen_batch, device, a.no_repeat, encdec, (a.sled_c, a.sled_rho), log.say)
    per = [[] for _ in range(n)]
    for k, o in zip(owner, outs):
        per[k].extend(o)
    texts = [" ".join(tok.decode([int(t) for t in p]).split()) for p in per]
    log.say(f"{split}: {n} summaries ({len(jobs)} generations) in {time.time() - t0:.0f}s, "
            f"{np.mean([len(t.split()) for t in texts]):.0f} words on average")
    return [sp.ids[k] for k in range(n)], texts, (time.time() - t0) / max(1, n)


def bertscore_f1(cands, refs, device):
    from bert_score import score
    _, _, F1 = score(cands, refs, lang="en", batch_size=16, device=str(device), verbose=False)
    return F1.numpy()


def main(argv=None):
    a = get_args(argv)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.model_dir, exist_ok=True)
    log = Log(os.path.join(a.out, "log.txt"))
    tag = os.path.basename(os.path.normpath(a.out))
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    D = SD.Data(a.cache)
    if a.input == "stage2":
        D.coarse = {sp: json.load(open(os.path.join(a.coarse_dir, f"coarse_{sp}.json"))) for sp in D.S
                    if os.path.exists(os.path.join(a.coarse_dir, f"coarse_{sp}.json"))}
        assert "fit" in D.coarse and "val" in D.coarse, f"stage-1 summaries missing in {a.coarse_dir}"
    from ner_data import load_tokenizer
    tok = load_tokenizer(a.tok_root)
    sep = np.array(SD.encode_text(SD.SEP_TEXT, a.tok_root), np.int64)
    log.say(f"run {tag}: {vars(a)}")

    holder, cfg, n_tr, n_all, wrapped = build_model(vars(a), a.ckpt)
    assert (a.input == "sled") == (cfg["kind"] == "encdec"), "sled is the encoder-decoder input (model A) and only it"
    holder.to(device).train()
    teacher = None
    if a.teacher:
        teacher, targs = load_saved(a.teacher, device, a.teacher_ckpt)
        assert targs["input"] == a.input, "teacher and student must read the same input"
        log.say(f"teacher {a.teacher}: {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M parameters; "
                f"KD alpha {a.kd_alpha}, T {a.kd_T}")
    log.say(f"model {cfg.get('kind')} {n_all / 1e6:.1f}M parameters, trainable {n_tr / 1e6:.2f}M ({a.method}, {wrapped} wrapped)")

    tr = examples(D, "fit", a, sep, train=True)
    va = examples(D, "val", a, sep, train=False)
    va_loss_ex = va if a.input != "chunk" else [va[i] for i in rng.permutation(len(va))[:600]]
    va_loss_ex = [e for e in va_loss_ex if e[1] is not None and len(e[1])]
    log.say(f"{len(tr)} training examples, {len(va)} val examples; mean prompt {np.mean([len(e[0]) for e in tr]):.0f} "
            f"target {np.mean([len(e[1]) for e in tr]):.0f} tokens")

    lr = a.lr if a.lr is not None else LR_DEFAULT.get(a.method, DEFAULT_LR.get(a.method, 1e-4))
    params = [p for p in holder.parameters() if p.requires_grad]
    groups = [dict(params=[p for p in params if p.ndim >= 2], weight_decay=a.wd),
              dict(params=[p for p in params if p.ndim < 2], weight_decay=0.0)]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    steps_ep = math.ceil(len(tr) / (a.micro * a.accum))
    total = max(1, int(round(a.epochs * steps_ep)))
    if a.max_steps:
        total = min(total, a.max_steps)
    warm = max(1, int(a.warmup * total))
    ctl = controller(holder, a.method, total, a.target_rank)
    log.say(f"lr {lr:g}; {total} steps ({steps_ep}/epoch, {a.micro}x{a.accum} sequences/step); warmup {warm}")

    best, best_loss, best_step, hist = None, float("inf"), 0, []
    step, t0, ntok, stop = 0, time.time(), 0, False
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    ep = 0
    while step < total and not stop:
        mbs = batches(tr, a.micro, rng)
        run_l, run_n = 0.0, 0
        for j in range(0, len(mbs), a.accum):
            group = mbs[j:j + a.accum]
            mult = (step + 1) / warm if step < warm else max(0.05, 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            for g in opt.param_groups:
                g["lr"] = lr * mult
            opt.zero_grad(set_to_none=True)
            ntot = sum(len(tr[i][1]) for mb in group for i in mb)
            for mb in group:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                    loss, n, ce = loss_of(holder, tr, mb, a, device, teacher)
                    loss = loss * n / max(1, ntot)
                    if ctl is not None:
                        reg = ctl.regularizer()
                        if torch.is_tensor(reg):
                            loss = loss + reg * len(mb) / max(1, sum(len(m) for m in group))
                scaler.scale(loss).backward()
                run_l += float(ce)
                run_n += n
                ntok += sum(len(tr[i][0]) + len(tr[i][1]) for i in mb)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, a.clip)
            scaler.step(opt)
            scaler.update()
            if ctl is not None:
                ctl.after_step(step)
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                log.say(f"step {step}/{total} train CE {run_l / max(1, run_n):.4f} {ntok / el:,.0f} tok/s "
                        f"mem {torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0:.1f}G {el / 60:.1f} min")
            if (time.time() - t0) / 60 > a.max_minutes:
                log.say("training time limit reached")
                stop = True
            if step >= total or stop:
                break
        ep += 1
        vl = val_loss(holder, va_loss_ex, a, device)
        hist.append(dict(epoch=ep, step=step, train_ce=run_l / max(1, run_n), val_ce=vl, minutes=(time.time() - t0) / 60))
        log.say(f"epoch {ep}: train CE {run_l / max(1, run_n):.4f} | val CE {vl:.4f} (ppl {math.exp(min(vl, 20)):.2f})")
        if vl < best_loss:
            best_loss, best_step, best = vl, step, trainable_state(holder)
    train_min = (time.time() - t0) / 60
    tok_s = ntok / max(1e-9, time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0
    if best is not None:
        pd = dict(holder.named_parameters())
        with torch.no_grad():
            for n_, v in best.items():
                pd[n_].copy_(v.to(pd[n_].device))
    sv = dict(args=vars(a), state=best if best is not None else trainable_state(holder), best_step=best_step, val_ce=best_loss)
    torch.save(sv, os.path.join(a.model_dir, f"{tag}.pt"))
    log.say(f"best epoch at step {best_step} (val CE {best_loss:.4f}); saved {a.model_dir}/{tag}.pt")
    del opt, scaler, teacher
    if use_cuda:
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- generation + length choice on val
    holder.eval()
    nm = SM.merge_for_decode(holder)
    if nm:
        log.say(f"merged {nm} adapted MLA up-projections into plain weights for cached decoding")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    vids, vgen, ms_doc = summaries(holder, D, "val", a, sep, device, tok, log)
    gen_peak = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0
    json.dump([{"id": i, "summary": s} for i, s in zip(vids, vgen)], open(os.path.join(a.out, "val_gen.json"), "w"))
    tgen = None
    if not a.no_test and "test" in D.S:
        tids, tgen, _ = summaries(holder, D, "test", a, sep, device, tok, log)
        json.dump([{"id": i, "summary": s} for i, s in zip(tids, tgen)], open(os.path.join(a.out, "test_gen.json"), "w"))
    del holder
    if use_cuda:
        torch.cuda.empty_cache()
    refs = [SD.clean(x) for x in D_val_refs(a, vids)]
    lengths = [int(w) for w in a.words.split(",")]
    table = []
    for W in lengths:
        cands = [SD.cut_words(g, W) or "." for g in vgen]
        r = np.array([SD.rouge(c, rf) for c, rf in zip(cands, refs)])
        row = dict(words=W, mean_words=float(np.mean([len(c.split()) for c in cands])),
                   rouge1=float(r[:, 0].mean()), rougeL=float(r[:, 1].mean()))
        if not a.no_bertscore:
            try:
                row["bertscore_f1"] = float(bertscore_f1(cands, refs, device).mean())
            except Exception as ex:
                log.say(f"BERTScore unavailable ({ex!r}); choosing the length by ROUGE-1")
                a.no_bertscore = True
        table.append(row)
        log.say(f"val, {W or 'all'} words: " + " ".join(f"{k} {v:.4f}" for k, v in row.items() if k not in ("words",)))
    key = "bertscore_f1" if not a.no_bertscore else "rouge1"
    bestW = max(table, key=lambda r: r[key])
    log.say(f"chosen length: {bestW['words'] or 'untruncated'} words ({key} {bestW[key]:.4f})")
    summary = dict(run=tag, args=vars(a), params_total=n_all, params_trainable=n_tr, train_minutes=train_min, tok_s=tok_s,
                   peak_mem_gb=peak, gen_peak_mem_gb=gen_peak, gen_s_per_doc=ms_doc, history=hist, best_step=best_step,
                   val_ce=best_loss, val_by_length=table, chosen=bestW, select_key=key, n_val=len(vids))
    if tgen is not None:
        pred = [{"id": i, "summary": SD.cut_words(g, bestW["words"]) or "."} for i, g in zip(tids, tgen)]
        pj = os.path.join(a.out, "prediction.json")
        json.dump(pred, open(pj, "w"), ensure_ascii=False)
        with zipfile.ZipFile(os.path.join(a.out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pj, arcname="prediction.json")
        summary["n_test"] = len(pred)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    log.say("done")


def D_val_refs(a, vids):
    """Reference summaries of the val documents, read back from train.json (path stored next to the cache)."""
    meta = json.load(open(a.cache + ".refs.json"))
    return [meta[i] for i in vids]


if __name__ == "__main__":
    main()
