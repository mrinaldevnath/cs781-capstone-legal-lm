"""
lsi_train.py - Phase-5 Legal Statute Identification: adapt a Phase-1 backbone on train.json (multi-label, one sigmoid
per statute), select the checkpoint and the decision thresholds on dev.json (macro-F1), predict test.json and write
submission.zip. Dev and test documents are never trained on (training documents that also occur in dev/test are
removed when the cache is built).

    python lsi_train.py --ckpt .../run_d_.../final.pt --tok_root .../cs781-legal-data --data_dir /tmp/p5data \
        --cache /tmp/p5_cache.npz --method lorma --lr 2e-4 --out runs/D01_lorma --model_dir models

Training: --epochs passes (default 1) over the training documents, batches of --batch_docs documents (micro-batches of
at most --micro_chunks chunks, gradients accumulated), linear warmup + linear decay, AdamW, fp16 autocast.
Checkpoint selection: every --eval_every of the steps, macro-F1 at the best global threshold on a fixed subset of
--eval_docs dev documents; the final report and thresholds use the whole dev set.

Losses (--loss): bce (binary relevance), bce_pw (BCE with per-statute positive weights sqrt(neg/pos), capped),
focal (gamma 2), asl (asymmetric loss: gamma- 4, gamma+ 0, negative-probability clip 0.05).

Decision rules saved for every run (dev_prob.npz + summary.json): threshold 0.5, best global threshold, per-statute
thresholds (in-sample and an honest 2-fold cross-fitted estimate); every document gets at least one statute.
The submission uses the best global threshold (robust); per-label thresholds are applied in the final notebook.

Knowledge distillation (optional, not part of the main experiments):
    teacher:  --save_train_probs writes train_prob.npz (per-statute probabilities for every training document)
    student:  --teacher_probs <file> --kd_alpha 0.5 --kd_T 2  ->  loss = (1-a) L(y) + a T^2 BCE(sigmoid(s/T), sigmoid(t/T))
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import argparse
import datetime
import json
import math
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lsi_data as D
import lsi_eval as E
from adapt import configure, controller, gradual_params, uses_prompt_slots, DEFAULT_LR, METHODS
from lsi_model import Backbone, LSIModel, HEADS


class Log:
    def __init__(self, out):
        os.makedirs(out, exist_ok=True)
        self.f = open(os.path.join(out, "train_log.txt"), "a", encoding="utf-8")
        self.m = open(os.path.join(out, "metrics.jsonl"), "a", encoding="utf-8")

    def say(self, s):
        line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def metric(self, d):
        self.m.write(json.dumps(d) + "\n")
        self.m.flush()


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Phase-1 final.pt")
    ap.add_argument("--tok_root", required=True, help="Phase-1 data folder (tokenizer)")
    ap.add_argument("--data_dir", required=True, help="folder holding train.json / dev.json / test.json (any depth)")
    ap.add_argument("--cache", default=None, help="tokenized-data .npz (built if missing)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="lorma", choices=METHODS)
    # document representation and head
    ap.add_argument("--chunk", type=int, default=512, help="tokens per chunk (one backbone window)")
    ap.add_argument("--n_chunks", type=int, default=4, help="chunks read per document")
    ap.add_argument("--select", default="spread", choices=D.SELECTS, help="which chunks of the document are read")
    ap.add_argument("--head", default="attn", choices=HEADS)
    ap.add_argument("--cooc", action="store_true", help="statute co-occurrence layer on top of the logits")
    ap.add_argument("--no_pos", action="store_true", help="drop the chunk-position features")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--loss", default="bce", choices=["bce", "bce_pw", "focal", "asl"])
    ap.add_argument("--pw_cap", type=float, default=10.0, help="bce_pw: cap of the positive weights")
    # optimisation
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_docs", type=int, default=8)
    ap.add_argument("--micro_chunks", type=int, default=8, help="max chunks per forward pass (memory)")
    ap.add_argument("--max_train_docs", type=int, default=None, help="random subset of the training documents")
    ap.add_argument("--lr", type=float, default=None, help="LR of the trainable backbone/PEFT params (default per method)")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=float, default=0.25, help="dev evaluation every this fraction of the steps")
    ap.add_argument("--eval_docs", type=int, default=3000, help="dev documents used for checkpoint selection")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--peft_dropout", type=float, default=0.05)
    ap.add_argument("--bottleneck", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--n_prompt", type=int, default=20, help="prompt / prefix length")
    ap.add_argument("--target_rank", type=int, default=8, help="AdaLoRA: average rank kept at the end (start = --rank)")
    # knowledge distillation (optional)
    ap.add_argument("--teacher_probs", default=None, help="train_prob.npz of a teacher run -> distillation")
    ap.add_argument("--kd_alpha", type=float, default=0.5)
    ap.add_argument("--kd_T", type=float, default=2.0)
    ap.add_argument("--save_train_probs", action="store_true", help="after training, predict every training document")
    # misc
    ap.add_argument("--max_docs", type=int, default=None, help="debug: documents per split")
    ap.add_argument("--max_minutes", type=float, default=600, help="training time limit (dev is evaluated at the stop)")
    ap.add_argument("--procs", type=int, default=None, help="tokenizer processes when the cache is built")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--act_ckpt", action="store_true", help="recompute attention in backward (less memory)")
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--model_dir", default=None, help="where <run name>.pt is saved (default: --out)")
    ap.add_argument("--load", default=None, help="saved model .pt to start from")
    ap.add_argument("--predict_only", action="store_true", help="with --load: no training, only dev/test prediction")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------ helpers
def pieces_of(cache, sp, idx, a):
    return [D.doc_chunks(cache.doc(sp, i), a.chunk, a.n_chunks, a.select) for i in idx]


def forward_docs(model, pieces, a, device):
    n_prompt = a.n_prompt if uses_prompt_slots(a.method) else 0
    ids, valid, pool, doc, slot, rel = D.batch_arrays(pieces, a.chunk, n_prompt)
    t = lambda x: torch.from_numpy(x).to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        logits, z, w = model(t(ids), t(valid), t(pool), t(doc), t(slot), t(rel), len(pieces))
    return logits, z, w, int(pool.sum())


@torch.no_grad()
def predict(model, cache, sp, idx, a, device):
    """-> P (n, K) float32 sigmoid scores, chunk weights (n, S) (NaN where a document has fewer chunks), positions."""
    model.eval()
    pieces = pieces_of(cache, sp, idx, a)
    S = max(1, max(len(p[0]) for p in pieces)) if pieces else 1
    P = np.zeros((len(idx), model.n_labels), dtype=np.float32)
    att = np.full((len(idx), S), np.nan, dtype=np.float32)
    pos = np.full((len(idx), S), np.nan, dtype=np.float32)
    for mb in D.micro_batches(pieces, 2 * a.micro_chunks):
        logits, _, w, _ = forward_docs(model, [pieces[j] for j in mb], a, device)
        P[mb] = torch.sigmoid(logits.float()).cpu().numpy()
        for r, j in enumerate(mb):
            n = len(pieces[j][0])
            pos[j, :n] = pieces[j][1]
            if w is not None:
                att[j, :n] = w[r, :n].float().cpu().numpy()
    return P, att, pos


def full_report(Y, P):
    """Dev report with every decision rule."""
    t_g, f_g = E.best_global(Y, P)
    t_l = E.per_label_thresholds(Y, P, fallback=t_g)
    r = {"at_0.5": E.report(Y, E.decide(P, 0.5)), "global": E.report(Y, E.decide(P, t_g)),
         "per_label": E.report(Y, E.decide(P, t_l)), "global_threshold": t_g,
         "per_label_thresholds": t_l.tolist(), "crossfit": E.crossfit_per_label(Y, P)}
    r["macro_f1"] = r["global"]["macro_f1"]                      # the selection metric
    return r


def loss_fn(logits, Y, a, pw):
    x, y = logits.float(), Y.float()
    if a.loss == "bce":
        return F.binary_cross_entropy_with_logits(x, y, reduction="none").sum(1)
    if a.loss == "bce_pw":
        return F.binary_cross_entropy_with_logits(x, y, pos_weight=pw, reduction="none").sum(1)
    p = torch.sigmoid(x)
    if a.loss == "focal":
        ce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
        pt = torch.where(y > 0, p, 1 - p)
        return ((1 - pt) ** 2 * ce).sum(1)
    # asl: positives log(p); negatives with shifted probability p_m = max(p - 0.05, 0) and focusing (p_m)^4
    p_m = (p - 0.05).clamp(min=0)
    lp = torch.log(p.clamp(min=1e-8))
    ln = torch.log((1 - p_m).clamp(min=1e-8))
    return -(y * lp + (1 - y) * (p_m ** 4) * ln).sum(1)


def kd_loss(logits, tp, T):
    """T^2 * BCE between softened teacher and student per-statute probabilities (teacher given as probabilities)."""
    tp = tp.clamp(1e-4, 1 - 1e-4)
    tq = torch.sigmoid((torch.log(tp) - torch.log1p(-tp)) / T)
    return (T * T) * F.binary_cross_entropy_with_logits(logits.float() / T, tq, reduction="none").sum(1)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    a = get_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    log = Log(a.out)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    use_cuda = device.type == "cuda"
    log.say(f"args {json.dumps(vars(a))}")

    # ---- data
    t0 = time.time()
    cache_path = a.cache or os.path.join(a.out, "p5_cache.npz")
    if not os.path.exists(cache_path):
        D.build_cache(a.data_dir, a.tok_root, cache_path, a.procs, log.say)
    cache = D.Cache(cache_path)
    K = cache.n_labels
    rng = np.random.default_rng(a.seed)
    tr_idx = np.arange(cache.n("train"))
    if a.max_train_docs and a.max_train_docs < len(tr_idx):
        tr_idx = np.sort(rng.choice(tr_idx, a.max_train_docs, replace=False))
    dev_idx = np.arange(cache.n("dev"))
    te_idx = np.arange(cache.n("test")) if cache.has("test") and not a.no_test else np.arange(0)
    if a.max_docs:
        tr_idx, dev_idx, te_idx = tr_idx[:a.max_docs], dev_idx[:a.max_docs], te_idx[:a.max_docs]
    sel_idx = dev_idx if len(dev_idx) <= a.eval_docs else np.sort(
        np.random.default_rng(0).choice(dev_idx, a.eval_docs, replace=False))
    Y_tr, Y_dev = cache.Y("train", tr_idx), cache.Y("dev", dev_idx)
    Y_sel = cache.Y("dev", sel_idx)
    freq = Y_tr.sum(0)
    log.say(f"data: train {len(tr_idx)} docs, dev {len(dev_idx)} (selection subset {len(sel_idx)}), test {len(te_idx)}; "
            f"{K} statutes, {Y_tr.sum(1).mean():.2f} per training document, rarest {int(freq.min())} / most frequent "
            f"{int(freq.max())} training documents; reading {a.n_chunks} x {a.chunk} tokens ({a.select}) [{time.time() - t0:.0f}s]")

    # ---- model
    bb = Backbone(a.ckpt, ckpt_act=a.act_ckpt)
    n_tr, n_all, n_wr = configure(bb, a.method, a.rank, a.alpha, a.peft_dropout, a.bottleneck, a.top_k, a.n_prompt)
    model = LSIModel(bb, K, a.head, a.hidden, a.dropout, not a.no_pos, a.cooc).to(device)
    if a.load:
        ck = torch.load(a.load, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"], strict=False)
        log.say(f"loaded {a.load} (step {ck.get('best_step')}, dev macro-F1 {ck.get('dev_macro_f1')})")
    n_head = sum(p.numel() for p in model.head_parameters())
    log.say(f"backbone {bb.cfg.get('kind')} d={bb.d}: {n_all / 1e6:.2f}M params; method={a.method} wrapped={n_wr} "
            f"trainable backbone {n_tr / 1e6:.3f}M + head {n_head / 1e6:.3f}M "
            f"({100 * (n_tr + n_head) / (n_all + n_head):.2f}% of {(n_all + n_head) / 1e6:.2f}M); head={a.head} "
            f"cooc={a.cooc} loss={a.loss}")
    summary = dict(args=vars(a), n_labels=K, params_total=n_all + n_head, params_trainable=n_tr + n_head,
                   params_backbone_trainable=n_tr, params_head=n_head, n_train_docs=len(tr_idx),
                   n_dev_docs=len(dev_idx), n_test_docs=len(te_idx), cache_info=cache.info,
                   train_label_freq=freq.astype(int).tolist())

    best_f1, best_step, rec = float("nan"), 0, {}
    if not a.predict_only:
        neg = len(tr_idx) - freq
        pw = torch.tensor(np.clip(np.sqrt(neg / np.maximum(freq, 1)), 1.0, a.pw_cap), dtype=torch.float32, device=device)
        teacher = None
        if a.teacher_probs:
            z = np.load(a.teacher_probs)
            tmap = {str(i): r for i, r in zip(z["ids"], z["P"])}
            ids_all = cache.ids("train")
            teacher = np.stack([tmap.get(ids_all[i], np.full(K, np.nan, np.float32)) for i in tr_idx]).astype(np.float32)
            log.say(f"distillation from {a.teacher_probs}: {int(np.isfinite(teacher[:, 0]).sum())}/{len(teacher)} "
                    f"documents covered, alpha {a.kd_alpha}, T {a.kd_T}")

        bb_p = gradual_params(model.backbone) if a.method == "gradual" else \
            [p for p in model.backbone.parameters() if p.requires_grad]
        hd_p = model.head_parameters()
        lr_bb = a.lr if a.lr is not None else DEFAULT_LR[a.method]
        groups = []
        for ps, lr in ((bb_p, lr_bb), (hd_p, a.lr_head)):
            for dec in (True, False):
                sel = [p for p in ps if (p.ndim >= 2) == dec]
                if sel:
                    groups.append(dict(params=sel, lr=lr, base_lr=lr, weight_decay=a.wd if dec else 0.0))
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
        total = max(1, math.ceil(a.epochs * len(tr_idx) / a.batch_docs))
        warm = max(1, int(a.warmup * total))
        every = max(1, int(round(a.eval_every * total)))
        ctl = controller(model.backbone, a.method, total, a.target_rank)
        order = np.concatenate([rng.permutation(len(tr_idx)) for _ in range(math.ceil(a.epochs))])[:total * a.batch_docs]
        log.say(f"lr backbone {lr_bb:g} head {a.lr_head:g}; {total} steps x {a.batch_docs} docs ({a.epochs:g} epochs); "
                f"warmup {warm}; dev every {every} steps")

        best_state, best_f1, step, t_start = None, -1.0, 0, time.time()
        params = [p for g in opt.param_groups for p in g["params"]]
        tl, tn, ntok, t_last, stop = 0.0, 0, 0, time.time(), False
        model.train()
        for step in range(1, total + 1):
            mult = step / warm if step <= warm else max(0.0, (total - step + 1) / max(1, total - warm + 1))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * mult
            pos_b = order[(step - 1) * a.batch_docs: step * a.batch_docs]
            pieces = pieces_of(cache, "train", tr_idx[pos_b], a)
            opt.zero_grad(set_to_none=True)
            ltot = 0.0
            for mb in D.micro_batches(pieces, a.micro_chunks):
                logits, _, _, nt = forward_docs(model, [pieces[j] for j in mb], a, device)
                y = torch.from_numpy(Y_tr[pos_b[mb]]).to(device)
                l = loss_fn(logits, y, a, pw)
                if teacher is not None:
                    tt = torch.from_numpy(teacher[pos_b[mb]]).to(device)
                    ok = torch.isfinite(tt[:, 0])
                    kd = kd_loss(logits, torch.nan_to_num(tt, 0.5), a.kd_T)
                    l = torch.where(ok, (1 - a.kd_alpha) * l + a.kd_alpha * kd, l)
                loss = l.sum() / len(pos_b)
                if ctl is not None:
                    reg = ctl.regularizer()
                    if torch.is_tensor(reg):
                        loss = loss + reg * len(mb) / len(pos_b)
                scaler.scale(loss).backward()
                ltot += float(loss.item())
                ntok += nt
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], a.clip)
            scaler.step(opt)
            scaler.update()
            if ctl is not None:
                ctl.after_step(step - 1)
            tl += ltot
            tn += 1
            over = (time.time() - t_start) / 60 > a.max_minutes
            if step % every == 0 or step == total or over:
                dt = time.time() - t_last
                rec = dict(step=step, frac=step / total, loss=tl / max(1, tn), tok_s=ntok / max(dt, 1e-9),
                           mem_gb=torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0,
                           minutes=(time.time() - t_start) / 60)
                t_ev = time.time()
                Ps, _, _ = predict(model, cache, "dev", sel_idx, a, device)
                thr, f = E.best_global(Y_sel, Ps)
                r5 = E.report(Y_sel, E.decide(Ps, 0.5))
                rec.update(dev_macro_f1=f, dev_thr=thr, dev_macro_f1_05=r5["macro_f1"], dev_micro_f1_05=r5["micro_f1"],
                           dev_s=time.time() - t_ev)
                msg = (f"step {step}/{total} loss {rec['loss']:.4f} {rec['tok_s']:,.0f} tok/s mem {rec['mem_gb']:.1f}G "
                       f"| dev[{len(sel_idx)}] macro-F1 {f:.4f} (thr {thr:.2f}; at 0.5: {r5['macro_f1']:.4f}, "
                       f"micro {r5['micro_f1']:.4f})")
                if f > best_f1:
                    best_f1, best_step = f, step
                    best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                                  if p.requires_grad or (a.method == "gradual" and n.startswith("backbone."))}
                    msg += "  *best*"
                log.say(msg)
                log.metric(rec)
                tl, tn, ntok, t_last = 0.0, 0, 0, time.time()
                model.train()
                if over:
                    log.say(f"time limit reached at step {step}/{total}")
                    stop = True
            if stop:
                break
        if a.method == "gradual":
            for p in gradual_params(model.backbone):
                p.requires_grad = True
            named = dict(model.named_parameters())
            best_state = {n: v for n, v in best_state.items() if named[n].requires_grad}
        model.load_state_dict(best_state, strict=False)
        if ctl is not None:
            summary.update(ctl.info())
        summary.update(steps_run=step, steps_planned=total, best_step=best_step, best_frac=best_step / total,
                       train_minutes=(time.time() - t_start) / 60, tok_s=rec.get("tok_s"), peak_mem_gb=rec.get("mem_gb"))
        mdir = a.model_dir or a.out
        os.makedirs(mdir, exist_ok=True)
        mpath = os.path.join(mdir, os.path.basename(os.path.normpath(a.out)) + ".pt")
        torch.save(dict(format="cs781-p5-lsi", args=vars(a), ckpt=a.ckpt, n_labels=K, state=best_state,
                        best_step=best_step, dev_macro_f1=best_f1), mpath)
        summary.update(model_path=mpath, model_mb=os.path.getsize(mpath) / 2**20)
        log.say(f"saved {mpath} ({summary['model_mb']:.1f} MB; step {best_step}, dev-subset macro-F1 {best_f1:.4f})")
    else:
        summary.update(steps_run=0, best_step=0, train_minutes=0.0, tok_s=None, peak_mem_gb=None)

    # ---- final dev report (whole dev set) with every decision rule
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    P, att, pos = predict(model, cache, "dev", dev_idx, a, device)
    summary["dev_predict_s"] = time.time() - t1
    summary["infer_ms_per_doc"] = 1000 * summary["dev_predict_s"] / max(1, len(dev_idx))
    summary["infer_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0
    dev_ids = [cache.ids("dev")[i] for i in dev_idx]
    np.savez_compressed(os.path.join(a.out, "dev_prob.npz"), ids=np.array(dev_ids), P=P.astype(np.float16), Y=Y_dev,
                        att=att, pos=pos, n_tok=cache.lengths("dev")[dev_idx])
    r = full_report(Y_dev, P)
    summary["dev"] = r
    thr = r["global_threshold"]
    pred_dev = E.decide(P, thr)
    json.dump([{"id": i, "pred": np.flatnonzero(row).tolist()} for i, row in zip(dev_ids, pred_dev)],
              open(os.path.join(a.out, "dev_pred.json"), "w"))
    log.say(f"DEV at 0.5:        {E.format_report(r['at_0.5'])}")
    log.say(f"DEV global {thr:.2f}:  {E.format_report(r['global'])}")
    log.say(f"DEV per-label:     {E.format_report(r['per_label'])}  (in-sample)")
    log.say(f"DEV cross-fitted (honest): global {r['crossfit']['global']:.4f}  per-label {r['crossfit']['per_label']:.4f} "
            f"| {summary['infer_ms_per_doc']:.1f} ms/doc")

    # ---- teacher outputs for distillation (optional)
    if a.save_train_probs:
        tp, _, _ = predict(model, cache, "train", tr_idx, a, device)
        np.savez_compressed(os.path.join(a.out, "train_prob.npz"), ids=np.array([cache.ids("train")[i] for i in tr_idx]),
                            P=tp.astype(np.float16))
        log.say(f"train_prob.npz written ({len(tr_idx)} documents)")

    # ---- test prediction (global threshold from dev; every document gets at least one statute)
    if len(te_idx):
        t1 = time.time()
        tp, tatt, tpos = predict(model, cache, "test", te_idx, a, device)
        test_ids = [cache.ids("test")[i] for i in te_idx]
        np.savez_compressed(os.path.join(a.out, "test_prob.npz"), ids=np.array(test_ids), P=tp.astype(np.float16),
                            att=tatt, pos=tpos, n_tok=cache.lengths("test")[te_idx])
        sets = [np.flatnonzero(row).tolist() for row in E.decide(tp, thr)]
        pj = os.path.join(a.out, "prediction.json")
        D.write_prediction(test_ids, sets, pj, K)
        with zipfile.ZipFile(os.path.join(a.out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pj, arcname="prediction.json")
        summary.update(test_predict_s=time.time() - t1, test_labels_per_doc=float(np.mean([len(s) for s in sets])))
        log.say(f"TEST: {len(sets)} documents, {summary['test_labels_per_doc']:.2f} statutes each (threshold {thr:.2f}) "
                f"-> submission.zip")
    else:
        log.say("TEST: no test.json - no submission written")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    log.say("DONE")


if __name__ == "__main__":
    main()
