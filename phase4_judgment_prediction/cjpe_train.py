"""
cjpe_train.py - Phase-4 Court Judgment Prediction: adapt a Phase-1 backbone on train.json, select the checkpoint on
dev.json (macro-F1), predict test.json and write submission.zip. Dev and test documents are never trained on
(training documents that also occur in dev/test are removed when the cache is built).

    python cjpe_train.py --ckpt .../run_d_.../final.pt --tok_root .../cs781-legal-data --data_dir /kaggle/working/p4data \
        --cache /kaggle/working/p4_cache.npz --method lorma --lr 2e-4 --out runs/D01_lorma --model_dir models

    reload a saved model and only predict:  --load models/D01_lorma.pt --predict_only   (same ckpt / method / head args)

Training: --epochs passes over the training documents (default 1), batches of --batch_docs documents (split into
micro-batches of at most --micro_chunks chunks, gradients accumulated), linear warmup + linear decay, AdamW, fp16
autocast. Dev is evaluated every --eval_every of the planned steps and at the end (or when --max_minutes is reached);
the best dev macro-F1 (decision 0.5) is kept.

Knowledge distillation (optional, Phase 4 allows skipping it for small models; the code is ready):
    teacher:  any run with --save_train_probs writes train_prob.npz (P(label=1) for every training document)
    student:  --teacher_probs <that file> --kd_alpha 0.5 --kd_T 2  ->  loss = (1-a) CE(y) + a T^2 KL(teacher_T || student_T)

Outputs in --out: train_log.txt, metrics.jsonl, summary.json, dev_prob.npz (+ chunk attention), dev_pred.json,
test_prob.npz, prediction.json, submission.zip. Model: <model_dir>/<run name>.pt (trainable parameters of the best
checkpoint + args), which with the Phase-1 checkpoint rebuilds the full model.
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
import cjpe_data as D
import cjpe_eval as E
from adapt import configure, controller, gradual_params, DEFAULT_LR, METHODS
from cjpe_model import Backbone, CJPEModel


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
    # document representation
    ap.add_argument("--chunk", type=int, default=512, help="tokens per chunk (one backbone window)")
    ap.add_argument("--n_chunks", type=int, default=2, help="chunks read per document")
    ap.add_argument("--select", default="last", choices=D.SELECTS, help="which chunks of the document are read")
    ap.add_argument("--pool", default="attn", choices=["attn", "mean", "max"], help="pooling of the chunk vectors")
    ap.add_argument("--no_pos", action="store_true", help="drop the chunk-position features")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--class_weight", default="balanced", choices=["none", "balanced"])
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
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--peft_dropout", type=float, default=0.05)
    ap.add_argument("--bottleneck", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=4)
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
    ids, valid, doc, slot, rel = D.batch_arrays(pieces, a.chunk)
    t = lambda x: torch.from_numpy(x).to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        logits, z, w = model(t(ids), t(valid), t(doc), t(slot), t(rel), len(pieces))
    return logits, z, w, int(valid.sum())


@torch.no_grad()
def predict(model, cache, sp, idx, a, device):
    """-> P(label = 1) (n,), chunk attention weights (n, K) (NaN where a document has fewer chunks), relative chunk
    positions (n, K)."""
    model.eval()
    pieces = pieces_of(cache, sp, idx, a)
    K = max(1, max(len(p[0]) for p in pieces)) if pieces else 1
    p1 = np.zeros(len(idx), dtype=np.float32)
    att = np.full((len(idx), K), np.nan, dtype=np.float32)
    pos = np.full((len(idx), K), np.nan, dtype=np.float32)
    for mb in D.micro_batches(pieces, 2 * a.micro_chunks):
        logits, _, w, _ = forward_docs(model, [pieces[j] for j in mb], a, device)
        p1[mb] = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
        for r, j in enumerate(mb):
            n = len(pieces[j][0])
            pos[j, :n] = pieces[j][1]
            if w is not None:
                att[j, :n] = w[r, :n].float().cpu().numpy()
    return p1, att, pos


def report(y, p1):
    r = E.binary_report(y, (p1 >= 0.5).astype(int))
    r["auc"] = E.auc(y, p1)
    r["best_threshold"], r["macro_f1_best_threshold"] = E.best_threshold(y, p1)
    return r


def kd_loss(logits, t1, T):
    """KL(teacher || student) at temperature T for binary outputs; t1 = teacher P(label = 1)."""
    t1 = t1.clamp(1e-4, 1 - 1e-4)
    tl = torch.log(t1) - torch.log1p(-t1)                              # teacher logit of class 1 vs 0
    tq = torch.stack([torch.sigmoid(-tl / T), torch.sigmoid(tl / T)], -1)
    ls = F.log_softmax(logits.float() / T, -1)
    return (T * T) * (tq * (torch.log(tq) - ls)).sum(-1)


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
    cache_path = a.cache or os.path.join(a.out, "p4_cache.npz")
    if not os.path.exists(cache_path):
        D.build_cache(a.data_dir, a.tok_root, cache_path, a.procs, log.say)
    cache = D.Cache(cache_path)
    rng = np.random.default_rng(a.seed)
    tr_idx = np.arange(cache.n("train"))
    if a.max_train_docs and a.max_train_docs < len(tr_idx):
        tr_idx = np.sort(rng.choice(tr_idx, a.max_train_docs, replace=False))
    dev_idx = np.arange(cache.n("dev"))
    te_idx = np.arange(cache.n("test")) if cache.has("test") and not a.no_test else np.arange(0)
    if a.max_docs:
        tr_idx, dev_idx, te_idx = tr_idx[:a.max_docs], dev_idx[:a.max_docs], te_idx[:a.max_docs]
    y_tr, y_dev = cache.y("train")[tr_idx], cache.y("dev")[dev_idx]
    log.say(f"data: train {len(tr_idx)} docs ({int((y_tr == 1).sum())} label 1), dev {len(dev_idx)} "
            f"({int((y_dev == 1).sum())} label 1), test {len(te_idx)}; reading {a.n_chunks} x {a.chunk} tokens "
            f"({a.select}) per document [{time.time() - t0:.0f}s]")

    # ---- model
    bb = Backbone(a.ckpt, ckpt_act=a.act_ckpt)
    n_tr, n_all, n_wr = configure(bb, a.method, a.rank, a.alpha, a.peft_dropout, a.bottleneck, a.top_k)
    model = CJPEModel(bb, 2, a.pool, a.hidden, a.dropout, not a.no_pos).to(device)
    if a.load:
        ck = torch.load(a.load, map_location="cpu", weights_only=False)
        if a.method == "gradual":
            for p in gradual_params(model.backbone):
                p.requires_grad = True
        model.load_state_dict(ck["state"], strict=False)
        log.say(f"loaded {a.load} (step {ck.get('best_step')}, dev macro-F1 {ck.get('dev_macro_f1')})")
    n_head = sum(p.numel() for p in model.head_parameters())
    log.say(f"backbone {bb.cfg.get('kind')} d={bb.d}: {n_all / 1e6:.2f}M params; method={a.method} wrapped={n_wr} "
            f"trainable backbone {n_tr / 1e6:.3f}M + head {n_head / 1e6:.3f}M "
            f"({100 * (n_tr + n_head) / (n_all + n_head):.2f}% of {(n_all + n_head) / 1e6:.2f}M); pool={a.pool}")
    summary = dict(args=vars(a), params_total=n_all + n_head, params_trainable=n_tr + n_head,
                   params_backbone_trainable=n_tr, params_head=n_head, n_train_docs=len(tr_idx),
                   n_dev_docs=len(dev_idx), n_test_docs=len(te_idx), cache_info=cache.info)

    best_f1, best_step = float("nan"), 0
    if not a.predict_only:
        # ---- class weights
        cnt = np.bincount(y_tr, minlength=2).astype(np.float64)
        w = np.ones(2) if a.class_weight == "none" else cnt.sum() / (2 * np.maximum(cnt, 1))
        cw = torch.tensor(w, dtype=torch.float32, device=device)
        log.say(f"class counts {cnt.astype(int).tolist()} -> weights {np.round(w, 3).tolist()}")
        teacher = None
        if a.teacher_probs:
            z = np.load(a.teacher_probs)
            tmap = {str(i): float(p) for i, p in zip(z["ids"], z["p1"])}
            ids_all = cache.ids("train")
            teacher = np.array([tmap.get(ids_all[i], np.nan) for i in tr_idx], dtype=np.float32)
            log.say(f"distillation from {a.teacher_probs}: {int(np.isfinite(teacher).sum())}/{len(teacher)} documents "
                    f"covered, alpha {a.kd_alpha}, T {a.kd_T}")

        # ---- optimizer (decay only on matrices); gradual unfreezing: every block is in the optimizer from the start
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
        n_ep = math.ceil(a.epochs)
        order = np.concatenate([rng.permutation(len(tr_idx)) for _ in range(n_ep)])[:total * a.batch_docs]
        log.say(f"lr backbone {lr_bb:g} head {a.lr_head:g}; {total} steps x {a.batch_docs} docs ({a.epochs:g} epochs); "
                f"warmup {warm}; dev every {every} steps")

        best_state, best_f1, best_step, step, t_start = None, -1.0, 0, 0, time.time()
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
                y = torch.from_numpy(y_tr[pos_b[mb]]).to(device)
                ce = F.cross_entropy(logits.float(), y, reduction="none") * cw[y]
                if teacher is not None:
                    tt = torch.from_numpy(teacher[pos_b[mb]]).to(device)
                    ok = torch.isfinite(tt)
                    kd = torch.where(ok, kd_loss(logits, torch.nan_to_num(tt, 0.5), a.kd_T), torch.zeros_like(ce))
                    ce = torch.where(ok, (1 - a.kd_alpha) * ce + a.kd_alpha * kd, ce)
                loss = ce.sum() / len(pos_b)
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
                p1, _, _ = predict(model, cache, "dev", dev_idx, a, device)
                r = report(y_dev, p1)
                rec.update(dev_macro_f1=r["macro_f1"], dev_auc=r["auc"], dev_best_thr=r["best_threshold"],
                           dev_f1_best_thr=r["macro_f1_best_threshold"], dev_s=time.time() - t_ev)
                msg = (f"step {step}/{total} loss {rec['loss']:.4f} {rec['tok_s']:,.0f} tok/s mem {rec['mem_gb']:.1f}G "
                       f"| dev macro-F1 {r['macro_f1']:.4f} AUC {r['auc']:.4f} (thr {r['best_threshold']:.2f}: "
                       f"{r['macro_f1_best_threshold']:.4f}) pred-1 {100 * r['pred_1_share']:.0f}%")
                if r["macro_f1"] > best_f1:
                    best_f1, best_step = r["macro_f1"], step
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
        if ctl is not None and a.method == "gradual":                  # the saved state must load into all blocks
            for p in gradual_params(model.backbone):
                p.requires_grad = True
        if a.method == "gradual" and best_state is not None:
            best_state = {n: v for n, v in best_state.items()
                          if dict(model.named_parameters())[n].requires_grad}
        model.load_state_dict(best_state, strict=False)
        if ctl is not None:
            summary.update(ctl.info())
        summary.update(steps_run=step, steps_planned=total, best_step=best_step, best_frac=best_step / total,
                       train_minutes=(time.time() - t_start) / 60, tok_s=rec.get("tok_s"), peak_mem_gb=rec.get("mem_gb"))
        mdir = a.model_dir or a.out
        os.makedirs(mdir, exist_ok=True)
        mpath = os.path.join(mdir, os.path.basename(os.path.normpath(a.out)) + ".pt")
        torch.save(dict(format="cs781-p4-cjpe", args=vars(a), ckpt=a.ckpt, state=best_state, best_step=best_step,
                        dev_macro_f1=best_f1), mpath)
        summary.update(model_path=mpath, model_mb=os.path.getsize(mpath) / 2**20)
        log.say(f"saved {mpath} ({summary['model_mb']:.1f} MB; step {best_step}, dev macro-F1 {best_f1:.4f})")
    else:
        summary.update(steps_run=0, best_step=0, train_minutes=0.0, tok_s=None, peak_mem_gb=None)

    # ---- final dev report
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    p1, att, pos = predict(model, cache, "dev", dev_idx, a, device)
    summary["dev_predict_s"] = time.time() - t1
    summary["infer_ms_per_doc"] = 1000 * summary["dev_predict_s"] / max(1, len(dev_idx))
    summary["infer_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0
    dev_ids = [cache.ids("dev")[i] for i in dev_idx]
    np.savez_compressed(os.path.join(a.out, "dev_prob.npz"), ids=np.array(dev_ids), p1=p1, y=y_dev, att=att, pos=pos,
                        n_tok=cache.lengths("dev")[dev_idx])
    r = report(y_dev, p1)
    summary["dev"] = r
    json.dump([{"id": i, "pred": int(p >= 0.5)} for i, p in zip(dev_ids, p1)], open(os.path.join(a.out, "dev_pred.json"), "w"))
    log.say("DEV " + E.format_report(r) + f" | AUC {r['auc']:.4f} | best threshold {r['best_threshold']:.2f} -> "
            f"{r['macro_f1_best_threshold']:.4f} | {summary['infer_ms_per_doc']:.1f} ms/doc")

    # ---- teacher outputs for distillation
    if a.save_train_probs:
        tp, _, _ = predict(model, cache, "train", tr_idx, a, device)
        np.savez_compressed(os.path.join(a.out, "train_prob.npz"), ids=np.array([cache.ids("train")[i] for i in tr_idx]),
                            p1=tp)
        log.say(f"train_prob.npz written ({len(tr_idx)} documents)")

    # ---- test prediction
    if len(te_idx):
        t1 = time.time()
        tp, tatt, tpos = predict(model, cache, "test", te_idx, a, device)
        test_ids = [cache.ids("test")[i] for i in te_idx]
        np.savez_compressed(os.path.join(a.out, "test_prob.npz"), ids=np.array(test_ids), p1=tp, att=tatt, pos=tpos,
                            n_tok=cache.lengths("test")[te_idx])
        pj = os.path.join(a.out, "prediction.json")
        rows = D.write_prediction(test_ids, (tp >= 0.5).astype(int), pj)
        with zipfile.ZipFile(os.path.join(a.out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pj, arcname="prediction.json")
        summary.update(test_predict_s=time.time() - t1, test_pred_1=int(sum(x["pred"] for x in rows)))
        log.say(f"TEST: {len(rows)} cases, {summary['test_pred_1']} predicted 1 -> submission.zip")
    else:
        log.say("TEST: no test.json - no submission written")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    log.say("DONE")


if __name__ == "__main__":
    main()
