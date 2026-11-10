"""
rr_train.py - Phase-3 Rhetorical Role prediction: adapt a Phase-1 backbone on CL_train + IT_train, select the
epoch on the official dev sets (CL_dev + IT_dev, official macro-F1 over all 13 labels), predict CL_test + IT_test
and write submission.zip. Dev and test documents are never trained on.

    python rr_train.py --ckpt .../run_d_.../final.pt --tok_root .../cs781-legal-data --data_dir /kaggle/working/p3data \
        --cache /kaggle/working/rr_cache.pkl --method lorma --out runs/D03_lorma --model_dir models

    reload a saved model and only predict:  --load models/D03_lorma.pt --predict_only   (same ckpt / method / head args)

Outputs in --out: train_log.txt, metrics.jsonl, summary.json (dev report per label / domain, confusions,
parameter counts, speed, memory), dev_prob.npz + dev_pred.json, test_prob.npz + prediction.json + submission.zip.
Model: <model_dir>/<run name>.pt = {args, labels, ckpt, state (trainable parameters of the selected epoch), dev score}
- with the Phase-1 checkpoint it rebuilds the full model (see load_model).
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
import rr_data as D
import rr_eval as E
from adapt import configure, controller, uses_prompt_slots, DEFAULT_LR, METHODS
from rr_model import Backbone, RRModel


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
    ap.add_argument("--data_dir", required=True, help="folder with {CL,IT}_{train,dev,test}.json + label_vocab.json")
    ap.add_argument("--cache", default=None, help="tokenized-data pickle (built if missing)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="lorma", choices=METHODS)
    ap.add_argument("--head", default="bilstm", choices=["bilstm", "mlp"])
    ap.add_argument("--mode", default="packed", choices=["packed", "single"], help="sentences per backbone window")
    ap.add_argument("--pool", default="mean", choices=["mean", "last"])
    ap.add_argument("--no_pos", action="store_true", help="drop the relative-position features")
    ap.add_argument("--no_dom", action="store_true", help="drop the domain embedding")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--seg_tokens", type=int, default=3072, help="token budget of one range of sentences (one step)")
    ap.add_argument("--max_tok", type=int, default=128, help="tokens kept per sentence (cache)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="LR of the trainable backbone/PEFT params (default per method)")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--peft_dropout", type=float, default=0.05)
    ap.add_argument("--bottleneck", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--n_prompt", type=int, default=20, help="prompt / prefix length")
    ap.add_argument("--target_rank", type=int, default=8, help="AdaLoRA: average rank kept at the end (start = --rank)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--rnn_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--class_weight", default="sqrt", choices=["none", "sqrt", "inv"])
    ap.add_argument("--weight_cap", type=float, default=10.0)
    ap.add_argument("--domains", default="CL,IT")
    ap.add_argument("--n_folds", type=int, default=0, help="cross-validation on the TRAIN documents: number of folds (0 = off)")
    ap.add_argument("--fold", type=int, default=0, help="with --n_folds: this fold is held out (predicted -> oof_prob.npz)")
    ap.add_argument("--tail_ranges", type=int, default=0,
                    help="extra training ranges per document and epoch that end at the document's last sentence")
    ap.add_argument("--max_docs", type=int, default=None, help="debug: documents per train/dev split")
    ap.add_argument("--max_sents", type=int, default=None, help="debug: sentences per train/dev document")
    ap.add_argument("--max_minutes", type=float, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--act_ckpt", action="store_true", help="recompute attention in backward (less memory)")
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--model_dir", default=None, help="where <run name>.pt is saved (default: --out)")
    ap.add_argument("--load", default=None, help="saved model .pt to start from")
    ap.add_argument("--predict_only", action="store_true", help="with --load: no training, only dev/test prediction")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------ helpers
def forward_range(model, doc, s, e, a, device):
    n_prompt = a.n_prompt if uses_prompt_slots(a.method) else 0
    ids, valid, win, st, en = D.batch_arrays(doc.sents[s:e], a.window, a.mode, n_prompt)
    t = lambda x: torch.from_numpy(x).to(device, non_blocking=True)
    rel = t(D.rel_positions(doc.n, s, e))
    dom = torch.tensor([doc.dom], device=device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        return model(t(ids), t(valid), t(win), t(st), t(en), rel, dom), int(valid.sum())


@torch.no_grad()
def predict_probs(model, docs, a, device):
    """Per document (n_sent, n_labels) probabilities; each sentence from its most central range."""
    model.eval()
    out = []
    for d in docs:
        probs = np.zeros((d.n, model.fc.out_features), dtype=np.float32)
        best = np.full(d.n, -1.0)
        for s, e in D.infer_segments(d.lens(), a.seg_tokens):
            logits, _ = forward_range(model, d, s, e, a, device)
            p = torch.softmax(logits.float(), -1).cpu().numpy()
            c = D.centrality(s, e)
            upd = c > best[s:e]
            probs[s:e][upd] = p[upd]
            best[s:e][upd] = c[upd]
        out.append(probs)
    return out


def labels_of(docs, probs, labels):
    return {d.id: [labels[i] for i in p.argmax(1)] for d, p in zip(docs, probs)}


def save_probs(path, docs, probs):
    np.savez_compressed(path, ids=np.array([d.id for d in docs]), doms=np.array([d.dom for d in docs]),
                        **{f"p{i}": p.astype(np.float16) for i, p in enumerate(probs)})


def write_submission(pred, docs, labels, out):
    D.validate_submission(pred, docs, labels)
    pj = os.path.join(out, "prediction.json")
    json.dump({d.id: pred[d.id] for d in docs}, open(pj, "w", encoding="utf-8"))
    zp = os.path.join(out, "submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    return zp


def dev_report(docs, probs, labels):
    pred = labels_of(docs, probs, labels)
    gold = {d.id: [labels[i] for i in d.y] for d in docs}
    r = E.evaluate(gold, pred, labels, {d.id: D.DOMAINS[d.dom] for d in docs})
    g = [x for k in sorted(gold) for x in gold[k]]
    p = [x for k in sorted(gold) for x in pred[k]]
    r["top_confusions"] = E.top_confusions(g, p)
    return r, pred


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
    cache_path = a.cache or os.path.join(a.out, "rr_cache.pkl")
    cache = D.load_cache(cache_path) if os.path.exists(cache_path) else D.build_cache(a.data_dir, a.tok_root, cache_path, a.max_tok)
    labels = cache["labels"]
    doms = [x for x in a.domains.split(",") if x]
    get = lambda sp, lim=True: [d for dom in doms for d in D.docs_from_cache(
        cache, dom, sp, a.max_docs if lim else None, a.max_sents if lim else None)]
    train_docs, dev_docs = get("train"), get("dev")
    oof_docs = []
    if a.n_folds:                                  # cross-validation inside the training documents only
        full = get("train", lim=False)
        rare = [labels.index(l) for l in ("Dissent", "PrecedentOverruled") if l in labels]
        fold = D.assign_folds(full, a.n_folds, rare)
        train_docs = [d for d in full if fold[d.id] != a.fold]
        oof_docs = [d for d in full if fold[d.id] == a.fold]
        if a.max_docs:
            train_docs = D.limit_docs(train_docs, 2 * a.max_docs, a.max_sents)
            oof_docs = D.limit_docs(oof_docs, 2 * a.max_docs, a.max_sents)
        log.say(f"fold {a.fold}/{a.n_folds}: train {len(train_docs)} docs, held out {len(oof_docs)} docs "
                f"({sum(int(np.isin(d.y, rare).any()) for d in oof_docs)} with Dissent/PrecedentOverruled)")
    test_docs = [] if a.no_test else get("test", lim=False)
    log.say(f"data: train {len(train_docs)} docs / {sum(d.n for d in train_docs):,} sentences / "
            f"{sum(int(d.lens().sum()) for d in train_docs):,} tokens; dev {len(dev_docs)} docs / "
            f"{sum(d.n for d in dev_docs):,} sent; test {len(test_docs)} docs / {sum(d.n for d in test_docs):,} sent "
            f"[{time.time() - t0:.0f}s]")

    # ---- model
    bb = Backbone(a.ckpt, ckpt_act=a.act_ckpt)
    n_tr, n_all, n_wr = configure(bb, a.method, a.rank, a.alpha, a.peft_dropout, a.bottleneck, a.top_k, a.n_prompt)
    model = RRModel(bb, len(labels), a.head, a.hidden, a.dropout, a.pool, not a.no_pos, not a.no_dom,
                    rnn_layers=a.rnn_layers).to(device)
    if a.load:
        ck = torch.load(a.load, map_location="cpu", weights_only=False)
        assert ck["labels"] == labels, "label vocabulary differs from the saved model"
        model.load_state_dict(ck["state"], strict=False)
        log.say(f"loaded {a.load} (epoch {ck.get('best_epoch')}, dev macro-F1 {ck.get('dev_macro_f1')})")
    n_head = sum(p.numel() for p in model.head_parameters())
    log.say(f"backbone {bb.cfg.get('kind')} d={bb.d}: {n_all / 1e6:.2f}M params; method={a.method} wrapped={n_wr} "
            f"trainable backbone {n_tr / 1e6:.3f}M + head {n_head / 1e6:.3f}M "
            f"({100 * (n_tr + n_head) / (n_all + n_head):.2f}% of {(n_all + n_head) / 1e6:.2f}M); "
            f"head={a.head} mode={a.mode} pool={a.pool} pos={not a.no_pos} dom={not a.no_dom}")
    summary = dict(args=vars(a), labels=labels, params_total=n_all + n_head, params_trainable=n_tr + n_head,
                   params_backbone_trainable=n_tr, params_head=n_head,
                   n_train_docs=len(train_docs), n_dev_docs=len(dev_docs))

    best_f1, best_ep = float("nan"), 0
    if not a.predict_only:
        # ---- class weights
        cnt = np.bincount(np.concatenate([d.y for d in train_docs]), minlength=len(labels)).astype(np.float64) + 1
        w = np.ones(len(labels)) if a.class_weight == "none" else \
            np.minimum((cnt.max() / cnt) ** (0.5 if a.class_weight == "sqrt" else 1.0), a.weight_cap)
        cw = torch.tensor(w, dtype=torch.float32, device=device)
        log.say("label weights: " + " ".join(f"{l}:{x:.1f}" for l, x in zip(labels, w)))

        # ---- optimizer (decay only on matrices)
        bb_p = [p for p in model.backbone.parameters() if p.requires_grad]
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
        rng = np.random.default_rng(a.seed)
        spe = sum(len(D.train_segments(d.lens(), a.seg_tokens)) for d in train_docs) + a.tail_ranges * len(train_docs)
        total = spe * a.epochs
        warm = max(1, int(a.warmup * total))
        ctl = controller(model.backbone, a.method, total, a.target_rank)     # AdaLoRA rank allocation (else None)
        log.say(f"lr backbone {lr_bb:g} head {a.lr_head:g}; ~{spe} steps/epoch x {a.epochs}; warmup {warm}; "
                f"seg_tokens {a.seg_tokens}; window {a.window}")

        best_state, best_f1, best_ep, bad, step, t_start = None, -1.0, 0, 0, 0, time.time()
        rec = {}
        params = [p for g in opt.param_groups for p in g["params"]]
        for ep in range(1, a.epochs + 1):
            model.train()
            items = [(di, s, e) for di, d in enumerate(train_docs) for s, e in D.train_segments(d.lens(), a.seg_tokens, rng)]
            items += [(di, *D.tail_segment(d.lens(), a.seg_tokens)) for di, d in enumerate(train_docs)] * a.tail_ranges
            order = rng.permutation(len(items))
            te, tl, tn, ntok, nsent = time.time(), 0.0, 0, 0, 0
            for k in order:
                di, s, e = items[k]
                d = train_docs[di]
                mult = (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * mult
                logits, nt = forward_range(model, d, s, e, a, device)
                y = torch.from_numpy(d.y[s:e]).to(device)
                loss = F.cross_entropy(logits.float(), y, weight=cw)
                if ctl is not None:
                    loss = loss + ctl.regularizer()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, a.clip)
                scaler.step(opt)
                scaler.update()
                if ctl is not None:
                    ctl.after_step(step)
                step += 1
                tl += loss.item()
                tn += 1
                ntok += nt
                nsent += e - s
            dt = time.time() - te
            rec = dict(epoch=ep, step=step, loss=tl / max(1, tn), tok_s=ntok / dt, sent_s=nsent / dt, epoch_s=dt,
                       mem_gb=torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0)
            msg = f"epoch {ep} loss {rec['loss']:.4f}  {rec['tok_s']:,.0f} tok/s  {dt:.0f}s  mem {rec['mem_gb']:.1f}G"
            if dev_docs:
                r, _ = dev_report(dev_docs, predict_probs(model, dev_docs, a, device), labels)
                rec.update(dev_macro_f1=r["macro_f1"], dev_CL=r.get("macro_f1_CL"), dev_IT=r.get("macro_f1_IT"), dev_acc=r["acc"])
                msg += (f"  | dev macro-F1 {r['macro_f1']:.4f} (CL {r.get('macro_f1_CL', 0):.4f} IT {r.get('macro_f1_IT', 0):.4f}) "
                        f"acc {r['acc']:.4f}")
                if r["macro_f1"] > best_f1:
                    best_f1, best_ep, bad = r["macro_f1"], ep, 0
                    best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
                    msg += "  *best*"
                else:
                    bad += 1
            log.say(msg)
            log.metric(rec)
            if dev_docs and bad >= a.patience:
                log.say(f"early stop (no improvement for {a.patience} epochs)")
                break
            if (time.time() - t_start) / 60 > a.max_minutes:
                log.say("time limit reached")
                break
        if best_state is None:                                     # no dev documents: keep the last epoch
            best_ep = ep
            best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        model.load_state_dict(best_state, strict=False)
        if ctl is not None:
            summary.update(ctl.info())
            log.say(f"AdaLoRA final ranks: mean {summary['adalora_rank_mean']:.2f} "
                    f"min {summary['adalora_rank_min']} max {summary['adalora_rank_max']}")
        summary.update(epochs_run=ep, best_epoch=best_ep, train_minutes=(time.time() - t_start) / 60,
                       tok_s=rec.get("tok_s"), sent_s=rec.get("sent_s"), peak_mem_gb=rec.get("mem_gb"))
        mdir = a.model_dir or a.out
        os.makedirs(mdir, exist_ok=True)
        mpath = os.path.join(mdir, os.path.basename(os.path.normpath(a.out)) + ".pt")
        torch.save(dict(format="cs781-p3-rr", args=vars(a), labels=labels, ckpt=a.ckpt, state=best_state,
                        best_epoch=best_ep, dev_macro_f1=best_f1), mpath)
        summary.update(model_path=mpath, model_mb=os.path.getsize(mpath) / 2**20)
        log.say(f"saved {mpath} ({summary['model_mb']:.1f} MB; epoch {best_ep}, dev macro-F1 {best_f1:.4f})")
    else:
        summary.update(epochs_run=0, best_epoch=0, train_minutes=0.0, tok_s=None, sent_s=None, peak_mem_gb=None)

    # ---- final dev report
    if dev_docs:
        t1 = time.time()
        probs = predict_probs(model, dev_docs, a, device)
        summary["dev_predict_s"] = time.time() - t1
        save_probs(os.path.join(a.out, "dev_prob.npz"), dev_docs, probs)
        r, pred = dev_report(dev_docs, probs, labels)
        summary["dev"] = r
        json.dump(pred, open(os.path.join(a.out, "dev_pred.json"), "w"))
        log.say("DEV\n" + E.format_report(r) + "\nconfusions " + " ".join(r["top_confusions"]))

    # ---- held-out training fold (cross-validation): out-of-fold predictions
    if oof_docs:
        probs = predict_probs(model, oof_docs, a, device)
        save_probs(os.path.join(a.out, "oof_prob.npz"), oof_docs, probs)
        r, pred = dev_report(oof_docs, probs, labels)
        summary.update(oof=r, oof_ids=[d.id for d in oof_docs], fold=a.fold, n_folds=a.n_folds)
        json.dump(pred, open(os.path.join(a.out, "oof_pred.json"), "w"))
        log.say("OOF (held-out train fold)\n" + E.format_report(r))

    # ---- test prediction
    if test_docs:
        t1 = time.time()
        probs = predict_probs(model, test_docs, a, device)
        save_probs(os.path.join(a.out, "test_prob.npz"), test_docs, probs)
        pred = labels_of(test_docs, probs, labels)
        zp = write_submission(pred, test_docs, labels, a.out)
        cnt = {l: sum(v.count(l) for v in pred.values()) for l in labels}
        summary.update(test_predict_s=time.time() - t1, test_label_counts=cnt)
        log.say(f"TEST: {sum(cnt.values())} sentences over {len(test_docs)} docs -> {zp}  {cnt}")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    log.say("DONE")


if __name__ == "__main__":
    main()
