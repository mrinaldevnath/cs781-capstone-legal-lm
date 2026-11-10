"""
ner_train.py - Phase-2 L-NER: fine-tune / adapt a Phase-1 backbone, select on a document-level dev split with
the official metric (strict span macro-F1), predict the test fold and write submission.zip.

    python ner_train.py --ckpt /kaggle/input/.../run_b_gpt_mha/final.pt --tok_root /kaggle/input/.../cs781-legal-data \
        --data_dir /kaggle/working/p2data --test_file /kaggle/working/p2data/fold3-text.json \
        --method lorma --head bilstm --out /kaggle/working/ner_b_lorma

Methods (peft.py): full | partial | norm | lora | lorma | adapter.   Heads (ner_model.py): linear | bilstm.
Pretokenization (ner_data.py): piece (default, 99.5% gold spans token-aligned) | native.
Decoding: viterbi (BIO-constrained, default) | greedy.

Outputs in --out: train_log.txt, metrics.jsonl (per epoch), summary.json (final dev report, error analysis,
parameter counts, speed, memory), dev_pred.json, prediction.json + submission.zip (if --test_file),
trainable.pt (best weights of the trainable parameters), test_logp.npz / dev_logp.npz (per-token log-probs,
for ensembling).
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
import ner_data as N
import ner_eval as E
from ner_model import Backbone, NERModel
from peft import configure_trainable

DEFAULT_LR = {"full": 5e-5, "partial": 1e-4, "norm": 1e-3, "lora": 5e-4, "lorma": 5e-4, "adapter": 1e-3}


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
    ap.add_argument("--tok_root", required=True, help="Phase-1 data folder (legal_bpe*.py + tokenizer/)")
    ap.add_argument("--data_dir", required=True, help="folder with fold{1,2}-{text,gold}.json and labels.txt")
    ap.add_argument("--test_file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="full", choices=list(DEFAULT_LR))
    ap.add_argument("--head", default="bilstm", choices=["linear", "bilstm"])
    ap.add_argument("--pretok", default="piece", choices=["piece", "native"])
    ap.add_argument("--decode", default="viterbi", choices=["viterbi", "greedy"])
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--prefix", type=int, default=0, help="prepend the first N document tokens (header) to every window")
    ap.add_argument("--eval_overlap", type=int, default=128, help="overlap of inference windows (each side)")
    ap.add_argument("--centrality", default="auto", choices=["auto", "center", "left"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None, help="LR of the trainable backbone/PEFT params (default per method)")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--peft_dropout", type=float, default=0.05)
    ap.add_argument("--peft_targets", default="all", choices=["all", "attn"])
    ap.add_argument("--bottleneck", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--lstm_hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--class_weight", default="sqrt", choices=["none", "sqrt", "inv"])
    ap.add_argument("--dev_frac", type=float, default=0.15)
    ap.add_argument("--all_data", action="store_true", help="train on every training document (no dev); fixed epochs")
    ap.add_argument("--max_docs", type=int, default=None, help="debug: use only this many training documents")
    ap.add_argument("--max_minutes", type=float, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--act_ckpt", action="store_true", help="recompute attention in backward (less memory, ~25% slower)")
    ap.add_argument("--load", default=None, help="trainable.pt of a finished run (with --predict_only)")
    ap.add_argument("--predict_only", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------ prediction
@torch.no_grad()
def predict_logp(model, docs, W, overlap, cent_mode, n_tags, device, bs=16, P=0):
    """Per-document (n_tokens, n_tags) log-probs. Each token takes the window where it is most central
    (cent_mode=center) or has the most left context (left)."""
    model.eval()
    use_cuda = device.type == "cuda"
    overlap = min(overlap, W // 4)
    stride = W - 2 * overlap
    cent = N.centrality(W, cent_mode)
    out = []
    for d in docs:
        n = len(d)
        lp = np.zeros((n, n_tags), dtype=np.float32)
        if n == 0:                                   # empty document: nothing to tag (no window, no spans)
            out.append(lp)
            continue
        best = np.full(n, -1.0)
        starts = N.window_starts(n, W, stride)
        off = N.prefix_len(d, P)
        for i in range(0, len(starts), bs):
            chunk = starts[i:i + bs]
            arr = [N.window_arrays(d, s, W, P) for s in chunk]
            ids = torch.from_numpy(np.stack([a[0] for a in arr])).to(device)
            valid = torch.from_numpy(np.stack([a[2] for a in arr])).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                logits = model(ids, valid)
            l = torch.log_softmax(logits.float(), -1).cpu().numpy()
            for k, s in enumerate(chunk):
                e = min(n, s + W)
                c = cent[:e - s]
                upd = c > best[s:e]
                lp[s:e][upd] = l[k, off:off + e - s][upd]
                best[s:e][upd] = c[upd]
        out.append(lp)
    return out


def spans_from_logp(docs, logps, labs, how):
    A, st = N.allowed_transitions(len(labs) * 2 + 1)
    pred = {}
    for d, lp in zip(docs, logps):
        if len(lp) == 0:
            pred[d.id] = []
            continue
        tags = N.viterbi(lp, A, st) if how == "viterbi" else N.greedy(lp)
        pred[d.id] = N.decode_spans(d, tags, labs)
    return pred


def write_submission(pred, test_ids, labs, out):
    pred = {str(k): [[int(s), int(e), l] for s, e, l in pred.get(str(k), [])] for k in test_ids}
    E.validate_submission(pred, test_ids, labs)
    pj = os.path.join(out, "prediction.json")
    json.dump(pred, open(pj, "w", encoding="utf-8"))
    zp = os.path.join(out, "submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    return zp


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    a = get_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    log = Log(a.out)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    use_cuda = device.type == "cuda"
    log.say(f"args {json.dumps(vars(a))}")

    labs, tags = N.load_labels(os.path.join(a.data_dir, "labels.txt"))
    tag_of = N.tag_ids(labs)
    n_tags = len(tags)
    otok = N.OffsetTokenizer(N.load_tokenizer(a.tok_root), a.pretok)

    # ---- data
    t0 = time.time()
    f_ids = []
    texts, gold = {}, {}
    for f in (1, 2):
        T, G = N.load_folds(a.data_dir, (f,))
        texts.update(T)
        gold.update(G)
        f_ids.append(sorted(T))
    if a.max_docs:
        keep = set(sorted(texts)[:a.max_docs])
        f_ids = [[i for i in ids if i in keep] for ids in f_ids]
        texts = {k: v for k, v in texts.items() if k in keep}
    dev_ids = set() if a.all_data else N.dev_split(f_ids, a.dev_frac, a.seed)
    docs = {k: N.Doc(k, texts[k], otok, gold[k], tag_of) for k in sorted(texts)}
    train_docs = [docs[k] for k in sorted(docs) if k not in dev_ids]
    dev_docs = [docs[k] for k in sorted(docs) if k in dev_ids]
    test_docs = []
    if a.test_file:
        tt = N.load_texts(a.test_file)
        test_docs = [N.Doc(k, tt[k], otok) for k in tt]
    ntr = sum(len(d) for d in train_docs)
    log.say(f"docs: train {len(train_docs)} ({ntr:,} tokens)  dev {len(dev_docs)}  test {len(test_docs)}  "
            f"[{a.pretok} pretok, {time.time() - t0:.0f}s]")
    oracle_docs = dev_docs or train_docs
    orc = E.score({d.id: d.gold for d in oracle_docs},
                  {d.id: N.decode_spans(d, d.tags, labs) for d in oracle_docs}, labs)
    log.say(f"alignment ceiling (gold -> BIO -> spans) on {'dev' if dev_docs else 'train'}: macro-F1 {orc['macro_f1']:.4f}")

    # ---- model
    bb = Backbone(a.ckpt, ckpt_act=a.act_ckpt)
    n_tr, n_all, n_wr = configure_trainable(bb, a.method, a.rank, a.alpha, a.peft_dropout, a.bottleneck, a.top_k, a.peft_targets)
    model = NERModel(bb, n_tags, a.head, a.lstm_hidden, a.dropout).to(device)
    n_head = sum(p.numel() for p in model.head.parameters())
    log.say(f"backbone {bb.cfg.get('kind')} d={bb.d}: {n_all / 1e6:.2f}M params ({'encoder only' if bb.is_encdec else 'trunk'}); "
            f"method={a.method} wrapped={n_wr} trainable backbone {n_tr / 1e6:.3f}M + head {n_head / 1e6:.3f}M "
            f"({100 * (n_tr + n_head) / (n_all + n_head):.2f}% of {(n_all + n_head) / 1e6:.2f}M)")
    cent_mode = a.centrality if a.centrality != "auto" else ("left" if (a.head == "linear" and not bb.is_encdec) else "center")

    if a.load:
        sd = torch.load(a.load, map_location="cpu", weights_only=False)
        missing = [k for k in sd if k not in model.state_dict()]
        assert not missing, f"unknown keys in {a.load}: {missing[:5]}"
        model.load_state_dict(sd, strict=False)
        log.say(f"loaded trained weights from {a.load}")

    summary = dict(args=vars(a), params_total=n_all + n_head, params_trainable=n_tr + n_head,
                   alignment_ceiling=orc["macro_f1"], n_train_docs=len(train_docs), n_dev_docs=len(dev_docs))
    best_state, best_f1, best_ep = None, -1.0, 0

    if not a.predict_only:
        # ---- class weights over tags (train docs)
        cnt = np.bincount(np.concatenate([d.tags for d in train_docs]), minlength=n_tags).astype(np.float64) + 1
        if a.class_weight == "none":
            w = np.ones(n_tags)
        else:
            w = (cnt.max() / cnt) ** (0.5 if a.class_weight == "sqrt" else 1.0)
            w = w / w[0]                                    # O has weight 1
            w = np.minimum(w, 20.0)
        cw = torch.tensor(w, dtype=torch.float32, device=device)
        log.say("tag weights: " + " ".join(f"{t}:{x:.1f}" for t, x in zip(tags, w)))

        # ---- optimizer
        head_p = [p for p in model.head.parameters()]
        bb_p = [p for p in model.backbone.parameters() if p.requires_grad]
        lr_bb = a.lr if a.lr is not None else DEFAULT_LR[a.method]
        groups = []
        for ps, lr in ((bb_p, lr_bb), (head_p, a.lr_head)):
            dec = [p for p in ps if p.ndim >= 2]
            nod = [p for p in ps if p.ndim < 2]
            if dec:
                groups.append(dict(params=dec, lr=lr, weight_decay=a.wd, base_lr=lr))
            if nod:
                groups.append(dict(params=nod, lr=lr, weight_decay=0.0, base_lr=lr))
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
        rng = np.random.default_rng(a.seed)
        spe = math.ceil(len(N.make_windows(train_docs, a.window, a.window)) / a.bs)
        total = spe * a.epochs
        warm = max(1, int(a.warmup * total))
        log.say(f"lr backbone {lr_bb:g}  head {a.lr_head:g}; ~{spe} steps/epoch x {a.epochs} epochs; warmup {warm}; "
                f"window {a.window}; centrality {cent_mode}")

        step, bad, t_start = 0, 0, time.time()
        for ep in range(1, a.epochs + 1):
            model.train()
            wins = N.make_windows(train_docs, a.window, a.window, rng)    # non-overlapping, random offset per epoch
            order = rng.permutation(len(wins))
            te, tl, tn, ntok = time.time(), 0.0, 0, 0
            for b0 in range(0, len(order), a.bs):
                bw = [wins[i] for i in order[b0:b0 + a.bs]]
                arr = [N.window_arrays(train_docs[di], s, a.window, a.prefix) for di, s in bw]
                ids = torch.from_numpy(np.stack([x[0] for x in arr])).to(device, non_blocking=True)
                tg = torch.from_numpy(np.stack([x[1] for x in arr])).to(device, non_blocking=True)
                valid = torch.from_numpy(np.stack([x[2] for x in arr])).to(device, non_blocking=True)
                lr_mult = (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * lr_mult
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                    logits = model(ids, valid)
                loss = F.cross_entropy(logits.reshape(-1, n_tags), tg.reshape(-1), weight=cw, ignore_index=N.IGNORE)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], a.clip)
                scaler.step(opt)
                scaler.update()
                step += 1
                tl += loss.item()
                tn += 1
                ntok += int(valid.sum())
            dt = time.time() - te
            rec = dict(epoch=ep, step=step, loss=tl / max(1, tn), tok_s=ntok / dt, epoch_s=dt,
                       mem_gb=torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0)
            msg = f"epoch {ep} loss {rec['loss']:.4f}  {rec['tok_s']:,.0f} tok/s  {dt:.0f}s  mem {rec['mem_gb']:.1f}G"
            if dev_docs:
                lps = predict_logp(model, dev_docs, a.window, a.eval_overlap, cent_mode, n_tags, device, P=a.prefix)
                pred = spans_from_logp(dev_docs, lps, labs, a.decode)
                r = E.score({d.id: d.gold for d in dev_docs}, pred, labs)
                rec.update(dev_macro_f1=r["macro_f1"], dev_micro_f1=r["micro_f1"])
                msg += f"  | dev macro-F1 {r['macro_f1']:.4f} (P {r['macro_p']:.3f} R {r['macro_r']:.3f}) micro {r['micro_f1']:.4f}"
                if r["macro_f1"] > best_f1:
                    best_f1, best_ep, bad = r["macro_f1"], ep, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
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
        summary.update(epochs_run=ep, best_epoch=best_ep if dev_docs else ep, train_minutes=(time.time() - t_start) / 60,
                       tok_s=rec["tok_s"], peak_mem_gb=rec["mem_gb"])
        if best_state is None:          # --all_data: keep the last epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
        model.load_state_dict(best_state, strict=False)
        torch.save(best_state, os.path.join(a.out, "trainable.pt"))
        log.say(f"saved trainable.pt (best epoch {summary['best_epoch']}, dev macro-F1 {best_f1:.4f})")

    # ---- final dev report
    if dev_docs:
        t1 = time.time()
        lps = predict_logp(model, dev_docs, a.window, a.eval_overlap, cent_mode, n_tags, device, P=a.prefix)
        np.savez_compressed(os.path.join(a.out, "dev_logp.npz"), **{d.id: x.astype(np.float16) for d, x in zip(dev_docs, lps)})
        g = {d.id: d.gold for d in dev_docs}
        for how in ("viterbi", "greedy"):
            pred = spans_from_logp(dev_docs, lps, labs, how)
            r = E.score(g, pred, labs)
            summary[f"dev_{how}"] = r
            if how == a.decode:
                json.dump(pred, open(os.path.join(a.out, "dev_pred.json"), "w"))
                summary["dev_errors"] = E.error_analysis(g, pred)
                log.say("DEV (" + how + ")\n" + E.format_report(r, labs))
                log.say("errors " + json.dumps(summary["dev_errors"]))
            else:
                log.say(f"DEV with {how} decoding: macro-F1 {r['macro_f1']:.4f}")
        summary["dev_predict_s"] = time.time() - t1

    # ---- test prediction
    if test_docs:
        t1 = time.time()
        lps = predict_logp(model, test_docs, a.window, a.eval_overlap, cent_mode, n_tags, device, P=a.prefix)
        np.savez_compressed(os.path.join(a.out, "test_logp.npz"), **{d.id: x.astype(np.float16) for d, x in zip(test_docs, lps)})
        pred = spans_from_logp(test_docs, lps, labs, a.decode)
        zp = write_submission(pred, [d.id for d in test_docs], labs, a.out)
        cnt = {l: sum(1 for v in pred.values() for e in v if e[2] == l) for l in labs}
        summary.update(test_predict_s=time.time() - t1, test_label_counts=cnt)
        log.say(f"TEST: {sum(cnt.values())} entities over {len(test_docs)} docs -> {zp}  {cnt}")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    log.say("DONE")


if __name__ == "__main__":
    main()
