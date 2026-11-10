"""
generate.py - cached, batched generation of 128-word continuations + Codabench submission zip.

    # official test set -> submission.zip (prediction.json at the zip root, {"<id>": {"text": "..."}})
    python generate.py --ckpt /kaggle/working/run_b/final.pt --data_root /kaggle/input/cs781-data-v2 \
        --input /kaggle/input/<test>/test.json --out_dir /kaggle/working/sub_b
    # quick check on validation-derived dev prompts (has references -> prints ROUGE-L / BLEU-4 proxy)
    python generate.py --ckpt ... --data_root ... --input /kaggle/input/cs781-data-v2/dev_prompts.jsonl --n 64 --out_dir /kaggle/working/dev_b

Decoding: KV-cache incremental decoding (prefill once, then one token per step). Left-padded batches with per-sequence
RoPE offsets and pad masks, prompts sorted by length. EOS is suppressed (the true continuation never ends the
document inside 128 words); output = first 128 whitespace-separated words, like the starter notebook.
The test file is read ONLY here, only to produce predictions.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import json
import time
import zipfile
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model, PAD_ID, BOS_ID, EOS_ID
from p1_loader import load_tokenizer

MASK_ID = 3


def load_model(path, device):
    ck = torch.load(path, map_location="cpu")
    cfg = ck["cfg"]
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.to_inference().to(device).eval()
    return model, cfg


def read_inputs(path):
    """-> list of (id, prompt, reference_or_None)"""
    if path.endswith(".jsonl"):
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        return [(f"{r.get('src', 'dev')}_{r.get('i', k)}", r["prompt"], r.get("reference")) for k, r in enumerate(rows)]
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        return [(str(k), (v["text"] if isinstance(v, dict) else v), None) for k, v in d.items()]
    return [(str(r.get("id", k)), r["text"], r.get("reference")) for k, r in enumerate(d)]


def make_normalizer(data_root):
    """Same text normalisation as the training data (mojibake repair + clean_v2), no filtering."""
    sys.path.insert(0, data_root)
    import legal_bpe as L

    def norm(t):
        try:
            t2, _ = L.repair_text(t)
            return L.clean_v2(L._FFFD_RE.sub("", t2))
        except Exception:
            return t
    return norm


def sample_next(logits, seen, temperature, top_p, rep_penalty, allow_eos):
    logits = logits.float()
    logits[:, PAD_ID] = float("-inf")
    logits[:, BOS_ID] = float("-inf")
    logits[:, MASK_ID] = float("-inf")
    if not allow_eos:
        logits[:, EOS_ID] = float("-inf")
    if rep_penalty != 1.0:
        pen = torch.where(logits > 0, logits / rep_penalty, logits * rep_penalty)
        logits = torch.where(seen, pen, logits)
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = probs.sort(dim=-1, descending=True)
        cum = sp.cumsum(-1)
        sp = sp.masked_fill((cum - sp) > top_p, 0.0)
        probs = torch.zeros_like(probs).scatter_(1, si, sp)
    return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1)[:, 0]


@torch.no_grad()
def generate_batch(model, tok, prompt_ids, device, max_new, target_words=128, temperature=0.0, top_p=1.0,
                   rep_penalty=1.0, allow_eos=False, is_encdec=False):
    """prompt_ids: list of token-id lists (already truncated). Returns list of continuation strings."""
    B = len(prompt_ids)
    P = max(len(p) for p in prompt_ids)
    ids = torch.full((B, P), PAD_ID, dtype=torch.long)
    padlen = torch.zeros(B, dtype=torch.long)
    for i, p in enumerate(prompt_ids):
        ids[i, P - len(p):] = torch.tensor(p, dtype=torch.long)
        padlen[i] = P - len(p)
    ids, padlen = ids.to(device), padlen.to(device)
    use_cuda = device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
        logits, st = model.begin(ids, padlen, max_new)
        seen = torch.zeros(B, model.cfg["vocab"], dtype=torch.bool, device=device)
        out = torch.zeros(B, max_new, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool)
        for t in range(max_new):
            nxt = sample_next(logits, seen, temperature, top_p, rep_penalty, allow_eos)
            out[:, t] = nxt
            seen.scatter_(1, nxt[:, None], True)
            if (t + 1) % 8 == 0 and t >= 100:
                for i in range(B):
                    if not finished[i] and len(tok.decode(out[i, :t + 1].tolist()).split()) > target_words:
                        finished[i] = True
                if bool(finished.all()):
                    break
            if t + 1 < max_new:
                logits = model.advance(nxt, st)
    res = []
    out = out.cpu().tolist()
    for i in range(B):
        toks = out[i]
        if EOS_ID in toks:
            toks = toks[:toks.index(EOS_ID)]
        res.append(" ".join(tok.decode(toks).split()[:target_words]))
    return res


# ---- simple proxy metrics for dev prompts (the official metrics are computed by Codabench)
def rouge_l(pred, ref):
    a, b = pred.lower().split(), ref.lower().split()
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def bleu4(pred, ref):
    """Sentence-level BLEU-4 with add-1 smoothing for n>1 (informational proxy only)."""
    from collections import Counter
    a, b = pred.lower().split(), ref.lower().split()
    if not a:
        return 0.0
    logp = 0.0
    for n in range(1, 5):
        ca = Counter(tuple(a[i:i + n]) for i in range(len(a) - n + 1))
        cb = Counter(tuple(b[i:i + n]) for i in range(len(b) - n + 1))
        ov = sum(min(c, cb[g]) for g, c in ca.items())
        tot = max(1, sum(ca.values()))
        pn = (ov + 1.0) / (tot + 1.0) if n > 1 else max(ov / tot, 1e-9)
        logp += float(np.log(pn))
    bp = 1.0 if len(a) >= len(b) else float(np.exp(1 - len(b) / len(a)))
    return bp * float(np.exp(logp / 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="/kaggle/input/cs781-data-v2")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_new", type=int, default=288)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--rep_penalty", type=float, default=1.0)
    ap.add_argument("--allow_eos", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bertscore", action="store_true", help="also compute BERTScore F1 on dev prompts (needs bert_score + internet)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    is_encdec = cfg["kind"] == "encdec"
    tok = load_tokenizer(args.data_root)
    norm = make_normalizer(args.data_root)
    rows = read_inputs(args.input)
    if args.n:
        rows = rows[:args.n]
    max_prompt = 768 if is_encdec else 1024 - args.max_new
    enc = []
    for _, prompt, _ in rows:
        ids = tok.encode_to_ids(norm(prompt), {})
        enc.append(ids[-max_prompt:])                               # keep the END of the prompt (nearest to the continuation)
    order = sorted(range(len(rows)), key=lambda i: len(enc[i]))
    preds = [None] * len(rows)
    t0 = time.time()
    for s in range(0, len(order), args.batch):
        idx = order[s:s + args.batch]
        outs = generate_batch(model, tok, [enc[i] for i in idx], device, args.max_new, 128, args.temperature,
                              args.top_p, args.rep_penalty, args.allow_eos, is_encdec)
        for i, o in zip(idx, outs):
            preds[i] = o
        print(f"generated {min(s + args.batch, len(order))}/{len(order)}  ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    pj = os.path.join(args.out_dir, "prediction.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump({rid: {"text": p} for (rid, _, _), p in zip(rows, preds)}, f, ensure_ascii=False, indent=2)
    zp = os.path.join(args.out_dir, "submission.zip")
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    words = [len(p.split()) for p in preds]
    print(f"wrote {pj} and {zp}; {len(preds)} predictions, words/prediction min {min(words)} mean {np.mean(words):.1f}")
    refs = [r for _, _, r in rows]
    if all(r for r in refs):
        rl = np.mean([rouge_l(p, r) for p, r in zip(preds, refs)])
        b4 = np.mean([bleu4(p, r) for p, r in zip(preds, refs)])
        print(f"DEV proxy: ROUGE-L F1 {rl:.4f}   BLEU-4 {b4:.4f}")
        if args.bertscore:
            from bert_score import score
            P, R, F = score(preds, refs, lang="en", verbose=False)
            print(f"DEV BERTScore F1 {F.mean().item():.4f}")
        print("sample:", preds[0][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
