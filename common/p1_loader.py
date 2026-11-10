"""
CS781 Phase 1 - reference loader for cs781_data_v2 (output of prep_p1.py).

Mixing, block length and packing are chosen HERE, at load time:
    data = P1Data("/kaggle/input/cs781-data-v2", split="train")          # default mix from manifest
    data = P1Data(root, "train", weights={"sc": .5, "hc": .3, "trib": .1, "acts": .1})
    x, y = data.get_batch(batch_size=16, block_len=1024, rng=rng, device="cuda")
    print(data.epochs(total_tokens=1.1e9))                               # repetition per source

Sampling: pick a source with probability = its mix weight, then a uniformly random window of
block_len+1 tokens inside that source's token stream. Documents are separated by </s> (id 2), so
a window may span a document boundary - standard GPT-style packing. For DDP give every rank its
own rng, e.g. np.random.default_rng([seed, rank]).

For BART-style / document-level models use data.doc(src, k) (one document, no </s>) and
data.n_docs(src), and build inputs/noise in your model code.

Validation: data_val = P1Data(root, "val"); data_val.val_windows(src, n, block_len) returns the
same fixed windows every call -> per-source validation loss comparable across runs.
"""
import json
import os

import numpy as np


class P1Data:
    def __init__(self, root, split="train", weights=None):
        self.root, self.split = root, split
        self.manifest = json.load(open(os.path.join(root, "manifest.json")))
        self.vocab_size = self.manifest["vocab_size"]
        self.eos_id = self.manifest["eos_id"]
        w = weights if weights is not None else self.manifest["default_mix"]
        self.sources = [s for s in self.manifest["sources"] if w.get(s, 0) > 0]
        tot = sum(w[s] for s in self.sources)
        self.weights = np.array([w[s] / tot for s in self.sources])
        self.bin, self.idx = {}, {}
        for s in self.sources:
            base = os.path.join(root, split, s)
            self.bin[s] = np.memmap(base + ".bin", dtype=np.uint16, mode="r")
            self.idx[s] = np.load(base + ".idx.npy")

    # ------------------------------------------------------------------ windows
    def sample(self, n, block_len, rng):
        """(n, block_len+1) int64 array of token windows drawn by mix weight."""
        src = rng.choice(len(self.sources), size=n, p=self.weights)
        out = np.empty((n, block_len + 1), dtype=np.int64)
        for r, k in enumerate(src):
            b = self.bin[self.sources[k]]
            st = int(rng.integers(0, b.size - block_len - 1))
            out[r] = b[st:st + block_len + 1]
        return out

    def get_batch(self, batch_size, block_len, rng, device="cpu"):
        import torch
        w = torch.from_numpy(self.sample(batch_size, block_len, rng))
        x, y = w[:, :-1], w[:, 1:]
        if str(device).startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    def val_windows(self, src, n, block_len):
        """Fixed, evenly spaced windows of one source (deterministic, no rng state)."""
        b = self.bin[src]
        starts = np.linspace(0, b.size - block_len - 1, num=n).astype(np.int64)
        return np.stack([np.asarray(b[s:s + block_len + 1], dtype=np.int64) for s in starts])

    # ------------------------------------------------------------------ documents
    def n_docs(self, src):
        return self.idx[src].size - 1

    def doc(self, src, k):
        """Token ids of document k without its trailing </s>."""
        i = self.idx[src]
        return np.asarray(self.bin[src][i[k]:i[k + 1] - 1], dtype=np.int64)

    # ------------------------------------------------------------------ accounting
    def tokens(self):
        return {s: int(self.bin[s].size) for s in self.sources}

    def epochs(self, total_tokens):
        """How many passes over each source a run of total_tokens makes at these weights."""
        return {s: round(float(w * total_tokens / self.bin[s].size), 3)
                for s, w in zip(self.sources, self.weights)}


def load_tokenizer(root):
    """The exact tokenizer used to build the data (legal_bpe_v2.py is copied next to the manifest)."""
    import sys
    sys.path.insert(0, root)
    import legal_bpe as L
    m = json.load(open(os.path.join(root, "manifest.json")))
    return L.LegalBPE.load(os.path.join(root, "tokenizer", f"legal_bpe_{m['vocab_size']}.json"))
