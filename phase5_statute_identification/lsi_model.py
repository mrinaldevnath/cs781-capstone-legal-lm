"""
lsi_model.py - hierarchical multi-label statute classifier on top of a Phase-1 backbone.

    chunks of one or more documents (n_chunks, P+L) --Backbone--> token states
        --masked mean over each chunk's own tokens (fp32)--> chunk vectors (n_chunks, d)
        --concat [relative position of the chunk in its document (16 sinusoids)] --> Linear + GELU (hidden h)
        --document head (one score per statute):
              attn      shared attention pooling over the document's chunks -> Linear(h, K)
              mean/max  mean / element-wise max of the chunk vectors -> Linear(h, K)
              lwan      label-wise attention: every statute has its own query over the chunks and its own output
                        vector (each statute looks for its evidence in its own part of the document)
              chunkmax  every chunk predicts all statutes; the document score of a statute is its maximum over the
                        chunks ("chunk-level predictions followed by document-level aggregation")
        --optional co-occurrence layer: logits + C . sigmoid(logits), C (K x K) zero-initialised (statutes inform
          each other; starts as the identity mapping)
        --> logits (n_docs, K); training uses independent sigmoid outputs (binary relevance).

Backbone (ner_model.py): model A uses its bidirectional encoder only, B-E their causal trunk.
The head runs in fp32 outside autocast.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import math

import torch
import torch.nn as nn

from ner_model import Backbone  # noqa: F401  (re-exported for lsi_train)

HEADS = ("attn", "mean", "max", "lwan", "chunkmax")


def position_features(rel, dim=16):
    k = torch.arange(1, dim // 2 + 1, device=rel.device, dtype=torch.float32)
    x = math.pi * rel.float()[:, None] * k[None]
    return torch.cat([x.sin(), x.cos()], dim=-1)


class LSIModel(nn.Module):
    def __init__(self, backbone, n_labels, head="attn", hidden=256, dropout=0.1, use_pos=True, cooc=False, feat_dim=16):
        super().__init__()
        assert head in HEADS
        self.backbone, self.head, self.use_pos, self.feat_dim, self.n_labels = backbone, head, use_pos, feat_dim, n_labels
        d_in = backbone.d + (feat_dim if use_pos else 0)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, hidden), nn.GELU())
        self.drop = nn.Dropout(dropout)
        if head == "attn":
            self.score = nn.Linear(hidden, 1)
        if head == "lwan":
            self.query = nn.Parameter(torch.randn(n_labels, hidden) / math.sqrt(hidden))
            self.out_w = nn.Parameter(torch.randn(n_labels, hidden) / math.sqrt(hidden))
            self.out_b = nn.Parameter(torch.zeros(n_labels))
        else:
            self.fc = nn.Linear(hidden, n_labels)
        self.cooc = nn.Linear(n_labels, n_labels, bias=False) if cooc else None
        if self.cooc is not None:
            nn.init.zeros_(self.cooc.weight)

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def chunk_vectors(self, ids, valid, pool):
        h = self.backbone(ids, valid).float()                       # fp32: a 512-token fp16 sum can overflow
        m = pool.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, ids, valid, pool, doc, slot, rel, n_docs):
        """-> logits (n_docs, K) fp32, document vectors (n_docs, h) or None, chunk weights (n_docs, n_slots) or None."""
        v = self.chunk_vectors(ids, valid, pool)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            x = v.float()
            if self.use_pos:
                x = torch.cat([x, position_features(rel, self.feat_dim)], dim=-1)
            x = self.proj(x)
            S = int(slot.max().item()) + 1
            grid = x.new_zeros(n_docs, S, x.shape[-1])
            mask = torch.zeros(n_docs, S, dtype=torch.bool, device=x.device)
            grid[doc, slot] = x
            mask[doc, slot] = True
            z, w = None, None
            if self.head == "attn":
                s = self.score(grid).squeeze(-1).masked_fill(~mask, float("-inf"))
                w = torch.softmax(s, dim=1)
                z = (w.unsqueeze(-1) * grid).sum(1)
                logits = self.fc(self.drop(z))
            elif self.head == "mean":
                z = grid.sum(1) / mask.sum(1, keepdim=True).clamp(min=1).float()
                logits = self.fc(self.drop(z))
            elif self.head == "max":
                z = grid.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
                logits = self.fc(self.drop(z))
            elif self.head == "lwan":
                s = torch.einsum("bsh,kh->bks", grid, self.query).masked_fill(~mask[:, None], float("-inf"))
                a = torch.softmax(s, dim=-1)                                # (B, K, S)
                zl = torch.einsum("bks,bsh->bkh", a, grid)                  # (B, K, h)
                logits = (self.drop(zl) * self.out_w[None]).sum(-1) + self.out_b
                w = a.mean(1)                                               # average over statutes
            else:                                                           # chunkmax
                cl = self.fc(self.drop(grid)).masked_fill(~mask.unsqueeze(-1), float("-inf"))   # (B, S, K)
                logits, arg = cl.max(1)
                w = torch.zeros_like(mask, dtype=torch.float32)             # share of statutes won by each chunk
                w.scatter_add_(1, arg, torch.ones_like(arg, dtype=torch.float32))
                w = w / self.n_labels
            if self.cooc is not None:
                logits = logits + self.cooc(torch.sigmoid(logits))
            return logits.float(), z, w
