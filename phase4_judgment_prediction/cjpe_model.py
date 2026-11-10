"""
cjpe_model.py - hierarchical judgment classifier on top of a Phase-1 backbone.

    chunks of one or more documents (n_chunks, L) --Backbone--> token states (n_chunks, L, d)
        --masked mean over each chunk's tokens--> chunk vectors (n_chunks, d)
        --concat [relative position of the chunk in its document (16 fixed sinusoids)] --> Linear + GELU (hidden h)
        --document pooling over the chunks of each document:
              attn   learned attention pooling: score = w . x_chunk, softmax over the document's chunks
                     (the weights are returned: which part of a judgment the model relies on)
              mean   average of the chunk vectors
              max    element-wise maximum
        --> dropout -> Linear(h, 2) -> logits (n_docs, 2)

Backbone (ner_model.py): model A uses its bidirectional encoder only, B-E their causal trunk.
The head runs in fp32 outside autocast (as in Phases 2 and 3).
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import math

import torch
import torch.nn as nn

from ner_model import Backbone  # noqa: F401  (re-exported for cjpe_train)


def position_features(rel, dim=16):
    """rel (n,) in [0, 1] -> (n, dim): sin/cos(pi * k * rel), k = 1..dim/2."""
    k = torch.arange(1, dim // 2 + 1, device=rel.device, dtype=torch.float32)
    x = math.pi * rel.float()[:, None] * k[None]
    return torch.cat([x.sin(), x.cos()], dim=-1)


class CJPEModel(nn.Module):
    def __init__(self, backbone, n_labels=2, pool="attn", hidden=256, dropout=0.1, use_pos=True, feat_dim=16):
        super().__init__()
        assert pool in ("attn", "mean", "max")
        self.backbone, self.pool, self.use_pos, self.feat_dim = backbone, pool, use_pos, feat_dim
        d_in = backbone.d + (feat_dim if use_pos else 0)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, hidden), nn.GELU())
        self.score = nn.Linear(hidden, 1) if pool == "attn" else None
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_labels)

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def chunk_vectors(self, ids, valid):
        h = self.backbone(ids, valid).float()                            # (n, L, d); fp32: a 512-token fp16 sum can overflow
        m = valid.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, ids, valid, doc, slot, rel, n_docs):
        """-> logits (n_docs, n_labels) fp32, document vectors (n_docs, hidden), chunk weights (n_docs, K) or None."""
        v = self.chunk_vectors(ids, valid)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            x = v.float()
            if self.use_pos:
                x = torch.cat([x, position_features(rel, self.feat_dim)], dim=-1)
            x = self.proj(x)                                             # (n_chunks, h)
            K = int(slot.max().item()) + 1
            grid = x.new_zeros(n_docs, K, x.shape[-1])
            mask = torch.zeros(n_docs, K, dtype=torch.bool, device=x.device)
            grid[doc, slot] = x
            mask[doc, slot] = True
            w = None
            if self.pool == "attn":
                s = self.score(grid).squeeze(-1).masked_fill(~mask, float("-inf"))
                w = torch.softmax(s, dim=1)
                z = (w.unsqueeze(-1) * grid).sum(1)
            elif self.pool == "mean":
                z = grid.sum(1) / mask.sum(1, keepdim=True).clamp(min=1).float()
            else:
                z = grid.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
            return self.fc(self.drop(z)).float(), z, w
