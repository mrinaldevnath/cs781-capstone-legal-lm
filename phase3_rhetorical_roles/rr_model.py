"""
rr_model.py - hierarchical rhetorical-role classifier on top of a Phase-1 backbone.

    windows of packed sentences --Backbone--> token states (n_win, L, d)
        --pool (mean | last over each sentence's tokens)--> sentence vectors (n_sent, d)
        --concat [relative position features (16, fixed sinusoids) | domain embedding (16)]-->
        --head: bilstm (document context over the range of sentences; 1 or 2 layers) | mlp (each sentence alone)--> logits

Backbone (ner_model.py): model A uses its bidirectional encoder only, B-E their causal trunk.
The head runs in fp32 outside autocast (as in Phase 2).
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ner_model import Backbone  # noqa: F401  (re-exported for rr_train)


def position_features(rel, dim=16):
    """rel (n,) in [0, 1] -> (n, dim): sin/cos(pi * k * rel), k = 1..dim/2."""
    k = torch.arange(1, dim // 2 + 1, device=rel.device, dtype=torch.float32)
    x = math.pi * rel.float()[:, None] * k[None]
    return torch.cat([x.sin(), x.cos()], dim=-1)


class RRModel(nn.Module):
    def __init__(self, backbone, n_labels, head="bilstm", hidden=256, dropout=0.1, pool="mean",
                 use_pos=True, use_dom=True, feat_dim=16, n_domains=2, rnn_layers=1):
        super().__init__()
        self.backbone, self.pool, self.feat_dim = backbone, pool, feat_dim
        self.use_pos = use_pos
        self.dom = nn.Embedding(n_domains, feat_dim) if use_dom else None
        d_in = backbone.d + (feat_dim if use_pos else 0) + (feat_dim if use_dom else 0)
        self.drop = nn.Dropout(dropout)
        self.head_kind = head
        if head == "bilstm":
            self.rnn = nn.LSTM(d_in, hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True,
                               dropout=dropout if rnn_layers > 1 else 0.0)
        elif head == "mlp":
            self.mlp = nn.Sequential(nn.Linear(d_in, 2 * hidden), nn.GELU())
        else:
            raise ValueError(head)
        self.fc = nn.Linear(2 * hidden, n_labels)

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def sentence_vectors(self, ids, valid, win, a, b):
        h = self.backbone(ids, valid).float()                         # (n_win, L, d)
        if self.pool == "last":
            return h[win, b - 1]
        hc = F.pad(h.cumsum(1), (0, 0, 1, 0))                         # hc[:, t] = sum of h[:, :t]
        return (hc[win, b] - hc[win, a]) / (b - a).clamp(min=1).unsqueeze(-1).float()

    def forward(self, ids, valid, win, a, b, rel, dom):
        """One range of consecutive sentences of one document -> logits (n_sent, n_labels)."""
        v = self.sentence_vectors(ids, valid, win, a, b)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            feats = [v.float()]
            if self.use_pos:
                feats.append(position_features(rel, self.feat_dim))
            if self.dom is not None:
                feats.append(self.dom(dom).expand(v.shape[0], -1))
            x = self.drop(torch.cat(feats, dim=-1))
            if self.head_kind == "bilstm":
                x, _ = self.rnn(x.unsqueeze(0))
                x = x[0]
            else:
                x = self.mlp(x)
            return self.fc(self.drop(x)).float()
