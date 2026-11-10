"""
pcr_model.py - a Phase-1 model as a dense case retriever (bi-encoder), written from scratch.

    Backbone    Phase-1 checkpoint -> contextual states (B, T, d). Model A: the bidirectional ENCODER only;
                B/C/D/E: the causal trunk (no LM head). Block-level activation checkpointing during training.
    Retriever   a document is cut into chunks of L tokens; each chunk -> mean-pooled (or last-token) state,
                L2-normalised; the document vector = normalised mean of its chunk vectors (hierarchical pooling).
                Scores: cosine of document vectors ("mean"), and MaxSim over chunks ("maxsim":
                mean over query chunks of the best-matching candidate chunk, as in ColBERT / BERT-MaxP).
    Adaptation  full | partial (top-k blocks) | lorma | ia3   (+ knowledge distillation in pcr_train.py)

IA3 (Liu et al., 2022, "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning"):
learned vectors rescale the attention keys and values and the FFN inner activation:
    k <- l_k * k,  v <- l_v * v,  ffn: down(act(gate x) * (l_ff * up x))      all l initialised to 1
In our blocks keys/values come from fused projections (qkv of MHA / Gated DeltaNet: layout [q|k|v]; kv of GQA;
w_uk / w_uv of MLA); only their k and v parts are rescaled.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import peft
from layers import Ctx
from models import build_model


class Backbone(nn.Module):
    def __init__(self, ckpt_path=None, cfg=None, state=None):
        super().__init__()
        if ckpt_path is not None:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg, state = ck["cfg"], ck["model"]
        self.cfg = dict(cfg)
        self.cfg["ckpt"] = False                    # we checkpoint whole blocks instead
        lm = build_model(self.cfg)
        if state is not None:
            missing, unexpected = lm.load_state_dict(state, strict=False)
            assert not missing and not unexpected, f"checkpoint mismatch: {missing[:5]} {unexpected[:5]}"
        self.is_encdec = self.cfg["kind"] == "encdec"
        if self.is_encdec:
            del lm.dec
            del lm.norm
        self.lm = lm
        self.d = self.cfg["d"]
        self.block_ckpt = False

    def blocks(self):
        return self.lm.enc if self.is_encdec else self.lm.blocks

    def forward(self, ids, valid):
        T = ids.shape[1]
        pad = not bool(valid.all())
        ctx = Ctx(torch.arange(T, device=ids.device), valid=valid if (self.is_encdec and pad) else None,
                  ckpt=False, qchunk=self.lm.qchunk)
        x = self.lm.emb(ids).float()
        for blk in self.blocks():
            if self.block_ckpt and torch.is_grad_enabled():
                x = checkpoint(blk, x, ctx, use_reentrant=False)
            else:
                x = blk(x, ctx)
        return self.lm.enc_norm(x) if self.is_encdec else self.lm.norm(x)


class IA3Linear(nn.Module):
    def __init__(self, base, skip_first=0):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.out_features - skip_first))
        self.skip = skip_first                      # leading output features left unscaled (the q part)

    def forward(self, x):
        y = self.base(x)
        if self.skip:
            return torch.cat([y[..., :self.skip], y[..., self.skip:] * self.l.to(y.dtype)], -1)
        return y * self.l.to(y.dtype)


def add_ia3(backbone):
    n = 0
    for blk in backbone.blocks():
        todo = []
        for mod in blk.modules():
            if isinstance(mod, IA3Linear):
                continue
            for name in ("qkv", "kv", "w_uk", "w_uv", "up"):
                lin = getattr(mod, name, None)
                if isinstance(lin, nn.Linear):
                    todo.append((mod, name, lin))
        for mod, name, lin in todo:
            skip = lin.out_features // 3 if name == "qkv" else 0
            setattr(mod, name, IA3Linear(lin, skip))
            n += 1
    return n


def configure(backbone, method, rank=16, alpha=32, dropout=0.05, top_k=4):
    """Returns (n_trainable, n_total, n_wrapped)."""
    if method == "ia3":
        for p in backbone.parameters():
            p.requires_grad = False
        n = add_ia3(backbone)
        for mod in backbone.modules():
            if isinstance(mod, IA3Linear):
                mod.l.requires_grad = True
        n_tr = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        return n_tr, sum(p.numel() for p in backbone.parameters()), n
    return peft.configure_trainable(backbone, method, r=rank, alpha=alpha, dropout=dropout, top_k=top_k)


class Retriever(nn.Module):
    def __init__(self, backbone, pool="mean"):
        super().__init__()
        self.backbone = backbone
        self.pool = pool

    def encode_chunks(self, ids, valid):
        """ids, valid (N, L) -> (N, d) fp32 unit vectors."""
        h = self.backbone(ids, valid)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            h = h.float()
            v = valid.float()
            if self.pool == "last":
                idx = valid.long().sum(1).clamp(min=1) - 1
                e = h[torch.arange(h.shape[0], device=h.device), idx]
            else:
                e = (h * v[..., None]).sum(1) / v.sum(1, keepdim=True).clamp(min=1.0)
            return F.normalize(e, dim=-1)


def doc_vectors(chunk_vecs, owner, n_docs):
    """Mean of each document's chunk vectors, normalised. owner (N,) long -> (n_docs, d)."""
    out = torch.zeros(n_docs, chunk_vecs.shape[1], device=chunk_vecs.device, dtype=chunk_vecs.dtype)
    out = out.index_add(0, owner, chunk_vecs)
    return F.normalize(out, dim=-1)


def build_retriever(base_ckpt, method, pool="mean", rank=16, alpha=32, top_k=4, saved=None):
    """A retriever with the adaptation wrappers of `method`; `saved` = state_dict of a trained run."""
    bb = Backbone(base_ckpt)
    info = configure(bb, method, rank=rank, alpha=alpha, top_k=top_k)
    r = Retriever(bb, pool)
    if saved is not None:
        sd = torch.load(saved, map_location="cpu", weights_only=False) if isinstance(saved, str) else saved
        r.load_state_dict(sd["state"] if "state" in sd else sd)
    return r, info
