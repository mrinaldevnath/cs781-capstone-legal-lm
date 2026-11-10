"""
peft.py - parameter-efficient adaptation of the Phase-1 backbones, written from scratch.

All Phase-1 projections are bias-free nn.Linear layers with stable names, so every method is a wrapper that
replaces selected nn.Linear modules in place (the frozen weight W0 stays inside as .base):

    LoRA     (Hu et al., 2021)            h = W0 x + (a/r) * B A x                 A: r x d_in,  B: d_out x r (=0)
    LoRMA+   (Bihany, Patel, Modi, 2025)  h = (I + (a/r) B A) W0 x
                                            = y + (a/r) * B (A y),   y = W0 x      A: r x d_out, B: d_out x r (=0)
             multiplicative low-rank adaptation; the additive-identity "rank inflation" (I + BA) keeps the
             initial model exact (B = 0) and avoids the rank-r bottleneck of a plain product (BA) W0.
             The product is never formed: B(A(W0 x)) costs the same as LoRA.
    Adapter  (Houlsby et al., 2019)       h = y + U act(D y),  y = W0 x            bottleneck m, U = 0 at init
             placed on the output projection of every token mixer and on the FFN down-projection, i.e. after
             each sub-layer and before the residual add (Houlsby placement).

Non-wrapper strategies (set by `configure_trainable`):
    full      every backbone parameter trains
    partial   only the top-k blocks (+ final norm) train          (layer-wise / partial fine-tuning)
    norm      only RMSNorm gains train                            (BitFit-style sparse tuning; the models have no biases)
    lora / lorma / adapter   backbone frozen, only the inserted modules train
The task head always trains.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# projection names inside the Phase-1 blocks (attn_mha, attn_gqa_local, attn_mla, gated_deltanet, layers.SwiGLU)
ATTN_TARGETS = {"qkv", "q", "kv", "out", "wq", "w_dkv", "w_uk", "w_uv", "g_proj"}
FFN_TARGETS = {"gate", "up", "down"}
ADAPTER_TARGETS = {"out", "down"}           # mixer output projection + FFN down projection


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)


class LoRMALinear(nn.Module):
    """LoRMA with additive rank inflation: h = (I + s * B A) W0 x."""

    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.out_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        return y + self.scale * F.linear(F.linear(self.drop(y), self.A), self.B)


class AdapterLinear(nn.Module):
    def __init__(self, base, bottleneck, dropout=0.0):
        super().__init__()
        self.base = base
        d = base.out_features
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        return y + self.up(self.drop(F.gelu(self.down(y))))


def _replace(module, targets, make):
    """Replace every nn.Linear child whose attribute name is in targets (recursively). Returns count."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, make(child))
            n += 1
        elif not isinstance(child, (LoRALinear, LoRMALinear, AdapterLinear)):
            n += _replace(child, targets, make)
    return n


def blocks_of(backbone):
    """The list of transformer blocks of a Phase-1 model (encoder blocks for model A)."""
    m = backbone.lm
    return m.enc if hasattr(m, "enc") else m.blocks


def configure_trainable(backbone, method, r=16, alpha=32, dropout=0.05, bottleneck=64, top_k=4, targets="all"):
    """Freeze / wrap the backbone for one adaptation method. Returns (n_trainable, n_total, n_wrapped)."""
    tg = ATTN_TARGETS | FFN_TARGETS if targets == "all" else ATTN_TARGETS
    wrapped = 0
    for p in backbone.parameters():
        p.requires_grad = method == "full"
    if method == "partial":
        for blk in list(blocks_of(backbone))[-top_k:]:
            for p in blk.parameters():
                p.requires_grad = True
        for name in ("norm", "enc_norm"):
            if hasattr(backbone.lm, name):
                for p in getattr(backbone.lm, name).parameters():
                    p.requires_grad = True
    elif method == "norm":
        for name, p in backbone.named_parameters():
            if p.ndim == 1 and ("n1." in name or "n2." in name or "nx." in name or name.endswith("norm.weight")
                                or "kv_norm" in name or "o_norm" in name):
                p.requires_grad = True
    elif method in ("lora", "lorma", "adapter"):
        for blk in blocks_of(backbone):
            if method == "lora":
                wrapped += _replace(blk, tg, lambda b: LoRALinear(b, r, alpha, dropout))
            elif method == "lorma":
                wrapped += _replace(blk, tg, lambda b: LoRMALinear(b, r, alpha, dropout))
            else:
                wrapped += _replace(blk, ADAPTER_TARGETS, lambda b: AdapterLinear(b, bottleneck, dropout))
        for mod in backbone.modules():                 # the new modules (and only they) train
            if isinstance(mod, (LoRALinear, LoRMALinear, AdapterLinear)):
                for n_, p in mod.named_parameters():
                    p.requires_grad = not n_.startswith("base.")
    elif method != "full":
        raise ValueError(method)
    n_tr = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in backbone.parameters())
    return n_tr, n_all, wrapped
