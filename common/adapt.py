"""
adapt.py - adaptation methods.

    full      every backbone weight trains
    partial   only the top-k blocks (+ final norm) train
    norm      only the RMSNorm gains train (BitFit-style; the Phase-1 models have no biases)
    lora      LoRA (Hu et al., 2021)
    lorma     LoRMA+ (Bihany et al., 2025), multiplicative low-rank update
    adapter   Houlsby bottleneck adapters

    rslora    Rank-stabilized LoRA (Kalajdzievski, 2023): LoRA with scale alpha / sqrt(r) instead of alpha / r,
              so the update does not shrink as the rank grows (used with a large rank, r = 64).
    adalora   AdaLoRA (Zhang et al., 2023): the update is in SVD form  P diag(lambda) Q  (P: d_out x r, Q: r x d_in),
              with an orthogonality penalty on P and Q, and a global rank budget: every 10 steps the singular values
              lambda with the lowest importance (EMA of |theta * grad| times its uncertainty, summed over the triplet
              (lambda_i, P[:, i], Q[i, :])) are set to 0, so ranks move to the matrices that need them. The budget falls
              from r_init to r_target per matrix with a cubic schedule (5% -> 50% of the planned steps), then is fixed.
    qlora     QLoRA (Dettmers et al., 2023): every frozen linear layer of the blocks is stored in 4-bit NormalFloat (NF4,
              blocks of 64 weights, one fp32 absmax per block) and dequantised on the fly; LoRA r16 trains on top in
              16/32-bit. The backward pass dequantises again instead of keeping the weights (as bitsandbytes does).
              Double quantisation of the absmax values and paged optimizers are not implemented (documented).
    ia3       IA^3 (Liu et al., 2022): learned element-wise rescaling vectors (init 1), backbone frozen:
                  keys / values   output of the K/V-producing projections  (qkv, kv, w_uk, w_uv)
                  FFN             input of the down projection, i.e. the gated hidden activation
              For fused qkv projections (MHA, Gated DeltaNet) the query part is rescaled too (one vector over the
              fused output) - a small, documented deviation from the paper.
    prompt    Prompt tuning (Lester et al., 2021): n learned input vectors prepended to every window, backbone frozen.
              The first n positions of every window carry placeholder ids and get the learned vectors.
    prefix    Prefix tuning (Li & Liang, 2021) in its deep-prompt form (P-Tuning v2, Liu et al., 2022): n learned
              vectors PER BLOCK are written into the first n positions of the hidden states at the input of every
              block, so every layer sees trainable prefix states that all later tokens attend to (or, for Gated
              DeltaNet, that enter the recurrent state first). This is architecture-independent; the original
              key/value-prefix form would need a separate change inside each token mixer (MHA, GQA, MLA, GDN).
    gradual   Gradual unfreezing (Howard & Ruder, 2018, ULMFiT; "layer freezing / unfreezing strategy" in the Phase-4
              brief): training starts with the whole backbone frozen (only the task head learns); between 10% and 60%
              of the planned steps the blocks are unfrozen one at a time from the TOP (the last block, with the final
              norm, first) down to the first; afterwards every block trains (embeddings stay frozen). Upper layers are
              the most task-specific, lower layers keep their pretrained features longest.
The task head always trains.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import peft as P

IA3_OUT = {"qkv", "kv", "w_uk", "w_uv"}
IA3_IN = {"down"}


# ------------------------------------------------------------------------------------------------ rsLoRA
class RSLoRALinear(P.LoRALinear):
    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__(base, r, alpha, dropout)
        self.scale = alpha / math.sqrt(r)


# ------------------------------------------------------------------------------------------------ AdaLoRA
class AdaLoRALinear(nn.Module):
    """h = W0 x + s * P diag(E) Q x ; E = 0 at init (exact start), P / Q near-orthonormal."""

    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.P = nn.Parameter(torch.randn(base.out_features, r) / math.sqrt(base.out_features))
        self.E = nn.Parameter(torch.zeros(r))
        self.Q = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.drop = nn.Dropout(dropout)
        self.register_buffer("ipt", torch.zeros(r), persistent=False)       # EMA of the sensitivity
        self.register_buffer("unc", torch.zeros(r), persistent=False)       # EMA of its uncertainty
        self.register_buffer("keep", torch.ones(r, dtype=torch.bool), persistent=False)

    def forward(self, x):
        z = F.linear(self.drop(x), self.Q) * self.E.to(x.dtype)
        return self.base(x) + self.scale * F.linear(z, self.P)

    def orth_penalty(self):
        I = torch.eye(self.r, device=self.P.device)
        P_, Q_ = self.P.float(), self.Q.float()
        return ((P_.t() @ P_ - I) ** 2).sum() + ((Q_ @ Q_.t() - I) ** 2).sum()


class AdaLoRAController:
    """Orthogonality penalty + global rank-budget allocation over all AdaLoRALinear modules."""

    def __init__(self, mods, r_target, total_steps, gamma=0.1, beta=0.85, t_init=0.05, t_final=0.5, every=10):
        self.mods, self.gamma, self.beta, self.every = mods, gamma, beta, every
        self.b0 = sum(m.r for m in mods)
        self.bT = sum(min(r_target, m.r) for m in mods)
        self.ti = max(1, int(t_init * total_steps))
        self.tf = max(self.ti + 1, int(t_final * total_steps))

    def regularizer(self):
        return self.gamma * sum(m.orth_penalty() for m in self.mods) / len(self.mods)

    def budget(self, t):
        if t < self.ti:
            return self.b0
        if t >= self.tf:
            return self.bT
        f = 1 - (t - self.ti) / (self.tf - self.ti)
        return int(self.bT + (self.b0 - self.bT) * f ** 3)

    @torch.no_grad()
    def after_step(self, t):
        """Call after optimizer.step() (gradients are still present and unscaled)."""
        if t < self.tf:
            for m in self.mods:
                if m.E.grad is None or m.P.grad is None or m.Q.grad is None:
                    continue
                s = ((m.E * m.E.grad).abs() + (m.P * m.P.grad).abs().mean(0) + (m.Q * m.Q.grad).abs().mean(1)).float()
                if not torch.isfinite(s).all():
                    continue                                        # AMP skipped this step
                m.ipt.mul_(self.beta).add_((1 - self.beta) * s)
                m.unc.mul_(self.beta).add_((1 - self.beta) * (s - m.ipt).abs())
            if t >= self.ti and (t % self.every == 0 or t == self.tf - 1):
                score = torch.cat([m.ipt * m.unc for m in self.mods])
                b = self.budget(t)
                keep = torch.zeros_like(score, dtype=torch.bool)
                keep[torch.topk(score, min(b, score.numel())).indices] = True
                i = 0
                for m in self.mods:
                    m.keep.copy_(keep[i:i + m.r])
                    i += m.r
        if t >= self.ti:
            for m in self.mods:                                     # masked singular values stay 0
                m.E.masked_fill_(~m.keep, 0.0)

    def info(self):
        ranks = [int(m.keep.sum()) for m in self.mods]
        return dict(adalora_budget_final=self.bT, adalora_ranks=ranks,
                    adalora_rank_mean=sum(ranks) / len(ranks), adalora_rank_min=min(ranks), adalora_rank_max=max(ranks))


# ------------------------------------------------------------------------------------------------ QLoRA
NF4_TABLE = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
             -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
             0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
             0.7229568362236023, 1.0]


class _NF4Matmul(torch.autograd.Function):
    """y = x W^T with W dequantised in forward AND again in backward (nothing full-precision is kept)."""

    @staticmethod
    def forward(ctx, x, mod):
        ctx.mod, ctx.xdtype = mod, x.dtype
        return F.linear(x, mod.dequant(x.dtype))

    @staticmethod
    def backward(ctx, g):
        W = ctx.mod.dequant(g.dtype)
        return (g @ W).to(ctx.xdtype), None


class NF4Linear(nn.Module):
    """Frozen bias-free linear layer stored as 4-bit NormalFloat codes (two per byte) + one absmax per 64 weights."""

    def __init__(self, lin, block=64):
        super().__init__()
        assert lin.bias is None, "Phase-1 projections are bias-free"
        self.in_features, self.out_features, self.block = lin.in_features, lin.out_features, block
        w = lin.weight.detach().float().reshape(-1)
        self.n = w.numel()
        w = F.pad(w, (0, (-self.n) % block)).view(-1, block)
        absmax = w.abs().amax(1, keepdim=True).clamp_min(1e-12)
        table = torch.tensor(NF4_TABLE, device=w.device)
        codes = torch.cat([((c / a).unsqueeze(-1) - table).abs().argmin(-1)              # chunked: bounded memory
                           for c, a in zip(w.split(4096), absmax.split(4096))]).to(torch.uint8).view(-1)
        self.register_buffer("packed", codes[0::2] | (codes[1::2] << 4))
        self.register_buffer("absmax", absmax.view(-1))
        self.register_buffer("table", table)

    def dequant(self, dtype=torch.float32):
        codes = torch.stack([self.packed & 15, self.packed >> 4], dim=-1).view(-1).long()
        w = (self.table[codes].view(-1, self.block) * self.absmax[:, None]).view(-1)[:self.n]
        return w.view(self.out_features, self.in_features).to(dtype)

    def forward(self, x):
        return _NF4Matmul.apply(x, self)


def _quantize_linears(module):
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, NF4Linear(child))
            n += 1
        else:
            n += _quantize_linears(child)
    return n


# ------------------------------------------------------------------------------------------------ IA3
class IA3Linear(nn.Module):
    def __init__(self, base, where):
        super().__init__()
        self.base, self.where = base, where
        n = base.out_features if where == "out" else base.in_features
        self.l = nn.Parameter(torch.ones(n))

    def forward(self, x):
        if self.where == "in":
            return self.base(x * self.l.to(x.dtype))
        return self.base(x) * self.l.to(x.dtype)


# ------------------------------------------------------------------------------------------------ prompt / prefix
def _vocab_init(emb, n, seed=0):
    idx = torch.randint(0, emb.num_embeddings, (n,), generator=torch.Generator().manual_seed(seed))
    return emb.weight.detach()[idx].clone().float()


class PromptEmbedding(nn.Module):
    """Wraps nn.Embedding; positions [0, n) of every sequence get the learned prompt vectors."""

    def __init__(self, emb, n):
        super().__init__()
        self.emb, self.n = emb, n
        self.prompt = nn.Parameter(_vocab_init(emb, n))

    @property
    def weight(self):
        return self.emb.weight

    def forward(self, ids):
        x = self.emb(ids)
        p = self.prompt.to(x.dtype).unsqueeze(0).expand(ids.shape[0], -1, -1)
        return torch.cat([p, x[:, self.n:]], dim=1)


class PrefixStates(nn.Module):
    """Deep prefix: learned states p[i] (n x d) replace positions [0, n) of the input of block i (forward pre-hooks)."""

    def __init__(self, blocks, emb, n):
        super().__init__()
        self.n = n
        self.p = nn.Parameter(_vocab_init(emb, n).unsqueeze(0).repeat(len(blocks), 1, 1).contiguous())
        self._handles = [blk.register_forward_pre_hook(self._hook(i)) for i, blk in enumerate(blocks)]

    def _hook(self, i):
        def f(module, args):
            x = args[0]
            p = self.p[i].to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
            return (torch.cat([p, x[:, self.n:]], dim=1),) + tuple(args[1:])
        return f


# ------------------------------------------------------------------------------------------------ configure
WRAPPERS = (P.LoRALinear, P.LoRMALinear, P.AdapterLinear, AdaLoRALinear, IA3Linear)


def _replace(module, targets, make):
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, make(child))
            n += 1
        elif not isinstance(child, WRAPPERS):
            n += _replace(child, targets, make)
    return n


def _train_only(backbone, cls, names):
    for mod in backbone.modules():
        if isinstance(mod, cls):
            for nm in names:
                getattr(mod, nm).requires_grad = True


def configure(backbone, method, r=16, alpha=32, dropout=0.05, bottleneck=64, top_k=4, n_prompt=20):
    """Freeze / wrap the backbone for one method. Returns (n_trainable, n_total, n_wrapped).
    For 'gradual' everything starts frozen; n_trainable is what trains at the end (every block + final norm)."""
    if method == "gradual":
        for p in backbone.parameters():
            p.requires_grad = False
        n_tr = sum(p.numel() for p in gradual_params(backbone))
        return n_tr, sum(p.numel() for p in backbone.parameters()), len(list(P.blocks_of(backbone)))
    if method in ("full", "partial", "norm", "lora", "lorma", "adapter"):
        return P.configure_trainable(backbone, method, r, alpha, dropout, bottleneck, top_k)
    for p in backbone.parameters():
        p.requires_grad = False
    wrapped = 0
    blocks = list(P.blocks_of(backbone))
    targets = P.ATTN_TARGETS | P.FFN_TARGETS
    if method == "rslora":
        for blk in blocks:
            wrapped += _replace(blk, targets, lambda b: RSLoRALinear(b, r, alpha, dropout))
        _train_only(backbone, RSLoRALinear, ("A", "B"))
    elif method == "adalora":
        for blk in blocks:
            wrapped += _replace(blk, targets, lambda b: AdaLoRALinear(b, r, alpha, dropout))
        _train_only(backbone, AdaLoRALinear, ("P", "E", "Q"))
    elif method == "qlora":
        for blk in blocks:
            wrapped += _replace(blk, targets, lambda b: P.LoRALinear(b, r, alpha, dropout))
            _quantize_linears(blk)                    # every frozen nn.Linear (incl. the LoRA bases) -> NF4
        _train_only(backbone, P.LoRALinear, ("A", "B"))
    elif method == "ia3":
        for blk in blocks:
            wrapped += _replace(blk, IA3_OUT, lambda b: IA3Linear(b, "out"))
            wrapped += _replace(blk, IA3_IN, lambda b: IA3Linear(b, "in"))
        _train_only(backbone, IA3Linear, ("l",))
    elif method == "prompt":
        backbone.lm.emb = PromptEmbedding(backbone.lm.emb, n_prompt)
        backbone.lm.emb.prompt.requires_grad = True
        wrapped = 1
    elif method == "prefix":
        backbone.prefix = PrefixStates(blocks, backbone.lm.emb, n_prompt)
        backbone.prefix.p.requires_grad = True
        wrapped = len(blocks)
    else:
        raise ValueError(method)
    n_tr = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in backbone.parameters()) + sum(
        m.n for m in backbone.modules() if isinstance(m, NF4Linear))          # quantised weights still count
    return n_tr, n_all, wrapped


def gradual_params(backbone):
    """Parameters gradual unfreezing will train: every block + the final norm(s) (not the embeddings)."""
    ps = [p for blk in P.blocks_of(backbone) for p in blk.parameters()]
    for name in ("norm", "enc_norm"):
        if hasattr(backbone.lm, name):
            ps += list(getattr(backbone.lm, name).parameters())
    return ps


class GradualUnfreeze:
    """Unfreezes blocks top-down: none before t_start, all after t_end (fractions of the planned steps)."""

    def __init__(self, backbone, total_steps, t_start=0.1, t_end=0.6):
        self.blocks = list(P.blocks_of(backbone))[::-1]                  # top block first
        self.norms = [getattr(backbone.lm, n) for n in ("norm", "enc_norm") if hasattr(backbone.lm, n)]
        self.ts, self.te = int(t_start * total_steps), max(int(t_start * total_steps) + 1, int(t_end * total_steps))
        self.k, self.log = 0, []

    def regularizer(self):
        return 0.0

    def target(self, t):
        if t < self.ts:
            return 0
        n = len(self.blocks)
        return min(n, 1 + int((t - self.ts) * n / max(1, self.te - self.ts)))

    def after_step(self, t):
        want = self.target(t + 1)
        while self.k < want:
            for p in self.blocks[self.k].parameters():
                p.requires_grad = True
            if self.k == 0:
                for m in self.norms:
                    for p in m.parameters():
                        p.requires_grad = True
            self.k += 1
            self.log.append((t + 1, self.k))

    def info(self):
        return {"gradual_unfrozen_blocks": self.k, "gradual_schedule": self.log}


def controller(backbone, method, total_steps, r_target=8):
    """Per-step extras of a method: AdaLoRA (rank allocation) and gradual unfreezing. Object with regularizer(),
    after_step(t), info(); or None."""
    if method == "gradual":
        return GradualUnfreeze(backbone, total_steps)
    if method != "adalora":
        return None
    mods = [m for m in backbone.modules() if isinstance(m, AdaLoRALinear)]
    return AdaLoRAController(mods, r_target, total_steps)


def uses_prompt_slots(method):
    return method in ("prompt", "prefix")


METHODS = ("full", "partial", "norm", "lora", "lorma", "adapter", "rslora", "adalora", "qlora", "ia3", "prompt", "prefix",
           "gradual")
DEFAULT_LR = {"full": 5e-5, "partial": 1e-4, "norm": 1e-3, "lora": 5e-4, "lorma": 5e-4, "adapter": 1e-3,
              "rslora": 5e-4, "adalora": 1e-3, "qlora": 5e-4, "ia3": 3e-3, "prompt": 1e-2, "prefix": 5e-3,
              "gradual": 1e-4}
