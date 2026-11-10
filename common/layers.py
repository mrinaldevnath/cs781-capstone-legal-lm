"""
layers.py - small building blocks shared by every model (all written from scratch).

    RMSNorm        pre-norm layer, computed in fp32
    RotaryEmbedding  RoPE (rotate-half form), per-sequence positions
    SwiGLU         gated feed-forward  down( silu(gate(x)) * up(x) )
    Ctx            per-forward-pass info handed to every attention layer
                   (positions for RoPE, padding masks, training/prefill/decode flags)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE on the last dim (must be even). x: (B, H, T, D); pos: (B, T) or (T,) integer positions."""

    def __init__(self, dim, max_len, base=10000.0):
        super().__init__()
        assert dim % 2 == 0
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        f = torch.outer(torch.arange(max_len, dtype=torch.float32), inv)   # (L, dim/2)
        emb = torch.cat([f, f], dim=-1)                                    # (L, dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        self.max_len = max_len

    def forward(self, x, pos):
        cos, sin = self.cos[pos], self.sin[pos]
        if cos.dim() == 2:            # (T, D) -> (1, 1, T, D)
            cos, sin = cos[None, None], sin[None, None]
        else:                         # (B, T, D) -> (B, 1, T, D)
            cos, sin = cos[:, None], sin[:, None]
        xf = x.float()
        return (xf * cos + rotate_half(xf) * sin).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Ctx:
    """What every mixer needs to know about the current forward pass.

    pos        (B, T) long   RoPE position of every current token (pad tokens clamp to 0)
    valid      (B, T) bool   True for real tokens of the current chunk (None = all real)
    key_valid  (B, S) bool   validity of every absolute cache slot (None = all real); S >= start+T
    start      int           absolute cache slot of the first current token (0 in train / prefill)
    decode     bool          True for the single-token incremental step (T == 1)
    ckpt       bool          recompute attention scores in backward (memory saver, training only)
    qchunk     int           query-block size inside the attention core
    enc_out / enc_valid      encoder output (B, Te, d) and its validity (B, Te) - model A only
    """

    def __init__(self, pos, valid=None, key_valid=None, start=0, decode=False,
                 ckpt=False, qchunk=256, enc_out=None, enc_valid=None):
        self.pos, self.valid, self.key_valid = pos, valid, key_valid
        self.start, self.decode, self.ckpt, self.qchunk = start, decode, ckpt, qchunk
        self.enc_out, self.enc_valid = enc_out, enc_valid
