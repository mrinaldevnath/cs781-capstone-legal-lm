"""
attn_cross.py - cross-attention of the encoder-decoder model A (written from scratch).

Queries come from the decoder stream, keys/values from the encoder output. No RoPE (positions are
already baked into the encoder states) and no causal mask; only encoder padding is masked.
At generation time K/V are computed ONCE from the encoder output and kept in a CrossCache.
"""
import math
import torch
import torch.nn as nn

from attn_core import attend, attend_blocks
from kv_cache import CrossCache


class CrossMHA(nn.Module):
    def __init__(self, d, n_heads, head_dim):
        super().__init__()
        self.H, self.D = n_heads, head_dim
        self.q = nn.Linear(d, n_heads * head_dim, bias=False)
        self.kv = nn.Linear(d, 2 * n_heads * head_dim, bias=False)
        self.out = nn.Linear(n_heads * head_dim, d, bias=False)
        self.scale = 1.0 / math.sqrt(head_dim)

    def make_cache(self, B, max_len, device, dtype=torch.float16):
        return CrossCache()

    def forward(self, x, ctx, cache=None):
        B, T, _ = x.shape
        H, D = self.H, self.D
        q = self.q(x).view(B, T, H, D).transpose(1, 2)
        if cache is not None and cache.k is not None:
            k, v = cache.k, cache.v
        else:
            Te = ctx.enc_out.shape[1]
            k, v = self.kv(ctx.enc_out).view(B, Te, 2, H, D).permute(2, 0, 3, 1, 4)
            if cache is not None:
                cache.k, cache.v = k, v
        if ctx.decode:
            mask = None if ctx.enc_valid is None else ctx.enc_valid[:, None, None, :]
            o = attend(q, k, v, mask, self.scale)
        else:
            o = attend_blocks(q, k, v, self.scale, causal=False, key_valid=ctx.enc_valid,
                              qchunk=ctx.qchunk, ckpt=ctx.ckpt)
        return self.out(o.transpose(1, 2).reshape(B, T, H * D))
