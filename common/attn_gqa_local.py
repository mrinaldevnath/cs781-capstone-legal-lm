"""
attn_gqa_local.py - sliding-window grouped-query attention, from scratch (model C local layers).

12 query heads share 4 key/value heads (G = 3 query heads per KV head); each token attends to the
`window` most recent tokens (itself included). Nothing is copied to match head counts: the grouped
matmul lives in attn_core._scores. In training, keys outside the window are never even multiplied
(attn_core.attend_blocks slices the key range per query block).

Inference cache: RingCache of `window` slots (constant memory, 1024 B/token-slot/layer at fp16),
keys stored after RoPE.
"""
import math
import torch
import torch.nn as nn

from attn_core import attend, attend_blocks
from kv_cache import RingCache


class LocalGQA(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, head_dim, window, rope):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.H, self.Hk, self.D, self.window, self.rope = n_heads, n_kv_heads, head_dim, window, rope
        self.q = nn.Linear(d, n_heads * head_dim, bias=False)
        self.kv = nn.Linear(d, 2 * n_kv_heads * head_dim, bias=False)
        self.out = nn.Linear(n_heads * head_dim, d, bias=False)
        self.scale = 1.0 / math.sqrt(head_dim)

    def make_cache(self, B, max_len, device, dtype=torch.float16):
        return RingCache(B, self.Hk, self.window, self.D, device, dtype)

    def forward(self, x, ctx, cache=None):
        B, T, _ = x.shape
        H, Hk, D = self.H, self.Hk, self.D
        q = self.q(x).view(B, T, H, D).transpose(1, 2)                       # (B,H,T,D)
        k, v = self.kv(x).view(B, T, 2, Hk, D).permute(2, 0, 3, 1, 4)        # (B,Hk,T,D)
        q, k = self.rope(q, ctx.pos), self.rope(k, ctx.pos)
        if cache is None or not ctx.decode:
            o = attend_blocks(q, k, v, self.scale, causal=True, window=self.window,
                              key_valid=ctx.valid, qchunk=ctx.qchunk, ckpt=ctx.ckpt)
            if cache is not None:
                cache.prefill(k, v)
        else:
            cache.append_one(k, v)
            o = attend(q, cache.k, cache.v, cache.valid_mask(ctx.key_valid), self.scale)
        return self.out(o.transpose(1, 2).reshape(B, T, H * D))
