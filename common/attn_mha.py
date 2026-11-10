"""
attn_mha.py - multi-head self-attention, written from scratch (models A encoder/decoder, B, and the
global layers of C).

    CausalMHA(bidirectional=False)   decoder-style causal attention with RoPE and a KV cache
    CausalMHA(bidirectional=True)    encoder-style (model A): every token sees every real token, no cache

Fused QKV projection -> RoPE on q,k -> attention core -> output projection.
Cache layout: StaticCache {"k": (B,H,L,D), "v": (B,H,L,D)}, keys stored after RoPE.
Decode cost per token per layer: reads 2*H*D*len values (3072 bytes/token/layer in fp16).
"""
import math
import torch
import torch.nn as nn

from attn_core import attend, attend_blocks
from kv_cache import StaticCache


class CausalMHA(nn.Module):
    def __init__(self, d, n_heads, head_dim, rope, bidirectional=False):
        super().__init__()
        self.H, self.D, self.rope, self.bidirectional = n_heads, head_dim, rope, bidirectional
        self.qkv = nn.Linear(d, 3 * n_heads * head_dim, bias=False)
        self.out = nn.Linear(n_heads * head_dim, d, bias=False)
        self.scale = 1.0 / math.sqrt(head_dim)

    @staticmethod
    def cache_spec(n_heads, head_dim):
        return {"k": (n_heads, head_dim), "v": (n_heads, head_dim)}

    def make_cache(self, B, max_len, device, dtype=torch.float16):
        return StaticCache(B, max_len, {"k": (self.H, self.D), "v": (self.H, self.D)}, device, dtype)

    def forward(self, x, ctx, cache=None):
        B, T, _ = x.shape
        H, D = self.H, self.D
        q, k, v = self.qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)    # each (B,H,T,D)
        q, k = self.rope(q, ctx.pos), self.rope(k, ctx.pos)
        if cache is None or not ctx.decode:
            # training / prefill / encoder: attend over the current chunk itself
            o = attend_blocks(q, k, v, self.scale, causal=not self.bidirectional,
                              key_valid=ctx.valid, qchunk=ctx.qchunk, ckpt=ctx.ckpt)
            if cache is not None:
                cache.append(k=k, v=v)
        else:
            cache.append(k=k, v=v)                                           # one new token
            n = cache.len
            mask = None if ctx.key_valid is None else ctx.key_valid[:, None, None, :n]
            o = attend(q, cache.get("k"), cache.get("v"), mask, self.scale)
        return self.out(o.transpose(1, 2).reshape(B, T, H * D))
