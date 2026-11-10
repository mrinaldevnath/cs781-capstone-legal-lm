"""
attn_mla.py - Multi-head Latent Attention (DeepSeek-V2 style), from scratch (model D, 1 layer in 4).

Keys/values are NOT cached per head. Each token stores one small latent vector c = RMSNorm(W_dkv x)
(dc = 256) plus a shared decoupled RoPE key k_r (dr = 32) -> 288 values/token/layer (576 B at fp16)
versus 1536 values for MHA with 12x64 heads.

    q       = W_q x                       per head: [q_nope (64) | q_rope (32)]
    k_nope  = W_uk c   (per head, 64)     v = W_uv c   (per head, 64)
    k_rope  = RoPE(W_kr x)  shared by all heads (32)
    score   = (q_nope . k_nope + q_rope . k_rope) / sqrt(64 + 32)

Training / prefill use the plain form above (K, V materialised from c).
Decode uses WEIGHT ABSORPTION so the cached latent is used directly:
    q_nope . (W_uk[h] c)  =  (q_nope W_uk[h]) . c     -> query lives in latent space (256-d)
    output  o_h = (sum_t p_t c_t) W_uv[h]^T           -> value up-projection applied after the sum
In absorbed form all 12 heads attend over ONE shared "KV head" (the latent), which is exactly the
MQA path of attn_core (Hk = 1, G = 12). tests.py checks absorbed decode == plain forward.
"""
import math
import torch
import torch.nn as nn

from attn_core import attend, attend_blocks
from layers import RMSNorm
from kv_cache import StaticCache


class MLA(nn.Module):
    def __init__(self, d, n_heads, d_latent, d_nope, d_rope, d_v, rope):
        super().__init__()
        self.H, self.dc, self.dn, self.dr, self.dv, self.rope = n_heads, d_latent, d_nope, d_rope, d_v, rope
        self.wq = nn.Linear(d, n_heads * (d_nope + d_rope), bias=False)
        self.w_dkv = nn.Linear(d, d_latent + d_rope, bias=False)      # latent | shared rope key
        self.kv_norm = RMSNorm(d_latent)
        self.w_uk = nn.Linear(d_latent, n_heads * d_nope, bias=False)
        self.w_uv = nn.Linear(d_latent, n_heads * d_v, bias=False)
        self.out = nn.Linear(n_heads * d_v, d, bias=False)
        self.scale = 1.0 / math.sqrt(d_nope + d_rope)

    @staticmethod
    def cache_spec(d_latent, d_rope):
        return {"c": (1, d_latent), "r": (1, d_rope)}

    def make_cache(self, B, max_len, device, dtype=torch.float16):
        return StaticCache(B, max_len, {"c": (1, self.dc), "r": (1, self.dr)}, device, dtype)

    def _project(self, x, ctx):
        B, T, _ = x.shape
        H, dn, dr, dc = self.H, self.dn, self.dr, self.dc
        q = self.wq(x).view(B, T, H, dn + dr).transpose(1, 2)            # (B,H,T,dn+dr)
        q_nope, q_rope = q[..., :dn], self.rope(q[..., dn:], ctx.pos)
        ckr = self.w_dkv(x)
        c = self.kv_norm(ckr[..., :dc])                                  # (B,T,dc)
        k_rope = self.rope(ckr[..., dc:].unsqueeze(1), ctx.pos)          # (B,1,T,dr)
        return q_nope, q_rope, c, k_rope

    def forward(self, x, ctx, cache=None):
        B, T, _ = x.shape
        H, dn, dv, dc = self.H, self.dn, self.dv, self.dc
        q_nope, q_rope, c, k_rope = self._project(x, ctx)
        if cache is None or not ctx.decode:
            # ---- plain form: materialise per-head K/V from the latent
            k_nope = self.w_uk(c).view(B, T, H, dn).transpose(1, 2)
            v = self.w_uv(c).view(B, T, H, dv).transpose(1, 2)
            o = attend_blocks(q_nope, k_nope, v, self.scale, causal=True, key_valid=ctx.valid,
                              q2=q_rope, k2=k_rope, qchunk=ctx.qchunk, ckpt=ctx.ckpt)
            if cache is not None:
                cache.append(c=c.unsqueeze(1), r=k_rope)
        else:
            # ---- absorbed form: attend directly over the cached latent
            cache.append(c=c.unsqueeze(1), r=k_rope)
            n = cache.len
            C, R = cache.get("c"), cache.get("r")                        # (B,1,n,dc), (B,1,n,dr)
            w_uk = self.w_uk.weight.view(H, dn, dc)
            w_uv = self.w_uv.weight.view(H, dv, dc)
            q_abs = torch.einsum("bhtn,hnc->bhtc", q_nope, w_uk.to(q_nope.dtype))      # (B,H,1,dc)
            mask = None if ctx.key_valid is None else ctx.key_valid[:, None, None, :n]
            o_lat = attend(q_abs, C, C, mask, self.scale, q2=q_rope, k2=R)             # (B,H,1,dc)
            o = torch.einsum("bhtc,hvc->bhtv", o_lat, w_uv.to(o_lat.dtype))            # (B,H,1,dv)
        return self.out(o.transpose(1, 2).reshape(B, T, H * dv))
