"""
attn_core.py - the one place where softmax attention is actually computed (from scratch: matmul,
mask, fp32 softmax, matmul). Every softmax-attention variant (MHA, local GQA, MLA, cross-attention,
MLA absorbed decode) calls it; no torch.nn attention module and no F.scaled_dot_product_attention.

Shapes (Hq query heads, Hk key/value heads, Hq = Hk * G):
    q  (B, Hq, Tq, D)      k  (B, Hk, Tk, D)      v  (B, Hk, Tk, Dv)
    optional second score term (used by MLA for the decoupled RoPE part):
    q2 (B, Hq, Tq, D2)     k2 (B, Hk2, Tk, D2)    Hk2 in {Hk, 1}
    scores = scale * (q k^T [+ q2 k2^T])
GQA/MQA without ever copying K/V: the G query heads that share a KV head are folded into the
query-length axis (B, Hk, G*Tq, D), so a plain batched matmul against (B, Hk, D, Tk) does the work.

Memory: scores of a whole 1024x1024 head would be stored for backward by autograd (12 layers x B x
12 heads x 1024^2 fp32 ...). Instead `attend_blocks` processes queries in blocks and (training) wraps
every block in activation checkpointing, so only q/k/v are kept and the block is recomputed in backward.
It also skips keys that can never be attended (causal / sliding window), which makes local layers cheap.
"""
import torch
from torch.utils.checkpoint import checkpoint

NEG = -1e30   # finite "minus infinity": fully masked rows (left padding) stay NaN-free


def _scores(q, k):
    """q (B,Hq,Tq,D), k (B,Hk,Tk,D) -> (B,Hq,Tq,Tk); K is shared by the G query heads of a group (no copy)."""
    B, Hq, Tq, D = q.shape
    Hk, Tk = k.shape[1], k.shape[2]
    G = Hq // Hk
    s = torch.matmul(q.reshape(B, Hk, G * Tq, D), k.transpose(-1, -2))      # (B,Hk,G*Tq,Tk)
    return s.reshape(B, Hq, Tq, Tk)


def attend(q, k, v, mask, scale, q2=None, k2=None):
    """Plain softmax attention. mask: bool broadcastable to (B,1,Tq,Tk), True = may attend (or None).
    Returns (B, Hq, Tq, Dv) in q's dtype. Softmax in fp32."""
    B, Hq, Tq, _ = q.shape
    Hk, Tk, Dv = k.shape[1], k.shape[2], v.shape[3]
    G = Hq // Hk
    s = _scores(q * scale, k)                       # scale folded into q: keeps fp16 scores far from overflow
    if q2 is not None:
        s = s + _scores(q2 * scale, k2)
    s = s if s.dtype == torch.float64 else s.float()
    if mask is not None:
        s = s.masked_fill(~mask, NEG)
    p = torch.softmax(s, dim=-1).to(v.dtype)                                 # (B,Hq,Tq,Tk)
    o = torch.matmul(p.reshape(B, Hk, G * Tq, Tk), v)                         # (B,Hk,G*Tq,Dv)
    return o.reshape(B, Hq, Tq, Dv)


def attend_blocks(q, k, v, scale, causal=True, window=None, key_valid=None,
                  q2=None, k2=None, qchunk=256, ckpt=False):
    """Self/cross attention over a whole sequence, query-block by query-block.

    causal=True assumes query i sits at the same absolute slot as key i (Tq == Tk, start 0).
    window=W (with causal) lets query i see keys i-W+1 .. i.   key_valid: (B, Tk) bool or None.
    causal=False: every query sees every (valid) key  (encoder, cross-attention).
    """
    Tq, Tk = q.shape[2], k.shape[2]
    dev = q.device
    outs = []
    for s0 in range(0, Tq, qchunk):
        e0 = min(s0 + qchunk, Tq)
        klo, khi = 0, Tk
        m = None
        if causal:
            khi = e0
            if window is not None:
                klo = max(0, s0 - window + 1)
            qi = torch.arange(s0, e0, device=dev)[:, None]
            kj = torch.arange(klo, khi, device=dev)[None, :]
            m = kj <= qi
            if window is not None:
                m = m & (kj > qi - window)
            m = m[None, None]                                                # (1,1,tq,tk)
        if key_valid is not None:
            kvm = key_valid[:, None, None, klo:khi]                          # (B,1,1,tk)
            m = kvm if m is None else (m & kvm)
        qc = q[:, :, s0:e0]
        kc, vc = k[:, :, klo:khi], v[:, :, klo:khi]
        q2c = q2[:, :, s0:e0] if q2 is not None else None
        k2c = k2[:, :, klo:khi] if k2 is not None else None
        if ckpt and torch.is_grad_enabled():
            o = checkpoint(attend, qc, kc, vc, m, scale, q2c, k2c, use_reentrant=False)
        else:
            o = attend(qc, kc, vc, m, scale, q2c, k2c)
        outs.append(o)
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
