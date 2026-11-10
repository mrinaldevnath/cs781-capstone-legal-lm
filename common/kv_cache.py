"""
kv_cache.py - cache containers for incremental generation (no torch.cat anywhere: every buffer is
pre-allocated once and written in place).

    StaticCache  pre-allocated (B, H, max_len, D) buffers addressed by a write pointer.
                 Used for: MHA keys/values, MLA latent + RoPE key, model-A decoder self-attention.
    RingCache    fixed-size ring buffer for sliding-window layers: slot = position mod window.
                 Memory is constant no matter how long the sequence gets.
    CrossCache   encoder-side keys/values of model A, computed once per sequence.
    GDNState     Gated-DeltaNet recurrent state (B, H, Dv, Dk) fp32 + conv tail.
"""
import torch


class StaticCache:
    def __init__(self, B, max_len, spec, device, dtype=torch.float16):
        """spec: {name: (n_heads, dim)}"""
        self.buf = {n: torch.zeros(B, h, max_len, d, device=device, dtype=dtype) for n, (h, d) in spec.items()}
        self.len = 0
        self.max_len = max_len

    def append(self, **new):
        T = next(iter(new.values())).shape[2]
        assert self.len + T <= self.max_len, "cache overflow"
        for n, t in new.items():
            self.buf[n][:, :, self.len:self.len + T] = t.to(self.buf[n].dtype)
        self.len += T

    def get(self, name):
        return self.buf[name][:, :, :self.len]


class RingCache:
    """Keeps the most recent `window` keys/values. Keys are stored AFTER RoPE (absolute rotation), so
    the order of the slots is irrelevant for softmax attention."""

    def __init__(self, B, n_kv, window, dim, device, dtype=torch.float16):
        self.k = torch.zeros(B, n_kv, window, dim, device=device, dtype=dtype)
        self.v = torch.zeros(B, n_kv, window, dim, device=device, dtype=dtype)
        self.slot_pos = torch.full((window,), -1, dtype=torch.long, device=device)  # absolute position per slot
        self.window = window
        self.len = 0   # absolute number of tokens seen so far

    def prefill(self, k, v):
        """k, v: (B, n_kv, T, D) for absolute positions 0..T-1 (cache must be empty)."""
        T = k.shape[2]
        m = min(T, self.window)
        pos = torch.arange(T - m, T, device=k.device)
        idx = pos % self.window
        self.k[:, :, idx] = k[:, :, T - m:].to(self.k.dtype)
        self.v[:, :, idx] = v[:, :, T - m:].to(self.v.dtype)
        self.slot_pos[idx] = pos
        self.len = T

    def append_one(self, k, v):
        """k, v: (B, n_kv, 1, D) for absolute position self.len."""
        p = self.len
        i = p % self.window
        self.k[:, :, i] = k[:, :, 0].to(self.k.dtype)
        self.v[:, :, i] = v[:, :, 0].to(self.v.dtype)
        self.slot_pos[i] = p
        self.len += 1

    def valid_mask(self, key_valid):
        """(B, 1, 1, window) bool: slot holds a real (non-pad) token. key_valid: (B, S) over absolute slots or None."""
        ok = self.slot_pos >= 0
        if key_valid is None:
            return ok[None, None, None, :]
        gathered = key_valid[:, self.slot_pos.clamp(min=0)]          # (B, window)
        return (gathered & ok[None, :])[:, None, None, :]


class CrossCache:
    def __init__(self):
        self.k = None
        self.v = None


class GDNState:
    def __init__(self, B, n_heads, dk, dv, conv_ch, conv_w, device):
        self.S = torch.zeros(B, n_heads, dv, dk, device=device, dtype=torch.float32)
        self.conv_tail = torch.zeros(B, conv_ch, conv_w - 1, device=device, dtype=torch.float32)
