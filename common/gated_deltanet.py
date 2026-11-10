"""
gated_deltanet.py - Gated DeltaNet token mixer (linear-recurrent / state-space style layer), from scratch.
Model D uses it for 3 of every 4 layers.

Per head the layer keeps a matrix state S (Dv x Dk). For every token t:
    S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T)  +  beta_t v_t k_t^T          (gated delta rule)
    o_t = S_t q_t
alpha_t = exp(-exp(A_log) * softplus(a_t + dt_bias)) in (0,1) is a data-dependent forget gate,
beta_t = sigmoid(b_t) in (0,1) is the "write strength"; q, k are L2-normalised.
The delta term (I - beta k k^T) first erases what S already stores for key k, then writes v:
that is what makes it a real memory update and not just decayed linear attention.

Around the recurrence (all from scratch):  x -> [q|k|v] projection -> depthwise causal conv(4) + SiLU
-> recurrence -> per-head RMSNorm -> * SiLU(output gate) -> output projection.
No RoPE: position information comes from the decay and the conv.

Training/prefill use the CHUNKWISE form (chunks of `chunk` tokens): everything that does not depend
on the carried state is computed for all chunks at once with batched matmuls and one batched
triangular solve; only a short loop over chunks (16 for 1024 tokens) carries the state.
Derivation (g_i = cumulative log-alpha inside the chunk, gamma_i = exp(g_i), S0 = state entering chunk):
    u_j = beta_j ( v_j - gamma_j S0 k_j - sum_{l<j} (gamma_j/gamma_l) (k_l.k_j) u_l )
        =>  U = (I + A)^-1 diag(beta) (V - diag(gamma) K S0^T),   A_jl = beta_j (gamma_j/gamma_l) k_j.k_l  (l<j)
    O   = diag(gamma) Q S0^T + (M * Q K^T) U,                    M_il = gamma_i/gamma_l (l<=i)
    S_C = gamma_C S0 + U^T (K * gamma_C/gamma)
Only ratios gamma_i/gamma_l <= 1 are formed, so it is numerically stable. Decode uses the plain recurrence
(constant memory: one Dv x Dk matrix per head + a 3-column conv tail). tests.py checks chunked == recurrent
(forward and gradients). `use_delta=False` drops the delta term (decay-only gated linear attention) - the
documented fallback.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from layers import RMSNorm
from kv_cache import GDNState


def _l2(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gdn_recurrent(q, k, v, beta, logal, S0=None, delta=True):
    """Reference step-by-step recurrence (slow; used for decode-style testing).
    q,k (B,H,T,Dk) v (B,H,T,Dv) beta, logal (B,H,T), all fp32. Returns o (B,H,T,Dv), S (B,H,Dv,Dk)."""
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    S = q.new_zeros(B, H, Dv, Dk) if S0 is None else S0
    outs = []
    for t in range(T):
        a = torch.exp(logal[:, :, t])[..., None, None]
        b = beta[:, :, t][..., None, None]
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        Sk = torch.einsum("bhvk,bhk->bhv", S, kt) if delta else 0.0
        S = a * (S - b * Sk[..., None] * kt[..., None, :] if delta else S) + b * vt[..., None] * kt[..., None, :]
        outs.append(torch.einsum("bhvk,bhk->bhv", S, qt))
    return torch.stack(outs, dim=2), S


def gdn_step(S, q, k, v, a, b, delta=True):
    """One recurrence step (decode). S (B,H,Dv,Dk); q,k (B,H,Dk); v (B,H,Dv); a,b (B,H,1,1). fp32, no autocast."""
    with torch.autocast(device_type=S.device.type, enabled=False):
        if delta:
            Sk = torch.einsum("bhvk,bhk->bhv", S, k)
            S = a * (S - b * Sk[..., None] * k[..., None, :]) + b * v[..., None] * k[..., None, :]
        else:
            S = a * S + b * v[..., None] * k[..., None, :]
        return S, torch.einsum("bhvk,bhk->bhv", S, q)


def gdn_chunked(q, k, v, beta, logal, S0=None, chunk=64, use_delta=True):
    """Chunkwise-parallel form of the same recurrence (see module docstring). Inputs fp32.
    Autocast is switched off inside: the state math must stay in fp32."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _gdn_chunked(q, k, v, beta, logal, S0, chunk, use_delta)


def _gdn_chunked(q, k, v, beta, logal, S0, chunk, use_delta):
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    C = chunk
    pad = (-T) % C
    if pad:   # padded positions: beta = 0 and alpha = 1  ->  state untouched
        q, k, v = F.pad(q, (0, 0, 0, pad)), F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))
        beta, logal = F.pad(beta, (0, pad)), F.pad(logal, (0, pad))
    N = (T + pad) // C
    q, k, v = q.reshape(B, H, N, C, Dk), k.reshape(B, H, N, C, Dk), v.reshape(B, H, N, C, Dv)
    beta, logal = beta.reshape(B, H, N, C), logal.reshape(B, H, N, C)
    g = torch.cumsum(logal, dim=-1)                                       # (B,H,N,C)
    gam = torch.exp(g)
    lower = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()     # i >= l
    diff = g[..., :, None] - g[..., None, :]
    Dm = torch.exp(diff.masked_fill(~lower, float("-inf")))                # gamma_i/gamma_l for l<=i else 0
    P = Dm * torch.matmul(q, k.transpose(-1, -2))                          # (M * Q K^T)
    if use_delta:
        strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
        A = (beta[..., :, None] * Dm * torch.matmul(k, k.transpose(-1, -2))).masked_fill(~strict, 0.0)
        eye = torch.eye(C, device=q.device, dtype=q.dtype)
        Tm = torch.linalg.solve_triangular(eye + A, eye.expand(A.shape), upper=False)   # (I+A)^-1
        Ut = torch.matmul(Tm, beta[..., None] * v)                          # (B,H,N,C,Dv)
        W = torch.matmul(Tm, (beta * gam)[..., None] * k)                   # (B,H,N,C,Dk)
    else:
        Ut, W = beta[..., None] * v, None
    Qg = gam[..., None] * q
    Kd = k * torch.exp(g[..., -1:] - g)[..., None]
    dC = torch.exp(g[..., -1])                                              # (B,H,N)
    S = q.new_zeros(B, H, Dv, Dk) if S0 is None else S0
    outs = []
    for n in range(N):
        St = S.transpose(-1, -2)                                            # (B,H,Dk,Dv)
        U = Ut[:, :, n] if W is None else Ut[:, :, n] - torch.matmul(W[:, :, n], St)
        outs.append(torch.matmul(Qg[:, :, n], St) + torch.matmul(P[:, :, n], U))
        S = dC[:, :, n, None, None] * S + torch.matmul(U.transpose(-1, -2), Kd[:, :, n])
    o = torch.stack(outs, dim=2).reshape(B, H, N * C, Dv)[:, :, :T]
    return o, S


class GatedDeltaNet(nn.Module):
    def __init__(self, d, n_heads, head_dim, conv_width=4, chunk=64, use_delta=True):
        super().__init__()
        self.H, self.Dh, self.W, self.chunk, self.use_delta = n_heads, head_dim, conv_width, chunk, use_delta
        C = 3 * n_heads * head_dim
        self.C = C
        self.qkv = nn.Linear(d, C, bias=False)
        self.conv = nn.Conv1d(C, C, conv_width, groups=C, bias=False)       # depthwise, causal via left padding
        self.g_proj = nn.Linear(d, n_heads * head_dim, bias=False)          # output gate
        self.b_proj = nn.Linear(d, n_heads, bias=False)                     # beta
        self.a_proj = nn.Linear(d, n_heads, bias=False)                     # decay input
        A = torch.empty(n_heads).uniform_(1.0, 16.0)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(n_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))      # inverse softplus
        self.o_norm = RMSNorm(head_dim)
        self.out = nn.Linear(n_heads * head_dim, d, bias=False)

    def make_cache(self, B, max_len, device, dtype=torch.float16):
        return GDNState(B, self.H, self.Dh, self.Dh, self.C, self.W, device)

    def _gates(self, x):
        beta = torch.sigmoid(self.b_proj(x).float())                                       # (B,T,H)
        logal = -torch.exp(self.A_log.float()) * F.softplus(self.a_proj(x).float() + self.dt_bias.float())
        return beta.transpose(1, 2), logal.transpose(1, 2)                                  # (B,H,T)

    def _split(self, y):
        """y (B,T,3*H*Dh) fp32 -> q,k,v each (B,H,T,Dh); q scaled, q/k L2-normalised."""
        B, T, _ = y.shape
        q, k, v = y.reshape(B, T, 3, self.H, self.Dh).permute(2, 0, 3, 1, 4)
        return _l2(q) * (self.Dh ** -0.5), _l2(k), v

    def forward(self, x, ctx, cache=None):
        B, T, _ = x.shape
        H, Dh, W = self.H, self.Dh, self.W
        qkv = self.qkv(x)                                                    # (B,T,C)
        if ctx.valid is not None:                                            # left padding: pads contribute nothing
            qkv = qkv * ctx.valid[..., None].to(qkv.dtype)
        beta, logal = self._gates(x)
        if ctx.valid is not None:
            vf = ctx.valid[:, None, :].float()
            beta, logal = beta * vf, logal * vf
        if cache is not None and ctx.decode:
            xt = qkv[:, 0, :].float()                                        # (B,C)
            win = torch.cat([cache.conv_tail, xt[:, :, None]], dim=2)         # (B,C,W)  (tiny tensor)
            y = F.silu((win * self.conv.weight[:, 0, :].float()[None]).sum(-1))
            cache.conv_tail = win[:, :, 1:]
            q, k, v = self._split(y[:, None, :])                             # (B,H,1,Dh)
            S = cache.S
            a = torch.exp(logal[:, :, 0])[..., None, None]
            b = beta[:, :, 0][..., None, None]
            S, ot = gdn_step(S, q[:, :, 0], k[:, :, 0], v[:, :, 0], a, b, self.use_delta)
            cache.S = S
            o = ot[:, :, None, :]                                            # (B,H,1,Dh)
        else:
            xc = qkv.transpose(1, 2)                                         # (B,C,T)
            y = F.silu(self.conv(F.pad(xc, (W - 1, 0))))
            q, k, v = self._split(y.transpose(1, 2).float())
            if ctx.ckpt and torch.is_grad_enabled():
                o, S = checkpoint(gdn_chunked, q, k, v, beta, logal, None, self.chunk, self.use_delta,
                                  use_reentrant=False)
            else:
                o, S = gdn_chunked(q, k, v, beta, logal, None, self.chunk, self.use_delta)
            if cache is not None:                                            # prefill: keep state + conv tail
                cache.S = S.detach()
                tail = xc[:, :, -(W - 1):].float()
                if tail.shape[2] < W - 1:
                    tail = F.pad(tail, (W - 1 - tail.shape[2], 0))
                cache.conv_tail = tail.detach()
        o = self.o_norm(o.transpose(1, 2))                                   # (B,T,H,Dh) fp32 per-head RMSNorm
        gate = F.silu(self.g_proj(x).float()).view(B, T, H, Dh)
        return self.out((o * gate).reshape(B, T, H * Dh).to(x.dtype))
