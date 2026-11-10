"""
models.py - assembles the four Phase-1 architectures from the from-scratch parts.

    A  EncDecLM    BART-style encoder-decoder (6 bidirectional MHA enc + 6 causal-MHA/cross-MHA dec)
    B  DecoderLM   12 x causal MHA                                   (GPT baseline)
    C  DecoderLM   12 x [3 local sliding-window GQA + 1 global MHA] -> 3 groups
    D  DecoderLM   12 x [3 Gated DeltaNet + 1 MLA]                  -> 3 groups

Shared skeleton: d=768, SwiGLU 2048, pre-norm RMSNorm, RoPE (softmax-attention layers only), tied
embeddings, no biases. The architecture is selected by a plain dict (see configs/*.json).

Common interface used by train.py / generate.py:
    model(window, split=0) -> mean next-token loss    window: (B, 1025) long. split is used by A only.
    model.shared_metric(window) -> (loss_sum, n_tokens)  loss on window tokens 768..1023 given tokens 0..767
    model.supervised_tokens(B, split) -> number of loss-carrying tokens of a micro-batch
    model.begin(prompt_ids, padlen, max_new) -> (logits, state);  model.advance(tok, state) -> logits
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from layers import RMSNorm, RotaryEmbedding, SwiGLU, Ctx
from attn_mha import CausalMHA
from attn_gqa_local import LocalGQA
from attn_mla import MLA
from gated_deltanet import GatedDeltaNet
from attn_cross import CrossMHA

BOS_ID, EOS_ID, PAD_ID = 1, 2, 0
WINDOW = 1024                  # training window (tokens); the loader returns WINDOW+1
SHARED_PREFIX = 768            # shared metric: predict tokens 768..1023 given 0..767


# ---------------------------------------------------------------------------------- building blocks
class Block(nn.Module):
    """x = x + mixer(norm(x)) [+ cross(norm(x))] ; x = x + ffn(norm(x))"""

    def __init__(self, d, mixer, ffn_hidden, cross=None):
        super().__init__()
        self.n1, self.mixer = RMSNorm(d), mixer
        self.cross = cross
        if cross is not None:
            self.nx = RMSNorm(d)
        self.n2, self.ffn = RMSNorm(d), SwiGLU(d, ffn_hidden)

    def make_cache(self, B, max_len, device, dtype=torch.float16):
        return (self.mixer.make_cache(B, max_len, device, dtype),
                self.cross.make_cache(B, max_len, device, dtype) if self.cross is not None else None)

    def forward(self, x, ctx, cache=None):
        mc, xc = cache if cache is not None else (None, None)
        x = x + self.mixer(self.n1(x), ctx, mc)
        if self.cross is not None:
            x = x + self.cross(self.nx(x), ctx, xc)
        return x + self.ffn(self.n2(x))


def make_mixer(kind, cfg, ropes):
    d, H, D = cfg["d"], cfg["n_heads"], cfg["head_dim"]
    if kind == "mha":
        return CausalMHA(d, H, D, ropes[D])
    if kind == "gqa":
        return LocalGQA(d, H, cfg["n_kv_heads"], D, cfg["window"], ropes[D])
    if kind == "mla":
        m = cfg["mla"]
        return MLA(d, H, m["d_latent"], m["d_nope"], m["d_rope"], m["d_v"], ropes[m["d_rope"]])
    if kind == "gdn":
        g = cfg["gdn"]
        return GatedDeltaNet(d, g["n_heads"], g["head_dim"], g.get("conv", 4), g.get("chunk", 64),
                             g.get("use_delta", True))
    raise ValueError(kind)


def make_ropes(cfg):
    dims = {cfg["head_dim"]}
    if "mla" in cfg:
        dims.add(cfg["mla"]["d_rope"])
    return {dd: RotaryEmbedding(dd, cfg.get("rope_len", 2048), cfg.get("rope_base", 10000.0)) for dd in dims}


def ce_chunked(h, weight, targets, chunk=4096, ckpt=True):
    """Summed cross-entropy of tied-head logits, 4096 tokens at a time (never builds (N,32768) in one go)."""
    def f(hc, w, tc):
        return F.cross_entropy(F.linear(hc, w).float(), tc, reduction="sum")
    tot = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], chunk):
        hc, tc = h[i:i + chunk], targets[i:i + chunk]
        if ckpt and torch.is_grad_enabled():
            tot = tot + checkpoint(f, hc, weight, tc, use_reentrant=False)
        else:
            tot = tot + f(hc, weight, tc)
    return tot


def init_weights(model, n_layers):
    std = 0.02
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=std)
    out_std = std / math.sqrt(2 * n_layers)
    for name, p in model.named_parameters():
        if name.endswith("out.weight") or name.endswith("down.weight"):
            nn.init.normal_(p, mean=0.0, std=out_std)


class GenState:
    pass


class _Base(nn.Module):
    cache_dtype = torch.float16          # dtype of KV caches during generation (tests switch to fp32)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def logits(self, h):
        return F.linear(h, self.emb.weight)

    def to_inference(self):
        """Matrices/embeddings to fp16, 1-D parameters (norms, GDN decay) stay fp32."""
        for p in self.parameters():
            if p.ndim >= 2:
                p.data = p.data.half()
        return self


# ---------------------------------------------------------------------------------- decoder-only (B, C, D)
class DecoderLM(_Base):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["d"]
        self.emb = nn.Embedding(cfg["vocab"], d)
        self.ropes = nn.ModuleDict({str(k): v for k, v in make_ropes(cfg).items()})
        ropes = {int(k): v for k, v in self.ropes.items()}
        self.blocks = nn.ModuleList([Block(d, make_mixer(k, cfg, ropes), cfg["ffn"]) for k in cfg["layout"]])
        self.norm = RMSNorm(d)
        self.qchunk, self.ckpt = cfg.get("qchunk", 256), cfg.get("ckpt", True)
        self.W, self.pref = cfg.get("train_len", WINDOW), cfg.get("shared_prefix", SHARED_PREFIX)
        init_weights(self, len(self.blocks))

    def trunk(self, ids, ctx, caches=None):
        x = self.emb(ids).float()
        for i, blk in enumerate(self.blocks):
            x = blk(x, ctx, None if caches is None else caches[i])
        return self.norm(x)

    def _train_ctx(self, T, device):
        return Ctx(torch.arange(T, device=device), ckpt=self.ckpt, qchunk=self.qchunk)

    def supervised_tokens(self, B, split=0):
        return B * self.W

    def forward(self, window, split=0):
        x, y = window[:, :self.W], window[:, 1:self.W + 1]
        h = self.trunk(x, self._train_ctx(self.W, x.device))
        return ce_chunked(h.reshape(-1, h.shape[-1]), self.emb.weight, y.reshape(-1)) / y.numel()

    @torch.no_grad()
    def shared_metric(self, window):
        x, y = window[:, :self.W], window[:, 1:self.W + 1]
        h = self.trunk(x, self._train_ctx(self.W, x.device))
        a, b = self.pref - 1, self.W - 1                       # positions predicting tokens 768..1023
        hs, ys = h[:, a:b].reshape(-1, h.shape[-1]), y[:, a:b].reshape(-1)
        return ce_chunked(hs, self.emb.weight, ys, ckpt=False), ys.numel()

    # ---- generation
    def new_caches(self, B, max_len, device):
        return [blk.make_cache(B, max_len, device, self.cache_dtype) for blk in self.blocks]

    @torch.no_grad()
    def begin(self, ids, padlen, max_new):
        """ids (B,P) left-padded prompt tokens, padlen (B,) number of pads. Returns (logits (B,V), state)."""
        B, P = ids.shape
        dev = ids.device
        st = GenState()
        st.max_len = P + max_new
        st.padlen = padlen
        ar = torch.arange(st.max_len, device=dev)
        st.key_valid = ar[None, :] >= padlen[:, None]                 # (B, max_len)
        st.caches = self.new_caches(B, st.max_len, dev)
        pos = (ar[None, :P] - padlen[:, None]).clamp(min=0)
        ctx = Ctx(pos, valid=st.key_valid[:, :P], key_valid=st.key_valid, start=0, decode=False,
                  ckpt=False, qchunk=self.qchunk)
        h = self.trunk(ids, ctx, st.caches)
        st.p = P
        return self.logits(h[:, -1]), st

    @torch.no_grad()
    def advance(self, tok, st):
        pos = (st.p - st.padlen)[:, None]
        ctx = Ctx(pos, valid=None, key_valid=st.key_valid, start=st.p, decode=True)
        h = self.trunk(tok[:, None], ctx, st.caches)
        st.p += 1
        return self.logits(h[:, -1])


# ---------------------------------------------------------------------------------- encoder-decoder (A)
class EncDecLM(_Base):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["d"]
        self.emb = nn.Embedding(cfg["vocab"], d)                   # shared by encoder, decoder and output head
        self.ropes = nn.ModuleDict({str(k): v for k, v in make_ropes(cfg).items()})
        rope = self.ropes[str(cfg["head_dim"])]
        H, D = cfg["n_heads"], cfg["head_dim"]
        self.enc = nn.ModuleList([Block(d, CausalMHA(d, H, D, rope, bidirectional=True), cfg["ffn"])
                                  for _ in range(cfg["n_enc"])])
        self.enc_norm = RMSNorm(d)
        self.dec = nn.ModuleList([Block(d, CausalMHA(d, H, D, rope), cfg["ffn"], cross=CrossMHA(d, H, D))
                                  for _ in range(cfg["n_dec"])])
        self.norm = RMSNorm(d)
        self.qchunk, self.ckpt = cfg.get("qchunk", 256), cfg.get("ckpt", True)
        self.W, self.pref = cfg.get("train_len", WINDOW), cfg.get("shared_prefix", SHARED_PREFIX)
        init_weights(self, cfg["n_enc"] + cfg["n_dec"])

    def encode(self, src, src_valid=None, pos=None):
        T = src.shape[1]
        if pos is None:
            pos = torch.arange(T, device=src.device)
        ctx = Ctx(pos, valid=src_valid, ckpt=self.ckpt and torch.is_grad_enabled(), qchunk=self.qchunk)
        x = self.emb(src).float()
        for blk in self.enc:
            x = blk(x, ctx)
        return self.enc_norm(x)

    def _decode(self, dec_in, enc_out):
        T = dec_in.shape[1]
        ctx = Ctx(torch.arange(T, device=dec_in.device), ckpt=self.ckpt, qchunk=self.qchunk, enc_out=enc_out)
        x = self.emb(dec_in).float()
        for blk in self.dec:
            x = blk(x, ctx)
        return self.norm(x)

    def supervised_tokens(self, B, split=0):
        return B * (self.W - split)

    def _hidden_and_targets(self, window, split):
        src = window[:, :split]
        tgt = window[:, split:self.W]                                           # what the decoder must produce
        bos = torch.full_like(tgt[:, :1], BOS_ID)
        dec_in = torch.cat([bos, tgt[:, :-1]], dim=1)
        return self._decode(dec_in, self.encode(src)), tgt

    def forward(self, window, split=512):
        h, tgt = self._hidden_and_targets(window, split)
        return ce_chunked(h.reshape(-1, h.shape[-1]), self.emb.weight, tgt.reshape(-1)) / tgt.numel()

    @torch.no_grad()
    def shared_metric(self, window):
        h, tgt = self._hidden_and_targets(window, self.pref)
        return ce_chunked(h.reshape(-1, h.shape[-1]), self.emb.weight, tgt.reshape(-1), ckpt=False), tgt.numel()

    # ---- generation
    @torch.no_grad()
    def begin(self, ids, padlen, max_new):
        B, P = ids.shape
        dev = ids.device
        st = GenState()
        ar = torch.arange(P, device=dev)
        st.enc_valid = ar[None, :] >= padlen[:, None]
        pos = (ar[None, :] - padlen[:, None]).clamp(min=0)
        st.enc_out = self.encode(ids, st.enc_valid, pos)
        st.caches = [blk.make_cache(B, max_new + 1, dev, self.cache_dtype) for blk in self.dec]
        st.p = 0
        st.B = B
        bos = torch.full((B,), BOS_ID, dtype=torch.long, device=dev)
        return self.advance(bos, st), st

    @torch.no_grad()
    def advance(self, tok, st):
        pos = torch.full((st.B, 1), st.p, dtype=torch.long, device=tok.device)
        ctx = Ctx(pos, decode=True, enc_out=st.enc_out, enc_valid=st.enc_valid)
        x = self.emb(tok[:, None]).float()
        for i, blk in enumerate(self.dec):
            x = blk(x, ctx, st.caches[i])
        st.p += 1
        return self.logits(self.norm(x)[:, -1])


def build_model(cfg):
    return EncDecLM(cfg) if cfg["kind"] == "encdec" else DecoderLM(cfg)
