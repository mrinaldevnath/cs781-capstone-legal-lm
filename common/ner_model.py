"""
ner_model.py - token-classification model on top of a Phase-1 backbone.

    Backbone   loads a Phase-1 checkpoint (final.pt: {"model", "cfg", ...}) and returns contextual states (B, T, d)
               A  -> the bidirectional ENCODER only (the decoder / LM head are dropped)
               B/C/D/E -> the causal trunk (DecoderLM.trunk); the tied LM head is not used
    Heads      linear : dropout -> Linear(d, n_tags)                       (sees only what the backbone sees)
               bilstm : dropout -> BiLSTM(d, h) -> dropout -> Linear(2h, n_tags)
                        gives causal backbones RIGHT context ("JAGE RAM & ORS. ..Appellants": the role comes
                        after the name), at a small cost (~2.6M params with h=256)
    NERModel   backbone + head; forward(ids, valid) -> fp32 logits (B, T, n_tags)
"""
import torch
import torch.nn as nn

from layers import Ctx
from models import build_model


class Backbone(nn.Module):
    def __init__(self, ckpt_path=None, cfg=None, state=None, ckpt_act=True):
        super().__init__()
        if ckpt_path is not None:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)   # Phase-1 files hold cfg/state dicts
            cfg, state = ck["cfg"], ck["model"]
        self.cfg = dict(cfg)
        self.cfg["ckpt"] = ckpt_act
        lm = build_model(self.cfg)
        if state is not None:
            missing, unexpected = lm.load_state_dict(state, strict=False)
            assert not missing and not unexpected, f"checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}"
        self.is_encdec = self.cfg["kind"] == "encdec"
        if self.is_encdec:                         # keep embeddings + encoder only
            del lm.dec
            del lm.norm
        self.lm = lm
        self.d = self.cfg["d"]

    def forward(self, ids, valid=None):
        T = ids.shape[1]
        pos = torch.arange(T, device=ids.device)
        if self.is_encdec:
            return self.lm.encode(ids, valid if (valid is not None and not bool(valid.all())) else None, pos)
        # causal: right padding can never influence real tokens, so no mask is needed
        ctx = Ctx(pos, ckpt=self.lm.ckpt and torch.is_grad_enabled(), qchunk=self.lm.qchunk)
        return self.lm.trunk(ids, ctx)


class LinearHead(nn.Module):
    def __init__(self, d, n_tags, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d, n_tags)

    def forward(self, h, valid=None):
        return self.fc(self.drop(h))


class BiLSTMHead(nn.Module):
    def __init__(self, d, n_tags, hidden=256, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(d, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden, n_tags)

    def forward(self, h, valid=None):
        x = self.drop(h)
        if valid is not None and not bool(valid.all()):
            lens = valid.sum(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        else:
            out, _ = self.lstm(x)
        return self.fc(self.drop(out))


class NERModel(nn.Module):
    def __init__(self, backbone, n_tags, head="bilstm", lstm_hidden=256, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        d = backbone.d
        self.head = BiLSTMHead(d, n_tags, lstm_hidden, dropout) if head == "bilstm" else LinearHead(d, n_tags, dropout)

    def forward(self, ids, valid=None):
        h = self.backbone(ids, valid)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            return self.head(h.float(), valid).float()
