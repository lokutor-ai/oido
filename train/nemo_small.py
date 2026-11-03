"""Standalone PyTorch reimplementation of NVIDIA NeMo stt_en_conformer_ctc_small (13 M params, CC-BY-4.0),
numerically matched to NeMo, used as the reference for the ESP32 port (no NeMo dependency).

Pipeline: preemph 0.97 -> centered STFT (n_fft 512, hop 160, win 400 symmetric Hann, zero pad) -> |X|^2 -> slaney mel
(80) -> log(x + 2^-24) -> per-utterance per-feature mean/std -> striding conv subsampling (2x conv3x3 s2, 176 ch)
-> 16 Conformer layers (rel-pos MHSA with untied biases, conv k=31 + BatchNorm, macaron FF) -> 1x1 conv -> CTC
(blank = last index).

"Oido v2" variants (cfg keys sub_ch / stack / n_layers; NemoSmall.convert() derives them from a trained model):
  - light front end: conv0 / conv2 with sub_ch (e.g. 64) channels instead of 176, sub_out = Linear(sub_ch * 20 -> d);
  - frame stacking (stack = 2): after sub_out * sqrt(d), pairs of consecutive 40 ms frames are concatenated and projected
    2d -> d (x'[i] = W [x[2i]; x[2i+1]] + b, odd tail paired with zeros, lens' = ceil(lens / 2)); everything downstream
    runs at 12.5 frames/s;
  - fewer layers (n_layers).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NemoPre(nn.Module):
    def __init__(self, window, fb):
        super().__init__()
        self.register_buffer("window", window.float())      # (400,)
        self.register_buffer("fb", fb.float().reshape(80, 257))

    def forward(self, wav):
        """wav (N,) float -> (T, 80) normalized features, T = floor(N/160) + 1 (last frame zeroed)."""
        x = torch.cat([wav[:1], wav[1:] - 0.97 * wav[:-1]])
        spec = torch.stft(x, n_fft=512, hop_length=160, win_length=400, window=self.window, center=True,
                          pad_mode="constant", return_complex=True)
        p = spec.real.square() + spec.imag.square()          # (257, T)
        mel = torch.log(self.fb @ p + 2.0 ** -24)            # (80, T)
        n = wav.shape[0] // 160
        mean = mel[:, :n].mean(1, keepdim=True)
        std = torch.sqrt(((mel[:, :n] - mean) ** 2).sum(1, keepdim=True) / (n - 1)) + 1e-5
        out = (mel - mean) / std
        out[:, n:] = 0.0
        return out.t()


def rel_pos_emb(T, d):
    pos = torch.arange(T - 1, -T, -1, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
    pe = torch.zeros(2 * T - 1, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # row m <-> relative position (T-1-m)


class Layer(nn.Module):
    def __init__(self, d=176, h=4, ff=704, k=31):
        super().__init__()
        self.h, self.dk = h, d // h
        self.norm_feed_forward1, self.norm_self_att, self.norm_conv = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.norm_feed_forward2, self.norm_out = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff1_l1, self.ff1_l2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.ff2_l1, self.ff2_l2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.linear_q, self.linear_k, self.linear_v, self.linear_out = [nn.Linear(d, d) for _ in range(4)]
        self.linear_pos = nn.Linear(d, d, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(h, self.dk))
        self.pos_bias_v = nn.Parameter(torch.zeros(h, self.dk))
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)   # BatchNorm folded in
        self.pw2 = nn.Linear(d, d)

    def attn(self, x, pe):
        T, d = x.shape
        q = self.linear_q(x).view(T, self.h, self.dk)
        k = self.linear_k(x).view(T, self.h, self.dk).transpose(0, 1)
        v = self.linear_v(x).view(T, self.h, self.dk).transpose(0, 1)
        p = self.linear_pos(pe).view(2 * T - 1, self.h, self.dk).transpose(0, 1)  # (H, 2T-1, dk)
        ac = (q + self.pos_bias_u).transpose(0, 1) @ k.transpose(1, 2)           # (H, T, T)
        bd = (q + self.pos_bias_v).transpose(0, 1) @ p.transpose(1, 2)           # (H, T, 2T-1)
        # rel_shift: score[i, j] uses relative position (i - j) -> pe row (T-1) - (i-j)
        idx = (T - 1) - torch.arange(T)[:, None] + torch.arange(T)[None, :]
        bd = torch.gather(bd, 2, idx[None].expand(self.h, T, T))
        a = torch.softmax((ac + bd) / math.sqrt(self.dk), -1)
        return self.linear_out((a @ v).transpose(0, 1).reshape(T, d))

    def forward(self, x, pe):
        x = x + 0.5 * self.ff1_l2(F.silu(self.ff1_l1(self.norm_feed_forward1(x))))
        x = x + self.attn(self.norm_self_att(x), pe)
        c = F.glu(self.pw1(self.norm_conv(x)), dim=-1)
        c = F.silu(self.dw(c.t()[None])[0].t())
        x = x + self.pw2(c)
        x = x + 0.5 * self.ff2_l2(F.silu(self.ff2_l1(self.norm_feed_forward2(x))))
        return self.norm_out(x)


F2 = 20  # frequency positions after the two stride-2 convs on 80 mel bins


def cfg_from_state(sd):
    """Architecture of a NemoSmall state dict (own format: conv0 / conv2 / sub_out / layers / head / stack_proj)."""
    n = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layers."))
    return dict(d=sd["sub_out.weight"].shape[0], h=sd["layers.0.pos_bias_u"].shape[0],
                ff=sd["layers.0.ff1_l1.weight"].shape[0], k=sd["layers.0.dw.weight"].shape[-1],
                vocab=sd["head.weight"].shape[0] - 1, sub_ch=sd["conv0.weight"].shape[0],
                stack=2 if "stack_proj.weight" in sd else 1, n_layers=n)


class NemoSmall(nn.Module):
    """Any NeMo Conformer-CTC with striding subsampling; sizes are read from the state dict (small: d=176, 16 layers,
    4 heads, FF 704; large: d=512, 18 layers, 8 heads, FF 2048).

    NemoSmall(sd)      builds the model from an NVIDIA state dict (sd) with NVIDIA's weights;
    NemoSmall(cfg=cfg) builds an empty model of that architecture (then load_state_dict from a checkpoint)."""
    def __init__(self, sd=None, cfg=None):
        super().__init__()
        if sd is not None:
            C = sd["encoder.pre_encode.conv.0.weight"].shape[0]
            d, sub_in = sd["encoder.pre_encode.out.weight"].shape
            n = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers."))
            cfg = dict(d=d, h=sd["encoder.layers.0.self_attn.pos_bias_u"].shape[0],
                       ff=sd["encoder.layers.0.feed_forward1.linear1.weight"].shape[0],
                       k=sd["encoder.layers.0.conv.depthwise_conv.weight"].shape[-1],
                       vocab=sd["decoder.decoder_layers.0.weight"].shape[0] - 1, sub_ch=C, stack=1, n_layers=n)
            assert sub_in == C * F2
            window, fb = sd["preprocessor.featurizer.window"], sd["preprocessor.featurizer.fb"]
        else:
            cfg = dict(cfg)
            window, fb = torch.zeros(400), torch.zeros(80, 257)    # filled by load_state_dict
        self.cfg = cfg
        C, d = cfg["sub_ch"], cfg["d"]
        self.d, self.vocab, self.stack = d, cfg["vocab"], cfg.get("stack", 1)  # blank = last index
        self.pre = NemoPre(window, fb)
        self.conv0 = nn.Conv2d(1, C, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(C, C, 3, stride=2, padding=1)
        self.sub_out = nn.Linear(C * F2, d)
        if self.stack == 2:
            self.stack_proj = nn.Linear(2 * d, d)
            with torch.no_grad():                       # average of the two frames
                self.stack_proj.weight.copy_(torch.cat([0.5 * torch.eye(d), 0.5 * torch.eye(d)], 1))
                self.stack_proj.bias.zero_()
        self.layers = nn.ModuleList([Layer(d, cfg["h"], cfg["ff"], cfg["k"]) for _ in range(cfg["n_layers"])])
        self.head = nn.Linear(d, self.vocab + 1)
        if sd is not None:
            self.load_nemo(sd)

    @staticmethod
    def from_ckpt(st):
        """Model from a checkpoint dict of this code ('model' state dict, optional 'cfg'); also loads old-format ones."""
        sd = st["model"] if "model" in st else st
        m = NemoSmall(cfg=st.get("cfg") or cfg_from_state(sd))
        m.load_state_dict(sd)
        return m

    def stack_frames(self, x):
        """(..., T, d) 40 ms frames -> (..., ceil(T / 2), d) 80 ms frames; an odd tail frame is paired with zeros."""
        if self.stack != 2:
            return x
        if x.shape[-2] % 2:
            x = F.pad(x, (0, 0, 0, 1))
        x = x.reshape(*x.shape[:-2], x.shape[-2] // 2, 2 * x.shape[-1])
        return F.linear(x, self.stack_proj.weight, self.stack_proj.bias)

    @torch.no_grad()
    def convert(self, sub_ch=0, stack=1, layers=None):
        """New model from this (stack 1) one: keep the sub_ch most important front-end channels (ranked by the L2 norm
        of the sub_out columns of each conv2 output channel, sub_out input index = c * F2 + f; the same channels are
        kept in conv0 and as conv2 inputs), add frame stacking (init: average of the pair), keep only the listed layers
        (list of indices, default all)."""
        assert self.stack == 1
        C = self.conv0.weight.shape[0]
        sub_ch = sub_ch or C
        layers = list(range(len(self.layers))) if layers is None else list(layers)
        n = NemoSmall(cfg=dict(self.cfg, sub_ch=sub_ch, stack=stack, n_layers=len(layers)))
        n.pre.load_state_dict(self.pre.state_dict())
        if sub_ch == C:
            sel = torch.arange(C)
        else:
            score = self.sub_out.weight.view(self.d, C, F2).norm(dim=(0, 2))
            sel = score.topk(sub_ch).indices.sort().values
        n.conv0.weight.copy_(self.conv0.weight[sel]); n.conv0.bias.copy_(self.conv0.bias[sel])
        n.conv2.weight.copy_(self.conv2.weight[sel][:, sel]); n.conv2.bias.copy_(self.conv2.bias[sel])
        n.sub_out.weight.copy_(self.sub_out.weight.view(self.d, C, F2)[:, sel].reshape(self.d, sub_ch * F2))
        n.sub_out.bias.copy_(self.sub_out.bias)
        for j, i in enumerate(layers):
            n.layers[j].load_state_dict(self.layers[i].state_dict())
        n.head.load_state_dict(self.head.state_dict())
        return n

    @torch.no_grad()
    def load_nemo(self, sd):
        g = lambda k: sd[k].float()
        self.conv0.weight.copy_(g("encoder.pre_encode.conv.0.weight")); self.conv0.bias.copy_(g("encoder.pre_encode.conv.0.bias"))
        self.conv2.weight.copy_(g("encoder.pre_encode.conv.2.weight")); self.conv2.bias.copy_(g("encoder.pre_encode.conv.2.bias"))
        self.sub_out.weight.copy_(g("encoder.pre_encode.out.weight")); self.sub_out.bias.copy_(g("encoder.pre_encode.out.bias"))
        for i, L in enumerate(self.layers):
            p = f"encoder.layers.{i}."
            for n in ["norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"]:
                getattr(L, n).weight.copy_(g(p + n + ".weight")); getattr(L, n).bias.copy_(g(p + n + ".bias"))
            for a, b in [("ff1_l1", "feed_forward1.linear1"), ("ff1_l2", "feed_forward1.linear2"),
                         ("ff2_l1", "feed_forward2.linear1"), ("ff2_l2", "feed_forward2.linear2"),
                         ("linear_q", "self_attn.linear_q"), ("linear_k", "self_attn.linear_k"),
                         ("linear_v", "self_attn.linear_v"), ("linear_out", "self_attn.linear_out")]:
                getattr(L, a).weight.copy_(g(p + b + ".weight")); getattr(L, a).bias.copy_(g(p + b + ".bias"))
            L.linear_pos.weight.copy_(g(p + "self_attn.linear_pos.weight"))
            L.pos_bias_u.copy_(g(p + "self_attn.pos_bias_u")); L.pos_bias_v.copy_(g(p + "self_attn.pos_bias_v"))
            L.pw1.weight.copy_(g(p + "conv.pointwise_conv1.weight")[:, :, 0]); L.pw1.bias.copy_(g(p + "conv.pointwise_conv1.bias"))
            L.pw2.weight.copy_(g(p + "conv.pointwise_conv2.weight")[:, :, 0]); L.pw2.bias.copy_(g(p + "conv.pointwise_conv2.bias"))
            s = g(p + "conv.batch_norm.weight") / torch.sqrt(g(p + "conv.batch_norm.running_var") + 1e-5)
            L.dw.weight.copy_(g(p + "conv.depthwise_conv.weight") * s[:, None, None])
            L.dw.bias.copy_((g(p + "conv.depthwise_conv.bias") - g(p + "conv.batch_norm.running_mean")) * s
                            + g(p + "conv.batch_norm.bias"))
        self.head.weight.copy_(g("decoder.decoder_layers.0.weight")[:, :, 0]); self.head.bias.copy_(g("decoder.decoder_layers.0.bias"))

    def encode(self, feats):
        """feats (T, 80) -> encoder output (T', 176); T' = ceil(T'' / 2) with frame stacking."""
        x = F.relu(self.conv0(feats[None, None]))
        x = F.relu(self.conv2(x))                       # (1, sub_ch, T'', 20)
        x = self.sub_out(x[0].permute(1, 0, 2).reshape(x.shape[2], -1))
        x = self.stack_frames(x * math.sqrt(float(self.d)))
        pe = rel_pos_emb(x.shape[0], self.d)
        for L in self.layers:
            x = L(x, pe)
        return x

    def forward(self, wav):
        f = self.pre(wav)
        n = wav.shape[0] // 160          # valid frames; the encoder sees all T = n+1 frames (last is zero)
        e = self.encode(f)
        return self.head(e), f, e        # logits (T', vocab + 1), blank = vocab


if __name__ == "__main__":
    import sys, soundfile as sf
    d = sys.argv[1] if len(sys.argv) > 1 else "../models/nemo_small"
    import numpy as np
    sd = torch.load(f"{d}/state_dict_plain.pt", map_location="cpu")
    m = NemoSmall(sd).eval()
    a, _ = sf.read(sys.argv[2] if len(sys.argv) > 2 else
                   "../data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac", dtype="float32")
    with torch.no_grad():
        lg, f, e = m(torch.from_numpy(a))
    import os
    if not os.path.exists(f"{d}/ref_1089.npz"):  # NeMo reference tensors (optional parity check)
        print("frames", tuple(lg.shape), "argmax ids", lg.argmax(-1)[:20].tolist())
        sys.exit(0)
    ref = {k: torch.from_numpy(v) for k, v in np.load(f"{d}/ref_1089.npz").items()}
    rf, re_, rl = ref["feat"][0].t(), ref["enc"][0].t(), ref["logits"][0]
    print("feat", tuple(f.shape), tuple(rf.shape), "max|d|", (f[:rf.shape[0]] - rf).abs().max().item())
    print("enc ", tuple(e.shape), tuple(re_.shape), "max|d|", (e - re_).abs().max().item(), "scale", re_.abs().mean().item())
    print("logit", tuple(lg.shape), "max|d|", (lg.log_softmax(-1) - rl).abs().max().item(),
          "argmax agree", (lg.argmax(-1) == rl.argmax(-1)).float().mean().item())
