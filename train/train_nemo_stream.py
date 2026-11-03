"""Fine-tune the NeMo Conformer-CTC port for a new language and/or for streaming, with the ESP32 engine's arithmetic.

New language: --tokenizer <spm.model> replaces the 1024-piece vocabulary; the CTC head is re-initialized and the
whole network is trained (the English acoustic layers are the starting point).

Streaming (--stream): the model learns to run on chunks of audio as it arrives, so only the last chunk is left to
compute when the speaker stops:
  - chunked self-attention: a frame sees the frames of its own chunk and of the previous chunks within a left context
    (--lefts, in 40 ms encoder frames, 0 = unlimited; the device uses 128 = 5.1 s so its caches have a fixed size);
    chunk size and left context are drawn per batch from --chunks / --lefts, or full context with probability
    --p_full, so one model runs in both modes;
  - chunk-limited depthwise convolution: the symmetric kernel (15 frames each side) sees any past frame but nothing
    after the end of its own chunk (zeros there), which keeps most of NVIDIA's pretrained kernel usable
    (--conv causal instead shifts the kernel to the past 30 frames, which needs far more retraining);
  - fixed feature normalization (global per-mel mean/std) instead of per-utterance statistics, plus random gain;
  - optional in-place distillation (--kd, "dual-mode ASR"): on chunked batches the EMA model's full-context posteriors
    teach the chunked posteriors (frame-level KL), which closes much of the streaming gap.
Quantization: --bits 0 trains in floating point (int8 post-training quantization is lossless for this model);
--bits 8/4 fake-quantize like train_nemo_qat.py.

Oido v2 architecture options (see nemo_small.py): --sub_ch 64 (light front end, channels kept by importance),
--stack 2 (80 ms frames after sub_out; --chunks / --lefts / --val_chunk / --val_left are then in 80 ms frames),
--n_layers N / --keep_layers i,j,k (fewer Conformer layers). Starting from a full-size checkpoint (--init) converts it.

Resumable for spot instances: out/ckpt.pt (+ optional --s3 copy) holds model, EMA, optimizer and step.

python train_nemo_stream.py --train a.jsonl,b.jsonl --val v.jsonl --nemo ../models/nemo_small --lang es \
    --tokenizer es1024.model --out ../exp/es_stream --stream --steps 40000 --lr 1e-3
"""
import argparse, copy, math, os, random, subprocess, time
import numpy as np, sentencepiece as spm, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

from data import AudioDataset, BucketBatchSampler, collate, load_manifest
from nemo_small import NemoSmall, cfg_from_state, rel_pos_emb
from quant_ops import fq_act, fq_weight, fq_rows


class BatchPre(nn.Module):
    """NeMo log-mel features. norm='utt': per-utterance per-mel mean/std (NeMo); norm='fixed': global mean/std."""
    def __init__(self, window, fb, mean=None, std=None):
        super().__init__()
        self.register_buffer("window", window.float())
        self.register_buffer("fb", fb.float().reshape(80, 257))
        self.fixed = mean is not None
        self.register_buffer("mean", (mean if mean is not None else torch.zeros(80)).float())
        self.register_buffer("std", (std if std is not None else torch.ones(80)).float())

    def logmel(self, wav):
        x = torch.cat([wav[:, :1], wav[:, 1:] - 0.97 * wav[:, :-1]], 1)
        spec = torch.stft(x, 512, 160, 400, window=self.window, center=True, pad_mode="constant", return_complex=True)
        return torch.log(torch.matmul(self.fb, spec.real.square() + spec.imag.square()) + 2.0 ** -24)  # (B, 80, T)

    def forward(self, wav, lens):
        mel = self.logmel(wav)
        n = lens // 160
        T = mel.shape[-1]
        valid = (torch.arange(T, device=wav.device)[None] < n[:, None]).float()[:, None]
        if self.fixed:
            out = (mel - self.mean[None, :, None]) / self.std[None, :, None] * valid
        else:
            mean = (mel * valid).sum(-1, keepdim=True) / n[:, None, None]
            std = torch.sqrt(((mel - mean) ** 2 * valid).sum(-1, keepdim=True) / (n[:, None, None] - 1).clamp(min=1)) + 1e-5
            out = ((mel - mean) / std) * valid
        return out.transpose(1, 2), n + 1  # (B, T, 80); the encoder sees n+1 frames (last is zero)


def make_causal(m: NemoSmall):
    """Symmetric depthwise kernel (offsets -15..+15) -> causal kernel (offsets -30..0) keeping the past half."""
    with torch.no_grad():
        for L in m.layers:
            w = L.dw.weight  # (d, 1, k)
            k = w.shape[-1]
            c = torch.zeros_like(w)
            c[:, :, k // 2:] = w[:, :, : k // 2 + 1]
            w.copy_(c)


def chunk_dwconv(g, w, bias, C):
    """Depthwise conv where output frame t sees inputs up to the end of its chunk (t // C + 1) * C, zeros beyond."""
    B, d, T = g.shape
    r = w.shape[-1] // 2
    nC = (T + C - 1) // C
    gp = F.pad(g, (r, nC * C - T + r))
    win = gp.unfold(2, C + 2 * r, C)                                  # (B, d, nC, C + 2r)
    win = win * (torch.arange(C + 2 * r, device=g.device) < r + C).to(g.dtype)
    out = F.conv1d(win.permute(0, 2, 1, 3).reshape(B * nC, d, C + 2 * r), w, bias, groups=d)
    return out.view(B, nC, d, C).permute(0, 2, 1, 3).reshape(B, d, nC * C)[..., :T]


class SModel(nn.Module):
    def __init__(self, m: NemoSmall, bits=0, causal=False):
        super().__init__()
        self.m, self.bits, self.causal = m, bits, causal
        self.chunk = 0

    def ql(self, lin, x, bits):
        if not bits:
            return F.linear(x, lin.weight, lin.bias)
        return F.linear(fq_act(x), fq_weight(lin.weight, bits), lin.bias)

    def fr(self, x):
        return fq_rows(x) if self.bits else x

    def layer(self, L, x, allowed, pad, pe):
        b = self.bits
        B, T, d = x.shape
        h = L.norm_feed_forward1(x)
        x = x + 0.5 * self.ql(L.ff1_l2, F.silu(self.ql(L.ff1_l1, h, b)), b)
        h = L.norm_self_att(x)
        q = self.ql(L.linear_q, h, b).view(B, T, L.h, L.dk)
        k = self.fr(self.ql(L.linear_k, h, b).view(B, T, L.h, L.dk).transpose(1, 2))
        v = self.ql(L.linear_v, h, b).view(B, T, L.h, L.dk).transpose(1, 2)
        p = self.fr(self.ql(L.linear_pos, pe, 8 if b else 0).view(2 * T - 1, L.h, L.dk).transpose(0, 1))
        qu = self.fr((q + L.pos_bias_u).transpose(1, 2))
        qv = self.fr((q + L.pos_bias_v).transpose(1, 2))
        ac = qu @ k.transpose(-1, -2)
        bd = qv @ p.transpose(-1, -2)[None]
        idx = (T - 1) - torch.arange(T, device=x.device)[:, None] + torch.arange(T, device=x.device)[None, :]
        bd = torch.gather(bd, 3, idx[None, None].expand(B, L.h, T, T))
        s = (ac + bd) / math.sqrt(L.dk)
        a = torch.softmax(s.masked_fill(~allowed, -1e4), -1)
        if b:
            sv = v.detach().abs().amax(-1, keepdim=True).clamp(min=1e-30) / 127.0
            vq = torch.clamp(torch.round(v / sv), -127, 127) + (v / sv - (v / sv).detach())  # STE
            o = fq_rows(a * sv.transpose(-1, -2)) @ vq
        else:
            o = a @ v
        x = x + self.ql(L.linear_out, o.transpose(1, 2).reshape(B, T, d), b)
        h = L.norm_conv(x)
        g = F.glu(self.ql(L.pw1, h, b), dim=-1).masked_fill(pad[..., None], 0.0).transpose(1, 2)  # (B, d, T)
        kz = L.dw.weight.shape[-1]
        if self.causal:
            c = F.conv1d(F.pad(g, (kz - 1, 0)), L.dw.weight, L.dw.bias, groups=d)
        elif self.chunk:
            c = chunk_dwconv(g, L.dw.weight, L.dw.bias, self.chunk)
        else:
            c = F.conv1d(g, L.dw.weight, L.dw.bias, padding=kz // 2, groups=d)
        x = x + self.ql(L.pw2, F.silu(c.transpose(1, 2)), b)
        h = L.norm_feed_forward2(x)
        x = x + 0.5 * self.ql(L.ff2_l2, F.silu(self.ql(L.ff2_l1, h, b)), b)
        return L.norm_out(x)

    def front(self, feats, flen):
        """feats (B, T, 80) -> sub_out * sqrt(d) (B, T2, d) at 40 ms, lengths (before frame stacking)."""
        m = self.m
        x = F.relu(m.conv0(feats[:, None]))
        B, C, T1, F1 = x.shape
        cols = F.unfold(x, 3, padding=1, stride=2)
        T2, F2 = (T1 + 2 - 3) // 2 + 1, (F1 + 2 - 3) // 2 + 1
        w2 = m.conv2.weight.flatten(1)
        if self.bits:
            y = F.linear(fq_act(cols.transpose(1, 2)), fq_weight(w2, 8), m.conv2.bias)
        else:
            y = F.linear(cols.transpose(1, 2), w2, m.conv2.bias)
        y = F.relu(y).view(B, T2, F2, C).permute(0, 1, 3, 2).reshape(B, T2, C * F2)
        x = self.ql(m.sub_out, y, 8 if self.bits else 0) * math.sqrt(float(m.d))
        return x, ((flen + 2 - 3) // 2 + 1 + 2 - 3) // 2 + 1

    def forward(self, feats, flen, chunk=0, left=0):
        """feats (B, T, 80) -> logits (B, T', V+1), lengths. chunk > 0: chunked attention with that many frames."""
        m = self.m
        x, lens = self.front(feats, flen)
        B, T2, _ = x.shape
        if m.stack == 2:  # 80 ms frames: [x[2i]; x[2i+1]] -> Linear; frames past the end are zero (odd tail pairs with 0)
            x = x.masked_fill((torch.arange(T2, device=x.device)[None] >= lens[:, None])[..., None], 0.0)
            if T2 % 2:
                x = F.pad(x, (0, 0, 0, 1))
            x = self.ql(m.stack_proj, x.reshape(B, x.shape[1] // 2, 2 * m.d), 8 if self.bits else 0)
            lens = (lens + 1) // 2
            T2 = x.shape[1]
        mask_k = torch.arange(T2, device=x.device)[None] < lens[:, None]
        allowed = mask_k[:, None, None, :]
        if chunk:
            cid = torch.arange(T2, device=x.device) // chunk
            ok = cid[None, :] <= cid[:, None]
            if left:  # whole chunks: the previous ceil(left / chunk) chunks
                ok = ok & (cid[None, :] >= cid[:, None] - (left + chunk - 1) // chunk)
            allowed = allowed & ok[None, None]
        pe = rel_pos_emb(T2, m.d).to(x.device)
        self.chunk = chunk
        for L in m.layers:
            x = self.layer(L, x, allowed, ~mask_k, pe)
        return self.ql(m.head, x, 8 if self.bits else 0), lens


def ctc_greedy(lg, n, blank):
    out, prev = [], -1
    for i in lg[:n].argmax(-1).tolist():
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def feature_stats(pre, items, dev, n=1500):
    """Global per-mel mean/std of log-mel features over n random training utterances (for fixed normalization)."""
    import soundfile as sf
    rng = random.Random(0)
    s, s2, cnt = torch.zeros(80, device=dev, dtype=torch.float64), torch.zeros(80, device=dev, dtype=torch.float64), 0
    for it in rng.sample(items, min(n, len(items))):
        w, _ = sf.read(it["wav"], dtype="float32")
        mel = pre.logmel(torch.from_numpy(w)[None].to(dev))[0, :, : len(w) // 160].double()
        s += mel.sum(1); s2 += (mel ** 2).sum(1); cnt += mel.shape[1]
    mean = s / cnt
    return mean.float().cpu(), torch.sqrt(s2 / cnt - mean ** 2).float().cpu()


class AugDataset(AudioDataset):
    def __init__(self, items, aug):
        super().__init__(items)
        self.aug = aug

    def __getitem__(self, i):
        a, ids, j = super().__getitem__(i)
        return (self.aug(a) if self.aug is not None and len(a) > 1600 else a), ids, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="comma-separated jsonl manifests")
    ap.add_argument("--val", default="")
    ap.add_argument("--nemo", required=True, help="dir with state_dict_plain.pt (+ tokenizer.model)")
    ap.add_argument("--tokenizer", default="", help="new SentencePiece model (1024 pieces); re-initializes the head")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--batch_sec", type=float, default=500)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--init", default="", help="start from a checkpoint's (EMA) weights")
    ap.add_argument("--keep_head", action="store_true",
                    help="with --tokenizer: only load that vocabulary, keep the CTC head (continuing a run)")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--dev", default="cuda", help="cuda (cpu only for smoke tests)")
    ap.add_argument("--conv", default="chunk", choices=["chunk", "causal"], help="streaming depthwise conv")
    ap.add_argument("--sub_ch", type=int, default=0, help="front-end channels (0 = as in the init model, 176)")
    ap.add_argument("--stack", type=int, default=1, choices=[1, 2], help="2 = stack pairs of 40 ms frames -> 80 ms")
    ap.add_argument("--n_layers", type=int, default=0, help="keep only the first N Conformer layers (0 = all)")
    ap.add_argument("--keep_layers", default="", help="comma list of source layer indices to keep (overrides --n_layers)")
    ap.add_argument("--chunks", default="", help="chunk sizes in encoder frames (default: 4,8,12,16,24,32; stack 2: 2,4,6,8,12,16)")
    ap.add_argument("--lefts", default="", help="left contexts in encoder frames (default: 0,0,64,128,256; stack 2: 0,0,32,64)")
    ap.add_argument("--p_full", type=float, default=0.25)
    ap.add_argument("--val_chunk", type=int, default=0, help="default 16 (stack 2: 8)")
    ap.add_argument("--val_left", type=int, default=0, help="default 128 (stack 2: 64)")
    ap.add_argument("--kd", type=float, default=0.0, help="in-place distillation weight (full-context EMA -> chunked)")
    ap.add_argument("--fe_steps", type=int, default=0,
                    help="with --sub_ch: first regress the light front end onto the full front end's output (steps)")
    ap.add_argument("--fe_lr", type=float, default=2e-3)
    ap.add_argument("--kd_start", type=int, default=0, help="step at which --kd turns on (the EMA teacher must be good first)")
    ap.add_argument("--max_twer", type=float, default=1e9, help="drop utterances whose teacher WER exceeds this")
    ap.add_argument("--max_dur", type=float, default=20.0)
    ap.add_argument("--s3", default="", help="s3://bucket/prefix to mirror checkpoints to")
    ap.add_argument("--aug_dir", default="", help="noise bank + RIRs built by augment.py (MUSAN noise/music/babble, reverb)")
    ap.add_argument("--p_noise", type=float, default=0.6)
    ap.add_argument("--p_rir", type=float, default=0.3)
    a = ap.parse_args()
    dev = a.dev
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0); random.seed(0)
    sd = torch.load(f"{a.nemo}/state_dict_plain.pt", map_location="cpu")
    base = NemoSmall(sd)
    tok = a.tokenizer or f"{a.nemo}/tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=tok)
    V = sp.get_piece_size()
    assert V == base.vocab, f"tokenizer has {V} pieces, model head {base.vocab}"
    if a.tokenizer and not a.keep_head:
        nn.init.xavier_uniform_(base.head.weight); nn.init.zeros_(base.head.bias)
        print("new vocabulary:", tok, flush=True)
    causal = a.stream and a.conv == "causal"
    if causal:
        assert a.stack == 1 and not a.sub_ch and not a.n_layers and not a.keep_layers, "causal conv: original architecture only"
        make_causal(base)
    if not a.chunks:
        a.chunks = "2,4,6,8,12,16" if a.stack == 2 else "4,8,12,16,24,32"
    if not a.lefts:
        a.lefts = "0,0,32,64" if a.stack == 2 else "0,0,64,128,256"
    a.val_chunk = a.val_chunk or (8 if a.stack == 2 else 16)
    a.val_left = a.val_left or (64 if a.stack == 2 else 128)
    enc_fps = 25.0 / a.stack  # encoder frames per second

    items = load_manifest(a.train.split(","), max_dur=a.max_dur, lang=a.lang)
    n0 = len(items)
    items = [it for it in items if it.get("twer", 0.0) <= a.max_twer]
    for it in items:
        it["ids"] = sp.encode(it["text"])
    items = [it for it in items if 0 < len(it["ids"]) < it["dur"] * enc_fps * 0.8]   # CTC needs >= 1 frame per token
    print(f"train utts {len(items)} of {n0} ({sum(it['dur'] for it in items)/3600:.0f} h), encoder {enc_fps} frames/s", flush=True)
    val = []
    for v in [x for x in a.val.split(",") if x]:
        vv = load_manifest([v], max_dur=35.0, lang=a.lang)
        val += vv[:: max(1, len(vv) // 150)][:150]

    mean = std = None
    ckpt_path = f"{a.out}/ckpt.pt"
    if not os.path.exists(ckpt_path) and a.s3:
        subprocess.run(["aws", "s3", "cp", f"{a.s3}/ckpt.pt", ckpt_path, "--only-show-errors"])
    ck = torch.load(ckpt_path, map_location="cpu") if os.path.exists(ckpt_path) else None
    if ck is not None:
        mean, std = ck.get("norm_mean"), ck.get("norm_std")
    elif a.init and torch.load(a.init, map_location="cpu").get("norm_mean") is not None:
        init_ck = torch.load(a.init, map_location="cpu")  # continue a streaming run: same fixed normalization
        mean, std = init_ck["norm_mean"], init_ck["norm_std"]
    elif a.stream:
        pre0 = BatchPre(sd["preprocessor.featurizer.window"], sd["preprocessor.featurizer.fb"]).to(dev)
        mean, std = feature_stats(pre0, items, dev)
        print("fixed normalization stats computed", flush=True)
    pre = BatchPre(sd["preprocessor.featurizer.window"], sd["preprocessor.featurizer.fb"], mean, std).to(dev)

    fe_teacher = None
    layers = [int(x) for x in a.keep_layers.split(",")] if a.keep_layers else list(range(a.n_layers or len(base.layers)))
    want = dict(sub_ch=a.sub_ch or base.cfg["sub_ch"], stack=a.stack, n_layers=len(layers))
    if ck is not None:  # resume: the architecture is the checkpoint's
        base = NemoSmall(cfg=ck.get("cfg") or cfg_from_state(ck["model"]))
    elif a.init:
        ik = torch.load(a.init, map_location="cpu")
        icfg = ik.get("cfg") or cfg_from_state(ik["model"])
        if all(icfg[k] == v for k, v in want.items()):      # already the wanted architecture
            base = NemoSmall(cfg=icfg)
            base.load_state_dict(ik["model"])
        else:                                              # full-size checkpoint -> convert
            assert icfg["stack"] == 1 and icfg["sub_ch"] == base.cfg["sub_ch"] and icfg["n_layers"] == len(base.layers), icfg
            base.load_state_dict(ik["model"])
            fe_teacher = copy.deepcopy(base)
            base = base.convert(want["sub_ch"], a.stack, layers)
        print("init from", a.init, "->", base.cfg, flush=True)
    else:
        fe_teacher = copy.deepcopy(base)
        base = base.convert(want["sub_ch"], a.stack, layers)
    if fe_teacher is not None and (not a.fe_steps or want["sub_ch"] == fe_teacher.cfg["sub_ch"]):
        fe_teacher = None
    print(f"model {sum(p.numel() for p in base.parameters())/1e6:.2f} M params, cfg {base.cfg}", flush=True)
    model = SModel(base, a.bits, causal=causal).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=1e-3)
    step = 0
    if ck is not None:
        model.m.load_state_dict(ck["model"]); ema.m.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print("resumed at step", step, flush=True)
    aug = None
    if a.aug_dir:
        from augment import NpAug
        aug = NpAug(a.aug_dir, a.p_noise, a.p_rir)  # applied per utterance inside the DataLoader workers
        print("augmentation: noise/music/babble p", a.p_noise, "reverb p", a.p_rir, flush=True)
    chunks = [int(c) for c in a.chunks.split(",")]
    lefts = [int(c) for c in a.lefts.split(",")]

    def save(final=False):
        st = {"model": model.m.state_dict(), "ema": ema.m.state_dict(), "opt": opt.state_dict(), "step": step,
              "norm_mean": mean, "norm_std": std, "causal": causal, "stream": a.stream, "tokenizer": os.path.basename(tok),
              "cfg": model.m.cfg}
        torch.save(st, ckpt_path + ".tmp"); os.replace(ckpt_path + ".tmp", ckpt_path)
        torch.save({"model": ema.m.state_dict(), "bits": a.bits, "step": step, "norm_mean": mean, "norm_std": std,
                    "causal": causal, "stream": a.stream, "cfg": model.m.cfg,
                    "chunk": a.val_chunk, "left": a.val_left}, f"{a.out}/last.pt")
        if a.s3:
            subprocess.Popen(["aws", "s3", "cp", ckpt_path, f"{a.s3}/ckpt.pt", "--only-show-errors"])
            if final:
                subprocess.run(["aws", "s3", "cp", f"{a.out}/last.pt", f"{a.s3}/last.pt", "--only-show-errors"])

    def evaluate(net, chunk, left=0):
        import jiwer, soundfile as sf
        net.eval()
        hyps, refs = [], []
        with torch.no_grad():
            for it in val:
                w, _ = sf.read(it["wav"], dtype="float32")
                f, fl = pre(torch.from_numpy(w)[None].to(dev), torch.tensor([len(w)], device=dev))
                lg, ol = net(f, fl, chunk, left)
                hyps.append(sp.decode(ctc_greedy(lg[0], int(ol[0]), V))); refs.append(it["text"])
        net.train()
        return 100 * jiwer.wer(refs, hyps)

    def report(tag, net):
        if not val:
            return
        msg = f"step {step} {tag} val WER full {evaluate(net, 0):.2f}"
        if a.stream:
            msg += f" | chunk{a.val_chunk}/left{a.val_left} {evaluate(net, a.val_chunk, a.val_left):.2f}"
        print(msg, flush=True)

    def make_feats(wav, lens):
        """waveform batch (B, N) in [-1, 1] -> augmented normalized log-mel features (B, T, 80), lengths."""
        with torch.no_grad():
            if a.stream:  # random gain: fixed normalization must not depend on the recording level
                wav = wav * (10.0 ** (torch.empty(wav.shape[0], 1, device=dev).uniform_(-12, 12) / 20.0))
            f, fl = pre(wav, lens)
            B, T, _ = f.shape
            fr = torch.arange(80, device=dev)[None]
            for _ in range(2):
                w = torch.randint(0, 16, (B, 1), device=dev); s0 = (torch.rand(B, 1, device=dev) * (80 - w)).long()
                f = f.masked_fill(((fr >= s0) & (fr < s0 + w))[:, None, :], 0.0)
            tr = torch.arange(T, device=dev)[None]
            for _ in range(max(1, T // 200)):
                w = torch.randint(0, 25, (B, 1), device=dev); s0 = (torch.rand(B, 1, device=dev) * (fl[:, None] - w).clamp(min=1)).long()
                f = f.masked_fill(((tr >= s0) & (tr < s0 + w))[:, :, None], 0.0)
        return f, fl

    if fe_teacher is not None and ck is None:
        # the pruned front end starts far from the full one (WER ~100 %): regress its output (sub_out * sqrt(d)) onto the
        # full front end's on training audio before anything else is trained
        teach = SModel(fe_teacher).to(dev).eval()
        fe_params = [p for n_, p in model.m.named_parameters() if n_.startswith(("conv0", "conv2", "sub_out"))]
        ofe = torch.optim.AdamW(fe_params, lr=a.fe_lr, betas=(0.9, 0.98), weight_decay=0.0)
        fs = BucketBatchSampler(items, 250, seed=7)
        n_fe, acc, t0 = 0, 0.0, time.time()
        model.train()
        while n_fe < a.fe_steps:
            fs.set_epoch(n_fe)
            for wav, lens, *_ in DataLoader(AugDataset(items, aug), batch_sampler=fs, collate_fn=collate, num_workers=a.workers,
                                            pin_memory=dev == "cuda", prefetch_factor=4):
                f, fl = make_feats(wav.to(dev).float() / 32768.0, lens.to(dev))
                with torch.no_grad():
                    yt, ln = teach.front(f, fl)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    ys, _ = model.front(f, fl)
                valid = (torch.arange(yt.shape[1], device=dev)[None] < ln[:, None])[..., None].float()
                loss = (((ys.float() - yt) ** 2) * valid).sum() / ((yt ** 2) * valid).sum().clamp(min=1e-9)
                for g in ofe.param_groups:
                    g["lr"] = a.fe_lr * min(1.0, (n_fe + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * n_fe / a.fe_steps)))
                ofe.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fe_params, 1.0)
                ofe.step()
                n_fe += 1
                acc += loss.item()
                if n_fe % 100 == 0:
                    print(f"front-end regression step {n_fe} relative MSE {acc / 100:.4f} {time.time() - t0:.0f}s", flush=True)
                    acc = 0.0
                if n_fe >= a.fe_steps:
                    break
        ema.m.load_state_dict(model.m.state_dict())
        del teach, fe_teacher, ofe
        print("front-end regression done", flush=True)

    if step == 0:
        report("init", model)
    model.train()
    sampler = BucketBatchSampler(items, a.batch_sec, seed=1)
    ep, t0, ag = step // max(1, len(sampler)), time.time(), 0.0
    while step < a.steps:
        sampler.set_epoch(ep)
        dl = DataLoader(AugDataset(items, aug), batch_sampler=sampler, collate_fn=collate, num_workers=a.workers,
                        pin_memory=dev == "cuda", prefetch_factor=4, persistent_workers=False)
        for wav, lens, tokb, tl, _ in dl:
            wav = wav.to(dev).float() / 32768.0
            lens, tokb, tl = lens.to(dev), tokb.to(dev), tl.to(dev)
            f, fl = make_feats(wav, lens)
            chunk = 0 if (not a.stream or random.random() < a.p_full) else random.choice(chunks)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                lg, ol = model(f, fl, chunk, random.choice(lefts) if chunk else 0)
            lp = lg.float().log_softmax(-1)
            loss = F.ctc_loss(lp.transpose(0, 1), tokb, ol, tl, blank=V, reduction="sum", zero_infinity=True) / tl.sum()
            if a.kd and chunk and step >= a.kd_start:
                with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    tg, _ = ema(f, fl, 0, 0)
                tp = tg.float().log_softmax(-1)
                valid = (torch.arange(lp.shape[1], device=dev)[None] < ol[:, None]).float()
                kl = (tp.exp() * (tp - lp)).sum(-1)
                loss = loss + a.kd * (kl * valid).sum() / valid.sum()
            lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                dec = min(0.999, (1 + step) / (10 + step))
                for pe_, pm in zip(ema.parameters(), model.parameters()):
                    pe_.mul_(dec).add_(pm.detach(), alpha=1 - dec)
            step += 1
            ag += loss.item()
            if step % 100 == 0:
                print(f"step {step} loss {ag/100:.4f} lr {lr:.2e} {time.time()-t0:.0f}s", flush=True)
                ag = 0.0
            if step % a.save_every == 0:
                save()
            if step % a.val_every == 0:
                report("EMA", ema)
            if step >= a.steps:
                break
        ep += 1
    save(final=True)
    report("final EMA", ema)


if __name__ == "__main__":
    main()
