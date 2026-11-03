"""Export the NeMo conformer-ctc-small port to the TNM1 blob for the C engine (tasr_nemo.c).

python export_nemo.py ../models/nemo_small out.tnm [bits] [checkpoint] [tokenizer.model]

Header (after "TNM1", 15 little-endian uint32 words, index = position): 0 version (1), 1 d (176), 2 heads (4), 3 FF (704),
4 depthwise kernel (31), 5 number of Conformer layers, 6 front-end conv channels C (176 for the NVIDIA model, 64 for
Oido v2), 7 vocab, 8 weight bits, 9 transducer flag, 10 flags, 11 / 12 default streaming chunk / left context in ENCODER
frames (80 ms with stacking), 13 stack factor (0 or 1 = none, 2 = pairs of 40 ms frames stacked), 14 reserved.
Flags (word 10): bit 0 = fixed feature normalization (80 means + 80 stds appended after the tokens), bit 1 = trained
for streaming (chunked attention, chunk-limited depthwise conv).
Weight order: window, fb, conv0 (float, C*9 + bias), conv2 qlin (K = C*9, (c, kh, kw) order), sub_out qlin (K = C*20,
input index c*20 + f), [stack qlin (176, 352) = W [x[2i]; x[2i+1]] + b, only if stack = 2], layers, head, tokens, norm.
"""
import os, struct, sys
import numpy as np, torch, sentencepiece as spm
from export import W
from nemo_small import NemoSmall


def export(d, out, bits=8, ckpt=None, tokenizer=None, chunk=None, left=None):
    # CTC model: state_dict_plain.pt (fetch_nemo_small.py); transducer: model_weights.ckpt unpacked from the .nemo
    path = f"{d}/state_dict_plain.pt" if os.path.exists(f"{d}/state_dict_plain.pt") else f"{d}/model_weights.ckpt"
    sd = dict(torch.load(path, map_location="cpu"))
    rnnt = "decoder.prediction.embed.weight" in sd
    if rnnt:  # NemoSmall builds a CTC head; the transducer has none
        sd["decoder.decoder_layers.0.weight"] = torch.zeros(1025, 176, 1)
        sd["decoder.decoder_layers.0.bias"] = torch.zeros(1025)
    st = {}
    if ckpt:  # fine-tuned weights (NemoSmall state dict; train_nemo_qat.py or train_nemo_stream.py)
        st = torch.load(ckpt, map_location="cpu")
        m = NemoSmall.from_ckpt(st).eval()   # architecture (sub_ch / stack / n_layers) from the checkpoint
        print("loaded fine-tuned weights from", ckpt, "step", st.get("step"), "cfg", m.cfg)
    else:
        m = NemoSmall(sd).eval()
    cfg = m.cfg
    assert (cfg["d"], cfg["h"], cfg["ff"], cfg["k"]) == (176, 4, 704, 31), cfg
    stack = cfg.get("stack", 1)
    if chunk is None:   # default streaming chunk / left in encoder frames (0.64 s chunks, 5.1 s left context)
        chunk, left = (8, 64) if stack == 2 else (32, 128)
    assert not st.get("causal"), "causal-conv checkpoints are not supported by the engine"
    fixed = st.get("norm_mean") is not None
    flags = (1 if fixed else 0) | (2 if st.get("stream") else 0)
    if tokenizer is None:
        tok = "tokenizer.model" if os.path.exists(f"{d}/tokenizer.model") else [f for f in os.listdir(d) if f.endswith("tokenizer.model")][0]
        tokenizer = f"{d}/{tok}"
    sp = spm.SentencePieceProcessor(model_file=tokenizer)
    V = sp.get_piece_size()
    w = W()
    w.buf += b"TNM1" + struct.pack("<15I", 1, 176, 4, 704, 31, len(m.layers), cfg["sub_ch"], V, bits, int(rnnt), flags,
                                   chunk if flags & 2 else 0, left if flags & 2 else 0, stack if stack == 2 else 0, 0)
    w.f32(m.pre.window)
    w.f32(m.pre.fb.t().contiguous().reshape(-1))            # (257, 80) row-major, like TASR
    w.f32(m.conv0.weight.reshape(-1)); w.f32(m.conv0.bias)
    sizes = {"header+window+fb+conv0": len(w.buf)}
    w.qlin(m.conv2.weight.flatten(1), m.conv2.bias, 8)       # im2col K = sub_ch*9 in (c, kh, kw) order
    sizes["conv2"] = len(w.buf) - sum(sizes.values())
    w.qlin(m.sub_out.weight, m.sub_out.bias, 8)              # K = sub_ch*20, input index c*20 + f
    sizes["sub_out"] = len(w.buf) - sum(sizes.values())
    if stack == 2:                                           # row = [x[2i]; x[2i+1]] (352), one dynamic scale per row
        w.qlin(m.stack_proj.weight, m.stack_proj.bias, 8)
        sizes["stack"] = len(w.buf) - sum(sizes.values())
    lay0 = len(w.buf)
    for L in m.layers:
        def ln(n):
            w.f32(n.weight); w.f32(n.bias)
        ln(L.norm_feed_forward1)
        w.qlin(L.ff1_l1.weight, L.ff1_l1.bias, bits); w.qlin(L.ff1_l2.weight, L.ff1_l2.bias, bits)
        ln(L.norm_self_att)
        w.qlin(torch.cat([L.linear_q.weight, L.linear_k.weight, L.linear_v.weight]),
               torch.cat([L.linear_q.bias, L.linear_k.bias, L.linear_v.bias]), bits)
        w.qlin(L.linear_out.weight, L.linear_out.bias, bits)
        w.qlin(L.linear_pos.weight, None, 8)
        w.f32(L.pos_bias_u.reshape(-1)); w.f32(L.pos_bias_v.reshape(-1))
        ln(L.norm_conv)
        w.qlin(L.pw1.weight, L.pw1.bias, bits)
        w.f32(L.dw.weight.reshape(-1)); w.f32(L.dw.bias)
        w.qlin(L.pw2.weight, L.pw2.bias, bits)
        ln(L.norm_feed_forward2)
        w.qlin(L.ff2_l1.weight, L.ff2_l1.bias, bits); w.qlin(L.ff2_l2.weight, L.ff2_l2.bias, bits)
        ln(L.norm_out)
    sizes[f"{len(m.layers)} layers"] = len(w.buf) - lay0
    if rnnt:  # RNN-T prediction network (embedding + 1-layer LSTM 320) and joint network, all int8
        g = lambda k: sd[k].float()
        w.qlin(g("decoder.prediction.embed.weight"), torch.zeros(V + 1), 8)
        w.qlin(g("decoder.prediction.dec_rnn.lstm.weight_ih_l0"), g("decoder.prediction.dec_rnn.lstm.bias_ih_l0"), 8)
        w.qlin(g("decoder.prediction.dec_rnn.lstm.weight_hh_l0"), g("decoder.prediction.dec_rnn.lstm.bias_hh_l0"), 8)
        w.qlin(g("joint.enc.weight"), g("joint.enc.bias"), 8)
        w.qlin(g("joint.pred.weight"), g("joint.pred.bias"), 8)
        w.qlin(g("joint.joint_net.2.weight"), g("joint.joint_net.2.bias"), 8)  # 1025 outputs, blank = 1024
    else:
        w.qlin(m.head.weight, m.head.bias, 8)                # 1025 outputs, blank = index 1024
    w.align()
    sizes["head" if not rnnt else "rnnt head"] = len(w.buf) - sum(sizes.values())
    for i in range(V):
        b = sp.id_to_piece(i).replace("▁", " ").encode()
        if sp.is_unknown(i) or sp.is_control(i):
            b = b""
        w.buf += bytes([len(b)]) + b
    if fixed:
        w.f32(st["norm_mean"].float()); w.f32(st["norm_std"].float())
    open(out, "wb").write(w.buf)
    print(f"wrote {out}: {len(w.buf)/1e6:.3f} MB (bits={bits}, vocab={V}, flags={flags}, layers={len(m.layers)}, "
          f"conv C={cfg['sub_ch']}, stack={stack}, default chunk/left={chunk}/{left})")
    sizes["tokens+norm"] = len(w.buf) - sum(sizes.values())
    for k, v in sizes.items():
        print(f"  {k:24s} {v/1e6:8.3f} MB")
    print(f"  per layer {sizes[f'{len(m.layers)} layers']/len(m.layers)/1e6:.3f} MB; params {sum(p.numel() for p in m.parameters())/1e6:.2f} M")


if __name__ == "__main__":
    a = sys.argv[1:]
    export(a[0], a[1], int(a[2]) if len(a) > 2 else 8, a[3] if len(a) > 3 and a[3] != "-" else None,
           a[4] if len(a) > 4 else None)
