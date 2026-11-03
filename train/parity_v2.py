"""Greedy transcripts of a checkpoint (or a .tnm file) under the C engine's int8 arithmetic emulation, to compare with
the engine:  esp32/host/tasr_cli models/<file>.tnm clip.wav

  - every Linear (incl. conv2 as an im2col GEMM, sub_out, the 2d->d stack projection, linear_pos, head): weights
    per-output-row symmetric int8, activations per-row dynamic int8 (quant_ops.fq_weight / fq_act, bits = 8); for the
    stack projection one row is the concatenated pair [x[2i]; x[2i+1]] (352 values, one scale);
  - attention: q+u, q+v, k, p per-row int8, probabilities*scale per-row int8, v per-row int8 (fq_rows);
  - LayerNorm, SiLU, softmax, depthwise conv in float.
Modes: full context and the streaming masks (chunk / left in encoder frames; defaults = the .tnm header or 8 / 64).

python parity_v2.py --ckpt last.pt [clips.wav ...]      (default clips: five LibriSpeech test-clean utterances)
python parity_v2.py --tnm oido_v2a.tnm [clips.wav ...]   (weights read back from the exported file)
"""
import argparse, glob, os, sys
import soundfile as sf, sentencepiece as spm, torch

HERE = os.path.dirname(os.path.abspath(__file__))
from nemo_small import NemoSmall  # noqa: E402
from train_nemo_stream import BatchPre, SModel, ctc_greedy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tnm", default="")
    ap.add_argument("--tok", default=os.path.join(HERE, "..", "models", "nemo_small", "tokenizer.model"))
    ap.add_argument("--bits", type=int, default=0, help="weight bits of the Linear layers (default: 8, or the .tnm's)")
    ap.add_argument("--chunk", type=int, default=0, help="streaming chunk (encoder frames); default from the model")
    ap.add_argument("--left", type=int, default=0)
    ap.add_argument("--ids", action="store_true", help="print the per-frame argmax ids (blank = 1024)")
    ap.add_argument("--float", action="store_true", help="also print the float (unquantized) transcript")
    a = ap.parse_args()
    if a.tnm:
        from tnm_read import read_tnm
        model, h, mean, std, _ = read_tnm(a.tnm)
        chunk, left = h["chunk"], h["left"]
        a.bits = a.bits or h["bits"]
    else:
        st = torch.load(a.ckpt, map_location="cpu")
        model = NemoSmall.from_ckpt(st).eval()
        mean, std = st.get("norm_mean"), st.get("norm_std")
        chunk, left = st.get("chunk", 0), st.get("left", 0)
    chunk = a.chunk or chunk or (8 if model.stack == 2 else 32)
    left = a.left or left or (64 if model.stack == 2 else 128)
    clips = a.clips or sorted(glob.glob(os.path.join(HERE, "..", "data", "LibriSpeech", "test-clean", "*", "*", "*.flac")))[::520][:5]
    pre = BatchPre(model.pre.window, model.pre.fb, mean, std)
    sp = spm.SentencePieceProcessor(model_file=a.tok)
    V = model.vocab
    a.bits = a.bits or 8
    q8, fl = SModel(model, a.bits).eval(), SModel(model, 0).eval()
    print(f"cfg {model.cfg}  weights int{a.bits}, activations int8  fixed norm {mean is not None}  stream chunk/left {chunk}/{left}")
    with torch.no_grad():
        for c in clips:
            w, sr = sf.read(c, dtype="float32")
            assert sr == 16000, c
            f, flen = pre(torch.from_numpy(w)[None], torch.tensor([len(w)]))
            print("==", os.path.basename(c), f"{len(w) / 16000:.2f} s")
            for name, net, ch, lf in [(f"int{a.bits} full  ", q8, 0, 0), (f"int{a.bits} {chunk}/{left}", q8, chunk, left)] + \
                    ([("float full ", fl, 0, 0)] if a.float else []):
                lg, ol = net(f, flen, ch, lf)
                ids = lg[0, : int(ol[0])].argmax(-1)
                print(f"  {name}: {sp.decode(ctc_greedy(lg[0], int(ol[0]), V))}")
                if a.ids:
                    print("   ids", ids.tolist())


if __name__ == "__main__":
    main()
