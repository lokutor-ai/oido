"""Parity test: C engine (host build) vs PyTorch fake-quant simulation, streaming chunk=32/left=4."""
import sys, os
import numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "train"))
from model import Encoder
from features import LogMel
from qsim import quantize_model, quantize_attention
from export import export
import pytasr

ckpt, sp_path = sys.argv[1], sys.argv[2]
bits = int(sys.argv[3]) if len(sys.argv) > 3 else 8
wavs = sys.argv[4:]
out = "/tmp/parity.tasr"
export(ckpt, sp_path, out, bits=bits)
st = torch.load(ckpt, map_location="cpu")
cfg = dict(st["cfg"]); cfg.pop("vocab")
m = Encoder(st["cfg"]["vocab"], **cfg)
sd = {k.replace("_orig_mod.", ""): v for k, v in st["ema"].items()}
m.load_state_dict(sd)
m.eval()
quantize_model(m, bits=bits)
quantize_attention(m)
fe = LogMel()
fe.mean.copy_(st["fe_mean"]); fe.inv_std.copy_(st["fe_inv_std"])
cm = pytasr.Model(out)
print("weight bytes", cm.weight_bytes())
cs = pytasr.Stream(cm, 32, 4)
for w in wavs:
    a, sr = sf.read(w, dtype="int16")
    with torch.no_grad():
        f = fe(torch.from_numpy(a.astype(np.float32) / 32768.0)[None])
        lg, ol = m(f, torch.tensor([f.shape[1]]), chunk=32, left_chunks=4)
    ref = lg[0].numpy()
    txt, clg = cs.transcribe(a, logits=True, vocab=ref.shape[1])
    n = min(len(ref), len(clg))
    diff = np.abs(ref[:n] - clg[:n])
    agree = (ref[:n].argmax(1) == clg[:n].argmax(1)).mean()
    print(f"{os.path.basename(w)} frames torch {len(ref)} c {len(clg)} maxdiff {diff.max():.4g} meandiff {diff.mean():.3g} "
          f"ref-scale {np.abs(ref).mean():.3g} argmax-agree {agree*100:.2f}%")
    print("  C text:", txt[:120])
