"""Greedy WER of a NemoSmall-family checkpoint (EMA weights in last.pt; also ckpt.pt with --key ema) with the training
code's forward (SModel): float (--bits 0) or the engine's int8 arithmetic emulation (--bits 8), full context and/or
streaming chunk masks.

python eval_v2.py --ckpt last.pt|--tnm model.tnm --sets test-clean,test-other,val_cv --modes full,8/64 [--bits 8] [--limit 200]
  sets: LibriSpeech splits under data/LibriSpeech (test-clean, test-other, dev-clean) or data/val/<name>.local.jsonl
        (val_cv, val_vox, val_ami, val_ps, val_e22, ...)
  modes: "full" or "<chunk>/<left>" in encoder frames (80 ms with stack 2)
A full-size checkpoint can be converted on the fly (--sub_ch / --stack / --keep_layers) to measure the initialization.
"""
import argparse, json, os, sys, time
import soundfile as sf, sentencepiece as spm, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
from wer_utils import load_librispeech, wer  # noqa: E402
from nemo_small import NemoSmall, cfg_from_state  # noqa: E402
from train_nemo_stream import BatchPre, SModel, ctc_greedy  # noqa: E402

DATA = os.path.join(HERE, "..", "data")


def load_set(name, limit=0):
    if os.path.exists(f"{DATA}/LibriSpeech/{name}"):
        items = [(p, t) for _, p, t in load_librispeech(name)]
    else:
        items = []
        for l in open(f"{DATA}/val/{name}.local.jsonl"):
            d = json.loads(l)
            items.append((d["wav"], d["text"]))
    if limit and len(items) > limit:
        items = items[:: len(items) // limit][:limit]  # evenly spread subset
    return items


def build(ckpt, key="model", sub_ch=0, stack=1, keep=None, dev="cpu"):
    """-> (model, norm_mean, norm_std)"""
    st = torch.load(ckpt, map_location="cpu")
    sd = st[key]
    m = NemoSmall(cfg=st.get("cfg") or cfg_from_state(sd))
    m.load_state_dict(sd)
    if sub_ch or stack != 1 or keep is not None:
        m = m.convert(sub_ch, stack, keep)
    return m.eval().to(dev), st.get("norm_mean"), st.get("norm_std")


def evaluate(model, pre, sp, items, chunk, left, bits, dev, batch_sec=200.0):
    net = SModel(model, bits).eval()
    V = model.vocab
    wavs = []
    for p, t in items:
        w, sr = sf.read(p, dtype="float32")
        if w.ndim > 1:
            w = w[:, 0]
        assert sr == 16000, (p, sr)
        wavs.append(torch.from_numpy(w))
    order = sorted(range(len(items)), key=lambda i: len(wavs[i]))
    hyps = [None] * len(items)
    i = 0
    with torch.no_grad():
        while i < len(order):
            j = i + 1
            while j < len(order) and (j - i + 1) * len(wavs[order[j]]) / 16000 <= batch_sec:
                j += 1
            idx = order[i:j]
            L = torch.tensor([len(wavs[k]) for k in idx])
            W = torch.zeros(len(idx), int(L.max()))
            for r, k in enumerate(idx):
                W[r, : len(wavs[k])] = wavs[k]
            f, fl = pre(W.to(dev), L.to(dev))
            lg, ol = net(f, fl, chunk, left)
            for r, k in enumerate(idx):
                hyps[k] = sp.decode(ctc_greedy(lg[r], int(ol[r]), V))
            i = j
    w, _ = wer([t for _, t in items], hyps)
    return w, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tnm", default="", help="evaluate the weights read back from an exported .tnm (int8 / int4 grid)")
    ap.add_argument("--key", default="model", help="'model' (last.pt holds the EMA weights there) or 'ema' (ckpt.pt)")
    ap.add_argument("--nemo", default=os.path.join(HERE, "..", "models", "nemo_small"), help="dir with tokenizer.model")
    ap.add_argument("--tok", default="")
    ap.add_argument("--sets", default="test-clean,test-other")
    ap.add_argument("--modes", default="full,8/64")
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--batch_sec", type=float, default=200.0)
    ap.add_argument("--sub_ch", type=int, default=0)
    ap.add_argument("--stack", type=int, default=1)
    ap.add_argument("--keep_layers", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    keep = [int(x) for x in a.keep_layers.split(",")] if a.keep_layers else None
    if a.tnm:
        from tnm_read import read_tnm
        model, h, mean, std, _ = read_tnm(a.tnm)
        model = model.to(a.dev)
        a.bits = a.bits or h["bits"]
    else:
        model, mean, std = build(a.ckpt, a.key, a.sub_ch, a.stack, keep, a.dev)
    pre = BatchPre(model.pre.window.cpu(), model.pre.fb.cpu(), mean, std).to(a.dev)
    sp = spm.SentencePieceProcessor(model_file=a.tok or f"{a.nemo}/tokenizer.model")
    print("cfg", model.cfg, "bits", a.bits, "fixed norm", mean is not None, flush=True)
    res = {}
    for s in a.sets.split(","):
        items = load_set(s, a.limit)
        for mode in a.modes.split(","):
            chunk, left = (0, 0) if mode == "full" else [int(x) for x in mode.split("/")]
            t0 = time.time()
            w, hyps = evaluate(model, pre, sp, items, chunk, left, a.bits, a.dev, a.batch_sec)
            res[f"{s}/{mode}"] = round(w, 2)
            print(f"{s:12s} {mode:8s} n={len(items):5d} WER {w:6.2f}  ({time.time() - t0:.0f}s)", flush=True)
    if a.json:
        json.dump({"ckpt": a.ckpt or a.tnm, "bits": a.bits, "cfg": model.cfg, "wer": res}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
