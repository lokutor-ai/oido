#!/usr/bin/env python3
"""Beam / top-k / LM sweep on the host build: WER and CPU time of tasr_cli for one model on LibriSpeech subsets.
usage: python tools/decoder_sweep.py MODEL.tnm [--lm lm.tlm] [--n 120] [--nstream]"""
import argparse, glob, os, random, subprocess, sys, tempfile, time
import numpy as np, soundfile as sf, jiwer
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--lm", default=""); ap.add_argument("--n", type=int, default=120)
ap.add_argument("--nstream", action="store_true"); ap.add_argument("--configs", default="none;4,6;3,6;2,6;3,4;2,4;2,3;1,3")
a = ap.parse_args()
rng = random.Random(3); tmp = tempfile.mkdtemp()
sets = {}
for split in ("test-clean", "test-other"):
    items = []
    for tf in glob.glob(f"{ROOT}/data/LibriSpeech/{split}/*/*/*.trans.txt"):
        for line in open(tf):
            u, t = line.strip().split(" ", 1); items.append((os.path.join(os.path.dirname(tf), u + ".flac"), t.lower()))
    rng.shuffle(items); sel = []
    for f, t in items:
        if sf.info(f).duration <= 12 and len(sel) < a.n // 2: sel.append((f, t))
    paths = []
    for i, (f, t) in enumerate(sel):
        x, _ = sf.read(f, dtype="float32"); p = f"{tmp}/{split}_{i}.wav"
        sf.write(p, np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16), 16000, subtype="PCM_16"); paths.append(p)
    sets[split] = (paths, [t for _, t in sel])
cli = f"{ROOT}/esp32/host/tasr_cli"
print(f"model {os.path.basename(a.model)} lm {os.path.basename(a.lm) or '-'} {'streaming' if a.nstream else 'utterance'}; {a.n // 2} clean + {a.n // 2} other utterances")
for cfg in a.configs.split(";"):
    cmd = [cli, a.model] + (["--nstream"] if a.nstream else [])
    if cfg != "none" and a.lm:
        b, k = cfg.split(","); cmd += ["--lm", a.lm, "--beam", b, "--topk", k]
    elif cfg != "none":
        continue
    row = []; tot_cpu = 0.0
    for split, (paths, refs) in sets.items():
        t0 = time.time(); r = subprocess.run(cmd + paths, capture_output=True, text=True); dt = time.time() - t0
        hyp = {l.split("\t")[0]: l.split("\t", 1)[1].strip() for l in r.stdout.splitlines() if "\t" in l}
        row.append(100 * jiwer.wer(refs, [hyp.get(p, "") for p in paths])); tot_cpu += dt
    print(f"  {'greedy (no LM)' if cfg == 'none' else 'beam %s topk %s' % tuple(cfg.split(',')):18s} WER clean {row[0]:5.2f}  other {row[1]:5.2f}   host time {tot_cpu:5.1f} s")
