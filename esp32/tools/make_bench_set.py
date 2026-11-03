#!/usr/bin/env python3
"""Build the speed-benchmark clip set for board_bench.py: a fixed mix of clean, other and noisy/meeting clips (16 kHz PCM16 wavs),
with the reference text and the transcript of the host build of the current engine (the bit-exactness reference for kernel changes).

usage: python tools/make_bench_set.py OUTDIR [--model ../models/nemo8.tnm] [--lm ../models/nemo_lm.tlm]
Needs numpy + soundfile; uses esp32/host/tasr_cli (make -C esp32/host).
"""
import argparse, glob, json, os, random, subprocess
import numpy as np, soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SETS = [  # (manifest, how many, min dur, max dur)
    ("data/val/val_ps.local.jsonl", 3, 3.5, 9.0),
    ("data/val/val_cv.local.jsonl", 3, 3.5, 9.0),
    ("data/val/val_vox.local.jsonl", 3, 4.0, 10.0),
    ("data/val/val_ami.local.jsonl", 3, 4.0, 10.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default=os.path.join(ROOT, "models/nemo8.tnm"))
    ap.add_argument("--lm", default=os.path.join(ROOT, "models/nemo_lm.tlm"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    clips = []
    for man, n, lo, hi in SETS:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, man))]
        rows = [r for r in rows if lo <= r["duration"] <= hi and os.path.exists(r["wav"])]
        for r in rng.sample(rows, n):
            clips.append(r)
    # LibriSpeech test-clean / test-other: refs from the .trans.txt files
    for split, n in (("test-clean", 3), ("test-other", 3)):
        items = []
        for tf in glob.glob(os.path.join(ROOT, "data/LibriSpeech", split, "*", "*", "*.trans.txt")):
            for line in open(tf):
                uid, text = line.strip().split(" ", 1)
                items.append((os.path.join(os.path.dirname(tf), uid + ".flac"), text.lower()))
        rng.shuffle(items)
        got = 0
        for wav, text in items:
            info = sf.info(wav)
            if 3.5 <= info.duration <= 9.0:
                clips.append({"wav": wav, "text": text, "duration": info.duration, "id": os.path.basename(wav)[:-5]})
                got += 1
                if got == n:
                    break
    out = []
    for i, r in enumerate(clips):
        x, sr = sf.read(r["wav"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        assert sr == 16000, (r["wav"], sr)
        path = os.path.join(a.out, f"b{i:02d}.wav")
        sf.write(path, np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16), 16000, subtype="PCM_16")
        out.append({"path": path, "ref": r["text"].lower().strip(), "dur": round(len(x) / 16000, 3), "id": r["id"]})
    cli = os.path.join(ROOT, "esp32/host/tasr_cli")
    cmd = [cli, a.model] + (["--lm", a.lm] if a.lm and os.path.exists(a.lm) else []) + [c["path"] for c in out]
    res = subprocess.run(cmd, capture_output=True, text=True)
    hyps = {l.split("\t", 1)[0]: l.split("\t", 1)[1] for l in res.stdout.splitlines() if "\t" in l}
    for c in out:
        c["host_hyp"] = hyps.get(c["path"], "")
    open(os.path.join(a.out, "host_cli.txt"), "w").write(res.stdout + res.stderr)
    print("tasr_cli output lines:", len(res.stdout.splitlines()), "(see host_cli.txt)")
    json.dump(out, open(os.path.join(a.out, "clips.json"), "w"), indent=1)
    print(len(out), "clips,", round(sum(c["dur"] for c in out), 1), "s")


if __name__ == "__main__":
    main()
