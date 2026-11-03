#!/usr/bin/env python3
"""Tune the LM weight / token bonus / beam of the on-chip beam search on a dev subset, with the host build (== chip arithmetic).
usage: python tools/tune_lm.py MODEL.tnm LM.tlm [--stream] [--sets dev-clean dev-other] [--limit 300]
Prints WER per setting; the best setting is what to store in the LM header / Kconfig."""
import argparse, itertools, json, os, subprocess, sys
HOST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "host")
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("lm"); ap.add_argument("--stream", action="store_true")
ap.add_argument("--sets", nargs="*", default=["dev-clean"]); ap.add_argument("--manifest", nargs="*", default=[]); ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--weights", default="0.2,0.3,0.4,0.5"); ap.add_argument("--bonus", default="0.3,0.6,1.0")
ap.add_argument("--beams", default="2,4"); ap.add_argument("--procs", type=int, default=8)
a = ap.parse_args()
rows = []
for beam, w, b in itertools.product(a.beams.split(","), a.weights.split(","), a.bonus.split(",")):
    cmd = [sys.executable, "eval_engine.py", a.model, "--lm", a.lm, "--beam", beam, "--lm_weight", w, "--token_bonus", b, "--topk", "3" if beam == "2" else "6",
           "--sets", *a.sets, "--manifest", *a.manifest, "--limit", str(a.limit), "--procs", str(a.procs)] + (["--stream"] if a.stream else [])
    out = subprocess.run(cmd, cwd=HOST, capture_output=True, text=True).stdout
    res = json.loads(out.strip().splitlines()[-1])
    avg = sum(res.values()) / len(res)
    rows.append((avg, beam, w, b, res))
    print(f"beam {beam} lm_weight {w} bonus {b}: " + "  ".join(f"{k} {v:.2f}" for k, v in res.items()) + f"   mean {avg:.2f}", flush=True)
rows.sort()
print("best:", rows[0][1:4], f"mean WER {rows[0][0]:.2f}")
