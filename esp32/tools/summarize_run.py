#!/usr/bin/env python3
"""Summarize one board_bench run: utterance and streaming RTF, compute left after the last sample, and the streaming-only
cycles per audio second of each stage (SPROF lines of the firmware, summed over batches).
usage: python tools/summarize_run.py bench/m2_b12i4.json"""
import json, re, sys
p = sys.argv[1]
d = json.load(open(p)); cl = d["clips"]; a = sum(c["audio"] for c in cl)
feed = sum(c.get("stream_feed_us", 0) for c in cl) / 1e6; fin = sum(c.get("stream_finish_us", 0) for c in cl) / 1e6
print(f"{p}: {len(cl)} clips, {a:.1f} s; utterance RTF {d['rtf_mean']:.3f}; streaming RTF {(feed + fin) / a:.3f} (feed {feed / a:.3f} + finish {fin / a:.3f}); "
      f"compute after the last sample {fin / len(cl) * 1000:.0f} ms avg; PSRAM/internal min free {d['min_free_psram_internal']}")
st, tot_cyc, tot_aud = {}, 0.0, 0.0
for line in open(p + ".log"):
    m = re.match(r"SPROF (\S+)\s+(\d+) cycles", line)
    if m and "total" not in line:
        st[m.group(1)] = st.get(m.group(1), 0) + int(m.group(2))
    m = re.match(r"SPROF total stream cycles (\d+) over ([\d.]+) s", line)
    if m:
        tot_cyc += float(m.group(1)); tot_aud += float(m.group(2))
if tot_aud:
    print(f"streaming only: {tot_cyc / tot_aud / 1e6:.1f} Mcycles per audio second (RTF {tot_cyc / tot_aud / 240e6:.3f}); per stage:")
    for k, v in sorted(st.items(), key=lambda kv: -kv[1]):
        print(f"   {k:10s} {v / tot_aud / 1e6:6.1f} Mc/s  {100 * v / tot_cyc:5.1f} %")
