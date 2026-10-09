#!/usr/bin/env python3
"""Summarize board_bench.py result files into one JSON (the numbers quoted in the README and the paper).

usage: summarize_board.py out.json name=results_a.json [name=results_b.json ...]
"""
import json, sys


def edits(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def wer(clips, key):
    e = n = 0
    for c in clips:
        if c.get(key) is not None:
            a, b = edits(c["ref"], c[key]); e += a; n += b
    return round(100.0 * e / n, 2) if n else None


def summarize(d):
    cl = d["clips"]
    aud = sum(c["audio"] for c in cl)
    sec = sum(c["rtf"] * c["audio"] for c in cl)  # compute seconds (wall clock when the firmware reports it)
    s = {"model": d["model"], "dual_core": d.get("dual", False), "clips": len(cl), "audio_s": round(aud, 1),
         "rtf_mean": round(sec / aud, 3),
         "rtf_min": round(min(c["rtf"] for c in cl), 3), "rtf_max": round(max(c["rtf"] for c in cl), 3),
         "wer_board": wer(cl, "hyp"), "wer_host": wer(cl, "host_hyp"), "wer_qemu": wer(cl, "qemu_hyp"),
         "min_free_psram": (d.get("min_free_psram_internal") or [None, None])[0],
         "min_free_internal": (d.get("min_free_psram_internal") or [None, None])[1]}
    q = [c for c in cl if c.get("qemu_hyp") is not None]
    if q:
        s["board_eq_qemu"] = f"{sum(c['hyp'] == c['qemu_hyp'] for c in q)}/{len(q)}"
        ins = [c for c in q if c.get("instr")]
        if ins:  # cycles per executed instruction (instructions summed over both emulated cores, board ran one core)
            s["cycles_per_instruction"] = round(sum(c["rtf"] * c["audio"] for c in ins) * 240e6 / sum(c["instr"] for c in ins), 2)
    h = [c for c in cl if c.get("host_hyp") is not None]
    if h:
        s["board_eq_host"] = f"{sum(c['hyp'] == c['host_hyp'] for c in h)}/{len(h)}"
    st = [c for c in cl if "stream_finish" in c]
    if st:
        fin = [c["stream_finish_us"] / 1e6 if "stream_finish_us" in c else c["stream_finish"] / 240e6 for c in st]
        s["stream"] = {"finish_s_mean": round(sum(fin) / len(fin), 2), "finish_s_min": round(min(fin), 2),
                       "finish_s_max": round(max(fin), 2),
                       "rtf_mean": round(sum((c["stream_feed_us"] + c["stream_finish_us"]) / 1e6 if "stream_feed_us" in c else (c["stream_feed"] + c["stream_finish"]) / 240e6 for c in st) / sum(c["audio"] for c in st), 3),
                       "wer_board_stream": wer([dict(c, s=c.get("shyp")) for c in st], "s"),
                       "stream_eq_utterance": f"{sum(c.get('shyp') == c['hyp'] for c in st)}/{len(st)}"}
        hs = [c for c in st if c.get("host_shyp") is not None]
        if hs:
            s["stream"]["stream_eq_host"] = f"{sum(c['shyp'] == c['host_shyp'] for c in hs)}/{len(hs)}"
            s["stream"]["wer_host_stream"] = wer([dict(c, s=c["host_shyp"]) for c in hs], "s")
    pc = [c for c in cl if "paced_final_ms" in c]
    if pc:
        f = [c["paced_final_ms"] / 1000 for c in pc]
        s["paced"] = {"clips": len(pc), "final_text_after_speech_s_mean": round(sum(f) / len(f), 2),
                      "min": round(min(f), 2), "max": round(max(f), 2),
                      "wer_board_paced": wer([dict(c, s=c.get("phyp")) for c in pc], "s"),
                      "per_clip": [{"audio": c["audio"], "final_s": round(c["paced_final_ms"] / 1000, 2)} for c in pc]}
    s["per_clip"] = [{"audio": c["audio"], "rtf": round(c["rtf"], 3), "hyp": c["hyp"], "ref": c["ref"],
                      "eq_qemu": (c["hyp"] == c["qemu_hyp"]) if c.get("qemu_hyp") is not None else None,
                      "eq_host": (c["hyp"] == c["host_hyp"]) if c.get("host_hyp") is not None else None} for c in cl]
    return s


if __name__ == "__main__":
    out = {}
    for a in sys.argv[2:]:
        name, path = a.split("=", 1)
        out[name] = summarize(json.load(open(path)))
        o = out[name]
        print(name, {k: v for k, v in o.items() if k != "per_clip"})
    json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
