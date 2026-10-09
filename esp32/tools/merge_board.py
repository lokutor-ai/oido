#!/usr/bin/env python3
"""Merge board_bench.py result files into research/board_s3.json (the numbers quoted in the README, the paper and the HF cards).

usage: merge_board.py out.json results_dir [profile_json]
  results_dir has results_<name>.json (and results_<name>.json.log with the PROF lines) as written by board_bench.py
"""
import collections, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_board import summarize

out, rd = sys.argv[1], sys.argv[2]
prof_json = sys.argv[3] if len(sys.argv) > 3 else None
res = {}
for name in ("en_nemo8", "en_nemo4", "en_stream", "es", "en_rnnt8", "en_stream_paced", "es_paced"):
    p = os.path.join(rd, f"results_{name}.json")
    if os.path.exists(p):
        res[name] = summarize(json.load(open(p)))

# compute seconds of the short commands (<= 3.9 s) and of the clips shared with the paced run, for the latency table
def short(name, cond=lambda c: c["audio"] <= 3.9):
    d = json.load(open(os.path.join(rd, f"results_{name}.json")))
    return [{"audio": round(c["audio"], 2), "compute_s": round(c["rtf"] * c["audio"], 2),
             "paced_final_s": (round(c["paced_final_ms"] / 1000, 2) if "paced_final_ms" in c else None)} for c in d["clips"] if cond(c)]
for name in ("en_nemo8", "en_nemo4", "en_stream_paced", "es_paced"):
    if name in res:
        res[name]["short_clips"] = short(name)
if "en_stream_paced" in res and "en_nemo8" in res:
    paced = json.load(open(os.path.join(rd, "results_en_stream_paced.json")))["clips"]
    n8 = {c["path"]: c for c in json.load(open(os.path.join(rd, "results_en_nemo8.json")))["clips"]}
    res["en_stream_paced"]["same_clips_compute_s"] = {
        "stream_model_utterance": [round(c["rtf"] * c["audio"], 2) for c in paced],
        "original_nemo8_utterance": [round(n8[c["path"]]["rtf"] * c["audio"], 2) for c in paced if c["path"] in n8]}

# cycles per stage on the board against instructions per stage in the emulator (same model, the 27 English clips)
if prof_json and "en_nemo8" in res:
    board = collections.Counter()
    for l in open(os.path.join(rd, "results_en_nemo8.json.log")):
        m = re.match(r"PROF (\S+)\s+(\d+) cycles", l)
        if m:
            board[m.group(1)] += int(m.group(2))
    aud = res["en_nemo8"]["audio_s"]
    q = json.load(open(prof_json))
    qs, qa = q["stages"], q["audio_s"]
    rows = [("log-mel", "features", "log-mel"), ("conv0", "conv0", "conv0"), ("im2col", "im2col", "im2col"),
            ("matrix products", "gemm*", "matrix products"), ("layernorm", "layernorm", "layernorm"),
            ("activation quant", "quant", "activation quant"), ("SiLU / GLU", "act", "SiLU / GLU"),
            ("q/k/v int8", "qkv_int8", "q/k/v int8"), ("pos. enc.", "pos", "pos. enc."), ("attention", "attention", "attention"),
            ("depthwise conv", "dwconv", "depthwise conv")]
    stages = []
    for label, bk, qk in rows:
        b = (board["gemm_fe"] + board["gemm_k704"] + board["gemm"] if bk == "gemm*" else board[bk]) / aud / 1e6
        i = qs[qk] * 25 / qa / 1e6
        stages.append({"stage": label, "emulator_Minstr_per_s": round(i, 1), "board_Mcycles_per_s": round(b, 1),
                       "cycles_per_instruction": round(b / i, 2)})
    lm_i = (qs["head + decode"] * 25 / qa / 1e6) + q["lm_extra_minstr_s"]
    lm_b = board["head+dec"] / aud / 1e6
    stages.append({"stage": "head + beam search with LM", "emulator_Minstr_per_s": round(lm_i, 1),
                   "board_Mcycles_per_s": round(lm_b, 1), "cycles_per_instruction": round(lm_b / lm_i, 2)})
    tot_b = sum(board.values()) / aud / 1e6
    tot_i = sum(s["emulator_Minstr_per_s"] for s in stages)
    res["profile"] = {"stages": stages, "board_total_Mcycles_per_s": round(tot_b, 1), "emulator_total_Minstr_per_s": round(tot_i, 1),
                      "cycles_per_instruction": round(tot_b / tot_i, 2), "audio_s": aud}
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
print({k: (v.get("rtf_mean") if isinstance(v, dict) and "rtf_mean" in v else "profile") for k, v in res.items()})
