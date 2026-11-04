#!/bin/bash
# Recognizer with ALL weights in flash (PSRAM free for something else): A2 (quick, model on board) then B3 (full flash). 18 clips, stream-only, chunk 8, beam 2 / top-k 3.
cd "$(dirname "$0")/.."
run() { CLIPS=bench/clips_det.json LIMIT=18 BATCH=9 bench/run_cfg.sh $1 $2 ../models/nemo_lm_pd_v2.tlm "sdkconfig.streamonly;sdkconfig.fastdec;sdkconfig.nopsram" $3 > /dev/null 2>&1
  echo "DONE $1: $(python3 tools/summarize_run.py bench/r_$1.json 2>&1 | head -1 | cut -c1-240)"; grep -h "moved to PSRAM" bench/r_$1.json.log | sort -u | cut -c1-200; }
run nops_a2 nemo4_v2a2_final quick
run nops_b3 nemo4_v2b3_final ""
echo ALLDONE11
