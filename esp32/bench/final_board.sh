#!/bin/bash
# Final board evaluation of one model (name under ../models without .tnm; int4 files must contain "nemo4" in the name for the 16 MB int4 layout):
#  1. all 18 clips, stream-only build, PD LM, beam 2 / top-k 3: WER vs references, equality with the host build, streaming RTF
#  2. 8 clips paced at real-time speed: final text delay after the end of speech
# usage: bench/final_board.sh MODEL TAG [quick]
cd "$(dirname "$0")/.."
M=$1; TAG=$2; Q=$3
CLIPS=bench/clips_det.json LIMIT=18 BATCH=9 bench/run_cfg.sh fin_$TAG $M ../models/nemo_lm_pd_v2.tlm "sdkconfig.streamonly;sdkconfig.fastdec" $Q > /dev/null 2>&1
echo "DONE fin_$TAG (18 clips): $(python3 tools/summarize_run.py bench/r_fin_$TAG.json 2>&1 | head -1 | cut -c1-240)"
TASR_PACED=1 LIMIT=8 bench/run_cfg.sh finp_$TAG $M ../models/nemo_lm_pd_v2.tlm "sdkconfig.streamonly;sdkconfig.fastdec" quick > /dev/null 2>&1
python3 - <<PY
import json
d=json.load(open('bench/r_finp_$TAG.json'))['clips']; v=[c['paced_final_ms'] for c in d if c.get('paced_final_ms') is not None]
print("DONE finp_$TAG: paced final text after end of speech: mean %.0f ms (min %d, max %d)" % (sum(v)/len(v), min(v), max(v)))
PY
echo ALLDONE
