#!/bin/bash
# A2 int4 (agent export 00:42): fast decoder (final_board), full decoder 4/6 (18 clips + paced), chunk 16 with 4/6 (paced 8 clips)
cd "$(dirname "$0")/.."
bench/final_board.sh nemo4_v2a2_final a2final
CLIPS=bench/clips_det.json LIMIT=18 BATCH=9 bench/run_cfg.sh fin_a2d46 nemo4_v2a2_final ../models/nemo_lm_pd_v2.tlm "sdkconfig.streamonly" quick > /dev/null 2>&1
echo "DONE fin_a2d46 (18 clips, beam 4/6): $(python3 tools/summarize_run.py bench/r_fin_a2d46.json 2>&1 | head -1 | cut -c1-240)"
for t in "finp_a2d46:sdkconfig.streamonly" "finp_a2c16:sdkconfig.streamonly;sdkconfig.chunk16"; do
  tag=${t%%:*}; frag=${t#*:}
  TASR_PACED=1 LIMIT=8 bench/run_cfg.sh $tag nemo4_v2a2_final ../models/nemo_lm_pd_v2.tlm "$frag" quick > /dev/null 2>&1
  python3 - <<PY
import json
d=json.load(open('bench/r_$tag.json'))['clips']; v=[c['paced_final_ms'] for c in d if c.get('paced_final_ms') is not None]
print("DONE $tag: paced final %.0f ms (min %d, max %d) | %s" % (sum(v)/len(v), min(v), max(v), "$tag"))
PY
  python3 tools/summarize_run.py bench/r_$tag.json 2>&1 | head -1 | cut -c1-200
done
echo ALLDONE7
