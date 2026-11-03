#!/bin/bash
# Long-utterance test: the first 4 bench clips (4-8 s) repeated 3x and 12x into ONE stream (15-90 s of continuous speech), audio fed at real-time speed.
# B3 (beam 2 / top-k 3), then A2 (beam 4 / top-k 6): final text delay after the end of speech.
cd "$(dirname "$0")/.."
rep() {  # tag model frags quick
  TASR_PACED=1 CLIPS=bench/clips_det.json LIMIT=4 BATCH=9 bench/run_cfg.sh $1 $2 ../models/nemo_lm_pd_v2.tlm "$3" $4 > /dev/null 2>&1
  python3 - <<PY
import json
d=json.load(open('bench/r_$1.json'))['clips']
for c in d:
    if c.get('paced_final_ms') is not None: print("DONE $1: audio %.1f s -> final text %d ms after the end of speech" % (c.get('paced_audio', c.get('audio', 0)), c['paced_final_ms']))
PY
}
rep long3_b nemo4_v2b3_final "sdkconfig.streamonly;sdkconfig.fastdec;sdkconfig.repeat3" quick
rep long12_b nemo4_v2b3_final "sdkconfig.streamonly;sdkconfig.fastdec;sdkconfig.repeat12" quick
rep long3_a nemo4_v2a2_final "sdkconfig.streamonly;sdkconfig.repeat3" ""
rep long12_a nemo4_v2a2_final "sdkconfig.streamonly;sdkconfig.repeat12" quick
echo ALLDONE9
