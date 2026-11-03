#!/bin/bash
# PSRAM headroom study (18 clips, stream-only, chunk 8, beam 2 / top-k 3): reserve 900 KB (live-microphone mode) and 100 KB, for A and B
cd "$(dirname "$0")/.."
run() {  # tag model fragments quick
  CLIPS=bench/clips_det.json LIMIT=18 BATCH=9 bench/run_cfg.sh $1 $2 ../models/nemo_lm_pd_v2.tlm "$3" $4 > /dev/null 2>&1
  echo "DONE $1: $(python3 tools/summarize_run.py bench/r_$1.json 2>&1 | head -1 | cut -c1-240)"
  grep -h "moved to PSRAM" bench/r_$1.json.log | sort -u | cut -c1-200
}
run mem_a900 nemo4_v2a_final "sdkconfig.streamonly;sdkconfig.fastdec;sdkconfig.micmem" quick
printf 'CONFIG_TASR_PSRAM_RESERVE_KB=100\n' > firmware/sdkconfig.res100
run mem_a100 nemo4_v2a_final "sdkconfig.streamonly;sdkconfig.fastdec;sdkconfig.res100" quick
run mem_b900 nemo4_v2b_final "sdkconfig.streamonly;sdkconfig.fastdec;sdkconfig.micmem" ""
run mem_b100 nemo4_v2b_final "sdkconfig.streamonly;sdkconfig.fastdec;sdkconfig.res100" quick
echo ALLDONE6
