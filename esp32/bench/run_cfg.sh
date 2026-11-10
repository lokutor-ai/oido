#!/bin/bash
# usage: run_cfg.sh TAG MODEL LM EXTRA_CFG [QUICK]   one board_bench run (8 clips) with an extra sdkconfig fragment list
cd "$(dirname "$0")/.."
. "$IDF_PATH/export.sh" >/dev/null 2>&1
export TASR_UART=1 PYTHON=$HOME/miniforge3/bin/python3
TAG=$1; MODEL=$2; LM=$3; CFG=$4
if [ -n "$5" ]; then export TASR_QUICK=1; fi
if [ -n "$CFG" ]; then export TASR_EXTRA_CFG="$CFG"; fi
LMARG="--no-lm"; [ "$LM" != none ] && LMARG="--lm $LM"
python tools/board_bench.py --port ${PORT:?set PORT} --model ../models/$MODEL.tnm $LMARG --clips ${CLIPS:-bench/clips.json} --limit ${LIMIT:-8} --batch ${BATCH:-9} --out bench/r_$TAG.json > bench/r_$TAG.out 2>&1
echo "DONE $TAG: $(grep TOTAL bench/r_$TAG.out)"
