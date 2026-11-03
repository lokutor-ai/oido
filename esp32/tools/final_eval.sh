#!/bin/bash
# WER of one model with the host build (== chip arithmetic) on the LibriSpeech test sets, in the decoding configurations that matter.
# usage: tools/final_eval.sh MODEL.tnm LM.tlm [out.txt]
cd "$(dirname "$0")/../host"
M=$1; LM=$2; OUT=${3:-/dev/stdout}
PY=${PYTHON:-$HOME/miniforge3/bin/python3}
run() { echo "== $1"; shift; $PY eval_engine.py "$M" "$@" --procs 8 2>&1 | grep -E "^test"; }
{
run "full context, greedy" 
run "stream chunk 8 (0.64 s), greedy" --stream
run "stream chunk 16 (1.28 s), greedy" --stream --sc 16
run "stream chunk 8, PD LM beam 4 top-k 6" --stream --lm $LM --beam 4 --topk 6
run "stream chunk 8, PD LM beam 2 top-k 3" --stream --lm $LM --beam 2 --topk 3
run "stream chunk 16, PD LM beam 4 top-k 6" --stream --sc 16 --lm $LM --beam 4 --topk 6
run "full context, PD LM beam 4 top-k 6" --lm $LM --beam 4 --topk 6
} | tee $OUT
