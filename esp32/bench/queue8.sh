#!/bin/bash
# B3 int4 (agent export 00:34): final_board (18 clips fast decoder + paced 8 clips)
cd "$(dirname "$0")/.."
bench/final_board.sh nemo4_v2b3_final b3final
echo ALLDONE8
