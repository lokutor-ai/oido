# esp32/bench

Raw outputs of `esp32/tools/board_bench.py` and the scripts that drive it.

## Known defect in the committed `r_fin_*.json` and `r_finp_*.json` files

In the streaming runs the per-clip `rtf` and `wall_us` fields are wrong (for example `"rtf": 1.9e-07` and `"wall_us": 1`), and so is the file-level `rtf_mean`. The parser took them from the `UTT` line, which only covers loading the clip, instead of from the `STREAM` line. Do not use these fields.

The headline real-time factors come from the `stream_feed_us` and `stream_finish_us` fields, summed over the clips and divided by the audio duration: the sum of the two gives 0.698 and 0.722 (the files for the A models) and 0.537 (the files for the B models), which are the figures in the root README (0.72, 0.54 and the re-run row) (feed alone gives 0.656 to 0.678 and 0.508). The final-text latencies come from `paced_final_ms` in the `r_finp_*.json` files and are not affected.

`board_bench.py` is fixed (`esp32/tools/test_board_bench_parse.py` covers it), so a re-run will write correct `rtf`, `wall_us` and `rtf_mean` fields. The committed files were not edited, because their numbers come from a board run that is not repeated here.
