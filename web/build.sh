#!/bin/bash
# Build the in-browser demo's engine (docs/demo/oido.js + oido.wasm) with emscripten:  source <emsdk>/emsdk_env.sh && web/build.sh
set -e
cd "$(dirname "$0")"
E=../esp32/components/tinyasr
emcc -O3 -std=c11 -D_DEFAULT_SOURCE -ffp-contract=off -I$E/include -I$E oido_wasm.c $E/tinyasr.c $E/kernels.c $E/tinyasr_lm.c $E/tasr_nemo.c \
  -o ../docs/demo/oido.js -sMODULARIZE=1 -sEXPORT_NAME=OidoModule -sENVIRONMENT=worker,web,node -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=67108864 -sSTACK_SIZE=1048576 \
  -sEXPORTED_FUNCTIONS=_oido_alloc,_oido_free,_oido_load,_oido_unload,_oido_reset,_oido_feed,_oido_partial,_oido_finish,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,UTF8ToString,HEAP16 -sFILESYSTEM=0 -lm
ls -la ../docs/demo/oido.js ../docs/demo/oido.wasm
