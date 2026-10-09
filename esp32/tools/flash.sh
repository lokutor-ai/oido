#!/bin/bash
# Build + flash firmware, model and LM to an ESP32-S3 N16R8 (16 MB flash, 8 MB octal PSRAM).
# usage: tools/flash.sh <port> <model.tnm|model.tasr> [lm.tlm] [clip.wav "reference" ...]
#   CTC models use models/nemo_lm.tlm by default when it exists; NO_LM=1 flashes greedy decoding only
#   TASR_MODE=mic  (default) live INMP441 microphone: speak, the transcript prints after each pause
#   TASR_MODE=file transcribe the given clips from flash and print WER and the measured real-time factor
#   TASR_OLED=1    also drive an SSD1306 128x64 I2C OLED (SDA GPIO8, SCL GPIO9) with the live transcript
#   TASR_DUAL=1    experimental: split the work across both cores (wrong transcripts on silicon for now, see README)
set -e
PORT=$1; MODEL=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift 2
LM=""; if [[ "${1:-}" == *.tlm ]]; then LM=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift; fi
DEFLM="$(cd "$(dirname "$0")/../.." && pwd)/models/nemo_lm.tlm"
[ -f "${MODEL%.tnm}.tlm" ] && DEFLM="${MODEL%.tnm}.tlm"   # a model's own LM (oido_es.tlm), else the English one
[ -z "$LM" ] && [ -z "$NO_LM" ] && [[ "$MODEL" == *.tnm ]] && [ -f "$DEFLM" ] && LM=$DEFLM
[ -n "$NO_LM" ] && LM=""
WAVS=(); REFS=()
while [ $# -gt 1 ]; do WAVS+=("$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"); REFS+=("$2"); shift 2; done
MODE=${TASR_MODE:-mic}
PY=${PYTHON:-$(command -v python3)}   # needs numpy + soundfile (ESP-IDF's own python does not have them)
cd "$(dirname "$0")/.."
. "${IDF_PATH:-$HOME/esp/esp-idf}/export.sh" > /dev/null
case "$MODEL" in
  *nemo4*.tnm) LAYOUT=nemo4_16; CFG="sdkconfig.defaults;sdkconfig.nemo4" ;;    # int4: 6 MB left for your app
  *rnnt*.tnm) LAYOUT=rnnt16; CFG="sdkconfig.defaults;sdkconfig.rnnt16"; LM="" ;;   # transducer: no external LM
  *.tnm)      LAYOUT=nemo16lm; CFG="sdkconfig.defaults;sdkconfig.nemo16lm" ;;
  *)          LAYOUT=tinyasr16; CFG="sdkconfig.defaults" ;;
esac
[ "$MODE" = mic ] && CFG="$CFG;sdkconfig.mic"
SUF=""; [ -n "$TASR_OLED" ] && CFG="$CFG;sdkconfig.oled" && SUF=_oled
[ -n "$TASR_PACED" ] && CFG="$CFG;sdkconfig.paced" && SUF=${SUF}_paced   # file mode: also stream at real-time pace, report the final-text delay
[ -n "$TASR_DUAL" ] && SUF=${SUF}_2c
B=build_${LAYOUT}_${MODE}${SUF}
(cd firmware && idf.py -B $B -D SDKCONFIG=sdkconfig_${LAYOUT}_${MODE}${SUF} -D SDKCONFIG_DEFAULTS="$CFG" -D TASR_2C=${TASR_DUAL:-0} build)
OUT=build_images/${LAYOUT}_${MODE}; mkdir -p $OUT
LMARG=""; [ -n "$LM" ] && LMARG="--lm $LM"
"$PY" tools/mkimages.py "$MODEL" $OUT --layout $LAYOUT --wavs "${WAVS[@]}" --refs "${REFS[@]}" $LMARG --merge firmware/$B --flash_mode keep
AFTER=hard_reset; case "$PORT" in *usbmodem*|*ttyACM*) AFTER=watchdog_reset ;; esac   # native USB: an RTS reset lands in download mode
esptool.py --chip esp32s3 -p "$PORT" -b 921600 --after $AFTER write_flash 0x0 $OUT/flash.bin
echo "flashed ($LAYOUT, $MODE). monitor: (cd firmware && idf.py -B $B -p $PORT monitor)"
