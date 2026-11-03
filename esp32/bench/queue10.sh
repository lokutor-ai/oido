#!/bin/bash
# Board 1 (Ito's board, now on a UART bridge; Dani authorized its use on 3 Nov): back up the whole flash, replicate the final v2 measurements on a second board
# (A2 int4 fast decoder 18 clips + paced, B3 int4 same), then restore the original flash image.
cd "$(dirname "$0")/.."
export PORT=/dev/cu.usbmodem5CF70382101
LOCK=/tmp/esp32-board.lock
BK=/private/tmp/claude-501/-Users-danivarela-Ita/198a477a-6428-4843-a085-aca3e7345087/scratchpad/board1_flash_backup.bin
if [ -e $LOCK ]; then echo "ABORT: $LOCK exists: $(cat $LOCK)"; exit 1; fi
echo "claude-oido-session: board 1 (UART $PORT, MAC f8:5b:1b:91:6d:a8) Oido v2 replication; flash backed up in $BK and restored afterwards; Dani authorized; started $(date '+%F %T')" > $LOCK
. ~/esp/esp-idf/export.sh >/dev/null 2>&1
esptool.py --chip esp32s3 -p $PORT -b 921600 read_flash 0x0 0x1000000 $BK > /tmp/board1_backup.log 2>&1
SZ=$(stat -f %z $BK 2>/dev/null)
echo "BACKUP size $SZ md5 $(md5 -q $BK)"
if [ "$SZ" != "16777216" ]; then echo "ABORT: backup incomplete"; rm -f $LOCK; exit 1; fi
bench/final_board.sh nemo4_v2a2_final b1a2
bench/final_board.sh nemo4_v2b3_final b1b3
esptool.py --chip esp32s3 -p $PORT -b 921600 write_flash 0x0 $BK > /tmp/board1_restore.log 2>&1
echo "RESTORE exit $?: $(grep -E 'Hash of data verified|Leaving' /tmp/board1_restore.log | tr '\n' ' ')"
rm -f $LOCK
echo ALLDONE10
