#!/bin/bash
# Flash only bootloader + partition table + app from an IDF build dir (model, LM and audio partitions stay untouched).
# usage: tools/quick_flash.sh PORT BUILD_DIR   (UART-bridge boards; hard reset after)
set -e
PORT=$1; B=$2
. "${IDF_PATH:-$HOME/esp/esp-idf}/export.sh" > /dev/null
APP=$(ls $B/*.bin | grep -v -E "bootloader|partition|ota_data" | head -1)
esptool.py --chip esp32s3 -p "$PORT" -b 921600 --after hard_reset write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m \
  0x0 $B/bootloader/bootloader.bin 0x8000 $B/partition_table/partition-table.bin 0x10000 $APP
