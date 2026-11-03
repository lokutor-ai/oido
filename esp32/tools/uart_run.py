#!/usr/bin/env python3
"""Reset a UART-bridge ESP32-S3 board (RTS -> EN, DTR low) and print its console until a marker line or a timeout.
usage: uart_run.py PORT MARKER [timeout_s]   (ESP-IDF python has pyserial)"""
import sys, time
import serial

port, marker = sys.argv[1], sys.argv[2]
timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 120
s = serial.Serial()
s.port, s.baudrate, s.timeout, s.dtr, s.rts = port, 115200, 0.5, False, False
s.open()
s.reset_input_buffer()
s.rts = True; time.sleep(0.15); s.rts = False
t0, buf = time.time(), b""
while time.time() - t0 < timeout:
    d = s.read(4096)
    if d:
        sys.stdout.write(d.decode("utf-8", "replace")); sys.stdout.flush()
        buf += d
        if marker.encode() in buf:
            break
s.close()
sys.exit(0 if marker.encode() in buf else 1)
