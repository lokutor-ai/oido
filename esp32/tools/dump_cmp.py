#!/usr/bin/env python3
"""Run a one-core and a two-core TASR_DEBUG_DUMP build on the same clip and print which elements of the dumped tensor differ.

  . ~/esp/esp-idf/export.sh
  (cd firmware; for v in 1 2; do TASR_DEBUG_SUMS=1 TASR_DEBUG_DUMP=1 idf.py -B build_dump$v -D SDKCONFIG=sdkconfig_dump$v \
      -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.nemo16lm" -D TASR_2C=$((v-1)) build; done)
  PYTHON=<python with numpy+soundfile> python tools/dump_cmp.py --port /dev/cu.usbmodemXXXX --model ../models/nemo8.tnm --lm ../models/nemo_lm.tlm --clip c.wav
"""
import argparse, collections, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_bench as bb

FW = os.path.join(HERE, "..", "firmware")


def run(port):
    for attempt in range(3):
        bb.reset(port)
        t = bb.capture(port, "DONE", 400, boot_timeout=60)
        if "Calling app_main" in t:
            break
    return t


def dumps(text):
    d = {}
    for l in text.splitlines():
        if l.startswith("DUMP "):
            p = l.split()
            d[(p[1], int(p[2]), int(p[3]))] = [int(x, 16) for x in p[4:]]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lm", default="")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--layout", default="nemo16lm")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "build_images", "dump"))
    a = ap.parse_args()
    py = os.environ.get("PYTHON") or sys.executable
    os.makedirs(a.out, exist_ok=True)
    cmd = [py, os.path.join(HERE, "mkimages.py"), os.path.abspath(a.model), a.out, "--layout", a.layout, "--wavs", a.clip, "--refs", "-",
           "--merge", os.path.join(FW, "build_dump1"), "--flash_mode", "keep"]
    if a.lm:
        cmd += ["--lm", os.path.abspath(a.lm)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["esptool.py", "--chip", "esp32s3", "-p", a.port, "-b", "921600", "--after", "no_reset", "write_flash", "0x0",
                    os.path.join(a.out, "flash.bin")], check=True, stdout=subprocess.DEVNULL)
    ref = run(a.port)
    open(os.path.join(a.out, "one_core.txt"), "w").write(ref)
    rd = dumps(ref)
    print(f"one core: {len(rd)} dumped rows")
    subprocess.run(["esptool.py", "--chip", "esp32s3", "-p", a.port, "-b", "921600", "--after", "no_reset", "write_flash", "0x10000",
                    os.path.join(FW, "build_dump2", "tinyasr_fw.bin")], check=True, stdout=subprocess.DEVNULL)
    for r in range(a.repeat):
        t = run(a.port)
        open(os.path.join(a.out, f"two_cores_{r}.txt"), "w").write(t)
        d = dumps(t)
        bad_rows, bad_cols, nbad = collections.Counter(), collections.Counter(), 0
        for k, v in rd.items():
            w = d.get(k)
            if w is None:
                continue
            for c, (x, y) in enumerate(zip(v, w)):
                if x != y:
                    bad_rows[k[2]] += 1; bad_cols[c] += 1; nbad += 1
        print(f"two cores run {r}: {len(d)} rows dumped, {nbad} differing words, rows affected {sorted(bad_rows)[:20]}"
              f"{'...' if len(bad_rows) > 20 else ''} ({len(bad_rows)} rows), columns {min(bad_cols) if bad_cols else None}..{max(bad_cols) if bad_cols else None} ({len(bad_cols)} cols)")


if __name__ == "__main__":
    main()
