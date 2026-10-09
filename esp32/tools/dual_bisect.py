#!/usr/bin/env python3
"""Find the first stage whose output differs between the one-core and the two-core firmware on a real board.

Builds with TASR_DEBUG_SUMS print an FNV hash of the log-mel features, the subsampled input and every Conformer layer
output ("DBG layer 3 1a2b3c4d"). Run the same clip once on each build and report the first stage that differs.

  . ~/esp/esp-idf/export.sh
  (cd firmware; for v in 1 2; do TASR_DEBUG_SUMS=1 idf.py -B build_dbg$v -D SDKCONFIG=sdkconfig_dbg$v \
      -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.nemo16lm" -D TASR_2C=$((v-1)) build; done)
  PYTHON=<python with numpy+soundfile> python tools/dual_bisect.py --port /dev/cu.usbmodemXXXX --model ../models/nemo8.tnm \
      --lm ../models/nemo_lm.tlm --clip clip.wav --ref "reference text"
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_bench as bb

FW = os.path.join(HERE, "..", "firmware")


def run(port):
    bb.reset(port)
    t = bb.capture(port, "DONE", 400)
    dbg = [l.strip() for l in t.splitlines() if l.startswith("DBG ")]
    utts, _ = bb.parse(t)
    return dbg, utts, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lm", default="")
    ap.add_argument("--clip", required=True, nargs="+")
    ap.add_argument("--ref", default=None, nargs="+")
    ap.add_argument("--layout", default="nemo16lm")
    ap.add_argument("--builds", default="dbg1,dbg2", help="firmware/build_<name> directories; the first one is the reference")
    ap.add_argument("--repeat", type=int, default=1, help="runs per build (races are not deterministic)")
    a = ap.parse_args()
    py = os.environ.get("PYTHON") or sys.executable
    names = a.builds.split(",")
    out = os.path.join(HERE, "..", "build_images", f"{a.layout}_dbg")
    cmd = [py, os.path.join(HERE, "mkimages.py"), os.path.abspath(a.model), out, "--layout", a.layout, "--wavs", *a.clip,
           "--refs", *(a.ref or ["-"] * len(a.clip)), "--merge", os.path.join(FW, "build_" + names[0]), "--flash_mode", "keep"]
    if a.lm:
        cmd += ["--lm", os.path.abspath(a.lm)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["esptool.py", "--chip", "esp32s3", "-p", a.port, "-b", "921600", "--after", "no_reset", "write_flash", "0x0",
                    os.path.join(out, "flash.bin")], check=True, stdout=subprocess.DEVNULL)
    ref = None
    report = []
    for bi, name in enumerate(names):
        if bi:  # only the application differs
            subprocess.run(["esptool.py", "--chip", "esp32s3", "-p", a.port, "-b", "921600", "--after", "no_reset", "write_flash",
                            "0x10000", os.path.join(FW, "build_" + name, "tinyasr_fw.bin")], check=True, stdout=subprocess.DEVNULL)
        for r in range(a.repeat):
            for attempt in range(3):
                bb.reset(a.port)
                t = bb.capture(a.port, "DONE", 400, boot_timeout=60)
                if "Calling app_main" in t:
                    break
            dbg = [l.strip() for l in t.splitlines() if l.startswith("DBG ")]
            utts, _ = bb.parse(t)
            if ref is None:
                ref = (dbg, [u.get("hyp") for u in utts])
            same = sum(x == y for x, y in zip(dbg, ref[0]))
            first = next((i for i, (x, y) in enumerate(zip(dbg, ref[0])) if x != y), None)
            hy = [u.get("hyp") for u in utts]
            line = (f"{name} run {r}: {len(dbg)} DBG lines, {same}/{len(ref[0])} equal to {names[0]}, first difference at "
                    f"{('line %d (%s)' % (first, dbg[first][:24])) if first is not None else 'none'}; "
                    f"transcripts equal to reference: {hy == ref[1]}; RTF {('%.3f' % utts[0]['rtf']) if utts else 'n/a'}")
            print(line, flush=True)
            report.append(line)
            open(os.path.join(out + "_" + name + f"_{r}.txt"), "w").write("\n".join(dbg) + "\n\n" + "\n".join(map(str, hy)) + "\n")
            if not utts:
                print(t[-800:])
    open(out + "_report.txt", "w").write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
