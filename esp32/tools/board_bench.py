#!/usr/bin/env python3
"""Run benchmark clips on a connected ESP32-S3 (TASR_MODE=file firmware) and compare with the QEMU / host transcripts.

usage (ESP-IDF's python has pyserial; set PYTHON to a python with numpy + soundfile):
  . ~/esp/esp-idf/export.sh
  PYTHON=$(command -v python3.x) python tools/board_bench.py --port /dev/cu.usbmodemXXXX --model ../models/nemo8.tnm \
      --clips clips.json --out board_nemo8.json [--no-lm] [--dual]

clips.json: [{"path": "c00.wav", "ref": "reference text", "dur": 4.7, "qemu_hyp": "...", "host_hyp": "..."}, ...]
The clips go in batches that fit the 'audio' partition (about 22 s). The first batch builds and flashes everything with
tools/flash.sh; later batches only rewrite the audio partition. Prints per-clip RTF measured from the CPU cycle counter
(on silicon one cycle is one cycle) and writes everything to --out.
"""
import argparse, json, os, re, subprocess, sys, time

import serial

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.join(HERE, "..")


def wer_counts(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def capture(port, until, timeout, boot_timeout=0):
    """Read the console until the DONE line (device was just reset); returns the text.
    boot_timeout > 0: give up after that many seconds if the application has not started (boot hang)."""
    t0 = time.time()
    s = serial.Serial()  # DTR/RTS low before opening, or the USB-Serial/JTAG bridge resets the chip into download mode
    s.port, s.baudrate, s.timeout, s.dtr, s.rts = port, 115200, 1, False, False
    while not s.is_open and time.time() - t0 < 20:  # the USB port re-enumerates after the reset
        try:
            s.open()
        except Exception:
            time.sleep(0.3)
    if not s.is_open:
        raise RuntimeError("serial port did not come back")
    buf = b""
    while time.time() - t0 < timeout:
        buf += s.read(4096)
        txt = buf.decode("utf-8", "replace")
        if until in txt:
            break
        if boot_timeout and time.time() - t0 > boot_timeout and "Calling app_main" not in txt:
            break
    s.close()
    return buf.decode("utf-8", "replace")


def reset(port):
    """Boot the application: a plain RTS reset leaves a native-USB board in download mode, a watchdog reset does not.
    After a boot hang the USB device can stop answering; TASR_USB_RESET names a helper that resets it with libusb."""
    cmd = ["esptool.py", "--chip", "esp32s3", "-p", port, "--connect-attempts", "3", "--after", "watchdog_reset", "read_mac"]
    for attempt in range(3):
        if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            return
        helper = os.environ.get("TASR_USB_RESET")
        if helper:
            subprocess.run([helper], stdout=subprocess.DEVNULL)
            time.sleep(3)
    raise RuntimeError("board does not answer (unplug and replug the USB cable)")


def parse(text):
    """UTT/STREAM lines of the file-mode firmware. Times come from the 64-bit wall clock when the firmware prints it (the
    32-bit cycle counter wraps every 17.9 s at 240 MHz, so older builds under-report clips whose compute exceeds that)."""
    utts, cur = [], None
    for line in text.splitlines():
        m = re.match(r"UTT (\d+) \| audio ([\d.]+)s \| cycles (\d+)(?: \| RTF [\d.]+)?(?: \| wall (\d+) us)?", line)
        if m:
            cur = {"audio": float(m.group(2)), "cycles": int(m.group(3)), "wall_us": int(m.group(4)) if m.group(4) else None}
            cur["rtf"] = (cur["wall_us"] / 1e6 if cur["wall_us"] is not None else cur["cycles"] / 240e6) / cur["audio"]
            utts.append(cur)
        elif line.startswith("HYP: ") and cur is not None:
            cur["hyp"] = line[5:].strip()
        m = re.match(r"STREAM (\d+) \| feed cycles (\d+) \| finish cycles (\d+) .*?(?:\| feed wall (\d+) us \| finish wall (\d+) us)?", line)
        if m and cur is not None:
            cur.update(stream_feed=int(m.group(2)), stream_finish=int(m.group(3)))
            w = re.search(r"feed wall (\d+) us \| finish wall (\d+) us", line)
            if w:
                cur.update(stream_feed_us=int(w.group(1)), stream_finish_us=int(w.group(2)))
        if line.startswith("SHYP: ") and cur is not None:
            cur["shyp"] = line[6:].strip()
        m = re.match(r"PACED (\d+) \| audio [\d.]+s \| last block fed (-?\d+) ms after the end of speech \| final text (-?\d+) ms", line)
        if m and cur is not None:
            cur.update(paced_last_block_ms=int(m.group(2)), paced_final_ms=int(m.group(3)))
        if line.startswith("PHYP: ") and cur is not None:
            cur["phyp"] = line[6:].strip()
    mem = re.search(r"MEM min free PSRAM (\d+) internal (\d+)", text)
    return utts, (int(mem.group(1)), int(mem.group(2))) if mem else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lm", default="")
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=float, default=22.0)
    ap.add_argument("--dual", action="store_true")
    ap.add_argument("--paced", action="store_true", help="streaming models: also feed at real-time pace (TASR_PACED build)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips")
    ap.add_argument("--pick", default="", help="comma-separated clip indices to run")
    a = ap.parse_args()
    py = os.environ.get("PYTHON") or sys.executable
    env = dict(os.environ, PYTHON=py, TASR_MODE="file")
    if a.no_lm:
        env["NO_LM"] = "1"
    if a.dual:
        env["TASR_DUAL"] = "1"
    if a.paced:
        env["TASR_PACED"] = "1"
    clips = json.load(open(a.clips))
    if a.limit:
        clips = clips[: a.limit]
    if a.pick:
        clips = [clips[int(i)] for i in a.pick.split(",")]
    batches, cur, tot = [], [], 0.0
    for c in clips:
        if cur and tot + c["dur"] > a.batch:
            batches.append(cur); cur, tot = [], 0.0
        cur.append(c); tot += c["dur"]
    batches.append(cur)
    model = os.path.abspath(a.model)
    mem, results, text_all = None, [], ""
    layout_audio = {}
    for bi, b in enumerate(batches):
        args = [c for cl in b for c in (cl["path"], cl["ref"])]
        lm = [os.path.abspath(a.lm)] if a.lm else []
        if bi == 0:
            subprocess.run([os.path.join(HERE, "flash.sh"), a.port, model] + lm + args, check=True, env=env)
            # the same layout rule as flash.sh
            name = os.path.basename(model)
            layout = "nemo4_16" if "nemo4" in name else "rnnt16" if "rnnt" in name else "nemo16lm" if name.endswith(".tnm") else "tinyasr16"
            out = os.path.join(FW, "build_images", f"{layout}_file")
            audio_off = int(subprocess.check_output([py, "-c", "import sys; sys.path.insert(0, %r); import mkimages; print(mkimages.LAYOUTS[%r]['AUDIO'][0])"
                                                     % (HERE, layout)]).decode())
        else:
            subprocess.run([py, os.path.join(HERE, "mkimages.py"), model, out, "--layout", layout, "--wavs"] + [c["path"] for c in b]
                           + ["--refs"] + [c["ref"] for c in b] + (["--lm", a.lm] if a.lm else []), check=True,
                           stdout=subprocess.DEVNULL)
            subprocess.run(["esptool.py", "--chip", "esp32s3", "-p", a.port, "-b", "921600", "write_flash", hex(audio_off),
                            os.path.join(out, "audio.bin")], check=True, stdout=subprocess.DEVNULL)
        text = ""
        for attempt in range(3):
            reset(a.port)
            text = capture(a.port, "DONE", 120 + 6.0 * sum(c["dur"] for c in b), boot_timeout=60)
            if "Calling app_main" in text:
                break
            print(f"batch {bi}: application did not start (attempt {attempt + 1}), resetting", flush=True)
        text_all += f"\n===== batch {bi} =====\n" + text
        utts, m = parse(text)
        mem = m or mem
        if len(utts) != len(b):
            print(f"batch {bi}: expected {len(b)} utterances, got {len(utts)}; tail of the console:\n" + text[-1500:])
        for c, u in zip(b, utts):
            c = dict(c); c.update(u)
            results.append(c)
            q = c.get("qemu_hyp")
            print(f"{os.path.basename(c['path'])} {c['dur']:5.1f}s RTF {u['rtf']:.2f}  "
                  f"{'==qemu' if q == u.get('hyp') else '!=QEMU'}  {'==host' if c.get('host_hyp') == u.get('hyp') else '!=host'}", flush=True)
    for key in ("hyp", "qemu_hyp", "host_hyp"):
        e = n = 0
        for c in results:
            if key in c and c.get("ref") is not None:
                d, k = wer_counts(c["ref"], c[key]); e += d; n += k
        print(f"WER vs reference, {key}: {100 * e / max(n, 1):.2f}% ({n} words)")
    tot_a = sum(c["audio"] for c in results)
    tot_c = sum(c["rtf"] * c["audio"] for c in results) * 240e6
    print(f"TOTAL {tot_a:.1f} s audio, mean RTF {tot_c / 240e6 / tot_a:.3f}, board == QEMU on "
          f"{sum(c.get('hyp') == c.get('qemu_hyp') for c in results)}/{len(results)}, == host on "
          f"{sum(c.get('hyp') == c.get('host_hyp') for c in results)}/{len(results)}; min free PSRAM/internal {mem}")
    json.dump({"clips": results, "rtf_mean": tot_c / 240e6 / tot_a, "min_free_psram_internal": mem, "model": os.path.basename(model),
               "dual": a.dual}, open(a.out, "w"), indent=1)
    open(a.out + ".log", "w").write(text_all)


if __name__ == "__main__":
    main()
