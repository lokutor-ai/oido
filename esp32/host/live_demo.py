"""Live demo of the ESP32-S3 recognizer on a laptop: microphone -> the firmware's own VAD/segmenter (tasr_seg.c) ->
the firmware's own Conformer-CTC engine + LM (tasr_nemo.c, tinyasr_lm.c). Same C code and arithmetic as the chip, so
the transcripts are what the board would print; the ESP32-S3 time is estimated from QEMU instruction counts.

python live_demo.py                        # speak; each utterance is transcribed after a 0.8 s pause
python live_demo.py --model fast           # int4 profile
python live_demo.py --wav session.wav      # run a recording through the same path instead of the microphone
python live_demo.py --save clips/          # also keep each utterance as a wav (e.g. to replay it in QEMU)
python live_demo.py --model es_stream.tnm  # streaming model: words appear while you speak, final text right after the pause
python live_demo.py --pause 0.5            # shorter end-of-utterance pause (default 0.8 s, like the firmware)
"""
import argparse, ctypes, os, queue, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "..", "models")
sys.path.insert(0, HERE)
import pytasr  # noqa: E402

lib = pytasr._lib
lib.tasr_seg_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.tasr_seg_feed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.tasr_seg_feed.restype = ctypes.c_int
lib.tasr_seg_next.argtypes = [ctypes.c_void_p]

# ESP32-S3 cost model, fitted to QEMU instruction counts of the firmware (int8 model, 4-20 s utterances):
# ccount = A*dur + B*dur^2 (attention grows with length^2); ccount x 25 = instructions summed over both cores,
# ~55% of them on the dual-core critical path; 1.3-1.6 cycles/instruction at 240 MHz; + ~0.08 s/s of flash/PSRAM stalls.
A_GREEDY, A_LM, B = 7.76e6, 8.6e6, 0.124e6

# Oido v2 models, measured on a board (one core, 240 MHz, language model on): streaming real-time factor and the final text's
# delay after the end of speech with audio fed at real-time speed (research/v2_speed_log.md); the model above is for v1 only.
V2_MEASURED = {"oido_v2a": (0.72, 0.43), "oido_v2b": (0.54, 0.26)}


def esp32_finish_seconds(n_samples, chunk, int4):
    """Streaming: compute left after the pause is detected (the last partial chunk). QEMU: ~17 M + 5.5 M instructions
    per remaining encoder frame (both cores, ~55% on the critical path), plus one pass over the weights from
    flash/PSRAM (estimate: 0.15-0.35 s int8, 0.08-0.2 s int4 until measured on a board)."""
    T = (((n_samples // 160 + 1) - 1) // 2 + 1 - 1) // 2 + 1
    last = T % chunk or chunk
    crit = (16.6e6 + 5.5e6 * last) * 0.55   # fitted to QEMU finish counts of 4 utterances (33-149 M instructions)
    mem = (0.08, 0.2) if int4 else (0.15, 0.35)
    return crit * 1.3 / 240e6 + mem[0], crit * 1.6 / 240e6 + mem[1]


def esp32_stream_rtf(chunk, int4):
    """Streaming real-time factor while speaking. Instructions are the same as the batch engine (~0.72-0.84 s per s of
    audio at 1.3-1.6 cycles/instruction); weight traffic is not: the batch engine streams every weight once per 64
    frames (7.5 MB/s, ~0.08 s/s of stalls), a chunk of C frames re-reads them every C frames, i.e. 64/C times more.
    Above 1.0 the device falls behind while you speak. Estimate until measured on a board."""
    stall = 0.08 * (64.0 / chunk) * (0.8 if int4 else 1.0)
    return 0.72 - 0.08 + stall, 0.84 - 0.08 + stall * 1.5


def esp32_seconds(dur, lm, int4):
    cc = (A_LM if lm else A_GREEDY) * dur + B * dur * dur
    if int4:
        cc *= 0.91
    crit = cc * 25 * 0.55
    return crit * 1.3 / 240e6 + 0.08 * dur, crit * 1.6 / 240e6 + 0.08 * dur


class Segmenter:
    def __init__(self, seconds=20):
        self.state = ctypes.create_string_buffer(256)  # tasr_seg_t (40 bytes) with room to spare
        self.buf = np.zeros(16000 * seconds, dtype=np.int16)
        lib.tasr_seg_init(self.state, self.buf.ctypes.data, len(self.buf))

    def feed(self, block):
        block = np.ascontiguousarray(block, dtype=np.int16)
        n = lib.tasr_seg_feed(self.state, block.ctypes.data, len(block))
        if not n:
            return None
        utt = self.buf[:n].copy()
        lib.tasr_seg_next(self.state)
        return utt

    def flush(self):  # end of input: hand over whatever speech is buffered
        n = int.from_bytes(self.state.raw[20:24], "little")  # tasr_seg_t.n
        if not self.in_speech or n < 3200:
            return None
        utt = self.buf[:n].copy()
        lib.tasr_seg_next(self.state)
        return utt

    @property
    def in_speech(self):
        return int.from_bytes(self.state.raw[24:28], "little") != 0  # tasr_seg_t.speech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="accurate", help="accurate (int8), fast (int4), stream (low-latency), es (Spanish) or a .tnm path")
    ap.add_argument("--no_lm", action="store_true")
    ap.add_argument("--wav", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--device", default=None, help="input device name/index (see python -m sounddevice)")
    ap.add_argument("--pause", type=float, default=0.0, help="end-of-utterance pause in s (default 0.8)")
    a = ap.parse_args()
    path = {"accurate": os.path.join(MODELS, "nemo8.tnm"), "fast": os.path.join(MODELS, "nemo4.tnm"),
            "stream": os.path.join(MODELS, "oido_stream.tnm"),
            "es": os.path.join(MODELS, "oido_es.tnm")}.get(a.model, a.model)
    int4 = os.path.getsize(path) < 10_000_000
    lm_path = os.path.splitext(path)[0] + ".tlm"  # a model's own LM (e.g. oido_es.tlm), else the English one
    if not os.path.exists(lm_path):
        lm_path = os.path.join(MODELS, "oido_v2.tlm" if os.path.basename(path).startswith("oido_v2") else "nemo_lm.tlm")
    if not a.no_lm and not os.path.exists(lm_path):
        print(f"note: {lm_path} not found, decoding greedily without the language model", flush=True)
    lm = pytasr.LM(lm_path) if not a.no_lm and os.path.exists(lm_path) else None
    eng = pytasr.Nemo(path, lm=lm, beam=4 if lm else 0)  # LM weights come from the LM file (English: 0.3 / 0.5)
    seg = Segmenter()
    if a.pause:
        seg.state[40:44] = int(round(a.pause * 50)).to_bytes(4, "little")  # tasr_seg_t.hang (320-sample blocks)
    stream = None
    if pytasr._lib.tasr_nemo_stream_supported(eng.ptr):
        stream = pytasr.NemoStream(eng)
        stream.reset()
    if a.save:
        os.makedirs(a.save, exist_ok=True)
    print(f"model {os.path.basename(path)} ({'int4' if int4 else 'int8'}{' + LM' if lm else ', greedy'}), "
          f"same engine and VAD as the ESP32-S3 firmware", flush=True)
    k = 0

    def handle(utt):
        nonlocal k
        dur = len(utt) / 16000
        t0 = time.time()
        text = eng.transcribe(utt)
        host = time.time() - t0
        lo, hi = esp32_seconds(dur, lm is not None, int4)
        v2 = V2_MEASURED.get(os.path.basename(path)[:8])
        if sys.stdout.isatty():
            sys.stdout.write("\r" + " " * 40 + "\r")
        print(f"[{dur:4.1f} s] {text or '(nothing recognized)'}")
        if v2:
            print(f"         ESP32-S3 measured: real-time factor {v2[0]:.2f} (streaming), final text ~{v2[1]:.2f} s after the end of speech "
                  f"(+ the {0.8:.1f} s pause wait) | laptop {host*1000:.0f} ms", flush=True)
        else:
            print(f"         ESP32-S3 est. {lo:.1f}-{hi:.1f} s compute (RTF {lo/dur:.2f}-{hi/dur:.2f}), "
                  f"text ~{0.8 + lo:.1f}-{0.8 + hi:.1f} s after you stop talking | laptop {host*1000:.0f} ms", flush=True)
        if a.save:
            import soundfile as sf
            sf.write(os.path.join(a.save, f"utt{k:03d}.wav"), utt, 16000, subtype="PCM_16")
        k += 1

    pause = a.pause or 0.8
    st = {"fed": 0, "hearing": False, "shown": ""}

    def stream_block(block):
        """Streaming path: segmenter -> stream while speaking; final text when the segmenter closes the utterance."""
        block = np.ascontiguousarray(block, dtype=np.int16)
        n_done = lib.tasr_seg_feed(seg.state, block.ctypes.data, len(block))
        speech = seg.in_speech
        if st["hearing"] and not speech and not n_done:  # a blip the segmenter discarded
            stream.reset(); st["fed"] = 0
        st["hearing"] = speech
        if not speech and not n_done:
            return
        have = n_done or int.from_bytes(seg.state.raw[20:24], "little")
        if have > st["fed"]:
            stream.feed(seg.buf[st["fed"]:have]); st["fed"] = have
            part = stream.partial()
            if part != st["shown"] and sys.stdout.isatty():
                sys.stdout.write("\r  ... " + part[-90:] + " " * 4); sys.stdout.flush()
                st["shown"] = part
        if n_done:
            t0 = time.time()
            text = stream.finish()
            host = time.time() - t0
            lo, hi = esp32_finish_seconds(n_done, stream_chunk, int4)
            if sys.stdout.isatty():
                sys.stdout.write("\r" + " " * 100 + "\r")
            print(f"[{n_done/16000:4.1f} s] {text or '(nothing recognized)'}")
            if v2m:
                print(f"         ESP32-S3 measured: final text {pause + v2m[1]:.2f} s after you stop talking "
                      f"({pause:.1f} s pause + {v2m[1]*1000:.0f} ms compute) | laptop {host*1000:.0f} ms", flush=True)
            else:
                print(f"         ESP32-S3 est.: final text {pause + lo:.2f}-{pause + hi:.2f} s after you stop talking "
                      f"({pause:.1f} s pause + {lo*1000:.0f}-{hi*1000:.0f} ms compute) | laptop {host*1000:.0f} ms", flush=True)
            if a.save:
                import soundfile as sf
                sf.write(os.path.join(a.save, f"utt{st.get('k', 0):03d}.wav"), seg.buf[:n_done], 16000, subtype="PCM_16")
                st["k"] = st.get("k", 0) + 1
            stream.reset(); st["fed"] = 0; st["shown"] = ""
            lib.tasr_seg_next(seg.state)

    with open(path, "rb") as fh:  # header word 11 = the chunk (encoder frames) the model was exported for
        stream_chunk = int.from_bytes(fh.read(64)[48:52], "little") or 16
    v2m = V2_MEASURED.get(os.path.basename(path)[:8])
    if stream is not None and v2m:
        print(f"streaming model ({stream_chunk} frames of 80 ms = {stream_chunk * 0.08:.2f} s chunks): partial text while you speak, final text right after the pause. "
              f"ESP32-S3 (measured): real-time factor {v2m[0]:.2f}", flush=True)
    elif stream is not None:
        lo, hi = esp32_stream_rtf(stream_chunk, int4)
        print(f"streaming model ({stream_chunk} frames = {stream_chunk * 0.04:.2f} s chunks): partial text while you speak, "
              f"final text right after the pause. ESP32-S3 est. real-time factor {lo:.2f}-{hi:.2f}"
              f"{' (may fall behind while you speak)' if hi > 1.0 else ''}", flush=True)

    if a.wav:
        import soundfile as sf
        x, sr = sf.read(a.wav, dtype="int16")
        if x.ndim > 1:
            x = x[:, 0]
        assert sr == 16000, "need 16 kHz audio"
        for i in range(0, len(x) - 319, 320):
            if stream is not None:
                stream_block(x[i:i + 320])
                continue
            utt = seg.feed(x[i:i + 320])
            if utt is not None:
                handle(utt)
        if stream is not None:
            if seg.in_speech:
                n = int.from_bytes(seg.state.raw[20:24], "little")
                stream.feed(seg.buf[st["fed"]:n]); print(f"[{n/16000:4.1f} s] {stream.finish()}")
            return
        utt = seg.flush()
        if utt is not None:
            handle(utt)
        return

    import sounddevice as sd
    q = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=320, callback=cb, device=a.device):
        print("listening... speak, pause to get the transcript (Ctrl-C to quit)", flush=True)
        shown = False
        try:
            while True:
                if stream is not None:
                    stream_block(q.get())
                    continue
                utt = seg.feed(q.get())
                if seg.in_speech and not shown and sys.stdout.isatty():
                    sys.stdout.write("\r  (hearing speech...)")
                    sys.stdout.flush()
                    shown = True
                if utt is not None:
                    handle(utt)  # audio keeps queueing meanwhile, like the firmware's 8 s ring buffer
                    shown = False
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
