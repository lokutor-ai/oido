"""Build the GitHub Pages demo (docs/index.html + docs/samples/*.mp3) of the released model.

Clips are drawn at random with a fixed seed (not cherry-picked): two LibriSpeech test-clean sentences under every
ambience of the robustness benchmark (eval/make_robust.py), plus real recordings from Common Voice, VoxPopuli and AMI.
Each clip is transcribed by the host build of the engine and by the real firmware in
Espressif's QEMU, which also gives the exact instruction count; the page shows reference, on-chip transcript (errors
marked), WER and the estimated on-chip time.

python make_demo_page.py --repo .. --out ../docs   # needs ESP-IDF + QEMU
Models, firmware and emulator come from --repo (the public release); clips come from this repository's data/.
"""
import argparse, html, json, os, random, re, subprocess, sys
import numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results" if os.path.isdir(os.path.join(ROOT, "results")) else "research")


def clean_text(t):  # same normalization as the benchmarks (lowercase, letters and apostrophes only)
    t = t.lower().replace("\u2019", "'").replace("`", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t)).strip()

CONDITIONS = [  # (directory, label, description)
    ("clean", "Quiet room", "the original recording"),
    ("car_5db", "Car, 5 dB SNR", "real car-interior noise (DEMAND)"),
    ("kitchen_5db", "Kitchen, 5 dB SNR", "real kitchen noise (DEMAND)"),
    ("living_5db", "Living room, 5 dB SNR", "real living-room noise (DEMAND)"),
    ("cafe_5db", "Cafeteria, 5 dB SNR", "real cafeteria noise with background talk (DEMAND)"),
    ("babble_10db", "Babble, 10 dB SNR", "four other talkers"),
    ("babble_5db", "Babble, 5 dB SNR", "four other talkers, louder (hardest case)"),
    ("reverb_0.4s", "Reverberant room, RT60 0.4 s", "5 x 4 x 3 m room, talker 2 m from the microphone"),
    ("reverb_0.8s", "Very reverberant room, RT60 0.8 s", "same room, harder walls"),
    ("farfield_kitchen", "Far field + kitchen noise", "0.6 s reverberation and kitchen noise at 10 dB"),
]
REAL = [  # (manifest, label, description, n)
    ("data/val/val_cv.local.jsonl", "Common Voice", "volunteers' own microphones, at home", 3),
    ("data/val/val_vox.local.jsonl", "VoxPopuli", "European Parliament microphones", 2),
    ("data/val/val_ami.local.jsonl", "AMI meeting", "spontaneous meeting speech, headset microphone", 2),
]
CPU_HZ, CRIT, CPI = 240e6, 0.55, (1.3, 1.6)


def est_seconds(instr, dur):
    crit = instr * CRIT
    return crit * CPI[0] / CPU_HZ + 0.08 * dur, crit * CPI[1] / CPU_HZ + 0.08 * dur


def marked(ref, hyp):
    """Transcript with substitutions/insertions in red and deletions as struck-out reference words."""
    import jiwer
    r, h = ref.split(), hyp.split()
    if not h:
        return '<span class="err">(nothing)</span>'
    out = []
    for c in jiwer.process_words(ref, hyp).alignments[0]:
        if c.type == "equal":
            out += [html.escape(w) for w in h[c.hyp_start_idx:c.hyp_end_idx]]
        elif c.type in ("substitute", "insert"):
            out += [f'<span class="err">{html.escape(w)}</span>' for w in h[c.hyp_start_idx:c.hyp_end_idx]]
        else:
            out += [f'<span class="del">{html.escape(w)}</span>' for w in r[c.ref_start_idx:c.ref_end_idx]]
    return " ".join(out)


def emulate(clips, tmp, repo):
    """Run the firmware in QEMU on the clips (batches of <= 22 s); returns {wav: (cycles, hyp)}."""
    res, batch, tot = {}, [], 0.0

    first = [True]

    def run(b):
        env = dict(os.environ, PYTHON=sys.executable)
        if first[0]:  # build the release's firmware from source once
            env["REBUILD"] = "1"
            first[0] = False
        subprocess.run([os.path.join(repo, "esp32", "tools", "emulate.sh")] + b, check=True, env=env,
                       stdout=subprocess.DEVNULL)
        lines = open("/tmp/emu_out.txt").read().splitlines()
        utts = [l for l in lines if l.startswith("UTT ")]
        hyps = [l[5:] for l in lines if l.startswith("HYP: ")]
        for w, u, h in zip(b, utts, hyps):
            res[w] = (int(re.search(r"cycles (\d+)", u).group(1)), h.strip())

    for c in clips:
        if batch and tot + c["dur"] > 22.0:
            run(batch); batch, tot = [], 0.0
        batch.append(c["tmpwav"]); tot += c["dur"]
    if batch:
        run(batch)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True, help="checkout of the public release (models, firmware, tools)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_dur", type=float, default=11.0)
    ap.add_argument("--render_only", action="store_true", help="rewrite index.html from samples/results.json")
    ap.add_argument("--board", default="", help="board_bench.py results for the same clips: use the board's transcripts and times")
    a = ap.parse_args()
    if a.render_only:
        clips = json.load(open(os.path.join(a.out, "samples", "results.json")))
        if a.board:
            apply_board(clips, a.board)
            json.dump(clips, open(os.path.join(a.out, "samples", "results.json"), "w"), indent=1)
        write_html(a.out, clips, load_robust())
        return
    repo = os.path.abspath(a.repo)
    sys.path.insert(0, os.path.join(repo, "esp32", "host"))
    import pytasr
    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, "samples"), exist_ok=True)
    tmp = "/tmp/oido_demo"
    os.makedirs(tmp, exist_ok=True)

    clean = [json.loads(l) for l in open(os.path.join(ROOT, "data/robust/clean.jsonl"))]
    pick = rng.sample([d for d in clean if 3.0 <= d["duration"] <= a.max_dur], 2)
    clips = []
    for d in pick:
        for cond, label, desc in CONDITIONS:
            wav = os.path.join(ROOT, "data/robust", cond, d["id"] + ".flac")
            clips.append({"group": "rooms", "sent": d["id"], "cond": cond, "label": label, "desc": desc, "wav": wav,
                          "ref": clean_text(d["text"])})
    for man, label, desc, n in REAL:
        its = [json.loads(l) for l in open(os.path.join(ROOT, man))]
        its = [d for d in its if 2.5 <= d["duration"] <= a.max_dur and len(clean_text(d["text"]).split()) >= 5]
        for d in rng.sample(its, n):
            clips.append({"group": "real", "sent": d["id"], "cond": label, "label": label, "desc": desc, "wav": d["wav"],
                          "ref": clean_text(d["text"])})

    eng = pytasr.Nemo(os.path.join(repo, "models/nemo8.tnm"), lm=pytasr.LM(os.path.join(repo, "models/nemo_lm.tlm")),
                      beam=4, lm_weight=0.3, token_bonus=0.5)
    for i, c in enumerate(clips):
        x, sr = sf.read(c["wav"], dtype="int16")
        if x.ndim > 1:
            x = x[:, 0]
        assert sr == 16000
        c["dur"] = len(x) / 16000
        c["tmpwav"] = os.path.join(tmp, f"c{i:02d}.wav")
        sf.write(c["tmpwav"], x, 16000, subtype="PCM_16")
        c["hyp"] = eng.transcribe(x)
        name = f"{c['group']}_{i:02d}.mp3"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", c["tmpwav"], "-codec:a", "libmp3lame", "-b:a", "48k",
                        os.path.join(a.out, "samples", name)], check=True)
        c["mp3"] = "samples/" + name
    emu = emulate(clips, tmp, repo)
    import jiwer
    for c in clips:
        cyc, h = emu[c["tmpwav"]]
        c["instr"] = cyc * 25  # QEMU -icount: the cycle counter reads 1/25 of the instructions of both cores
        c["host_hyp"] = c["hyp"]
        c["hyp"] = h  # the page shows what the firmware transcribes (the chip's own math library)
        c["chip_match"] = h == c["host_hyp"].strip()
        c["wer"] = 100 * jiwer.wer(c["ref"], c["hyp"]) if c["hyp"] else 100.0
        c["t"] = est_seconds(c["instr"], c["dur"])
    json.dump([{k: v for k, v in c.items() if k not in ("tmpwav", "wav")} for c in clips],  # no local paths
              open(os.path.join(a.out, "samples", "results.json"), "w"), indent=1)
    write_html(a.out, clips, load_robust())
    print(f"{len(clips)} clips, chip == host on {sum(c['chip_match'] for c in clips)}/{len(clips)}")


def load_robust():
    rob = json.load(open(os.path.join(RES, "robustness.json")))["wer"]
    rob["pdlm"] = {}  # the released (public-domain) LM, as used on this page
    for l in open(os.path.join(RES, "robust_nemo8_pdlm.txt")):
        m = re.match(r"\S+ (\S+)\.jsonl: WER ([\d.]+)%", l)
        if m:
            rob["pdlm"][m.group(1)] = float(m.group(2))
    return rob


def apply_board(clips, board_json):
    """Replace the emulator's transcripts and the estimated times by what a real ESP32-S3 board produced and took
    (board_bench.py output for the same clips, one core)."""
    import jiwer
    board = json.load(open(board_json))["clips"]
    assert len(board) == len(clips)
    for c, b in zip(clips, board):
        assert abs(b["dur"] - c["dur"]) < 0.01 and b["ref"] == c["ref"], (b["ref"], c["ref"])
        c["emu_hyp"] = c.get("emu_hyp", c["hyp"])
        c["hyp"] = b["hyp"]
        c["board_s"] = b["rtf"] * b["audio"]
        c["chip_match"] = c["hyp"] == c["host_hyp"].strip()
        c["wer"] = 100 * jiwer.wer(c["ref"], c["hyp"]) if c["hyp"] else 100.0
        c["emu_match"] = c["hyp"] == c["emu_hyp"]
        c.pop("t", None)
    return clips


def write_html(out, clips, rob):
    def row(c, first_col):
        t = c["board_s"]
        return (f"<tr><td>{first_col}</td><td><audio controls preload=\"none\" src=\"{c['mp3']}\"></audio>"
                f"<div class=\"small\">{c['dur']:.1f} s</div></td>"
                f"<td>{html.escape(c['ref'])}</td><td>{marked(c['ref'], c['hyp'])}</td>"
                f"<td class=\"num\">{c['wer']:.0f}%</td>"
                f"<td class=\"num\">{t:.1f} s<div class=\"small\">RTF {t/c['dur']:.2f}"
                f"<br>{c['instr']/1e6:.0f} M instructions</div></td></tr>")

    head = ("<tr><th>{}</th><th>Audio</th><th>Reference</th><th>Oído on the ESP32-S3</th><th>WER</th>"
            "<th>Compute on the board</th></tr>")
    rooms = [c for c in clips if c["group"] == "rooms"]
    sents = list(dict.fromkeys(c["sent"] for c in rooms))
    parts = []
    for k, s in enumerate(sents):
        cs = [c for c in rooms if c["sent"] == s]
        parts.append(f"<h3>Sentence {k + 1}</h3><table>" + head.format("Ambience") +
                     "".join(row(c, f"<b>{c['label']}</b><div class=\"small\">{c['desc']}</div>") for c in cs) + "</table>")
    real = [c for c in clips if c["group"] == "real"]
    real_t = "<table>" + head.format("Source") + "".join(
        row(c, f"<b>{c['label']}</b><div class=\"small\">{c['desc']}</div>") for c in real) + "</table>"
    names = {c: l for c, l, _ in CONDITIONS}
    summ = "".join(
        f"<tr><td>{names.get(k, k)}</td>" + "".join(f"<td class=\"num\">{rob[m][k]:.1f}</td>" for m in
                                                     ["pdlm", "nemo8_greedy", "whisper_tiny_en", "moonshine_tiny"]) + "</tr>"
        for k in [c for c, _, _ in CONDITIONS] if k in rob["pdlm"])
    match = sum(c["chip_match"] for c in clips)
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oído: audio samples</title>
<style>
body {{ font-family: Georgia, "Times New Roman", serif; color: #111; background: #fff; max-width: 1080px;
       margin: 0 auto; padding: 24px 16px 64px; line-height: 1.5; }}
h1 {{ font-size: 1.7em; margin-bottom: 0.2em; }} h2 {{ margin-top: 2em; border-bottom: 1px solid #ccc; }}
.links a {{ margin-right: 1.2em; }} .small {{ font-size: 0.8em; color: #555; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.6em 0 1.4em; font-size: 0.92em; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; vertical-align: top; text-align: left; }}
th {{ background: #f4f4f4; }} td.num {{ text-align: right; white-space: nowrap; }}
audio {{ width: 210px; height: 32px; }}
.err {{ color: #c00; font-weight: bold; }} .del {{ color: #c00; text-decoration: line-through; }}
.wrap {{ overflow-x: auto; }}
</style></head><body>
<h1>Oído: open-vocabulary speech recognition on a $5 microcontroller</h1>
<p>Lokutor &middot; <span class="small">ESP32-S3, 240 MHz, 8 MB PSRAM, no neural accelerator; measured on one core</span></p>
<p class="links"><a href="https://github.com/lokutor-ai/oido">Code</a>
<a href="https://huggingface.co/lokutor-ai/oido-ctc-small-int8">Model (int8)</a>
<a href="https://huggingface.co/lokutor-ai/oido-ctc-small-int4">Model (int4)</a>
<a href="https://huggingface.co/lokutor-ai/oido-ctc-small-int8/blob/main/demo.mp4">Demo video</a></p>

<p>Oído runs NVIDIA's 13M-parameter Conformer-CTC Small speech recognizer entirely on an ESP32-S3, with an int8
engine written for the chip's vector instructions. It transcribes any English sentence, not a list of commands.
On LibriSpeech it scores 3.7% / 8.2% word error rate (test-clean / test-other), or 3.3% / 7.2% with the on-chip
language model used on this page.</p>

<p><b>How to read this page.</b> Every clip below was picked at random with a fixed seed, not chosen for looking
good, and every error is shown: <span class="err">wrong or extra words</span> in red,
<span class="del">missed words</span> struck through. Transcripts come from the released firmware (int8 model +
on-chip language model) running on a physical ESP32-S3 board. The laptop build of the same C code gives an
identical transcript on {match} of these {len(clips)} clips; on the others the laptop's math library rounds the last
bit differently from the chip's and a word or two changes, always on the hard (noisy) clips. <i>Compute on the
board</i> is the time the chip took, measured with its 64-bit timer, on one of its two cores; the chip does not run in
real time yet (RTF is compute time divided by the length of the audio, here about 2). The same firmware in the
emulator produced the same transcripts. Text appears after a 0.8 s pause plus this compute time.</p>

<h2>Same sentence, different rooms</h2>
<p>Two LibriSpeech test-clean sentences mixed with real noise recordings (DEMAND), other talkers, and simulated
room reverberation, as in the robustness benchmark of the repository.</p>
<div class="wrap">{''.join(parts)}</div>

<h2>Real-world recordings</h2>
<p>Unmodified recordings from public corpora: home microphones (Common Voice), parliament (VoxPopuli) and
meetings (AMI).</p>
<div class="wrap">{real_t}</div>

<h2>Over the whole benchmark</h2>
<p>Word error rate (%) over 300 sentences per condition. Oído on the chip, with and without the language model;
Whisper tiny.en and Moonshine tiny in full precision on a laptop.</p>
<div class="wrap"><table><tr><th>Ambience</th><th>Oído + LM (ESP32-S3)</th><th>Oído, no LM (ESP32-S3)</th>
<th>Whisper tiny.en (laptop)</th><th>Moonshine tiny (laptop)</th></tr>{summ}</table></div>

<h2>Credits</h2>
<p class="small">Model: NVIDIA stt_en_conformer_ctc_small (CC-BY-4.0). Speech: LibriSpeech (CC-BY-4.0), Common Voice 17
(CC0), VoxPopuli (CC0), AMI Meeting Corpus (CC-BY-4.0). Noise: DEMAND (CC-BY-4.0). Reverberation: pyroomacoustics.
Page generated by <a href="https://github.com/lokutor-ai/oido/blob/main/eval/make_demo_page.py">eval/make_demo_page.py</a>.</p>
</body></html>
"""
    open(os.path.join(out, "index.html"), "w").write(doc)


if __name__ == "__main__":
    main()
