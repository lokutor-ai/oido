# Oído: speech recognition that fits in a $5 chip

*¡Oído!* is what cooks call out in a Spanish kitchen to confirm an order: *heard, got it*.

[![Oído demo: speech recognition on a $5 chip](docs/demo.gif)](https://huggingface.co/lokutor-ai/oido-ctc-small-int8/blob/main/demo.mp4)

*Animated demo; [full video with sound](https://huggingface.co/lokutor-ai/oido-ctc-small-int8/blob/main/demo.mp4).
The transcripts are Oído's output, sped up. Footage from a physical board is coming.*

Speech-to-text for any English or Spanish sentence, running entirely on an **ESP32-S3** (240 MHz dual-core Xtensa LX7, 8 MB PSRAM,
16 MB flash). No cloud, no command list, no neural accelerator. Built by [Lokutor](https://lokutor.com).
Models on Hugging Face: [int8](https://huggingface.co/lokutor-ai/oido-ctc-small-int8) ·
[int4](https://huggingface.co/lokutor-ai/oido-ctc-small-int4).

**[Audio samples](https://lokutor-ai.github.io/oido/)**: random clips in a quiet room, a car, a kitchen, a cafeteria,
reverberant rooms and real meetings, each with the reference, the on-chip transcript (errors marked) and the compute
it took on the chip.

> **Status (2 October 2026).** Accuracy numbers come from the host build of the engine, which compiles the same C code
> as the firmware (same int8 arithmetic). Under Espressif's QEMU emulator the firmware gives identical transcripts on
> most utterances; the laptop's math library and the chip's round the last bit differently, which can change a word on
> some hard, noisy clips (22 of 27 random clips identical, see the [audio samples](https://lokutor-ai.github.io/oido/)).
> Real-time speed is **estimated** from exact emulator instruction counts. Measurements on physical boards follow in the
> next days and will be added here.

## Which model should I use?

There is **one model per language**, and the models that stream also run in full-context ("utterance") mode.

| I want | Use | Language | Modes | Size | License |
|---|---|---|---|---|---|
| The best English accuracy | `models/nemo8.tnm` + `nemo_lm.tlm` ([Hugging Face](https://huggingface.co/lokutor-ai/oido-ctc-small-int8)) | English | utterance only: text about 3 s after you stop | 14.0 + 1.3 MB | CC-BY-4.0 |
| English with the least flash | `models/nemo4.tnm` ([Hugging Face](https://huggingface.co/lokutor-ai/oido-ctc-small-int4)) | English | utterance only | 8.3 MB | CC-BY-SA-4.0 |
| English, low latency (voice agents) | `models/oido_stream.tnm` ([Hugging Face](https://huggingface.co/lokutor-ai/oido-ctc-small-stream-int8)) | English | **streaming** (final text about 1.1-1.4 s after you stop, estimated) and full-context | 14.0 MB | CC-BY-SA-4.0 |
| Spanish | `models/oido_es.tnm` + `oido_es.tlm` ([Hugging Face](https://huggingface.co/lokutor-ai/oido-es-ctc-small-int8)) | Spanish | **streaming and full-context, in the same file** | 14.0 + 1.3 MB | CC-BY-4.0 |

- **Streaming vs utterance:** `oido_stream.tnm` and `oido_es.tnm` carry a streaming flag. The firmware's microphone mode and
  `live_demo.py` stream with them automatically; the full-context (utterance) accuracy figures in the tables come from the
  same files run through `tasr_nemo_transcribe` / `eval_engine.py`. The original English models (`nemo8`, `nemo4`) are
  utterance-only.
- **What streaming costs:** some accuracy (English LibriSpeech 4.9 / 11.0 instead of 3.3 / 7.2 with the language model),
  and estimated speed on the real chip is still unmeasured.
- **No bilingual model:** English and Spanish have different vocabularies, so each is its own file.

## Accuracy

Word error rate (%) on LibriSpeech, same text normalization for every system.

| System | Runs on | test-clean | test-other | Size |
|---|---|---|---|---|
| **Oído**: NVIDIA Conformer-CTC Small, int8, greedy (this repo) | ESP32-S3 | **3.7** | **8.2** | 14.0 MB |
| **Oído int4** (`models/nemo4.tnm`, 4-bit quantization-aware fine-tune), greedy | ESP32-S3 | 4.6 | 10.0 | **8.3 MB** |
| **Oído int8 + language model** (`models/nemo_lm.tlm`, beam search on chip) | ESP32-S3 | 3.3 | 7.2 | 15.3 MB |
| **Oído int4 + language model** | ESP32-S3 | 3.8 | 8.4 | 9.6 MB |
| **Oído streaming** (`models/oido_stream.tnm`) + language model, 32-frame chunks (see [below](#low-latency-streaming-mode)) | ESP32-S3 | 4.9 | 11.0 | 15.3 MB |
| Oído with NVIDIA Conformer-Transducer Small, int8 (weights not included, see below) | ESP32-S3 | 3.0 | 6.7 | 15.5 MB |
| Espressif MultiNet7 (ESP-SR benchmark; its API takes fixed command lists) | ESP32-S3 | 8.5 | 21.3 | 2.7 MB |
| Moonshine tiny, fp32 | laptop | 5.0 | 12.1 | 27 M params |
| Whisper tiny.en, fp32 | laptop | 6.3 | 15.9 | 39 M params |
| Vosk small (Kaldi) | laptop | 9.9 | 21.6 | 40 MB |

- On-chip rows use the full test sets. Laptop baselines use 500 evenly spaced utterances per set. MultiNet7 figures are from
  Espressif's ESP-SR benchmark page.
- The int8 engine is within 0.1 points of full precision: 3.70 / 8.23 on chip vs 3.68 / 8.11 for the original fp32 model.
- The language model is a 1.3 M-parameter GRU trained only on public-domain books (the LibriSpeech LM corpus). The
  device runs CTC prefix beam search with it (beam 4). It is optional: without it the engine decodes greedily.
- To our knowledge this is the most accurate LibriSpeech result published for any microcontroller. It is not the first
  open-vocabulary recognizer on one (Arm has shown Conformer models on Cortex-M55 + Ethos-U NPUs).

**Robustness** (300 LibriSpeech utterances under real DEMAND noise, babble and room reverb; `eval/make_robust.py`,
full numbers in [`results/robustness.json`](results/robustness.json)):

| Mean WER over 14 conditions | CTC int8 (chip) | CTC int8 + LM (chip) | Transducer int8 (chip) | Whisper tiny.en | Moonshine tiny | Vosk small |
|---|---|---|---|---|---|---|
| | **8.4** | 7.5 | 6.7 | 12.1 | 12.2 | 21.7 |

For the transducer, car and kitchen noise at 5 dB SNR cost under 1 point, and living-room noise about 1.7. Four-talker babble at 5 dB and very
reverberant rooms are the hard cases.

## Spanish

`models/oido_es.tnm` + `models/oido_es.tlm` is a Spanish model for the same chip and engine: NVIDIA's Conformer
fine-tuned on 2,492 hours of Spanish (Common Voice, VoxPopuli, Multilingual LibriSpeech, FLEURS) with noise and
reverberation augmentation and a new Spanish vocabulary, plus a Spanish language model. 14.0 MB + 1.3 MB, the same speed
as the English model.

Word error rate (%), same normalization for every system (lowercase, accents kept, punctuation removed, references with
digits excluded), on the same 400 evenly spaced utterances per test set (333 for FLEURS, after dropping references with digits):

| System | Runs on | Common Voice | MLS | VoxPopuli | FLEURS |
|---|---|---|---|---|---|
| **Oído Spanish + language model** | ESP32-S3 | **13.3** | **10.6** | **15.5** | **11.3** |
| Oído Spanish, greedy | ESP32-S3 | 20.3 | 14.2 | 20.0 | 16.9 |
| Oído Spanish, streaming (32-frame chunks) + language model | ESP32-S3 | 15.2 | 12.1 | 16.3 | 12.8 |
| Whisper tiny (multilingual), fp32 | laptop | 33.0 | 21.5 | 28.7 | 17.1 |

On the complete test sets (60 hours) the numbers are 13.8 / 10.9 / 15.7 / 11.3 with the language model
([`results/es_benchmark.json`](results/es_benchmark.json)).

- **Read this fairly.** Oído was fine-tuned on the training splits of these corpora, while Whisper tiny is zero-shot, so
  the comparison favors us on these domains; on phone calls, strong regional accents or specialized vocabulary expect
  higher error. We have not built a Spanish noise benchmark; the model is trained with the same noise and reverberation
  augmentation as the English streaming model.
- The language model weights (0.5 / 1.5) were picked from a small grid on subsets of the test sets; the optimum is flat
  (all of 0.4–0.6 / 1.5–2.0 are within 0.2 points).
- The model writes numbers as words (*veinte*), not digits.
- Under Espressif's emulator the firmware's transcripts match the laptop build on most utterances; on uncertain ones the
  last-bit rounding of the math library can change a word (2 of 3 clips we compared). Speed on silicon is estimated, as
  for English.
- It works in both modes: `python live_demo.py --model es` streams, and utterance mode runs the same file.

## Speed and memory

| | |
|---|---|
| Flash | 14.0 MB (int8), or **8.3 MB (int4), which leaves a 6 MB app partition for your own code** (`partitions_nemo4.csv`) |
| PSRAM | 2.4 MB working memory peak for a 20 s utterance (measured in QEMU); the rest caches the most-reused weights |
| Compute | ~225 M instructions per second of audio across both cores, ~121 M on the dual-core critical path (exact, QEMU `-icount`) |
| Real-time factor | **1.97 measured on an ESP32-S3 N16R8 board, one core** (Spanish model with the language model, 4.7 s clip, `TASR_MODE=file`); streaming 1.93. Firmware builds use one core by default: see [Dual core](#dual-core) |
| Latency | Utterance mode. Text appears after a 0.8 s pause plus compute: about 3 s for a 2–4 s command. [Streaming mode](#low-latency-streaming-mode) cuts this to about 1.1–1.4 s |

## Low-latency streaming mode

For voice-to-voice and other interactive uses, the time between the end of a sentence and its final text matters more
than the last tenth of a percent of accuracy. `models/oido_stream.tnm` is the same Conformer, fine-tuned to run on
audio **as it arrives**: the encoder processes 1.28 s chunks (32 frames) with 5 s of left context while you speak, so
when you stop only the last partial chunk is left to compute. Partial text appears while you talk.

| Mode (all on the ESP32-S3 engine, with the language model) | LibriSpeech clean / other | Mean WER, 14 noise and reverb conditions | Final text after you stop (estimated) |
|---|---|---|---|
| Utterance mode, released `nemo8.tnm` | 3.3 / 7.2 | 7.5 | ~3.0–3.4 s for a 2–4 s command |
| **Streaming, 32-frame chunks** | 4.9 / 11.0 | 9.0 | **~1.1–1.4 s** |
| Streaming, 16-frame chunks (0.64 s) | 6.1 / 13.2 | n/a | ~1.1–1.3 s, but see the speed note |
| Same `oido_stream.tnm`, full-context mode | 3.4 / 7.8 | 6.2 | as utterance mode |

- The final text arrives 0.8 s (the end-of-speech pause, `CONFIG_TASR_SEG_HANG_MS`, or `live_demo.py --pause`) plus
  0.25–0.6 s of compute for the last chunk and a pass over the weights. Voice agents with their own turn detector can
  use a shorter pause.
- It was trained with the original model's weights as a starting point and noise, music, babble and reverberation
  augmentation (MUSAN, simulated rooms), so it is as robust as the utterance model. The accuracy cost is the chunking:
  the encoder cannot see what comes after the chunk.
- **Speed is the open question.** Instruction counts are exact, but this chip is limited by how fast weights can be read
  from flash and PSRAM, and a chunk re-reads them every chunk (4× as often at 16 frames as the 64-frame blocks of
  utterance mode). Estimated real-time factor while speaking: **0.80–1.00 at 32 frames**, 0.96–1.24 at 16 frames (which
  may fall behind). All of this is estimated, not measured on silicon; board numbers will replace it.
- The transcripts of the firmware under QEMU match the laptop build on the clips we compared (see
  [`results/en_stream_v2.json`](results/en_stream_v2.json)).

## Dual core

`TASR_DUAL=1 esp32/tools/flash.sh ...` splits each job across both cores. On silicon this mode is **not correct yet**:
transcripts come out wrong, or the firmware crashes, when both cores run the PIE kernels at the same time. The same
split is correct when its two halves run one after the other, when all the work runs on the second core, and when the
PIE kernels are compiled out (`TASR_NO_SIMD`), so the cause is concurrent PIE execution on the two cores, not the work
split. Until it is fixed, firmware builds run on one core. QEMU's estimate for two cores was a real-time factor of
0.7–0.95.

## How it works

- **Front end:** log-mel features, then 2× (3×3, stride 2) convolution subsampling to 25 Hz.
- **Encoder:** 16 Conformer layers (d = 176, 4 heads, relative-position attention, conv kernel 31).
- **Decoding:** CTC over 1024 BPE tokens. The engine also supports an RNN-T head (LSTM 320 + joint network) and a GRU
  language model with CTC prefix beam search.

The engine (`esp32/components/tinyasr`) is new C written for the ESP32-S3's PIE vector unit:
- int8 matrix kernels on `EE.VMULAS.S8.ACCX` (16 MACs per instruction), plus int4 outer-product kernels;
- int8 relative-position attention with a lookup-table softmax;
- dual-core scheduling (experimental, see [Dual core](#dual-core));
- tiling so that each weight is streamed from flash once per 64 frames (weight traffic cut from 18 to 7.5 MB/s);
- a VAD/AGC utterance segmenter;
- an optional SSD1306 OLED that shows the live transcript.

## Try it

**On a laptop, with the same engine as the chip.** Needs Python with numpy, soundfile, sentencepiece and sounddevice.

```bash
cd esp32/host && make
./tasr_cli ../../models/nemo8.tnm recording.wav      # 16 kHz mono PCM16 wav
python live_demo.py                                   # microphone -> the firmware's VAD + engine, with ESP32 time estimates
python live_demo.py --model fast --no_lm              # int4, greedy decoding
python live_demo.py --model stream                    # low latency: words appear while you speak (--pause 0.5 for a shorter pause)
python live_demo.py --model es                        # Spanish (models/oido_es.tnm + oido_es.tlm)
```

**On a board.** ESP32-S3-DevKitC-1 **N16R8**, an INMP441 I2S microphone (SCK→GPIO4, WS→GPIO5, SD→GPIO6, L/R→GND),
and optionally a 0.96" SSD1306 OLED (SDA→GPIO8, SCL→GPIO9). Needs ESP-IDF v5.5.

```bash
esp32/tools/flash.sh /dev/ttyUSB0 models/nemo8.tnm              # live microphone, most accurate (+ models/nemo_lm.tlm)
esp32/tools/flash.sh /dev/ttyUSB0 models/nemo4.tnm              # int4: 8.3 MB, leaves 6 MB of flash for your app
TASR_OLED=1 esp32/tools/flash.sh /dev/ttyUSB0 models/nemo8.tnm  # + transcript on the OLED
TASR_MODE=file esp32/tools/flash.sh /dev/ttyUSB0 models/nemo8.tnm clip.wav "reference"   # prints measured RTF
NO_LM=1 esp32/tools/flash.sh /dev/ttyUSB0 models/nemo8.tnm      # greedy decoding, no language model
```

**In the emulator** (Espressif QEMU 9.x): runs the real firmware, then reports the transcript and instruction counts.

```bash
esp32/tools/emulate.sh clip.wav
```

**The transducer model.** Its weights are NVIDIA's, distributed on NGC under NVIDIA's terms, so they are not included
here. You can fetch and convert them yourself:

```bash
cd train && python fetch_nemo_small.py ../models/nemo_rnnt --transducer
python export_nemo.py ../models/nemo_rnnt ../models/rnnt8.tnm 8
```

## Repository

```
esp32/components/tinyasr/  on-chip engine: tasr_nemo.c (Conformer CTC/RNN-T), kernels.c (PIE SIMD), tinyasr_lm.c
                           (GRU LM + beam search), tasr_seg.c (VAD), tinyasr.c (streaming engine)
esp32/firmware/            ESP-IDF app: live I2S microphone or benchmark mode, OLED, partition layouts
esp32/host/                host build of the engine: tasr_cli, live_demo.py, seg_test, eval_engine.py, benchmark.py
esp32/tools/               flash.sh, emulate.sh, run_qemu.sh, bench_latency.py, mkimages.py
train/                     PyTorch port of NVIDIA's model (nemo_small.py, rnnt_small.py), exporters, GRU LM training,
                           int4 QAT (train_nemo_qat.py), streaming + new-language fine-tuning (train_nemo_stream.py,
                           augment.py, filter_teacher.py, make_tok.py, prep_es.py)
eval/                      WER normalization, robustness benchmark builder, laptop baselines
results/                   benchmark outputs behind the numbers above
models/                    nemo8.tnm (int8, 14.0 MB), nemo4.tnm (int4, 8.3 MB), oido_stream.tnm (streaming int8, 14.0 MB),
                           nemo_lm.tlm (language model, 1.3 MB), oido_es.tnm + oido_es.tlm (Spanish, 14.0 + 1.3 MB), and the tokenizers
```

## Limitations

- English and Spanish only.
- Utterance mode prints text after each utterance; streaming mode shows partial text but is less accurate (see above).
- Very noisy crowds and reverberant rooms remain hard.
- Speed is estimated until board measurements are published.
- Requires an ESP32-S3 with 16 MB flash and 8 MB octal PSRAM (N16R8).

## License

- **Code** is licensed under the **GNU GPL v3** ([`LICENSE`](LICENSE)).
- For products that cannot meet GPLv3 terms (for example, consumer devices that do not allow users to install modified
  firmware), Lokutor offers commercial licenses and support. See [`COMMERCIAL.md`](COMMERCIAL.md).
- **Model weights** are derived from NVIDIA's `stt_en_conformer_ctc_small`: `nemo8.tnm` is under **CC-BY-4.0**,
  while `nemo4.tnm` and `oido_stream.tnm` (fine-tuned on public corpora that include share-alike data) are under
  **CC-BY-SA-4.0**. The language model `nemo_lm.tlm` is under **CC-BY-4.0**. The Spanish model `oido_es.tnm`
  (fine-tuned only on CC0 and CC-BY data) and its language model `oido_es.tlm` are under **CC-BY-4.0**. See [`NOTICE`](NOTICE).

Lokutor also has models for other languages (Catalan, Basque, Galician and more) and an on-device TTS for the same chip. Contact us for these.
