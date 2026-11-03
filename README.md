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

> **Status (3 November 2026): real time on one core.** Version 2 of the models and of the engine runs **faster than speech** on a single 240 MHz core of the
> ESP32-S3-WROOM-1-N16R8 (one board, no neural accelerator): streaming real-time factor **0.54 (variant B) and 0.72 (variant A)** with the on-chip language model, and the final text
> appears **259 ms (B) / 431 ms (A) after the end of speech** with audio fed at the speed of speech (version 1: 1.8-2.0× *slower* than speech, 4-11 s). Streaming costs accuracy:
> LibriSpeech test-clean / test-other 7.1 / 16.4 % (A) and 10.6 / 22.4 % (B) with 0.64 s chunks and the language model
> (test-other stays above 10 % in every streaming mode). See [Real time (v2)](#real-time-v2). The v1 numbers further down describe the version-1 models.

<details><summary>Status of version 1 (9 October 2026): not real time</summary>

> **Status (9 October 2026).** Measured on an ESP32-S3-WROOM-1-N16R8 board, one core at 240 MHz: the English int8 model with
> the language model runs at **1.97× real time** in utterance mode, i.e. one-shot batch transcription of clips stored in flash (1.91–2.02 over 27 clips of 3.6–10.6 s; this is not the live microphone path). Spanish runs at 2.06×, int4 at 1.76×,
> the transducer at 1.89× (see [Speed and memory](#speed-and-memory)). **That is not real time yet:** a 4-second command shows its
> text about 8.7 s after it ends (including the 0.8 s end-of-speech wait). The two-core mode, which emulation had predicted would
> approach real time, gives wrong transcripts on silicon and is disabled (see [Dual core](#dual-core)).
> The board's transcripts are identical to the instruction-exact emulator's on all 27 clips. Accuracy numbers come from the host
> build of the same C code; it differs from the board on some hard clips because of the laptop's math library (22 of the 27
> clips are identical, and the word error rate on those 27 clips is the same, 7.14 %).

</details>

## Which model should I use?

There is **one model per language**, and the models that stream also run in full-context ("utterance") mode.

| I want | Use | Language | Modes | Size | License |
|---|---|---|---|---|---|
| **Real time on the chip**, best accuracy | `models/oido_v2a.tnm` + `oido_v2.tlm` ([Hugging Face](https://huggingface.co/lokutor-ai/oido-v2-realtime-int4)) | English | **streaming** (RTF 0.72; final text 431 ms after the end of speech) and full-context | 7.7 + 1.3 MB | CC-BY-SA-4.0 |
| **Real time on the chip**, fastest | `models/oido_v2b.tnm` + `oido_v2.tlm` ([Hugging Face](https://huggingface.co/lokutor-ai/oido-v2-realtime-int4)) | English | **streaming** (RTF 0.54; final text 259 ms) and full-context | 5.9 + 1.3 MB | CC-BY-SA-4.0 |
| The best English accuracy | `models/nemo8.tnm` + `nemo_lm.tlm` ([Hugging Face](https://huggingface.co/lokutor-ai/oido-ctc-small-int8)) | English | utterance only: text about 3 s after you stop | 14.0 + 1.3 MB | CC-BY-4.0 |
| English with the least flash | `models/nemo4.tnm` ([Hugging Face](https://huggingface.co/lokutor-ai/oido-ctc-small-int4)) | English | utterance only | 8.3 MB | CC-BY-SA-4.0 |
| English, low latency (voice agents) | `models/oido_stream.tnm` ([Hugging Face](https://huggingface.co/lokutor-ai/oido-ctc-small-stream-int8)) | English | **streaming** (final text about 4 s after a 4 s command ends, measured; see [below](#low-latency-streaming-mode)) and full-context | 14.0 MB | CC-BY-SA-4.0 |
| Spanish | `models/oido_es.tnm` + `oido_es.tlm` ([Hugging Face](https://huggingface.co/lokutor-ai/oido-es-ctc-small-int8)) | Spanish | **streaming and full-context, in the same file** | 14.0 + 1.3 MB | CC-BY-4.0 |

- **Streaming vs utterance:** `oido_stream.tnm` and `oido_es.tnm` carry a streaming flag. The firmware's microphone mode and
  `live_demo.py` stream with them automatically; the full-context (utterance) accuracy figures in the tables come from the
  same files run through `tasr_nemo_transcribe` / `eval_engine.py`. The original English models (`nemo8`, `nemo4`) are
  utterance-only.
- **What streaming costs:** some accuracy (English LibriSpeech 4.9 / 11.0 instead of 3.3 / 7.2 with the language model).
  On the board it runs at 1.96× real time on one core, so the final text arrives about half as late as in utterance mode
  (4.1–11.1 s after clips of 3.6–10.6 s, against 7.2–22.4 s), not within 1–2 s.
- **No bilingual model:** English and Spanish have different vocabularies, so each is its own file.

## Real time (v2)

Version 1 spent its time on memory traffic, not on arithmetic: a streaming recognizer re-reads every weight for every chunk, and the chip reads internal SRAM at 3.8 GB/s,
PSRAM at 89 MB/s and flash at 32 MB/s (measured, [`esp32/tools/gemm_bench`](esp32/tools/gemm_bench)). Version 2 changes the model and the engine to fit that:

- **Model** (`models/oido_v2a.tnm`, `models/oido_v2b.tnm`, int4 quantization-aware training; license CC-BY-SA-4.0): 80 ms encoder frames (pairs of frames are stacked), a 64-channel convolutional front end, 16 blocks (A) or 12 blocks (B), 7.7 / 5.9 MB.
- **Engine**: the ROM's asynchronous data-cache preload overlaps the weight stream with computation (layers 38-41 % faster, bit-identical), streaming state first and weights in the rest of PSRAM, work buffers in SRAM, table-driven exp/tanh/sigmoid, a blank-frame shortcut and a beam-2 option in the decoder, deterministic math so that the host build and the chip print identical transcripts (18 of 18 test clips).

Measured on a board (one core, 240 MHz, 18 clips of 120 s, language model on):

| | A: 16 blocks, int4 | B: 12 blocks, int4 |
|---|---|---|
| Streaming real-time factor | **0.72** (beam 4 / top 6), 0.70 (beam 2 / top 3) | **0.54** (beam 2 / top 3) |
| Compute left after the last sample | 289 ms | 198 ms |
| Final text after the end of speech, audio fed at real-time speed (8 clips) | 431 ms (281-574) | 259 ms (159-391) |
| 1.28 s chunks instead of 0.64 s (`sdkconfig.chunk16`) | 0.61, 658 ms | |
| Long utterances: 8 runs of 15-89 s of continuous speech into one stream, audio at real-time speed | 353-634 ms (mean 500) | 178-448 ms (mean 336) |
| Board transcripts equal the host build's | 18 / 18 | 18 / 18 |

LibriSpeech word error rate (%), host build of the same engine, test-clean / test-other:

| Mode | A greedy | A + language model | B greedy | B + language model |
|---|---|---|---|---|
| Streaming, 0.64 s chunks | 10.01 / 19.90 | **7.14 / 16.37** (beam 4 / top 6); 7.81 / 17.29 (2 / 3) | 13.43 / 26.10 | 9.74 / 21.28 (4 / 6); **10.63 / 22.42** (2 / 3) |
| Streaming, 1.28 s chunks | 8.02 / 17.00 | 5.96 / 14.03 | 10.68 / 22.37 | 8.08 / 18.45 |
| Full context | 5.66 / 12.79 | 4.58 / 10.71 | 7.72 / 17.34 | 6.15 / 14.41 |

The delay above is measured from the end of speech as the engine receives it. The firmware's own end-of-speech detector waits 0.8 s of silence by default (`CONFIG_TASR_SEG_HANG_MS`), so the text appears about 1.1 s (B) to 1.3 s (A) after the speaker stops; to get under 1 s, lower that wait (about 0.4 s for B, 0.3 s for A), at the cost of cutting slow speakers at pauses.

What this does not give you: noisy ("other") speech below 10 % in streaming mode, a bilingual or Spanish v2 model, two-core operation, or the live-microphone mode measured (the microphone ring and the 20 s recording buffer take
0.9 MB of PSRAM, which pushes more weights to flash; with the microphone's PSRAM budget the benchmark build measures about 0.015 more, 0.71 for A and 0.55 for B; the I2S capture path itself was not measured because the test board has no microphone). Details, the ablation of each change and the bandwidth table are in the paper (Section 6) and in
[`results/v2`](results/v2) (evaluation tables, parity checks); every board run is in [`esp32/bench`](esp32/bench).

Run it:

```bash
# board: ESP32-S3-DevKitC-1 N16R8 (+ INMP441 on GPIO 4/5/6 for the live microphone)
esp32/tools/flash.sh /dev/ttyUSB0 models/oido_v2a.tnm models/oido_v2.tlm            # variant A, live microphone
TASR_MODE=file TASR_PACED=1 esp32/tools/flash.sh /dev/ttyUSB0 models/oido_v2b.tnm models/oido_v2.tlm clip.wav "reference"   # B, clip at real-time pace
# laptop (same engine, same transcripts as the chip)
cd esp32/host && make && python live_demo.py --model models/oido_v2a.tnm
```

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
- On the board the Spanish model runs at 2.06× real time (12 FLEURS clips, one core). Its transcripts equal the host build's on
  9 of the 12 clips; on the others a single uncertain word differs because of the math library. On these 12 clips the board
  scores 9.3 % WER against 7.3 % for the host build (205 words, a difference of four words).
- It works in both modes: `python live_demo.py --model es` streams, and utterance mode runs the same file.

## Speed and memory

Measured on one ESP32-S3-WROOM-1-N16R8 board (DevKitC-1 class), ESP-IDF 5.5.1, CPU 240 MHz, flash QIO, octal PSRAM 80 MHz,
**one core**, `TASR_MODE=file` (batch utterance transcription of clips stored in flash, not the live microphone path; timed with the 64-bit microsecond timer). Raw numbers:
[`results/board_esp32s3.json`](results/board_esp32s3.json).

| Model (with its language model unless noted) | Clips | Real-time factor (range) | Same transcript as the host build |
|---|---|---|---|
| English int8 (`nemo8.tnm`) | 27 (215 s) | **1.97** (1.91–2.02) | 22 / 27 (**27 / 27 identical to the emulator**) |
| English int4 (`nemo4.tnm`) | 10 (59 s) | 1.76 (1.73–1.83) | 8 / 10 |
| Transducer int8, no LM | 10 (70 s) | 1.89 (1.88–1.91) | 10 / 10 |
| Streaming-trained model, utterance mode | 12 (90 s) | 2.04 (1.98–2.11) | 12 / 12 |
| Streaming-trained model, streaming (32-frame chunks) | 12 (90 s) | 1.96 | 10 / 12 |
| Spanish, utterance mode | 12 (103 s) | 2.06 (1.98–2.13) | 9 / 12 |
| Spanish, streaming | 12 (103 s) | 1.97 | not compared |

| | |
|---|---|
| Flash | 14.0 MB (int8), or **8.3 MB (int4), which leaves a 6 MB app partition for your own code** (`partitions_nemo4.csv`) |
| PSRAM | 2.4 MB working memory peak for a 20 s utterance (measured in QEMU); the rest caches the most-reused weights. On the board, at least 1.4 MB of PSRAM stayed free on clips of up to 10.6 s (0.3–0.4 MB with the streaming caches) |
| Internal RAM | at least 122 KB free (72 KB with the streaming caches) |
| Compute | the board spends 461 M cycles per second of audio, **2.0 cycles per executed instruction**: every stage costs 1.3–2.4 cycles per instruction (the im2col gather 3.7), so on one core the time is set by the number of instructions the kernels issue, not by waiting for weights. 220 M instructions per second of audio across both cores (exact, QEMU `-icount`) |
| Latency, utterance mode | the text of a 3.6–3.8 s command appears 6.9–7.4 s after it ends, plus the 0.8 s end-of-speech wait |
| Latency, streaming mode | 4.1–4.2 s after a 3.6–3.8 s command ends (see [below](#low-latency-streaming-mode)) |
| What would make it real time | the second core (a perfect split of the 55 % of instructions on the critical path would give about 1.1×; not measured) and faster kernels. Both are open, see [Dual core](#dual-core) |

The firmware's transcripts equal the emulator's on all 27 clips, and for the two clips we traced layer by layer the activations
of every stage are bit-identical between the board and the emulator. The host build differs in the last bit of the log-mel
features (the laptop's math library), which changes a word on some hard clips.

## Low-latency streaming mode

For voice-to-voice and other interactive uses, the time between the end of a sentence and its final text matters more
than the last tenth of a percent of accuracy. `models/oido_stream.tnm` is the same Conformer, fine-tuned to run on
audio **as it arrives**: the encoder processes 1.28 s chunks (32 frames) with 5 s of left context while you speak, so
when you stop only the last partial chunk is left to compute. Partial text appears while you talk.

| Mode (all on the ESP32-S3 engine, with the language model) | LibriSpeech clean / other | Mean WER, 14 noise and reverb conditions | Final text after the clip ends, measured on the board (one core) |
|---|---|---|---|
| Utterance mode, released `nemo8.tnm` | 3.3 / 7.2 | 7.5 | 6.9–21.3 s for clips of 3.6–10.6 s (about 7 s for a 3.7 s command) |
| **Streaming, 32-frame chunks** | 4.9 / 11.0 | 9.0 | **4.1–11.1 s** (about 4 s for a 3.7 s command) |
| Streaming, 16-frame chunks (0.64 s) | 6.1 / 13.2 | n/a | not measured |
| Same `oido_stream.tnm`, full-context mode | 3.4 / 7.8 | 6.2 | 7.2–22.4 s |

- **What we measured.** When the chip has kept up with the audio, the last step after the final chunk takes 1.1 s on average on the board
  (0.8–1.6 s over 12 clips). But on one core the chip runs at 1.96× real time, so it falls behind the speaker and the backlog grows
  with the length of the utterance. Feeding six clips at the rate of speech (one 20 ms block when its last sample would have been spoken), the final text appeared
  4.1–11.1 s after the last sample (3.6–10.6 s clips; mean 6.7 s), roughly (RTF − 1) × duration plus that last step. The 0.8 s end-of-speech
  wait (`CONFIG_TASR_SEG_HANG_MS`, or `live_demo.py --pause`) comes on top. Streaming halves the wait; it does not give the 1.1–1.4 s we had estimated
  from instruction counts before we had a board. That needs a real-time factor below one.
- It was trained with the original model's weights as a starting point and noise, music, babble and reverberation
  augmentation (MUSAN, simulated rooms), so it is as robust as the utterance model. The accuracy cost is the chunking:
  the encoder cannot see what comes after the chunk.
- **Cost of chunking.** A chunk re-reads every weight, so weight traffic is 2× the 64-frame blocks of utterance mode at 32 frames and 4× at 16 frames.
  On the board this does not show at the chip's present speed: streaming at 32 frames runs at 1.96× real time, slightly faster than the same model in
  utterance mode (2.04×), because one core is limited by instruction issue. It may matter once both cores work.
- On the board, the streaming model's transcripts equal the host build's on 10 of 12 clips in streaming mode and 12 of 12 in utterance mode.

## Dual core

`TASR_DUAL=1 esp32/tools/flash.sh ...` splits each job across both cores. On our boards at 240 MHz this mode is **not correct**: transcripts come out
wrong, or the firmware crashes, when both cores run the heavy matrix products at the same time (Guillermo found this first). The same split is correct
when its two halves run one after the other, when all the work runs on the second core, and with the PIE kernels compiled out (`TASR_NO_SIMD`). Until it is
understood, firmware builds run on one core. QEMU's estimate for two cores was a real-time factor of 0.7–0.95; QEMU executes the cores in coarse
alternation and cannot show this.

What we established on 9 October with one board (tools: `esp32/tools/dual_bisect.py`, `dump_cmp.py`; builds with `TASR_DEBUG_SUMS=1` print a hash of every
layer's output; clip: a 3.8 s VoxPopuli sentence; every line below is 3 runs):

- **One core is exactly reproducible**: 166 hashes identical in every run, and identical to the emulator's.
- **Two cores at 240 MHz corrupt activations at random places** (the hash values differ from run to run), always first in layer 1 or 2, almost always in the
  feed-forward block of layer 1; in roughly one run in eight the firmware crashes with `IllegalInstruction` and a corrupted backtrace, which looks like corrupted
  instruction or data fetches rather than a logic error.
- **Two cores at 160 MHz are correct**: 166 of 166 hashes identical to the one-core run, 3 of 3 runs (but no faster than one core at 240 MHz).
- Parallelizing everything except the matrix products is correct; parallelizing only the matrix products fails. Among them, the K=176 products and the
  front-end K=3520 products are correct on two cores, and the K=704 ones (the second linear layer of each feed-forward block) corrupt. These move the most
  weight data per cycle.
- **Not the cause**: PSRAM speed (80 → 40 MHz), data-cache line size (64 → 32 B), weights in flash versus PSRAM, FreeRTOS semaphores (a spin-wait barrier
  behaves the same), interrupts (masked while a job runs), where the activation tile lives (PSRAM, or a private copy for the second core), instruction
  fetch (kernel in IRAM, or a separate copy of the kernel per core), the split itself (by rows instead of by output channels).
- **Also not the cause** (a second investigation on another board of the same kind, ESP32-S3 rev v0.2 with octal PSRAM):
  - the engine's C code: two real threads on a laptop give the reference transcripts, and Valgrind's race detector (`helgrind`) finds no data race;
  - erratum CACHE-126: writing the whole data cache back with interrupts masked before every job, as the erratum's workaround describes, does not help;
  - flash and PSRAM at 40 MHz;
  - a FreeRTOS context switch: the failure persists with both cores spinning on shared flags.

  Two results from that board narrow it down:
  - Espressif's esp-nn bit-exactness tests with its own two-core mode pass on it (326 of 326, buffers in PSRAM). The chip can run PIE kernels on both cores; the trigger is specific to our workload.
  - With the whole engine linked into IRAM, the two-core language-model, front-end and im2col jobs pass repeatedly, but the encoder matrix products still crash inside the dot-product loop.
- Even a correct two-core mode would not reach real time on this evidence: the incorrect spin-wait build ran at 0.78 × the one-core time, not the 0.55 the
  instruction counts promised, so the cores contend for memory.

What the failure looks like, and what we now think it is:

- A stress test with no Oído code in it, on the same module class, reproduced silent hangs and brownout resets under dense PIE load on both cores, while plain loads on both cores, or PIE on one core, passed. The failure follows the power drawn (activity times clock frequency) and not the memory the data lives in.
- The chip's own brownout detector, read at each of its thresholds, showed the 3.3 V rail falling below its lowest threshold (about 2.44 V) when both cores ran dense PIE loads, and staying above 3.19 V at idle. This is an on-chip indication, not a measurement of the rail with an instrument.
- The failure is therefore consistent with a collapse of the board's 3.3 V supply, not with a fault in the engine. That fits the 160 MHz result, the random values, the `IllegalInstruction` crashes and the dependence on timing and memory layout.
- Ito's own long soak (a 21-minute synthetic two-core stress and 600 iterations of a golden sentence) passed on a different board with a better supply. Both statements are true: whether a board survives depends on its regulator, bulk capacitance, cable and host port. See Ito's [`esp32/DUAL_CORE_POWER.md`](https://github.com/lokutor-ai/ito/blob/main/esp32/DUAL_CORE_POWER.md).

If you can, try `TASR_DUAL=1` on another board or supply and tell us what you see (`python esp32/tools/dual_bisect.py --help`).

**A split that works on this board (measured).** The parallel call sites go through a runtime mask, `tasr_par_mask` (one bit per site; a cleared bit runs that site on core 0 only). With the encoder matrix products kept on core 0 and the other parallel sites (features, layer norm, SiLU, quantise, convolutions, attention) split across both cores, the Spanish streaming run gave a transcript identical to the host's and a real-time factor of 1.84 against 1.92 on one core. The matrix products are about 54 % of the cycles, which bounds this kind of split to a saving of about a quarter of the time. Most split stages lose 30 to 50 % of their cycles, but im2col is 2.5 times slower on two cores, because its gather writes external memory from both cores. The matrix products on both cores fail at a fixed early call (the first encoder GEMM, the 7th parallel call of the streaming path) with a hang, or a cache error in the language model's GEMM.

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
TASR_MODE=file TASR_PACED=1 esp32/tools/flash.sh <port> models/oido_stream.tnm clip.wav "reference"   # streaming models: also feed at the rate of speech and print when the final text appears
```

Boards with the **native USB port** (`/dev/cu.usbmodem*`, `/dev/ttyACM*`): after flashing, a normal reset leaves the chip in download mode, so `flash.sh` boots the
application with `esptool --after watchdog_reset`; if you read the console yourself, open the serial port with DTR and RTS low
(pyserial: `Serial()`, set `dtr = rts = False`, then `open()`), or the bridge resets the chip into download mode. After a crash the USB device can stop
answering until it is reset (libusb `libusb_reset_device`) or the cable is replugged.

To benchmark a set of clips and compare with the emulator and the host build: `python esp32/tools/board_bench.py --port <port> --model models/nemo8.tnm --lm models/nemo_lm.tlm --clips clips.json --out result.json`
(run it with ESP-IDF's Python, which has pyserial; `PYTHON=` points the tools that need numpy and soundfile to another interpreter), then `tools/merge_board.py`.

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
esp32/tools/               flash.sh, emulate.sh, run_qemu.sh, bench_latency.py, mkimages.py, board_bench.py (clips on a real board),
                           summarize_board.py / merge_board.py, dual_bisect.py / dump_cmp.py (two-core debugging)
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
- Speed: about twice slower than real time on one core today (see [Speed and memory](#speed-and-memory)); the two-core mode is not correct on silicon yet.
- Requires an ESP32-S3 with 16 MB flash and 8 MB octal PSRAM (N16R8).
- Known issue: the one-shot batch `transcribe` call inside a live microphone capture loop hangs on the chip, although the same call works on clips stored in flash and streaming works in the live loop. The cause is not understood; use streaming for live capture.

## License

- **Code** is licensed under the **GNU GPL v3** ([`LICENSE`](LICENSE)).
- For products that cannot meet GPLv3 terms (for example, consumer devices that do not allow users to install modified
  firmware), Lokutor offers commercial licenses and support. See [`COMMERCIAL.md`](COMMERCIAL.md).
- **Model weights** are derived from NVIDIA's `stt_en_conformer_ctc_small`: `nemo8.tnm` is under **CC-BY-4.0**,
  while `nemo4.tnm` and `oido_stream.tnm` (fine-tuned on public corpora that include share-alike data) are under
  **CC-BY-SA-4.0**. The language model `nemo_lm.tlm` is under **CC-BY-4.0**. The Spanish model `oido_es.tnm`
  (fine-tuned only on CC0 and CC-BY data) and its language model `oido_es.tlm` are under **CC-BY-4.0**. See [`NOTICE`](NOTICE).

Lokutor also has models for other languages (Catalan, Basque, Galician and more) and an on-device TTS for the same chip. Contact us for these.
