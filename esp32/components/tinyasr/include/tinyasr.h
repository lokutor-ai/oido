// tinyasr: streaming open-vocabulary English ASR small enough for an ESP32-S3.
// Portable C99; optional ESP32-S3 PIE SIMD kernels.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tasr_model tasr_model_t;
typedef struct tasr_stream tasr_stream_t;

// blob must stay valid for the model lifetime (weights are referenced in place, e.g. mmapped flash / PSRAM).
tasr_model_t *tasr_model_load(const uint8_t *blob, size_t size);
void tasr_model_free(tasr_model_t *m);
size_t tasr_model_weight_bytes(const tasr_model_t *m);
size_t tasr_model_blob_bytes(const tasr_model_t *m);
// Relocate weight matrices into tasr_alloc(kind 0) memory (PSRAM on ESP32) up to budget bytes; returns bytes moved.
size_t tasr_model_place_weights(tasr_model_t *m, size_t budget);

// chunk: encoder frames per step at 25 Hz (even, e.g. 32 = 1.28 s); left_chunks: attention history in chunks.
tasr_stream_t *tasr_stream_create(const tasr_model_t *m, int chunk, int left_chunks);
void tasr_stream_free(tasr_stream_t *s);
void tasr_stream_reset(tasr_stream_t *s);

// Feed 16 kHz mono PCM. Returns number of new tokens emitted.
int tasr_stream_feed(tasr_stream_t *s, const int16_t *pcm, int n);
// Flush remaining audio at end of utterance.
int tasr_stream_finish(tasr_stream_t *s);
// Transcript so far (NUL terminated, leading space trimmed).
const char *tasr_stream_text(tasr_stream_t *s);

// Optional beam-search decoder with LM fusion (see tinyasr_lm.h); NULL = greedy CTC.
struct tasr_decoder;
void tasr_stream_set_decoder(tasr_stream_t *s, struct tasr_decoder *d);

// Debug: if set, every encoder output frame's logits (V+1 floats) are appended here (host testing).
void tasr_stream_set_logit_sink(tasr_stream_t *s, float *buf, int max_frames, int *n_frames);

// Profiling (compile with -DTASR_PROFILE): per-stage cycle (ESP32) / ns (host) totals.
const char *tasr_profile_name(int i);
uint64_t tasr_profile_value(int i);
void tasr_profile_reset(void);

// Parallelism hook: run fn over [0, n) split across workers (worker ids 0..1). Default: serial.
typedef void (*tasr_job_fn)(void *ctx, int begin, int end, int worker);
extern void (*tasr_parallel)(tasr_job_fn fn, void *ctx, int n);
// Which call sites of the neural-network paths use tasr_parallel (bit = site id; the others run on worker 0 only).
// Sites: 0 features, 1 layer norm, 2 SiLU, 3 GEMM K<=256, 4 quantise, 5 conv0, 6 im2col, 7 attention, 8 depthwise conv,
// 9 GEMM K<=1024, 10 GEMM K>1024, 11 int4 GEMM, 12 language-model GEMM. Default: all.
extern uint32_t tasr_par_mask;
#define TASR_PAR(site, fn, ctx, n) \
    do { if (tasr_par_mask & (1u << (site))) tasr_parallel(fn, ctx, n); else (fn)((ctx), 0, (n), 0); } while (0)

// Memory hooks (default: aligned malloc). kind: 0 = large/slow ok (PSRAM), 1 = hot (internal SRAM preferred)
extern void *(*tasr_alloc)(size_t size, int kind);
extern void (*tasr_free)(void *p);

#ifdef __cplusplus
}
#endif
