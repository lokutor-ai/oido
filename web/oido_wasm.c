// WebAssembly build of the Oído streaming recognizer: the same C engine as the ESP32-S3 firmware, for the in-browser demo (docs/demo).
// build: web/build.sh (needs emscripten). All functions work on one global stream; JS owns the model and LM buffers.
#include <emscripten.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "tinyasr.h"
#include "tinyasr_lm.h"
#include "tasr_nemo.h"

static tasr_nemo_t *g_m;
static tasr_lm_t *g_lm;
static tasr_decoder_t *g_dec;
static tasr_nemo_stream_t *g_s;
static char g_text[8192];

EMSCRIPTEN_KEEPALIVE void *oido_alloc(size_t n) { return aligned_alloc(16, (n + 15) & ~(size_t)15); }
EMSCRIPTEN_KEEPALIVE void oido_free(void *p) { free(p); }

EMSCRIPTEN_KEEPALIVE void oido_unload(void)
{
    if (g_s) tasr_nemo_stream_free(g_s);
    if (g_dec) tasr_decoder_free(g_dec);
    if (g_lm) tasr_lm_free(g_lm);
    if (g_m) tasr_nemo_free(g_m);
    g_s = NULL; g_dec = NULL; g_lm = NULL; g_m = NULL;
}

// model/lm: 16-byte aligned buffers from oido_alloc that stay alive while loaded. Returns 0, or a negative error.
EMSCRIPTEN_KEEPALIVE int oido_load(const uint8_t *model, size_t mlen, const uint8_t *lm, size_t llen, int beam, int topk)
{
    oido_unload();
    g_m = tasr_nemo_load(model, mlen);
    if (!g_m) return -1;
    if (!tasr_nemo_stream_supported(g_m)) return -2;
    if (lm && llen) {
        g_lm = tasr_lm_load(lm, llen);
        if (g_lm) g_dec = tasr_decoder_create(g_lm, 1025, beam, topk, tasr_lm_weight(g_lm, 0.3f), tasr_lm_bonus(g_lm, 0.5f));
    }
    g_s = tasr_nemo_stream_new(g_m, 0, 0, g_dec);
    if (!g_s) return -3;
    tasr_nemo_stream_reset(g_s);
    return 0;
}

EMSCRIPTEN_KEEPALIVE void oido_reset(void) { if (g_s) tasr_nemo_stream_reset(g_s); }

EMSCRIPTEN_KEEPALIVE int oido_feed(const int16_t *pcm, int n) { return g_s ? tasr_nemo_stream_feed(g_s, pcm, n) : -1; }

EMSCRIPTEN_KEEPALIVE const char *oido_partial(void)
{
    g_text[0] = 0;
    if (g_s) tasr_nemo_stream_text(g_s, g_text, sizeof(g_text));
    return g_text;
}

EMSCRIPTEN_KEEPALIVE const char *oido_finish(void)
{
    g_text[0] = 0;
    if (g_s) tasr_nemo_stream_finish(g_s, g_text, sizeof(g_text));
    return g_text;
}
