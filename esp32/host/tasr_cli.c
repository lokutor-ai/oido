// tasr_cli: transcribe 16 kHz mono PCM16 WAV files with the exact ESP32 engine arithmetic.
// usage: tasr_cli model.{tasr,tnm} [--lm lm.tlm] [--beam 4] [--lm_weight w] [--token_bonus b] [--chunk 32] [--left 4]
//                 file1.wav [file2.wav ...]
// .tasr = streaming tinyasr model, .tnm = NeMo Conformer-CTC small (utterance mode)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tinyasr.h"
#include "tinyasr_lm.h"
#include "tasr_nemo.h"

static uint8_t *read_file(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = NULL;
    if (posix_memalign((void **)&buf, 16, sz + 16)) { fclose(f); return NULL; }
    *n = fread(buf, 1, sz, f);
    fclose(f);
    return buf;
}

// minimal RIFF/WAVE reader: PCM16, mono, 16 kHz
static int16_t *read_wav(const char *path, int *ns)
{
    size_t n;
    uint8_t *b = read_file(path, &n);
    if (!b || n < 44 || memcmp(b, "RIFF", 4) || memcmp(b + 8, "WAVE", 4)) { free(b); return NULL; }
    size_t p = 12;
    int ch = 0, rate = 0, bits = 0;
    while (p + 8 <= n) {
        uint32_t len;
        memcpy(&len, b + p + 4, 4);
        if (!memcmp(b + p, "fmt ", 4)) {
            ch = b[p + 10] | (b[p + 11] << 8);
            rate = b[p + 12] | (b[p + 13] << 8) | (b[p + 14] << 16) | (b[p + 15] << 24);
            bits = b[p + 22] | (b[p + 23] << 8);
        } else if (!memcmp(b + p, "data", 4)) {
            if (ch != 1 || rate != 16000 || bits != 16) {
                fprintf(stderr, "%s: need 16 kHz mono PCM16 (got %d ch, %d Hz, %d bit)\n", path, ch, rate, bits);
                free(b);
                return NULL;
            }
            if (p + 8 + len > n) len = (uint32_t)(n - p - 8);
            int16_t *pcm = malloc(len);
            memcpy(pcm, b + p + 8, len);
            *ns = len / 2;
            free(b);
            return pcm;
        }
        p += 8 + len + (len & 1);
    }
    free(b);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.{tasr,tnm} [--lm lm.tlm] [--beam 4] [--lm_weight w] [--token_bonus b] file.wav ...\n", argv[0]);
        return 1;
    }
    const char *lm_path = NULL;
    int chunk = 32, left = 4, first = 2, beam = 4, topk = 6;
    float lw = -1.f, tb = -1.f;
    int nstream = 0;   // --nstream: streaming engine for a NeMo model (chunk/left from --chunk/--left in encoder frames, 0 = header defaults)
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--lm") && i + 1 < argc) { lm_path = argv[++i]; first = i + 1; }
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) { chunk = atoi(argv[++i]); first = i + 1; }
        else if (!strcmp(argv[i], "--left") && i + 1 < argc) { left = atoi(argv[++i]); first = i + 1; }
        else if (!strcmp(argv[i], "--nstream")) { nstream = 1; first = i + 1; chunk = 0; left = 0; }
        else if (!strcmp(argv[i], "--beam") && i + 1 < argc) { beam = atoi(argv[++i]); first = i + 1; }
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) { topk = atoi(argv[++i]); first = i + 1; }
        else if (!strcmp(argv[i], "--lm_weight") && i + 1 < argc) { lw = (float)atof(argv[++i]); first = i + 1; }
        else if (!strcmp(argv[i], "--token_bonus") && i + 1 < argc) { tb = (float)atof(argv[++i]); first = i + 1; }
    }
    size_t msz;
    uint8_t *mb = read_file(argv[1], &msz);
    if (!mb) { fprintf(stderr, "cannot read model %s\n", argv[1]); return 1; }
    const int nemo = msz >= 4 && !memcmp(mb, "TNM1", 4);
    tasr_lm_t *lm = NULL;
    if (lm_path) {
        size_t lsz;
        uint8_t *lb = read_file(lm_path, &lsz);
        lm = lb ? tasr_lm_load(lb, lsz) : NULL;
        if (!lm) { fprintf(stderr, "cannot load LM %s\n", lm_path); return 1; }
    }
    tasr_nemo_t *nm = NULL;
    tasr_stream_t *s = NULL;
    tasr_decoder_t *dec = NULL;
    if (nemo) {  // defaults tuned on dev-clean: beam 4, LM weight 0.3, token bonus 0.5
        nm = tasr_nemo_load(mb, msz);
        if (!nm) { fprintf(stderr, "cannot load NeMo model %s\n", argv[1]); return 1; }
        if (lm) dec = tasr_decoder_create(lm, 1025, beam, topk, lw < 0 ? tasr_lm_weight(lm, 0.3f) : lw, tb < 0 ? tasr_lm_bonus(lm, 0.5f) : tb);
    } else {
        tasr_model_t *m = tasr_model_load(mb, msz);
        if (!m) { fprintf(stderr, "cannot load model %s\n", argv[1]); return 1; }
        s = tasr_stream_create(m, chunk, left);
        if (lm) tasr_stream_set_decoder(s, tasr_decoder_create(lm, 257, beam, 6, lw < 0 ? 0.5f : lw, tb < 0 ? 2.0f : tb));
    }
    static char text[16384];
    tasr_nemo_stream_t *ns_ = (nemo && nstream) ? tasr_nemo_stream_new(nm, chunk, left, dec) : NULL;
    if (nemo && nstream && !ns_) { fprintf(stderr, "model is not a streaming model\n"); return 1; }
    for (int i = first; i < argc; i++) {
        int ns = 0;
        int16_t *pcm = read_wav(argv[i], &ns);
        if (!pcm) { fprintf(stderr, "skip %s\n", argv[i]); continue; }
        clock_t t0 = clock();
        if (ns_) {
            tasr_nemo_stream_reset(ns_);
            for (int j = 0; j < ns; j += 320) tasr_nemo_stream_feed(ns_, pcm + j, ns - j < 320 ? ns - j : 320);
            tasr_nemo_stream_finish(ns_, text, sizeof(text));
        } else if (nemo) {
            tasr_nemo_transcribe(nm, pcm, ns, dec, text, sizeof(text), NULL, 0, NULL);
        } else {
            tasr_stream_reset(s);
            for (int j = 0; j < ns; j += 320) tasr_stream_feed(s, pcm + j, ns - j < 320 ? ns - j : 320);
            tasr_stream_finish(s);
            snprintf(text, sizeof(text), "%s", tasr_stream_text(s));
        }
        double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("%s\t%s\n", argv[i], text);
        fprintf(stderr, "  %.2f s audio, %.3f s CPU (host)\n", ns / 16000.0, sec);
        free(pcm);
    }
    return 0;
}
