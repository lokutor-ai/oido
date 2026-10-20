// tinyasr engine: log-mel frontend -> causal conv subsampling -> U-Net conformer (chunked streaming) -> CTC greedy.
// Mirrors train/model.py (batch-1 forward with chunk_mask(chunk, left_chunks)) with the arithmetic of train/qsim.py:
// int8/int4 weights, dynamic per-row int8 activations, int8 attention (quantize_attention).
#include "tinyasr.h"
#include "kernels.h"
#include "tinyasr_lm.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void *default_alloc(size_t n, int kind)
{
    (void)kind;
    void *p = NULL;
    if (posix_memalign(&p, 16, n ? n : 16)) return NULL;
    memset(p, 0, n);
    return p;
}
static void default_free(void *p) { free(p); }
void *(*tasr_alloc)(size_t, int) = default_alloc;
void (*tasr_free)(void *) = default_free;

static void serial_parallel(tasr_job_fn fn, void *ctx, int n) { fn(ctx, 0, n, 0); }
void (*tasr_parallel)(tasr_job_fn fn, void *ctx, int n) = serial_parallel;
#ifdef CONFIG_TASR_PAR_MASK
uint32_t tasr_par_mask = CONFIG_TASR_PAR_MASK;
#else
uint32_t tasr_par_mask = 0xFFFFFFFFu;
#endif

#ifdef TASR_PROFILE
#ifdef ESP_PLATFORM
#include "esp_cpu.h"
static inline uint32_t tasr_ts(void) { return esp_cpu_get_cycle_count(); }
#else
#include <time.h>
static inline uint32_t tasr_ts(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint32_t)(t.tv_sec * 1000000000ull + t.tv_nsec);
}
#endif
enum { P_MEL, P_FECONV, P_GEMM, P_QUANT, P_ATT, P_LN, P_ACT, P_DWC, P_MISC, P_N };
static const char *tasr_prof_names[P_N] = {"mel", "fe_conv", "gemm", "quant", "attention", "layernorm", "act", "dwconv", "misc"};
static uint64_t tasr_prof[P_N];
#define PB(v) uint32_t v = tasr_ts()
#define PE(v, c) tasr_prof[c] += (uint32_t)(tasr_ts() - v)
const char *tasr_profile_name(int i) { return i < P_N ? tasr_prof_names[i] : 0; }
uint64_t tasr_profile_value(int i) { return i < P_N ? tasr_prof[i] : 0; }
void tasr_profile_reset(void) { memset(tasr_prof, 0, sizeof(tasr_prof)); }
#else
#define PB(v)
#define PE(v, c)
const char *tasr_profile_name(int i) { (void)i; return 0; }
uint64_t tasr_profile_value(int i) { (void)i; return 0; }
void tasr_profile_reset(void) {}
#endif

#define NMEL 80
#define WIN 400
#define HOP 160
#define NFFT 512
#define NBIN 257
#define LN_EPS 1e-5f
#define NW 2       // max parallel workers
#define TMAX 64    // max chunk

typedef struct {
    const float *g, *b;
} ln_t;

typedef struct {
    ln_t ln_ff1, ln_att, ln_conv, conv_norm, ln_ff2, ln_out;
    tasr_qlin_t ff1_l1, ff1_l2, qkv, att_out, pw1, pw2, ff2_l1, ff2_l2;
    const float *dw_w, *dw_b;
    float *dw_wt;  // [k][d] transposed copy
} block_t;

struct tasr_model {
    int V, d, h, dh, dhp, ff, k, n_a, n_b, n_c, fc, bits, head_bits, fe_bits, f1, f2;
    const float *fe_mean, *fe_inv_std;
    int mel_start[NMEL], mel_len[NMEL];
    float *mel_w;
    const float *c1_w, *c1_b, *fdw_w, *fdw_b;
    tasr_qlin_t fe_pw, fe_out, head;
    block_t *blk;
    const float *down;
    const char **tok;
    uint8_t *tok_len;
    float window[WIN];
    float tw_re[NFFT / 2], tw_im[NFFT / 2];   // e^{-2 pi i k / 512}, k < 256 (also used by the 256-pt FFT with stride 2)
    int16_t bitrev[NFFT / 2];
    float *rope_inv;
    int kp_max;
    size_t weight_bytes, blob_bytes;
};

typedef struct {
    int8_t *kq;  // [cap][h][dhp]
    float *ks;   // [cap][h]
    int8_t *vt;  // [h][dh][capp]  (transposed so P.V is an int8 dot over slots)
    float *vs;   // [cap][h]
    int cap, capp;
    long n;      // frames appended so far
    float *conv; // [k-1][d]
} layer_state_t;

typedef struct {  // per-worker scratch
    int8_t *wtmp, *qbuf, *pq;
    float *sc;
    int32_t *acc, *gacc;
} worker_t;

struct tasr_stream {
    const tasr_model_t *m;
    int chunk, left, ldq;
    float pcm[WIN];
    int npcm;
    long n_mel, n_c1, n_c2;
    float mel_hist[3][NMEL];
    float *c1_hist;  // [3][fc][f1]
    float *dwtmp, *pwout, *fe_row;
    int8_t *enc_q;
    float *enc_qs;
    int n_enc, kp_fe;
    float *x, *skip, *xb, *h, *h2, *att, *ext, *xs, *logits, *rcs25, *rcs12;
    int8_t *xq;
    worker_t wk[NW];
    layer_state_t *ls;
    long pos25, pos12;
    int prev_tok, n_new;
    char *text;
    int text_len, text_cap;
    float *sink;
    int sink_max, *sink_n;
    tasr_decoder_t *dec;
    int *dec_toks;
    float fft_re[NFFT], fft_im[NFFT];
};

static void sig_init(void);

// ------------------------------------------------------------------ model loading
typedef struct {
    const uint8_t *p, *end;
    int err;
} cur_t;

static void cur_align(cur_t *c) { c->p = (const uint8_t *)(((uintptr_t)c->p + 15) & ~(uintptr_t)15); }
static const float *take_f32(cur_t *c, size_t n, size_t *acc)
{
    cur_align(c);
    const float *r = (const float *)c->p;
    c->p += n * 4;
    if (c->p > c->end) c->err = 1;
    if (acc) *acc += n * 4;
    return r;
}
static int g_version;
static tasr_qlin_t take_qlin(cur_t *c, int n, int k, int bits, size_t *acc, int *kp_max)
{
    tasr_qlin_t L;
    L.blocked = bits == 4 && g_version >= 2 && n % 16 == 0;
    int blk = (bits == 8 || L.blocked) ? 16 : 32;
    L.n = n; L.k = k; L.bits = bits;
    L.kp = (k + blk - 1) / blk * blk;
    cur_align(c);
    L.w = (const int8_t *)c->p;
    size_t wb = (size_t)n * (bits == 8 ? L.kp : L.kp / 2);
    c->p += wb;
    *acc += wb;
    L.s = take_f32(c, n, acc);
    L.b = take_f32(c, n, acc);
    if (L.kp > *kp_max) *kp_max = L.kp;
    return L;
}
static ln_t take_ln(cur_t *c, int d, size_t *acc)
{
    ln_t l;
    l.g = take_f32(c, d, acc);
    l.b = take_f32(c, d, acc);
    return l;
}

tasr_model_t *tasr_model_load(const uint8_t *blob, size_t size)
{
    if (size < 64 || memcmp(blob, "TASR", 4)) return NULL;
    uint32_t hd[15];
    memcpy(hd, blob + 4, sizeof(hd));
    if (hd[0] != 1 && hd[0] != 2) return NULL;
    g_version = hd[0];
    tasr_model_t *m = (tasr_model_t *)tasr_alloc(sizeof(tasr_model_t), 1);
    m->V = hd[1]; m->d = hd[2]; m->h = hd[3]; m->ff = hd[4]; m->k = hd[5];
    m->n_a = hd[6]; m->n_b = hd[7]; m->n_c = hd[8]; m->fc = hd[9];
    m->bits = hd[11]; m->head_bits = hd[12]; m->fe_bits = hd[13];
    m->dh = m->d / m->h;
    m->dhp = (m->dh + 15) & ~15;
    m->f1 = (NMEL + 2 - 3) / 2 + 1;
    m->f2 = (m->f1 + 2 - 3) / 2 + 1;
    cur_t c = {blob + 64, blob + size, 0};
    size_t acc = 0;
    int kpm = 0;
    const int d = m->d;
    m->fe_mean = take_f32(&c, NMEL, &acc);
    m->fe_inv_std = take_f32(&c, NMEL, &acc);
    const float *fb = take_f32(&c, NBIN * NMEL, NULL);
    int tot = 0;
    for (int j = 0; j < NMEL; j++) {
        int s = -1, e = -1;
        for (int b = 0; b < NBIN; b++)
            if (fb[b * NMEL + j] != 0.f) { if (s < 0) s = b; e = b; }
        m->mel_start[j] = s < 0 ? 0 : s;
        m->mel_len[j] = s < 0 ? 0 : e - s + 1;
        tot += m->mel_len[j];
    }
    m->mel_w = (float *)tasr_alloc(sizeof(float) * (tot + 1), 1);
    for (int j = 0, o = 0; j < NMEL; j++)
        for (int b = 0; b < m->mel_len[j]; b++) m->mel_w[o++] = fb[(m->mel_start[j] + b) * NMEL + j];
    m->c1_w = take_f32(&c, m->fc * 9, &acc);
    m->c1_b = take_f32(&c, m->fc, &acc);
    m->fdw_w = take_f32(&c, m->fc * 9, &acc);
    m->fdw_b = take_f32(&c, m->fc, &acc);
    m->fe_pw = take_qlin(&c, m->fc, m->fc, m->fe_bits, &acc, &kpm);
    int kpm_fe = 0;
    m->fe_out = take_qlin(&c, d, m->fc * m->f2, m->fe_bits, &acc, &kpm_fe);
    int nl = m->n_a + m->n_b + m->n_c;
    m->blk = (block_t *)tasr_alloc(sizeof(block_t) * nl, 1);
    for (int i = 0; i < nl; i++) {
        if (i == m->n_a) m->down = take_f32(&c, 2 * d, &acc);
        block_t *B = &m->blk[i];
        B->ln_ff1 = take_ln(&c, d, &acc);
        B->ff1_l1 = take_qlin(&c, m->ff, d, m->bits, &acc, &kpm);
        B->ff1_l2 = take_qlin(&c, d, m->ff, m->bits, &acc, &kpm);
        B->ln_att = take_ln(&c, d, &acc);
        B->qkv = take_qlin(&c, 3 * d, d, m->bits, &acc, &kpm);
        B->att_out = take_qlin(&c, d, d, m->bits, &acc, &kpm);
        B->ln_conv = take_ln(&c, d, &acc);
        B->pw1 = take_qlin(&c, 2 * d, d, m->bits, &acc, &kpm);
        B->dw_w = take_f32(&c, d * m->k, &acc);
        B->dw_b = take_f32(&c, d, &acc);
        B->conv_norm = take_ln(&c, d, &acc);
        B->pw2 = take_qlin(&c, d, d, m->bits, &acc, &kpm);
        B->ln_ff2 = take_ln(&c, d, &acc);
        B->ff2_l1 = take_qlin(&c, m->ff, d, m->bits, &acc, &kpm);
        B->ff2_l2 = take_qlin(&c, d, m->ff, m->bits, &acc, &kpm);
        B->ln_out = take_ln(&c, d, &acc);
    }
    m->head = take_qlin(&c, m->V + 1, d, m->head_bits, &acc, &kpm);
    cur_align(&c);
    m->tok = (const char **)tasr_alloc(sizeof(char *) * m->V, 1);
    m->tok_len = (uint8_t *)tasr_alloc(m->V, 1);
    for (int i = 0; i < m->V; i++) {
        m->tok_len[i] = *c.p++;
        m->tok[i] = (const char *)c.p;
        c.p += m->tok_len[i];
    }
    if (c.err || c.p > c.end) { tasr_model_free(m); return NULL; }
    m->blob_bytes = (size_t)(c.p - blob);
    m->kp_max = kpm;
    m->weight_bytes = acc;
    for (int n = 0; n < WIN; n++) m->window[n] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * n / WIN));
    for (int i = 0; i < NFFT / 2; i++) {
        m->tw_re[i] = (float)cos(-2.0 * M_PI * i / NFFT);
        m->tw_im[i] = (float)sin(-2.0 * M_PI * i / NFFT);
    }
    for (int i = 0; i < NFFT / 2; i++) {
        int r = 0;
        for (int b = 0; b < 8; b++) r |= ((i >> b) & 1) << (7 - b);
        m->bitrev[i] = (int16_t)r;
    }
    sig_init();
    for (int i = 0; i < nl; i++) {
        float *wt = (float *)tasr_alloc(sizeof(float) * m->k * d, 0);
        for (int ch = 0; ch < d; ch++)
            for (int j = 0; j < m->k; j++) wt[j * d + ch] = m->blk[i].dw_w[ch * m->k + j];
        m->blk[i].dw_wt = wt;
    }
    m->rope_inv = (float *)tasr_alloc(sizeof(float) * (m->dh / 2), 1);
    for (int i = 0; i < m->dh / 2; i++) m->rope_inv[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)m->dh);
    return m;
}

void tasr_model_free(tasr_model_t *m)
{
    if (!m) return;
    if (m->blk)
        for (int i = 0; i < m->n_a + m->n_b + m->n_c; i++) tasr_free(m->blk[i].dw_wt);
    tasr_free(m->mel_w); tasr_free(m->blk); tasr_free((void *)m->tok); tasr_free(m->tok_len); tasr_free(m->rope_inv);
    tasr_free(m);
}

size_t tasr_model_weight_bytes(const tasr_model_t *m) { return m->weight_bytes; }

// Copy weight matrices (in execution order) into tasr_alloc(kind 0) memory until `budget` bytes are used.
// On the ESP32 this moves weights from memory-mapped flash (~20-30 MB/s) into PSRAM (~50-80 MB/s).
static size_t move_qlin(tasr_qlin_t *L, size_t *left)
{
    size_t wb = (size_t)L->n * (L->bits == 8 ? L->kp : L->kp / 2);
    if (wb > *left) return 0;
    int8_t *p = (int8_t *)tasr_alloc(wb, 0);
    if (!p) return 0;
    memcpy(p, L->w, wb);
    L->w = p;
    *left -= wb;
    return wb;
}
size_t tasr_model_place_weights(tasr_model_t *m, size_t budget)
{
    size_t left = budget, moved = 0;
    moved += move_qlin(&m->fe_pw, &left);
    moved += move_qlin(&m->fe_out, &left);
    for (int i = 0; i < m->n_a + m->n_b + m->n_c; i++) {
        block_t *B = &m->blk[i];
        tasr_qlin_t *Ls[8] = {&B->ff1_l1, &B->ff1_l2, &B->qkv, &B->att_out, &B->pw1, &B->pw2, &B->ff2_l1, &B->ff2_l2};
        for (int j = 0; j < 8; j++) moved += move_qlin(Ls[j], &left);
    }
    moved += move_qlin(&m->head, &left);
    return moved;
}
size_t tasr_model_blob_bytes(const tasr_model_t *m) { return m->blob_bytes; }

// ------------------------------------------------------------------ small float ops
// sigmoid via 1/16-step linear-interpolated table on [-12, 12] (max abs error ~5e-5)
#define SIG_N 384
static float sig_tab[SIG_N + 1], sig_slope[SIG_N];
static int sig_init_done;
static void sig_init(void)
{
    if (sig_init_done) return;
    for (int i = 0; i <= SIG_N; i++) sig_tab[i] = (float)(1.0 / (1.0 + exp(-(-12.0 + i / 16.0))));
    for (int i = 0; i < SIG_N; i++) sig_slope[i] = sig_tab[i + 1] - sig_tab[i];
    sig_init_done = 1;
}
static inline float sigm(float x)
{
    float u = (x + 12.0f) * 16.0f;
    if (u <= 0.f) return 0.f;
    if (u >= (float)SIG_N) return 1.f;
    int i = (int)u;
    return sig_tab[i] + (u - (float)i) * sig_slope[i];
}
// exp(x) for x <= 0: 2^(x*log2e) = 2^i * poly(f); rel. error ~2e-6
static inline float fexp_neg(float x)
{
    if (x < -80.f) return 0.f;
    float t = x * 1.44269504f;
    int i = (int)t;
    if ((float)i > t) i--;
    float f = t - (float)i;
    float p = 1.5403530e-4f;
    p = p * f + 1.3333558e-3f;
    p = p * f + 9.6181291e-3f;
    p = p * f + 5.5504109e-2f;
    p = p * f + 2.4022651e-1f;
    p = p * f + 6.9314718e-1f;
    p = p * f + 1.0f;
    union { float f; int32_t i; } u;
    u.i = (i + 127) << 23;
    return p * u.f;
}

// ------------------------------------------------------------------ parallel jobs
typedef struct {
    tasr_stream_t *s;
    const tasr_qlin_t *L;
    const float *x;
    int T, ldx, ldy;
    float *y;
    layer_state_t *Ls;
    const float *qkv;
    const int8_t *xq;
    const float *xs;
    int ldq;
} job_t;

static void job_quant(void *c, int b, int e, int w)
{
    (void)w;
    job_t *j = (job_t *)c;
    tasr_quant_rows(j->x + (size_t)b * j->ldx, e - b, j->L->k, j->ldx, j->s->xq + (size_t)b * j->s->ldq, j->s->ldq,
                    j->L->kp, j->s->xs + b);
}
static void job_gemm(void *c, int b, int e, int w)
{
    job_t *j = (job_t *)c;
    const int u = j->L->blocked ? 16 : 1;  // work units: blocks of 16 outputs for blocked layers
    tasr_qlin_range(j->L, j->xq, j->xs, j->T, j->ldq, j->y, j->ldy, j->s->wk[w].wtmp, j->s->wk[w].gacc, b * u, e * u);
}
static inline int gemm_units(const tasr_qlin_t *L) { return L->blocked ? L->n / 16 : L->n; }
static void qlinf(tasr_stream_t *s, const tasr_qlin_t *L, const float *x, int T, int ldx, float *y, int ldy)
{
    job_t j = {.s = s, .L = L, .x = x, .T = T, .ldx = ldx, .y = y, .ldy = ldy, .xq = s->xq, .xs = s->xs, .ldq = s->ldq};
    PB(t0);
    tasr_parallel(job_quant, &j, T);
    PE(t0, P_QUANT);
    PB(t1);
    tasr_parallel(job_gemm, &j, gemm_units(L));
    PE(t1, P_GEMM);
}

typedef struct {
    const float *x;
    float *y;
    int d, ldx, ldy;
    ln_t p;
} ln_job_t;
static void job_ln(void *c, int b, int e, int w)
{
    (void)w;
    ln_job_t *j = (ln_job_t *)c;
    const int d = j->d;
    for (int t = b; t < e; t++) {
        const float *r = j->x + (size_t)t * j->ldx;
        float *o = j->y + (size_t)t * j->ldy;
        float mean = 0.f;
        for (int i = 0; i < d; i++) mean += r[i];
        mean /= (float)d;
        float var = 0.f;
        for (int i = 0; i < d; i++) { float v = r[i] - mean; var += v * v; }
        var /= (float)d;
        float inv = 1.0f / sqrtf(var + LN_EPS);
        for (int i = 0; i < d; i++) o[i] = (r[i] - mean) * inv * j->p.g[i] + j->p.b[i];
    }
}
static void layernorm(const float *x, int T, int d, int ldx, ln_t p, float *y, int ldy)
{
    ln_job_t j = {x, y, d, ldx, ldy, p};
    PB(t0);
    tasr_parallel(job_ln, &j, T);
    PE(t0, P_LN);
}
static void job_silu(void *c, int b, int e, int w)
{
    (void)w;
    float *x = (float *)c;
    for (int i = b; i < e; i++) x[i] = x[i] * sigm(x[i]);
}
static void silu_inplace(float *x, int n)
{
    PB(t0);
    tasr_parallel(job_silu, x, n);
    PE(t0, P_ACT);
}

// ------------------------------------------------------------------ frontend
// power spectrum |X[k]|^2, k = 0..256, of a real 512-point frame via one 256-point complex FFT
static void power_spectrum(const tasr_model_t *m, const float *x, float *re, float *im, float *pw)
{
    const int N2 = NFFT / 2;
    for (int n = 0; n < N2; n++) {
        int j = m->bitrev[n];
        re[j] = x[2 * n];
        im[j] = x[2 * n + 1];
    }
    for (int len = 2; len <= N2; len <<= 1) {
        const int half = len >> 1, step = 2 * (N2 / len);  // twiddles of the 512 table at even indices
        for (int i = 0; i < N2; i += len) {
            for (int j = 0; j < half; j++) {
                const float wr = m->tw_re[j * step], wi = m->tw_im[j * step];
                float *ar = &re[i + j], *ai = &im[i + j], *br = &re[i + j + half], *bi = &im[i + j + half];
                const float tr = *br * wr - *bi * wi, ti = *br * wi + *bi * wr;
                *br = *ar - tr; *bi = *ai - ti;
                *ar += tr; *ai += ti;
            }
        }
    }
    // X[k] = (Z[k] + conj Z[N2-k])/2 - i/2 * W^k (Z[k] - conj Z[N2-k])
    pw[0] = (re[0] + im[0]) * (re[0] + im[0]);
    pw[N2] = (re[0] - im[0]) * (re[0] - im[0]);
    for (int k = 1; k < N2; k++) {
        const float zr = re[k], zi = im[k], cr = re[N2 - k], ci = -im[N2 - k];
        const float er = 0.5f * (zr + cr), ei = 0.5f * (zi + ci);
        const float dr = 0.5f * (zr - cr), di = 0.5f * (zi - ci);
        const float wr = m->tw_re[k], wi = m->tw_im[k];
        // -i * W * D  = -i*(wr*dr - wi*di + i(wr*di + wi*dr)) = (wr*di + wi*dr) - i(wr*dr - wi*di)
        const float or_ = er + (wr * di + wi * dr), oi = ei - (wr * dr - wi * di);
        pw[k] = or_ * or_ + oi * oi;
    }
}

static void run_chunk(tasr_stream_t *s, int T);

typedef struct {
    tasr_stream_t *s;
    long idx;
    const float (*P)[NMEL + 2];
    float *out;
} fe_job_t;

static void job_fdw(void *c, int b, int e, int w)
{
    (void)w;
    fe_job_t *j = (fe_job_t *)c;
    tasr_stream_t *s = j->s;
    const tasr_model_t *m = s->m;
    const int fc = m->fc, f1 = m->f1, f2 = m->f2;
    for (int ch = b; ch < e; ch++) {
        const float *wt = m->fdw_w + ch * 9;
        const float bias = m->fdw_b[ch];
        const float *r[3];
        for (int dt = 0; dt < 3; dt++) {
            long ti = 2 * j->idx - 2 + dt;
            r[dt] = ti < 0 ? NULL : s->c1_hist + (size_t)(ti % 3) * fc * f1 + (size_t)ch * f1;
        }
        for (int f = 0; f < f2; f++) {
            float acc = bias;
            for (int dt = 0; dt < 3; dt++) {
                const float *y = r[dt];
                if (!y) continue;
                if (f > 0) acc += wt[dt * 3] * y[2 * f - 1];
                acc += wt[dt * 3 + 1] * y[2 * f] + wt[dt * 3 + 2] * y[2 * f + 1];
            }
            s->dwtmp[f * fc + ch] = acc;
        }
    }
}

static void frontend_frame_out(tasr_stream_t *s)
{
    // conv2 (depthwise 3x3 stride 2) output j from conv1 outputs 2j-2..2j, then pw + relu, flatten, quantize
    const tasr_model_t *m = s->m;
    const int fc = m->fc, f2 = m->f2;
    fe_job_t j = {.s = s, .idx = s->n_c2};
    PB(t0);
    tasr_parallel(job_fdw, &j, fc);
    PE(t0, P_FECONV);
    qlinf(s, &m->fe_pw, s->dwtmp, f2, fc, s->pwout, fc);
    for (int f = 0; f < f2; f++)
        for (int ch = 0; ch < fc; ch++) {
            float v = s->pwout[f * fc + ch];
            s->fe_row[ch * f2 + f] = v > 0.f ? v : 0.f;
        }
    tasr_quant_rows(s->fe_row, 1, fc * f2, fc * f2, s->enc_q + (size_t)s->n_enc * s->kp_fe, s->kp_fe, s->kp_fe,
                    s->enc_qs + s->n_enc);
    s->n_enc++;
    s->n_c2++;
    if (s->n_enc == s->chunk) run_chunk(s, s->chunk);
}

static void job_conv1(void *c, int b, int e, int w)
{
    (void)w;
    fe_job_t *j = (fe_job_t *)c;
    const tasr_model_t *m = j->s->m;
    const int f1 = m->f1;
    for (int ch = b; ch < e; ch++) {
        const float *wt = m->c1_w + ch * 9;
        const float w0 = wt[0], w1 = wt[1], w2 = wt[2], w3 = wt[3], w4 = wt[4], w5 = wt[5], w6 = wt[6], w7 = wt[7], w8 = wt[8];
        const float bias = m->c1_b[ch];
        float *o = j->out + ch * f1;
        for (int f = 0; f < f1; f++) {
            const float *p0 = &j->P[0][2 * f], *p1 = &j->P[1][2 * f], *p2 = &j->P[2][2 * f];
            float acc = bias + w0 * p0[0] + w1 * p0[1] + w2 * p0[2] + w3 * p1[0] + w4 * p1[1] + w5 * p1[2] + w6 * p2[0] +
                        w7 * p2[1] + w8 * p2[2];
            o[f] = acc > 0.f ? acc : 0.f;
        }
    }
}

static void frontend_conv1(tasr_stream_t *s)
{
    const tasr_model_t *m = s->m;
    long k = s->n_c1;
    PB(t0);
    // padded input rows: P[dt][1 + f] = mel(2k-2+dt, f), P[dt][0] = 0 (left freq pad)
    float P[3][NMEL + 2];
    for (int dt = 0; dt < 3; dt++) {
        long ti = 2 * k - 2 + dt;
        P[dt][0] = 0.f;
        P[dt][NMEL + 1] = 0.f;
        if (ti < 0) memset(&P[dt][1], 0, sizeof(float) * NMEL);
        else memcpy(&P[dt][1], s->mel_hist[ti % 3], sizeof(float) * NMEL);
    }
    fe_job_t j = {.s = s, .idx = k, .P = (const float(*)[NMEL + 2])P, .out = s->c1_hist + (size_t)(k % 3) * m->fc * m->f1};
    tasr_parallel(job_conv1, &j, m->fc);
    PE(t0, P_FECONV);
    s->n_c1++;
    if ((k & 1) == 0) frontend_frame_out(s);
}

static void mel_frame(tasr_stream_t *s)
{
    const tasr_model_t *m = s->m;
    PB(t0);
    float xw[NFFT], pw[NBIN];
    for (int i = 0; i < WIN; i++) xw[i] = s->pcm[i] * m->window[i];
    for (int i = WIN; i < NFFT; i++) xw[i] = 0.f;
#ifdef TASR_REF_DFT
    for (int k = 0; k < NBIN; k++) {  // reference O(N^2) DFT in double (testing only)
        double ar = 0, ai = 0;
        for (int n = 0; n < NFFT; n++) { ar += xw[n] * cos(2 * M_PI * k * n / NFFT); ai -= xw[n] * sin(2 * M_PI * k * n / NFFT); }
        pw[k] = (float)(ar * ar + ai * ai);
    }
#else
    power_spectrum(m, xw, s->fft_re, s->fft_im, pw);
#endif
    float *out = s->mel_hist[s->n_mel % 3];
    const float *w = m->mel_w;
    for (int j = 0; j < NMEL; j++) {
        float acc = 0.f;
        const float *p = pw + m->mel_start[j];
        for (int b = 0; b < m->mel_len[j]; b++) acc += p[b] * w[b];
        w += m->mel_len[j];
        out[j] = (logf(acc + 1e-6f) - m->fe_mean[j]) * m->fe_inv_std[j];
    }
    PE(t0, P_MEL);
    long t = s->n_mel++;
    if ((t & 1) == 0) frontend_conv1(s);
}

// ------------------------------------------------------------------ encoder
static void rope_table(const tasr_model_t *m, long pos0, int T, float *cs)
{
    // cs[t][i] = (cos, sin) of (pos0+t) * inv_freq[i]; angle reduced in double for long streams
    const int nh = m->dh / 2;
    for (int t = 0; t < T; t++)
        for (int i = 0; i < nh; i++) {
            double a = fmod((double)(pos0 + t) * (double)m->rope_inv[i], 2.0 * M_PI);
            cs[(t * nh + i) * 2] = (float)cos(a);
            cs[(t * nh + i) * 2 + 1] = (float)sin(a);
        }
}

static void rope(const tasr_model_t *m, float *v, const float *cs)
{
    for (int i = 0; i < m->dh / 2; i++) {
        const float c = cs[2 * i], sn = cs[2 * i + 1];
        for (int hh = 0; hh < m->h; hh++) {
            float *p = v + hh * m->dh + 2 * i;
            float x1 = p[0], x2 = p[1];
            p[0] = x1 * c - x2 * sn;
            p[1] = x1 * sn + x2 * c;
        }
    }
}

// quantize n floats to int8 (symmetric, per vector); returns scale
static inline float quant_vec(const float *x, int n, int8_t *q)
{
    float mx = 0.f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); mx = a > mx ? a : mx; }
    if (mx < 1e-30f) mx = 1e-30f;
    const float inv = 127.0f / mx;
    for (int i = 0; i < n; i++) {
        float y = x[i] * inv + 12582912.0f;
        q[i] = (int8_t)(int)(y - 12582912.0f);
    }
    return mx / 127.0f;
}

static void attn_append(const tasr_model_t *m, layer_state_t *L, const float *qkv, int T)
{
    const int d = m->d, dh = m->dh, dhp = m->dhp, H = m->h;
    int8_t tmp[64];
    for (int t = 0; t < T; t++) {
        const int slot = (int)(L->n % L->cap);
        const float *k = qkv + (size_t)t * 3 * d + d, *v = qkv + (size_t)t * 3 * d + 2 * d;
        for (int hh = 0; hh < H; hh++) {
            int8_t *kq = L->kq + ((size_t)slot * H + hh) * dhp;
            L->ks[slot * H + hh] = quant_vec(k + hh * dh, dh, kq);
            for (int e = dh; e < dhp; e++) kq[e] = 0;
            L->vs[slot * H + hh] = quant_vec(v + hh * dh, dh, tmp);
            int8_t *vt = L->vt + (size_t)hh * dh * L->capp + slot;
            for (int e = 0; e < dh; e++) vt[(size_t)e * L->capp] = tmp[e];
        }
        L->n++;
    }
}

static void job_attn(void *c, int b, int e, int w)
{
    // work units = (head, query) pairs
    job_t *j = (job_t *)c;
    tasr_stream_t *s = j->s;
    const tasr_model_t *m = s->m;
    layer_state_t *L = j->Ls;
    worker_t *W = &s->wk[w];
    const int d = m->d, dh = m->dh, dhp = m->dhp, H = m->h, T = j->T;
    const int nv = L->n < L->cap ? (int)L->n : L->cap;
    const float scale = 1.0f / sqrtf((float)dh);
    for (int u = b; u < e; u++) {
        const int hh = u / T, t = u % T;
        const float sq = quant_vec(j->qkv + (size_t)t * 3 * d + hh * dh, dh, W->qbuf);
        for (int i = dh; i < dhp; i++) W->qbuf[i] = 0;
        tasr_dot_rows_s8(W->qbuf, L->kq + (size_t)hh * dhp, H * dhp, nv, dhp, W->acc);
        float mx = -1e30f;
        const float *ks = L->ks + hh;
        for (int i = 0; i < nv; i++) {
            float a = (float)W->acc[i] * sq * ks[i * H] * scale;
            W->sc[i] = a;
            mx = a > mx ? a : mx;
        }
        float sum = 0.f, pm = 0.f;
        const float *vs = L->vs + hh;
        for (int i = 0; i < nv; i++) {
            float ex = fexp_neg(W->sc[i] - mx);
            sum += ex;
            float p2 = ex * vs[i * H];
            W->sc[i] = p2;
            pm = p2 > pm ? p2 : pm;
        }
        if (pm < 1e-30f) pm = 1e-30f;
        const float inv = 127.0f / pm;
        for (int i = 0; i < nv; i++) {
            float y = W->sc[i] * inv + 12582912.0f;
            W->pq[i] = (int8_t)(int)(y - 12582912.0f);
        }
        for (int i = nv; i < L->capp; i++) W->pq[i] = 0;
        tasr_dot_rows_s8(W->pq, L->vt + (size_t)hh * dh * L->capp, L->capp, dh, L->capp, W->acc);
        const float os = pm / 127.0f / sum;
        float *y = s->att + (size_t)t * d + hh * dh;
        for (int i = 0; i < dh; i++) y[i] = (float)W->acc[i] * os;
    }
}

typedef struct {
    tasr_stream_t *s;
    const block_t *B;
    float *h2;
} conv_job_t;

static void job_glu(void *c, int b, int e, int w)
{
    (void)w;
    conv_job_t *j = (conv_job_t *)c;
    const tasr_model_t *m = j->s->m;
    const int d = m->d, K = m->k;
    float *ext = j->s->ext;
    for (int t = b; t < e; t++) {
        const float *g = j->h2 + (size_t)t * 2 * d;
        float *a = ext + (size_t)(K - 1 + t) * d;
        for (int ch = 0; ch < d; ch++) a[ch] = g[ch] * sigm(g[d + ch]);
    }
}
static void job_dw(void *c, int b, int e, int w)
{
    (void)w;
    conv_job_t *j = (conv_job_t *)c;
    const tasr_model_t *m = j->s->m;
    const int d = m->d, K = m->k;
    const float *ext = j->s->ext;
    for (int t = b; t < e; t++) {
        float *o = j->s->h + (size_t)t * d;
        memcpy(o, j->B->dw_b, sizeof(float) * d);
        for (int jj = 0; jj < K; jj++) {
            const float *wt = j->B->dw_wt + jj * d, *ex = ext + (size_t)(t + jj) * d;
            for (int ch = 0; ch < d; ch++) o[ch] += wt[ch] * ex[ch];
        }
    }
}

static void block_forward(tasr_stream_t *s, const block_t *B, layer_state_t *L, float *x, int T, const float *rcs)
{
    const tasr_model_t *m = s->m;
    const int d = m->d, ff = m->ff, K = m->k;
    float *h = s->h, *h2 = s->h2;
    // FF1 (half step)
    layernorm(x, T, d, d, B->ln_ff1, h, d);
    qlinf(s, &B->ff1_l1, h, T, d, h2, ff);
    silu_inplace(h2, T * ff);
    qlinf(s, &B->ff1_l2, h2, T, ff, h, d);
    for (int i = 0; i < T * d; i++) x[i] += 0.5f * h[i];
    // MHSA (int8; keys/values of the current chunk are appended to the ring first)
    layernorm(x, T, d, d, B->ln_att, h, d);
    qlinf(s, &B->qkv, h, T, d, h2, 3 * d);
    PB(ta);
    for (int t = 0; t < T; t++) {
        rope(m, h2 + (size_t)t * 3 * d, rcs + t * m->dh);
        rope(m, h2 + (size_t)t * 3 * d + d, rcs + t * m->dh);
    }
    attn_append(m, L, h2, T);
    job_t aj = {.s = s, .T = T, .Ls = L, .qkv = h2};
    tasr_parallel(job_attn, &aj, m->h * T);
    PE(ta, P_ATT);
    qlinf(s, &B->att_out, s->att, T, d, h, d);
    for (int i = 0; i < T * d; i++) x[i] += h[i];
    // conv module
    layernorm(x, T, d, d, B->ln_conv, h, d);
    qlinf(s, &B->pw1, h, T, d, h2, 2 * d);
    PB(tc);
    memcpy(s->ext, L->conv, sizeof(float) * (K - 1) * d);
    conv_job_t cj = {s, B, h2};
    tasr_parallel(job_glu, &cj, T);
    tasr_parallel(job_dw, &cj, T);
    memcpy(L->conv, s->ext + (size_t)T * d, sizeof(float) * (K - 1) * d);
    PE(tc, P_DWC);
    layernorm(h, T, d, d, B->conv_norm, h, d);
    silu_inplace(h, T * d);
    qlinf(s, &B->pw2, h, T, d, s->att, d);
    for (int i = 0; i < T * d; i++) x[i] += s->att[i];
    // FF2 (half step)
    layernorm(x, T, d, d, B->ln_ff2, h, d);
    qlinf(s, &B->ff2_l1, h, T, d, h2, ff);
    silu_inplace(h2, T * ff);
    qlinf(s, &B->ff2_l2, h2, T, ff, h, d);
    for (int i = 0; i < T * d; i++) x[i] += 0.5f * h[i];
    layernorm(x, T, d, d, B->ln_out, x, d);
}

static void emit_token(tasr_stream_t *s, int id)
{
    const tasr_model_t *m = s->m;
    int n = m->tok_len[id];
    if (s->text_len + n + 1 > s->text_cap) {
        int cap = (s->text_cap + n + 1) * 2;
        char *t = (char *)tasr_alloc(cap, 0);
        memcpy(t, s->text, s->text_len + 1);
        tasr_free(s->text);
        s->text = t;
        s->text_cap = cap;
    }
    memcpy(s->text + s->text_len, m->tok[id], n);
    s->text_len += n;
    s->text[s->text_len] = 0;
}

static void run_chunk(tasr_stream_t *s, int T)
{
    const tasr_model_t *m = s->m;
    const int d = m->d;
    float *x = s->x;
    {   // frontend output projection on the already-quantized rows
        job_t j = {.s = s, .L = &m->fe_out, .T = T, .y = x, .ldy = d, .xq = s->enc_q, .xs = s->enc_qs, .ldq = s->kp_fe};
        PB(tg);
        tasr_parallel(job_gemm, &j, gemm_units(&m->fe_out));
        PE(tg, P_GEMM);
    }
    s->n_enc = 0;
    int li = 0;
    const int T2 = (T + 1) / 2;
    PB(tr);
    rope_table(m, s->pos25, T, s->rcs25);
    rope_table(m, s->pos12, T2, s->rcs12);
    PE(tr, P_MISC);
    for (int i = 0; i < m->n_a; i++, li++) block_forward(s, &m->blk[li], &s->ls[li], x, T, s->rcs25);
    memcpy(s->skip, x, sizeof(float) * T * d);
    for (int i = 0; i < T2; i++)
        for (int ch = 0; ch < d; ch++) {
            float a = x[(size_t)(2 * i) * d + ch] * m->down[ch];
            float b = (2 * i + 1 < T) ? x[(size_t)(2 * i + 1) * d + ch] * m->down[d + ch] : 0.f;
            s->xb[(size_t)i * d + ch] = a + b;
        }
    for (int i = 0; i < m->n_b; i++, li++) block_forward(s, &m->blk[li], &s->ls[li], s->xb, T2, s->rcs12);
    for (int t = 0; t < T; t++)
        for (int ch = 0; ch < d; ch++) x[(size_t)t * d + ch] = s->xb[(size_t)(t / 2) * d + ch] + s->skip[(size_t)t * d + ch];
    for (int i = 0; i < m->n_c; i++, li++) block_forward(s, &m->blk[li], &s->ls[li], x, T, s->rcs25);
    s->pos25 += T;
    s->pos12 += T2;
    const int V1 = m->V + 1;
    qlinf(s, &m->head, x, T, d, s->logits, V1);
    for (int t = 0; t < T; t++) {
        const float *lg = s->logits + (size_t)t * V1;
        int best = 0;
        float bv = lg[0];
        for (int v = 1; v < V1; v++)
            if (lg[v] > bv) { bv = lg[v]; best = v; }
        if (s->sink && *s->sink_n < s->sink_max) {
            memcpy(s->sink + (size_t)(*s->sink_n) * V1, lg, sizeof(float) * V1);
            (*s->sink_n)++;
        }
        if (s->dec) {
            // log-softmax for the beam decoder
            float lz = 0.f;
            for (int v = 0; v < V1; v++) lz += expf(lg[v] - bv);
            lz = bv + logf(lz);
            float *lp = s->logits + (size_t)t * V1;  // in place (sink already copied)
            for (int v = 0; v < V1; v++) lp[v] -= lz;
            tasr_decoder_step(s->dec, lp);
            if (best != 0 && best != s->prev_tok) s->n_new++;
        } else if (best != s->prev_tok && best != 0) {
            emit_token(s, best - 1);
            s->n_new++;
        }
        s->prev_tok = best;
    }
}

void tasr_stream_set_decoder(tasr_stream_t *s, tasr_decoder_t *d)
{
    s->dec = d;
    if (!s->dec_toks) s->dec_toks = (int *)tasr_alloc(sizeof(int) * 4096, 0);
    if (d) tasr_decoder_reset(d);
}

// ------------------------------------------------------------------ stream API
tasr_stream_t *tasr_stream_create(const tasr_model_t *m, int chunk, int left_chunks)
{
    if (chunk < 2 || (chunk & 1) || chunk > TMAX || left_chunks < 1) return NULL;
    tasr_stream_t *s = (tasr_stream_t *)tasr_alloc(sizeof(tasr_stream_t), 0);
    s->m = m;
    s->chunk = chunk;
    s->left = left_chunks;
    const int d = m->d, fc = m->fc;
    s->ldq = m->kp_max;
    s->kp_fe = m->fe_out.kp;
    int wmax = 3 * d > m->ff ? 3 * d : m->ff;
    int V1 = m->V + 1;
    if (V1 > wmax) wmax = V1;
    s->c1_hist = (float *)tasr_alloc(sizeof(float) * 3 * fc * m->f1, 1);
    s->dwtmp = (float *)tasr_alloc(sizeof(float) * m->f2 * fc, 1);
    s->pwout = (float *)tasr_alloc(sizeof(float) * m->f2 * fc, 1);
    s->fe_row = (float *)tasr_alloc(sizeof(float) * m->f2 * fc, 1);
    s->enc_q = (int8_t *)tasr_alloc((size_t)chunk * s->kp_fe, 1);
    s->enc_qs = (float *)tasr_alloc(sizeof(float) * chunk, 1);
    s->x = (float *)tasr_alloc(sizeof(float) * chunk * d, 1);
    s->skip = (float *)tasr_alloc(sizeof(float) * chunk * d, 1);
    s->xb = (float *)tasr_alloc(sizeof(float) * chunk * d, 1);
    s->h = (float *)tasr_alloc(sizeof(float) * chunk * d, 1);
    s->h2 = (float *)tasr_alloc(sizeof(float) * chunk * wmax, 1);
    s->att = (float *)tasr_alloc(sizeof(float) * chunk * d, 1);
    s->logits = (float *)tasr_alloc(sizeof(float) * chunk * V1, 0);
    s->ext = (float *)tasr_alloc(sizeof(float) * (m->k - 1 + chunk) * d, 1);
    s->rcs25 = (float *)tasr_alloc(sizeof(float) * chunk * m->dh, 1);
    s->rcs12 = (float *)tasr_alloc(sizeof(float) * chunk * m->dh, 1);
    int rows = chunk > m->f2 ? chunk : m->f2;
    s->xq = (int8_t *)tasr_alloc((size_t)rows * s->ldq, 1);
    s->xs = (float *)tasr_alloc(sizeof(float) * rows, 1);
    const int capmax = (left_chunks + 1) * chunk;
    const int cappmax = (capmax + 15) & ~15;
    const int accn = cappmax > TMAX ? cappmax : TMAX;
    int tile = 16 * s->ldq + 16;
    if (s->kp_fe > tile) tile = s->kp_fe;
    for (int w = 0; w < NW; w++) {
        s->wk[w].wtmp = (int8_t *)tasr_alloc(tile, 1);
        s->wk[w].gacc = (int32_t *)tasr_alloc(sizeof(int32_t) * TMAX * 16, 1);
        s->wk[w].qbuf = (int8_t *)tasr_alloc(m->dhp, 1);
        s->wk[w].pq = (int8_t *)tasr_alloc(cappmax, 1);
        s->wk[w].sc = (float *)tasr_alloc(sizeof(float) * cappmax, 1);
        s->wk[w].acc = (int32_t *)tasr_alloc(sizeof(int32_t) * accn, 1);
    }
    int nl = m->n_a + m->n_b + m->n_c;
    s->ls = (layer_state_t *)tasr_alloc(sizeof(layer_state_t) * nl, 0);
    for (int i = 0; i < nl; i++) {
        int stage_chunk = (i >= m->n_a && i < m->n_a + m->n_b) ? chunk / 2 : chunk;
        layer_state_t *L = &s->ls[i];
        L->cap = (left_chunks + 1) * stage_chunk;
        L->capp = (L->cap + 15) & ~15;
        L->kq = (int8_t *)tasr_alloc((size_t)L->cap * m->h * m->dhp, 0);
        L->ks = (float *)tasr_alloc(sizeof(float) * L->cap * m->h, 0);
        L->vt = (int8_t *)tasr_alloc((size_t)m->h * m->dh * L->capp, 0);
        L->vs = (float *)tasr_alloc(sizeof(float) * L->cap * m->h, 0);
        L->conv = (float *)tasr_alloc(sizeof(float) * (m->k - 1) * d, 0);
    }
    s->text_cap = 256;
    s->text = (char *)tasr_alloc(s->text_cap, 0);
    tasr_stream_reset(s);
    return s;
}

void tasr_stream_reset(tasr_stream_t *s)
{
    const tasr_model_t *m = s->m;
    s->npcm = 0;
    s->n_mel = s->n_c1 = s->n_c2 = 0;
    s->n_enc = 0;
    s->pos25 = s->pos12 = 0;
    s->prev_tok = 0;
    s->text_len = 0;
    s->text[0] = 0;
    if (s->dec) tasr_decoder_reset(s->dec);
    int nl = m->n_a + m->n_b + m->n_c;
    for (int i = 0; i < nl; i++) {
        layer_state_t *L = &s->ls[i];
        L->n = 0;
        memset(L->conv, 0, sizeof(float) * (m->k - 1) * m->d);
        memset(L->vt, 0, (size_t)m->h * m->dh * L->capp);
    }
}

void tasr_stream_free(tasr_stream_t *s)
{
    if (!s) return;
    const tasr_model_t *m = s->m;
    int nl = m->n_a + m->n_b + m->n_c;
    for (int i = 0; i < nl; i++) {
        layer_state_t *L = &s->ls[i];
        tasr_free(L->kq); tasr_free(L->ks); tasr_free(L->vt); tasr_free(L->vs); tasr_free(L->conv);
    }
    tasr_free(s->ls);
    tasr_free(s->dec_toks);
    for (int w = 0; w < NW; w++) {
        tasr_free(s->wk[w].wtmp); tasr_free(s->wk[w].qbuf); tasr_free(s->wk[w].pq); tasr_free(s->wk[w].sc);
        tasr_free(s->wk[w].acc); tasr_free(s->wk[w].gacc);
    }
    void *bufs[] = {s->c1_hist, s->dwtmp, s->pwout, s->fe_row, s->enc_q, s->enc_qs, s->x, s->skip, s->xb, s->h,
                    s->h2, s->att, s->logits, s->ext, s->xq, s->xs, s->text, s->rcs25, s->rcs12};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) tasr_free(bufs[i]);
    tasr_free(s);
}

int tasr_stream_feed(tasr_stream_t *s, const int16_t *pcm, int n)
{
    s->n_new = 0;
    for (int i = 0; i < n; i++) {
        s->pcm[s->npcm++] = (float)pcm[i] / 32768.0f;
        if (s->npcm == WIN) {
            mel_frame(s);
            memmove(s->pcm, s->pcm + HOP, sizeof(float) * (WIN - HOP));
            s->npcm = WIN - HOP;
        }
    }
    return s->n_new;
}

int tasr_stream_finish(tasr_stream_t *s)
{
    s->n_new = 0;
    if (s->n_enc > 0) run_chunk(s, s->n_enc);
    return s->n_new;
}

const char *tasr_stream_text(tasr_stream_t *s)
{
    if (s->dec) {
        int n = tasr_decoder_best(s->dec, s->dec_toks, 4096);
        s->text_len = 0;
        s->text[0] = 0;
        for (int i = 0; i < n; i++) emit_token(s, s->dec_toks[i]);
    }
    return s->text[0] == ' ' ? s->text + 1 : s->text;
}

void tasr_stream_set_logit_sink(tasr_stream_t *s, float *buf, int max_frames, int *n_frames)
{
    s->sink = buf;
    s->sink_max = max_frames;
    s->sink_n = n_frames;
    *n_frames = 0;
}
