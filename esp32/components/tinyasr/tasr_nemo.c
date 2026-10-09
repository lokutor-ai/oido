// tasr_nemo: utterance-level engine for NVIDIA NeMo Conformer-CTC small (rel-pos MHSA, full context), int8.
// Mirrors train/nemo_small.py + train/nemo_eval.py (--bits 8 --att8) arithmetic.
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.h"
#include "tasr_nemo.h"
#include "tinyasr.h"
#include "tinyasr_lm.h"

#define NMEL 80
#define NBIN 257
#define WIN 400
#define HOP 160
#define NW 2
#define RB 64  // row block for position-wise sublayers

#ifdef TASR_PROFILE
#ifdef ESP_PLATFORM
#include "esp_cpu.h"
static inline uint32_t nts(void) { return esp_cpu_get_cycle_count(); }
#else
#include <time.h>
static inline uint32_t nts(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint32_t)(t.tv_sec * 1000000000ull + t.tv_nsec); }
#endif
enum { N_FEAT, N_CONV0, N_IM2COL, N_CONV2, N_SUB, N_LN, N_GEMM, N_QUANT, N_ACT, N_QKV8, N_POS, N_ATT, N_DW, N_HEAD, N_NP };
static const char *nprof_names[N_NP] = {"features", "conv0", "im2col", "gemm_fe", "gemm_k704", "layernorm", "gemm", "quant",
                                        "act", "qkv_int8", "pos", "attention", "dwconv", "head+dec"};
static uint64_t nprof[N_NP];
#define NB(v) uint32_t v = nts()
#define NE(v, c) nprof[c] += (uint32_t)(nts() - v)
const char *tasr_nemo_profile_name(int i) { return i < N_NP ? nprof_names[i] : 0; }
uint64_t tasr_nemo_profile_value(int i) { return i < N_NP ? nprof[i] : 0; }
#else
#define NB(v)
#define NE(v, c)
const char *tasr_nemo_profile_name(int i) { (void)i; return 0; }
uint64_t tasr_nemo_profile_value(int i) { (void)i; return 0; }
#endif

typedef struct {
    const float *g, *b;
} nln_t;

typedef struct {
    nln_t n_ff1, n_att, n_conv, n_ff2, n_out;
    tasr_qlin_t ff1_1, ff1_2, qkv, out, pos, pw1, pw2, ff2_1, ff2_2;
    const float *pbu, *pbv, *dw_w, *dw_b;
    float *dw_wt;  // [k][d]
} nlayer_t;

struct tasr_nemo {
    int d, h, dh, dhp, ff, k, nl, sc, V, f1, f2;
    const float *window;
    int mel_start[NMEL], mel_len[NMEL];
    float *mel_w;
    const float *c0_w, *c0_b;
    tasr_qlin_t c2, sub, head;
    int rnnt;                                              // transducer model (RNN-T head instead of CTC)
    tasr_qlin_t r_emb, r_ih, r_hh, r_jenc, r_jpred, r_jout;  // prediction net (embedding, LSTM) + joint net
    nlayer_t *L;
    const char **tok;
    uint8_t *tok_len;
    int flags, s_chunk, s_left;  // header words 10-12: bit 0 fixed normalization, bit 1 trained for streaming
    const float *nmean, *nstd;   // fixed per-mel normalization (flags & 1)
    float tw_re[256], tw_im[256];
    int16_t bitrev[256];
    size_t weight_bytes, blob_bytes;
};

// ------------------------------------------------------------------ loading
typedef struct {
    const uint8_t *p, *end;
    int err;
} ncur_t;
static void nalign(ncur_t *c) { c->p = (const uint8_t *)(((uintptr_t)c->p + 15) & ~(uintptr_t)15); }
static const float *nf32(ncur_t *c, size_t n, size_t *acc)
{
    nalign(c);
    const float *r = (const float *)c->p;
    c->p += n * 4;
    if (c->p > c->end) c->err = 1;
    if (acc) *acc += n * 4;
    return r;
}
static tasr_qlin_t nload_qlin(ncur_t *c, int n, int k, int bits, size_t *acc)
{
    tasr_qlin_t L;
    memset(&L, 0, sizeof(L));
    L.blocked = bits == 4 && n % 16 == 0;
    int blk = (bits == 8 || L.blocked) ? 16 : 32;
    L.n = n; L.k = k; L.bits = bits;
    L.kp = (k + blk - 1) / blk * blk;
    nalign(c);
    L.w = (const int8_t *)c->p;
    size_t wb = (size_t)n * (bits == 8 ? L.kp : L.kp / 2);
    c->p += wb;
    *acc += wb;
    L.s = nf32(c, n, acc);
    L.b = nf32(c, n, acc);
    return L;
}
static nln_t nln(ncur_t *c, int d, size_t *acc)
{
    nln_t l;
    l.g = nf32(c, d, acc);
    l.b = nf32(c, d, acc);
    return l;
}

tasr_nemo_t *tasr_nemo_load(const uint8_t *blob, size_t size)
{
    if (size < 64 || memcmp(blob, "TNM1", 4)) return NULL;
    uint32_t hd[15];
    memcpy(hd, blob + 4, sizeof(hd));
    tasr_nemo_t *m = (tasr_nemo_t *)tasr_alloc(sizeof(tasr_nemo_t), 1);
    m->d = hd[1]; m->h = hd[2]; m->ff = hd[3]; m->k = hd[4]; m->nl = hd[5]; m->sc = hd[6]; m->V = hd[7];
    const int bits = hd[8];
    m->flags = hd[10]; m->s_chunk = hd[11]; m->s_left = hd[12];
    m->dh = m->d / m->h;
    m->dhp = (m->dh + 15) & ~15;
    m->f1 = (NMEL + 2 - 3) / 2 + 1;
    m->f2 = (m->f1 + 2 - 3) / 2 + 1;
    ncur_t c = {blob + 64, blob + size, 0};
    size_t acc = 0;
    const int d = m->d;
    m->window = nf32(&c, WIN, &acc);
    const float *fb = nf32(&c, NBIN * NMEL, NULL);
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
    m->c0_w = nf32(&c, m->sc * 9, &acc);
    m->c0_b = nf32(&c, m->sc, &acc);
    m->c2 = nload_qlin(&c, m->sc, m->sc * 9, 8, &acc);
    m->sub = nload_qlin(&c, d, m->sc * m->f2, 8, &acc);
    m->L = (nlayer_t *)tasr_alloc(sizeof(nlayer_t) * m->nl, 1);
    for (int i = 0; i < m->nl; i++) {
        nlayer_t *L = &m->L[i];
        L->n_ff1 = nln(&c, d, &acc);
        L->ff1_1 = nload_qlin(&c, m->ff, d, bits, &acc);
        L->ff1_2 = nload_qlin(&c, d, m->ff, bits, &acc);
        L->n_att = nln(&c, d, &acc);
        L->qkv = nload_qlin(&c, 3 * d, d, bits, &acc);
        L->out = nload_qlin(&c, d, d, bits, &acc);
        L->pos = nload_qlin(&c, d, d, 8, &acc);
        L->pbu = nf32(&c, d, &acc);
        L->pbv = nf32(&c, d, &acc);
        L->n_conv = nln(&c, d, &acc);
        L->pw1 = nload_qlin(&c, 2 * d, d, bits, &acc);
        L->dw_w = nf32(&c, d * m->k, &acc);
        L->dw_b = nf32(&c, d, &acc);
        L->pw2 = nload_qlin(&c, d, d, bits, &acc);
        L->n_ff2 = nln(&c, d, &acc);
        L->ff2_1 = nload_qlin(&c, m->ff, d, bits, &acc);
        L->ff2_2 = nload_qlin(&c, d, m->ff, bits, &acc);
        L->n_out = nln(&c, d, &acc);
        float *wt = (float *)tasr_alloc(sizeof(float) * m->k * d, 0);
        for (int ch = 0; ch < d; ch++)
            for (int j = 0; j < m->k; j++) wt[j * d + ch] = L->dw_w[ch * m->k + j];
        L->dw_wt = wt;
    }
    if (hd[9] == 1) {  // RNN-T: embedding (V+1 x 320), LSTM 320 (4 gates), joint (enc 176->320, pred 320->320, 320->V+1)
        const int P = 320;
        m->rnnt = 1;
        m->r_emb = nload_qlin(&c, m->V + 1, P, 8, &acc);
        m->r_ih = nload_qlin(&c, 4 * P, P, 8, &acc);
        m->r_hh = nload_qlin(&c, 4 * P, P, 8, &acc);
        m->r_jenc = nload_qlin(&c, P, d, 8, &acc);
        m->r_jpred = nload_qlin(&c, P, P, 8, &acc);
        m->r_jout = nload_qlin(&c, m->V + 1, P, 8, &acc);
    } else {
        m->head = nload_qlin(&c, m->V + 1, d, 8, &acc);
    }
    nalign(&c);
    m->tok = (const char **)tasr_alloc(sizeof(char *) * m->V, 1);
    m->tok_len = (uint8_t *)tasr_alloc(m->V, 1);
    for (int i = 0; i < m->V; i++) {
        m->tok_len[i] = *c.p++;
        m->tok[i] = (const char *)c.p;
        c.p += m->tok_len[i];
    }
    if (m->flags & 1) {  // fixed normalization stats appended after the tokens
        m->nmean = nf32(&c, NMEL, NULL);
        m->nstd = nf32(&c, NMEL, NULL);
    }
    if (c.err || c.p > c.end) { tasr_nemo_free(m); return NULL; }
    m->blob_bytes = (size_t)(c.p - blob);
    m->weight_bytes = acc;
    for (int i = 0; i < 256; i++) {
        m->tw_re[i] = (float)cos(-2.0 * M_PI * i / 512);
        m->tw_im[i] = (float)sin(-2.0 * M_PI * i / 512);
    }
    for (int i = 0; i < 256; i++) {
        int r = 0;
        for (int b = 0; b < 8; b++) r |= ((i >> b) & 1) << (7 - b);
        m->bitrev[i] = (int16_t)r;
    }
    return m;
}

void tasr_nemo_free(tasr_nemo_t *m)
{
    if (!m) return;
    if (m->L)
        for (int i = 0; i < m->nl; i++) tasr_free(m->L[i].dw_wt);
    tasr_free(m->mel_w); tasr_free(m->L); tasr_free((void *)m->tok); tasr_free(m->tok_len);
    tasr_free(m);
}
size_t tasr_nemo_weight_bytes(const tasr_nemo_t *m) { return m->weight_bytes; }
size_t tasr_nemo_blob_bytes(const tasr_nemo_t *m) { return m->blob_bytes; }

static size_t nmove(tasr_qlin_t *L, size_t *left)
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
size_t tasr_nemo_place_weights(tasr_nemo_t *m, size_t budget)
{
    // front end first (re-read every few frames), then the layers
    size_t left = budget, moved = nmove(&m->c2, &left) + nmove(&m->sub, &left);
    if (m->rnnt)  // joint output and LSTM weights are re-read every frame / every token
        moved += nmove(&m->r_jout, &left) + nmove(&m->r_jpred, &left) + nmove(&m->r_ih, &left) + nmove(&m->r_hh, &left) +
                 nmove(&m->r_jenc, &left);
    else
        moved += nmove(&m->head, &left);
    for (int i = 0; i < m->nl; i++) {
        nlayer_t *L = &m->L[i];
        tasr_qlin_t *q[9] = {&L->qkv, &L->pos, &L->out, &L->pw1, &L->pw2, &L->ff1_1, &L->ff1_2, &L->ff2_1, &L->ff2_2};
        for (int j = 0; j < 9; j++) moved += nmove(q[j], &left);
    }
    return moved;
}

#ifdef TASR_DEBUG_SUMS  // bit-exact comparison of intermediate activations between the host build and the firmware
#include <stdio.h>
static void ndbg(const char *what, int i, const float *x, size_t n)
{
    uint32_t h = 2166136261u;
    for (size_t k = 0; k < n; k++) { uint32_t b; memcpy(&b, &x[k], 4); h = (h ^ b) * 16777619u; }
    printf("DBG %s %d %08x\n", what, i, (unsigned)h);
}
static void ndbg8(const char *what, int i, const int8_t *x, int rows, int stride, int kp)
{
    uint32_t h = 2166136261u;
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < kp; k++) h = (h ^ (uint8_t)x[(size_t)r * stride + k]) * 16777619u;
    printf("DBG %s %d %08x\n", what, i, (unsigned)h);
}
#ifdef TASR_DEBUG_DUMP  // print every word of a tensor ("DUMP name layer row hex hex ..."), to find which elements a bug corrupts
static void ndump(const char *what, int i, const float *x, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        printf("DUMP %s %d %d", what, i, r);
        for (int c = 0; c < cols; c++) { uint32_t b; memcpy(&b, &x[(size_t)r * cols + c], 4); printf(" %x", (unsigned)b); }
        printf("\n");
    }
}
#else
#define ndump(what, i, x, rows, cols)
#endif
#else
#define ndbg(what, i, x, n)
#define ndbg8(what, i, x, rows, stride, kp)
#define ndump(what, i, x, rows, cols)
#endif

// Experiment: CONFIG_TASR_PAR_MASK selects which call sites of the utterance path really use both cores (bit = site id)
#ifdef CONFIG_TASR_PAR_MASK
#define NPAR(site, fn, ctx, n) do { if (CONFIG_TASR_PAR_MASK & (1u << (site))) tasr_parallel(fn, ctx, n); else fn(ctx, 0, n, 0); } while (0)
#else
#define NPAR(site, fn, ctx, n) tasr_parallel(fn, ctx, n)
#endif

// ------------------------------------------------------------------ helpers
static float nsig_tab[385], nsig_slope[384];
static int sig_ok;
static void nsig_init(void)
{
    if (sig_ok) return;
    for (int i = 0; i <= 384; i++) nsig_tab[i] = (float)(1.0 / (1.0 + exp(-(-12.0 + i / 16.0))));
    for (int i = 0; i < 384; i++) nsig_slope[i] = nsig_tab[i + 1] - nsig_tab[i];
    sig_ok = 1;
}
static inline float nsig(float x)
{
    float u = (x + 12.0f) * 16.0f;
    if (u <= 0.f) return 0.f;
    if (u >= 384.f) return 1.f;
    int i = (int)u;
    return nsig_tab[i] + (u - (float)i) * nsig_slope[i];
}
// exp(-i/64) for the attention softmax; nearest-entry lookup (error <= 0.8%, below the int8 probability step)
#define NEXP_STEPS 64
#define NEXP_N (20 * NEXP_STEPS)
static float nexp_tab[NEXP_N];
static void nexp_init(void)
{
    static int done;
    if (done) return;
    for (int i = 0; i < NEXP_N; i++) nexp_tab[i] = (float)exp(-(double)i / NEXP_STEPS);
    done = 1;
}
static inline float qvec(const float *x, int n, int8_t *q, int np)
{
    float mx = 0.f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); mx = a > mx ? a : mx; }
    if (mx < 1e-30f) mx = 1e-30f;
    const float inv = 127.0f / mx;
    for (int i = 0; i < n; i++) { float y = x[i] * inv + 12582912.0f; q[i] = (int8_t)(int)(y - 12582912.0f); }
    for (int i = n; i < np; i++) q[i] = 0;
    return mx / 127.0f;
}
static void layernorm_row(const float *r, int d, nln_t p, float *o)
{
    float mean = 0.f;
    for (int i = 0; i < d; i++) mean += r[i];
    mean /= (float)d;
    float var = 0.f;
    for (int i = 0; i < d; i++) { float v = r[i] - mean; var += v * v; }
    var /= (float)d;
    const float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < d; i++) o[i] = (r[i] - mean) * inv * p.g[i] + p.b[i];
}

// ------------------------------------------------------------------ work context
typedef struct {
    const tasr_nemo_t *m;
    int T;                       // encoder frames
    int ldq;
    int8_t *xq;                  // [RB][ldq]
    int8_t *xq1;                 // experiment: private copy of the activation tile for worker 1
    float *xs;                   // [RB]
    int8_t *wtmp[NW];
    int32_t *acc[NW];
    // attention
    int8_t *qu, *qv, *k8, *vt, *p8;  // [h][T][dhp], [h][T][dhp], [h][T][dhp], [h][dh][Tp], [h][2T-1][dhp]
    float *squ, *sqv, *sk, *sv, *sp; // scales
    int Tp;
    float *sc[NW];
    int8_t *pq[NW];
    int32_t *iacc[NW];
    int iacc_n;  // stride of the second (positional) score buffer in iacc[w]
    float *att;                  // [T][d]
} nctx_t;

typedef struct {
    nctx_t *c;
    const tasr_qlin_t *L;
    int T;
    float *y;
    int ldy;
} ngemm_t;
static void ngemm_job(void *p, int b, int e, int w)
{
    ngemm_t *j = (ngemm_t *)p;
    const int u = j->L->blocked ? 16 : 1;
    const int8_t *xq = j->c->xq;
#ifdef CONFIG_TASR_EXP_XQ_COPY
    if (w == 1 && j->c->xq1 && !j->L->blocked && j->L->kp > 256 && j->L->kp <= 1024) {
        memcpy(j->c->xq1, xq, (size_t)j->T * j->c->ldq);   // the two cores then never read the same activation addresses
        xq = j->c->xq1;
    }
#endif
    tasr_qlin_range(j->L, xq, j->c->xs, j->T, j->c->ldq, j->y, j->ldy, j->c->wtmp[w], j->c->acc[w], b * u, e * u);
}
#ifdef CONFIG_TASR_EXP_ROWSPLIT
// experiment: split the K=704 products by rows of the tile instead of by output channels
static void ngemm_rows_job(void *p, int b, int e, int w)
{
    ngemm_t *j = (ngemm_t *)p;
    tasr_qlin_range(j->L, j->c->xq + (size_t)b * j->c->ldq, j->c->xs + b, e - b, j->c->ldq, j->y + (size_t)b * j->ldy, j->ldy,
                    j->c->wtmp[w], j->c->acc[w], 0, j->L->n);
}
#endif
// y[t][:] = L(x[t][:]) for T <= RB rows (x rows already quantized into c->xq / c->xs)
static void nqlin(nctx_t *c, const tasr_qlin_t *L, int T, float *y, int ldy)
{
    NB(t0);
    ngemm_t j = {c, L, T, y, ldy};
#ifdef CONFIG_TASR_EXP_ROWSPLIT
    if (!L->blocked && L->bits == 8 && L->kp > 256 && L->kp <= 1024) {
        tasr_parallel(ngemm_rows_job, &j, T);
        NE(t0, L->kp > 1024 ? N_CONV2 : L->kp > 256 ? N_SUB : N_GEMM);
        return;
    }
#endif
    NPAR(L->blocked ? 11 : (L->bits == 8 && L->kp <= 256) ? 3 : L->kp > 1024 ? 10 : 9, ngemm_job, &j, L->blocked ? L->n / 16 : L->n);
    NE(t0, L->kp > 1024 ? N_CONV2 : L->kp > 256 ? N_SUB : N_GEMM);  // front-end GEMMs / long-K layers / K <= 256
}
static void nq_job(void *p, int b, int e, int w);
static void nquant(nctx_t *c, const float *x, int T, int K, int ldx, int kp)
{
    NB(t0);
    typedef struct { const float *x; int K, ldx, kp; nctx_t *c; } nqj2_t;
    nqj2_t j = {x, K, ldx, kp, c};
    NPAR(4, nq_job, &j, T);
    NE(t0, N_QUANT);
}

// ------------------------------------------------------------------ attention job (per head)
typedef struct {
    nctx_t *c;
} natt_t;
static void natt_job(void *p, int b, int e, int w)
{
    nctx_t *c = ((natt_t *)p)->c;
    const tasr_nemo_t *m = c->m;
    const int T = c->T, dh = m->dh, dhp = m->dhp, d = m->d;
    const float scale = 1.0f / sqrtf((float)dh);
    float *sc = c->sc[w];
    int8_t *pq = c->pq[w];
    int32_t *ia = c->iacc[w];
    for (int hh = b; hh < e; hh++) {
        const int8_t *K = c->k8 + (size_t)hh * T * dhp;
        const int8_t *P = c->p8 + (size_t)hh * (2 * T - 1) * dhp;
        const int8_t *VT = c->vt + (size_t)hh * dh * c->Tp;
        const float *sk = c->sk + (size_t)hh * T, *sv = c->sv + (size_t)hh * T, *sp = c->sp + (size_t)hh * (2 * T - 1);
        for (int i = 0; i < T; i++) {
            const int8_t *qu = c->qu + ((size_t)hh * T + i) * dhp, *qv = c->qv + ((size_t)hh * T + i) * dhp;
            const float squ = c->squ[hh * T + i], sqv = c->sqv[hh * T + i];
            int32_t *ib = ia + c->iacc_n;
            const int m0 = T - 1 - i;  // score(i, j) uses pos row m0 + j
            if (dhp == 48) {
                tasr_dot48_rows(qu, K, 48, T, ia);
                tasr_dot48_rows(qv, P + (size_t)m0 * 48, 48, T, ib);
            } else {
                tasr_dot_rows_s8(qu, K, dhp, T, dhp, ia);
                tasr_dot_rows_s8(qv, P + (size_t)m0 * dhp, dhp, T, dhp, ib);
            }
            const float A = squ * scale, B = sqv * scale;
            const float *spm = sp + m0;
            float mx = -1e30f;
            for (int j = 0; j < T; j++) {
                float a = (float)ia[j] * A * sk[j] + (float)ib[j] * B * spm[j];
                sc[j] = a;
                mx = a > mx ? a : mx;
            }
            float sum = 0.f, pm = 0.f;
            const float mx64 = mx * NEXP_STEPS + 0.5f;
            for (int j = 0; j < T; j++) {
                const int ix = (int)(mx64 - sc[j] * NEXP_STEPS);  // >= 0: round((mx - s) * 64)
                const float ex = ix < NEXP_N ? nexp_tab[ix] : 0.f;
                sum += ex;
                float p2 = ex * sv[j];
                sc[j] = p2;
                pm = p2 > pm ? p2 : pm;
            }
            if (pm < 1e-30f) pm = 1e-30f;
            const float inv = 127.0f / pm;
            for (int j = 0; j < T; j++) { float y = sc[j] * inv + 12582912.0f; pq[j] = (int8_t)(int)(y - 12582912.0f); }
            for (int j = T; j < c->Tp; j++) pq[j] = 0;
            tasr_dot_rows_s8(pq, VT, c->Tp, dh, c->Tp, ia);
            const float os = pm / 127.0f / sum;
            float *y = c->att + (size_t)i * d + hh * dh;
            for (int e2 = 0; e2 < dh; e2++) y[e2] = (float)ia[e2] * os;
        }
    }
}

// depthwise conv (k, symmetric padding) + swish over all frames: job over channel ranges
typedef struct {
    const float *in;  // [T][d]
    float *out;       // [T][d]
    const nlayer_t *L;
    int T, d, k;
} ndw_t;
static void ndw_job(void *p, int b, int e, int w)
{
    (void)w;
    ndw_t *j = (ndw_t *)p;
    const int d = j->d;
    int t = b;
    for (; t + 1 < e; t += 2) {  // two frames per pass: each weight load feeds both (same summation order per output)
        float *o = j->out + (size_t)t * d, *o2 = o + d;
        const float *xin = j->in + (size_t)t * d;
        for (int ch = 0; ch < d; ch += 4) {
            float a0 = j->L->dw_b[ch], a1 = j->L->dw_b[ch + 1], a2 = j->L->dw_b[ch + 2], a3 = j->L->dw_b[ch + 3];
            float c0 = a0, c1 = a1, c2 = a2, c3 = a3;
            const float *wt = j->L->dw_wt + ch, *x = xin + ch;
            for (int kk = 0; kk < j->k; kk++, wt += d, x += d) {
                const float w0 = wt[0], w1 = wt[1], w2 = wt[2], w3 = wt[3];
                a0 += w0 * x[0]; a1 += w1 * x[1]; a2 += w2 * x[2]; a3 += w3 * x[3];
                c0 += w0 * x[d]; c1 += w1 * x[d + 1]; c2 += w2 * x[d + 2]; c3 += w3 * x[d + 3];
            }
            o[ch] = a0; o[ch + 1] = a1; o[ch + 2] = a2; o[ch + 3] = a3;
            o2[ch] = c0; o2[ch + 1] = c1; o2[ch + 2] = c2; o2[ch + 3] = c3;
        }
        for (int ch = 0; ch < d; ch++) o[ch] = o[ch] * nsig(o[ch]);
        for (int ch = 0; ch < d; ch++) o2[ch] = o2[ch] * nsig(o2[ch]);
    }
    for (; t < e; t++) {  // j->in has k/2 zero rows before frame 0 and after frame T-1
        float *o = j->out + (size_t)t * d;
        const float *xin = j->in + (size_t)t * d;
        for (int ch = 0; ch < d; ch += 4) {  // d % 4 == 0; 4 accumulators stay in registers across the k taps
            float a0 = j->L->dw_b[ch], a1 = j->L->dw_b[ch + 1], a2 = j->L->dw_b[ch + 2], a3 = j->L->dw_b[ch + 3];
            const float *wt = j->L->dw_wt + ch, *x = xin + ch;
            for (int kk = 0; kk < j->k; kk++, wt += d, x += d) {
                a0 += wt[0] * x[0]; a1 += wt[1] * x[1]; a2 += wt[2] * x[2]; a3 += wt[3] * x[3];
            }
            o[ch] = a0; o[ch + 1] = a1; o[ch + 2] = a2; o[ch + 3] = a3;
        }
        for (int ch = 0; ch < d; ch++) o[ch] = o[ch] * nsig(o[ch]);
    }
}

// ------------------------------------------------------------------ front end
static void power_spectrum512(const tasr_nemo_t *m, const float *x, float *re, float *im, float *pw)
{
    const int N2 = 256;
    for (int n = 0; n < N2; n++) {
        int j = m->bitrev[n];
        re[j] = x[2 * n];
        im[j] = x[2 * n + 1];
    }
    for (int len = 2; len <= N2; len <<= 1) {
        const int half = len >> 1, step = 2 * (N2 / len);
        for (int i = 0; i < N2; i += len)
            for (int j = 0; j < half; j++) {
                const float wr = m->tw_re[j * step], wi = m->tw_im[j * step];
                float *ar = &re[i + j], *ai = &im[i + j], *br = &re[i + j + half], *bi = &im[i + j + half];
                const float tr = *br * wr - *bi * wi, ti = *br * wi + *bi * wr;
                *br = *ar - tr; *bi = *ai - ti;
                *ar += tr; *ai += ti;
            }
    }
    pw[0] = (re[0] + im[0]) * (re[0] + im[0]);
    pw[N2] = (re[0] - im[0]) * (re[0] - im[0]);
    for (int k = 1; k < N2; k++) {
        const float zr = re[k], zi = im[k], cr = re[N2 - k], ci = -im[N2 - k];
        const float er = 0.5f * (zr + cr), ei = 0.5f * (zi + ci), dr = 0.5f * (zr - cr), di = 0.5f * (zi - ci);
        const float wr = m->tw_re[k], wi = m->tw_im[k];
        const float o_r = er + (wr * di + wi * dr), oi = ei - (wr * dr - wi * di);
        pw[k] = o_r * o_r + oi * oi;
    }
}

// features: preemph -> centered STFT -> mel -> log -> per-feature normalization; returns T = n/160 + 1 frames
typedef struct {
    const tasr_nemo_t *m;
    const int16_t *pcm;
    int n;
    float *F;
} nfeat_t;
static void nfeat_job(void *p, int b, int e, int w)
{
    (void)w;
    nfeat_t *j = (nfeat_t *)p;
    const tasr_nemo_t *m = j->m;
    float xw[512], re[256], im[256], pw[NBIN];
    for (int t = b; t < e; t++) {
        // frame t covers original samples [160t - 200, 160t + 200) (window offset inside the 512 frame does not
        // change |X|); samples outside [0, n) are zero (center=True, zero padding)
        for (int i = 0; i < WIN; i++) {
            const int si = HOP * t - 200 + i;
            float v = 0.f;
            if (si >= 0 && si < j->n) {
                const float x0 = j->pcm[si] / 32768.0f;
                v = si == 0 ? x0 : x0 - 0.97f * (j->pcm[si - 1] / 32768.0f);
            }
            xw[i] = v * m->window[i];
        }
        for (int i = WIN; i < 512; i++) xw[i] = 0.f;
        power_spectrum512(m, xw, re, im, pw);
        float *o = j->F + (size_t)t * NMEL;
        const float *wm = m->mel_w;
        for (int k = 0; k < NMEL; k++) {
            float acc = 0.f;
            const float *pp = pw + m->mel_start[k];
            for (int q = 0; q < m->mel_len[k]; q++) acc += pp[q] * wm[q];
            wm += m->mel_len[k];
            o[k] = logf(acc + 5.9604644775390625e-08f);
        }
    }
}
static float *nemo_features(const tasr_nemo_t *m, const int16_t *pcm, int n, int *T_out)
{
    const int nvalid = n / HOP, T = nvalid + 1;
    float *F = (float *)tasr_alloc(sizeof(float) * (size_t)T * NMEL, 0);
    nfeat_t fj = {m, pcm, n, F};
    NPAR(0, nfeat_job, &fj, T);
    ndbg("twre", 0, m->tw_re, 256);
    ndbg("twim", 0, m->tw_im, 256);
    ndbg("logmel", 0, F, (size_t)T * NMEL);
    for (int j = 0; j < NMEL; j++) {
        if (m->nmean) {  // streaming-trained models: global per-mel statistics
            const float mu = m->nmean[j], inv = 1.0f / m->nstd[j];
            for (int t = 0; t < nvalid; t++) F[(size_t)t * NMEL + j] = (F[(size_t)t * NMEL + j] - mu) * inv;
            F[(size_t)nvalid * NMEL + j] = 0.f;
            continue;
        }
        double mean = 0, var = 0;
        for (int t = 0; t < nvalid; t++) mean += F[(size_t)t * NMEL + j];
        mean /= nvalid;
        for (int t = 0; t < nvalid; t++) { double v = F[(size_t)t * NMEL + j] - mean; var += v * v; }
        const float sd = (float)sqrt(var / (nvalid > 1 ? nvalid - 1 : 1)) + 1e-5f;
        for (int t = 0; t < nvalid; t++) F[(size_t)t * NMEL + j] = (F[(size_t)t * NMEL + j] - (float)mean) / sd;
        F[(size_t)nvalid * NMEL + j] = 0.f;
    }
    *T_out = T;
    return F;
}

typedef struct {
    const tasr_nemo_t *m;
    const float (*P)[NMEL + 2];  // 3 zero-padded mel rows
    float *out;                  // [sc][f1 + 2] (zero column on both sides for the next conv)
    float *cmw;                  // [NW][f1 + 2] per-worker max over its channels of each output column
} nc0_t;
static void nc0_job(void *p, int b, int e, int w)
{
    nc0_t *j = (nc0_t *)p;
    const tasr_nemo_t *m = j->m;
    const int f1 = m->f1;
    float *cm = j->cmw + (size_t)w * (f1 + 2);
    for (int f = 0; f < f1 + 2; f++) cm[f] = 0.f;
    for (int ch = b; ch < e; ch++) {
        const float *wt = m->c0_w + ch * 9;
        const float w0 = wt[0], w1 = wt[1], w2 = wt[2], w3 = wt[3], w4 = wt[4], w5 = wt[5], w6 = wt[6], w7 = wt[7], w8 = wt[8];
        const float bias = m->c0_b[ch];
        float *o = j->out + (size_t)ch * (f1 + 2);
        o[0] = 0.f;
        o[f1 + 1] = 0.f;
        for (int f = 0; f < f1; f++) {
            const float *p0 = &j->P[0][2 * f], *p1 = &j->P[1][2 * f], *p2 = &j->P[2][2 * f];
            float acc = bias + w0 * p0[0] + w1 * p0[1] + w2 * p0[2] + w3 * p1[0] + w4 * p1[1] + w5 * p1[2] + w6 * p2[0] +
                        w7 * p2[1] + w8 * p2[2];
            const float v = acc > 0.f ? acc : 0.f;
            o[1 + f] = v;
            cm[1 + f] = v > cm[1 + f] ? v : cm[1 + f];
        }
    }
}

// im2col for 20 output positions of one frame + per-row int8 quantization (job over positions). conv0 outputs are
// ReLU'd (>= 0), so each patch row's max is the max of 3 x 3 per-(row, column) channel maxima (cmax, computed once per
// conv0 row); patches are then quantized while gathering, with exactly the arithmetic of tasr_quant_rows.
typedef struct {
    const tasr_nemo_t *m;
    const float *rows[3];  // conv0 rows 2t2-1..2t2+1 (padded layout) or NULL
    const float *cmax[3];  // [f1 + 2] max over channels of each padded column of those rows
    int8_t *col;           // [f2][kp]
    float *xs;
} nim_t;
static void nim_job(void *p, int b, int e, int w)
{
    (void)w;
    nim_t *j = (nim_t *)p;
    const tasr_nemo_t *m = j->m;
    const int sc = m->sc, f1p = m->f1 + 2, K = m->c2.k, kp = m->c2.kp;
    for (int f = b; f < e; f++) {
        float mx = 0.f;
        for (int dt = 0; dt < 3; dt++)
            if (j->rows[dt])
                for (int df = 0; df < 3; df++) {
                    const float v = j->cmax[dt][2 * f + df];
                    mx = v > mx ? v : mx;
                }
        if (mx < 1e-30f) mx = 1e-30f;
        const float inv = 127.0f / mx;
        int8_t *q = j->col + (size_t)f * kp;
        for (int dt = 0; dt < 3; dt++) {
            const float *r = j->rows[dt];
            int8_t *qd = q + dt * 3;
            if (!r) {
                for (int ch = 0; ch < sc; ch++) qd[ch * 9] = qd[ch * 9 + 1] = qd[ch * 9 + 2] = 0;
                continue;
            }
            const float *x = r + 2 * f;
            for (int ch = 0; ch < sc; ch++, x += f1p, qd += 9) {
                qd[0] = (int8_t)(int)((x[0] * inv + 12582912.0f) - 12582912.0f);
                qd[1] = (int8_t)(int)((x[1] * inv + 12582912.0f) - 12582912.0f);
                qd[2] = (int8_t)(int)((x[2] * inv + 12582912.0f) - 12582912.0f);
            }
        }
        for (int k = K; k < kp; k++) q[k] = 0;
        j->xs[f] = mx / 127.0f;
    }
}

// generic row-parallel helpers for the encoder
typedef struct {
    const float *x;
    float *y;
    int d, ldx, ldy;
    nln_t p;
} nlnj_t;
static void nln_job(void *p, int b, int e, int w)
{
    (void)w;
    nlnj_t *j = (nlnj_t *)p;
    for (int t = b; t < e; t++) layernorm_row(j->x + (size_t)t * j->ldx, j->d, j->p, j->y + (size_t)t * j->ldy);
}
static void nln_rows(const float *x, int T, int d, nln_t p, float *y)
{
    NB(t0);
    nlnj_t j = {x, y, d, d, d, p};
    NPAR(1, nln_job, &j, T);
    NE(t0, N_LN);
}
static void nsilu_job(void *p, int b, int e, int w)
{
    (void)w;
    float *x = (float *)p;
    for (int i = b; i < e; i++) x[i] = x[i] * nsig(x[i]);
}
static void nsilu(float *x, int n)
{
    NB(t0);
    NPAR(2, nsilu_job, x, n);
    NE(t0, N_ACT);
}
typedef struct {
    const float *x;
    int K, ldx, kp;
    nctx_t *c;
} nqj_t;
static void nq_job(void *p, int b, int e, int w)
{
    (void)w;
    nqj_t *j = (nqj_t *)p;
    tasr_quant_rows(j->x + (size_t)b * j->ldx, e - b, j->K, j->ldx, j->c->xq + (size_t)b * j->c->ldq, j->c->ldq, j->kp,
                    j->c->xs + b);
}

// ------------------------------------------------------------------ RNN-T greedy decoding
// Exactly NeMo's greedy transducer search (up to 5 symbols per frame), with int8 GEMVs for the LSTM and joint network.
#define RNNT_P 320
#define RNNT_MAXSYM 5
#define RNNT_B 8  // joint-network frames evaluated per pass (see rnnt_greedy)
static float nsigm(float x) { return 1.0f / (1.0f + expf(-x)); }
typedef struct {
    float h[RNNT_P], c[RNNT_P], gp[RNNT_P];
} rnnt_state_t;
static void rnnt_pred_step(const tasr_nemo_t *m, nctx_t *C, const float *xin, rnnt_state_t *st)
{
    const int P = RNNT_P;
    static float gi[4 * RNNT_P], gh[4 * RNNT_P];
    nquant(C, xin, 1, P, P, m->r_ih.kp);
    nqlin(C, &m->r_ih, 1, gi, 4 * P);
    nquant(C, st->h, 1, P, P, m->r_hh.kp);
    nqlin(C, &m->r_hh, 1, gh, 4 * P);
    for (int j = 0; j < P; j++) {  // PyTorch gate order i, f, g, o
        const float i = nsigm(gi[j] + gh[j]), f = nsigm(gi[P + j] + gh[P + j]);
        const float g = tanhf(gi[2 * P + j] + gh[2 * P + j]), o = nsigm(gi[3 * P + j] + gh[3 * P + j]);
        st->c[j] = f * st->c[j] + i * g;
        st->h[j] = o * tanhf(st->c[j]);
    }
    nquant(C, st->h, 1, P, P, m->r_jpred.kp);
    nqlin(C, &m->r_jpred, 1, st->gp, P);
}
static void rnnt_greedy(const tasr_nemo_t *m, nctx_t *C, const float *x, int T, char *text, int maxlen, int *len)
{
    const int P = RNNT_P, d = m->d, V1 = m->V + 1;
    float *fj = (float *)tasr_alloc(sizeof(float) * (size_t)T * P, 0);  // joint encoder projection, all frames
    for (int t0 = 0; t0 < T; t0 += RB) {
        const int tn = T - t0 < RB ? T - t0 : RB;
        nquant(C, x + (size_t)t0 * d, tn, d, d, m->r_jenc.kp);
        nqlin(C, &m->r_jenc, tn, fj + (size_t)t0 * P, P);
    }
    static rnnt_state_t st;
    static float e[RNNT_P];
    float *z = (float *)tasr_alloc(sizeof(float) * RNNT_B * P, 0);
    float *lo = (float *)tasr_alloc(sizeof(float) * RNNT_B * V1, 0);
    memset(&st, 0, sizeof(st));
    memset(e, 0, sizeof(e));
    rnnt_pred_step(m, C, e, &st);  // start of sequence: zero input
    // Most frames emit only blank and leave the prediction state unchanged, so the joint network is evaluated for up to
    // RNNT_B frames at once with the current state (one pass over its 328 KB output matrix instead of one per frame).
    // Rows after the first frame that emits a token are discarded and recomputed, so the result is exactly greedy search.
    int t = 0;
    while (t < T) {
        const int nb = T - t < RNNT_B ? T - t : RNNT_B;
        for (int i = 0; i < nb; i++) {
            const float *f = fj + (size_t)(t + i) * P;
            float *zi = z + (size_t)i * P;
            for (int j = 0; j < P; j++) { const float v = f[j] + st.gp[j]; zi[j] = v > 0.f ? v : 0.f; }
        }
        nquant(C, z, nb, P, P, m->r_jout.kp);
        nqlin(C, &m->r_jout, nb, lo, V1);
        int i = 0, k = m->V;
        for (; i < nb; i++) {
            const float *l = lo + (size_t)i * V1;
            k = 0;
            for (int v = 1; v < V1; v++) if (l[v] > l[k]) k = v;
            if (k != m->V) break;
        }
        if (i == nb) { t += nb; continue; }  // whole window blank
        // frame t+i emits k: continue that frame symbol by symbol with the updated prediction state
        const float *f = fj + (size_t)(t + i) * P;
        for (int s = 0; s < RNNT_MAXSYM; s++) {
            if (s > 0) {
                for (int j = 0; j < P; j++) { const float v = f[j] + st.gp[j]; z[j] = v > 0.f ? v : 0.f; }
                nquant(C, z, 1, P, P, m->r_jout.kp);
                nqlin(C, &m->r_jout, 1, lo, V1);
                k = 0;
                for (int v = 1; v < V1; v++) if (lo[v] > lo[k]) k = v;
                if (k == m->V) break;
            }
            if (maxlen) {
                const int nl = m->tok_len[k];
                if (*len + nl + 1 < maxlen) { memcpy(text + *len, m->tok[k], nl); *len += nl; text[*len] = 0; }
            }
            const int8_t *er = m->r_emb.w + (size_t)k * m->r_emb.kp;  // embedding row, dequantized
            for (int j = 0; j < P; j++) e[j] = (float)er[j] * m->r_emb.s[k];
            rnnt_pred_step(m, C, e, &st);
        }
        t += i + 1;
    }
    tasr_free(z);
    tasr_free(lo);
    tasr_free(fj);
}

// ------------------------------------------------------------------ main entry
int tasr_nemo_transcribe(const tasr_nemo_t *m, const int16_t *pcm, int n, tasr_decoder_t *dec, char *text, int maxlen,
                         float *logit_sink, int max_frames, int *n_frames)
{
    nsig_init();
    nexp_init();
    if (n < 2 * HOP) { if (maxlen) text[0] = 0; return 0; }
    const int d = m->d, H = m->h, dh = m->dh, dhp = m->dhp, sc = m->sc, f1 = m->f1, f2 = m->f2;
    int T0;
    NB(tf);
    float *F = nemo_features(m, pcm, n, &T0);
    NE(tf, N_FEAT);
    ndbg("feat", 0, F, (size_t)T0 * NMEL);
    const int T1 = (T0 - 1) / 2 + 1, T = (T1 - 1) / 2 + 1;
    nctx_t C;
    memset(&C, 0, sizeof(C));
    C.m = m; C.T = T;
    C.ldq = m->ff > 2 * d ? m->ff : 2 * d;   // layer activations (K <= ff)
#ifdef CONFIG_TASR_EXP_XQ_PSRAM
    C.xq = (int8_t *)tasr_alloc((size_t)RB * C.ldq, 0);
#else
    C.xq = (int8_t *)tasr_alloc((size_t)RB * C.ldq, 1);
#endif
#ifdef CONFIG_TASR_EXP_XQ_COPY
    C.xq1 = (int8_t *)tasr_alloc((size_t)RB * C.ldq, 1);
#endif
    C.xs = (float *)tasr_alloc(sizeof(float) * RB, 1);
    int8_t *xq_sub = (int8_t *)tasr_alloc((size_t)RB * m->sub.kp, 0);   // front-end projection rows (K = 3520)
    for (int w = 0; w < NW; w++) {
        C.wtmp[w] = (int8_t *)tasr_alloc(16 * C.ldq + 16, 1);
        C.acc[w] = (int32_t *)tasr_alloc(sizeof(int32_t) * 64 * 16, 1);
    }
    float *x = (float *)tasr_alloc(sizeof(float) * (size_t)T * d, 0);
    // ---- striding conv subsampling, pipelined over output frames
    {
        float *ring = (float *)tasr_alloc(sizeof(float) * 3 * sc * (f1 + 2), 0);  // conv0 rows (r % 3), padded
        // conv2 runs as one GEMM per C2B output frames (C2B * f2 <= RB patch rows), so its weights are fetched once per
        // C2B frames instead of once per frame
        const int C2B = RB / f2 < 1 ? 1 : RB / f2;
        float *c2out = (float *)tasr_alloc(sizeof(float) * C2B * f2 * sc, 0);    // [C2B * f2][sc]
        float *fr = (float *)tasr_alloc(sizeof(float) * sc * f2, 0);       // one sub_out input row (float)
        float *xs_sub = (float *)tasr_alloc(sizeof(float) * RB, 1);          // its per-row scales
        int8_t *col = (int8_t *)tasr_alloc((size_t)C2B * f2 * m->c2.kp, 0);
        int nbat = 0;
        float *cmaxr = (float *)tasr_alloc(sizeof(float) * 3 * (f1 + 2), 1);  // per ring row: max over channels
        float *cmw = (float *)tasr_alloc(sizeof(float) * NW * (f1 + 2), 1);
        int have = -1, nflat = 0, t_flat0 = 0;
        for (int t2 = 0; t2 < T; t2++) {
            for (int r = 2 * t2 - 1; r <= 2 * t2 + 1; r++) {  // conv0 output rows needed
                if (r < 0 || r >= T1 || r <= have) continue;
                NB(tc0);
                float P[3][NMEL + 2];
                for (int dt = 0; dt < 3; dt++) {
                    const int ti = 2 * r - 1 + dt;
                    P[dt][0] = 0.f;
                    P[dt][NMEL + 1] = 0.f;
                    if (ti < 0 || ti >= T0) memset(&P[dt][1], 0, sizeof(float) * NMEL);
                    else memcpy(&P[dt][1], F + (size_t)ti * NMEL, sizeof(float) * NMEL);
                }
                nc0_t j = {m, (const float(*)[NMEL + 2])P, ring + (size_t)(r % 3) * sc * (f1 + 2), cmw};
                memset(cmw, 0, sizeof(float) * NW * (f1 + 2));  // a worker that gets no channels leaves zeros
                NPAR(5, nc0_job, &j, sc);
                {
                    float *cm = cmaxr + (size_t)(r % 3) * (f1 + 2);
                    for (int f = 0; f < f1 + 2; f++) {
                        float v = cmw[f];
                        for (int w = 1; w < NW; w++) v = cmw[(size_t)w * (f1 + 2) + f] > v ? cmw[(size_t)w * (f1 + 2) + f] : v;
                        cm[f] = v;
                    }
                }
                NE(tc0, N_CONV0);
                have = r;
            }
            NB(tim);
            {
                nim_t j;
                j.m = m;
                j.col = col + (size_t)nbat * f2 * m->c2.kp;
                j.xs = C.xs + nbat * f2;
                for (int dt = 0; dt < 3; dt++) {
                    const int r = 2 * t2 - 1 + dt;
                    j.rows[dt] = (r >= 0 && r < T1) ? ring + (size_t)(r % 3) * sc * (f1 + 2) : NULL;
                    j.cmax[dt] = cmaxr + (size_t)((r + 3) % 3) * (f1 + 2);
                }
                NPAR(6, nim_job, &j, f2);
            }
            NE(tim, N_IM2COL);
            if (++nbat < C2B && t2 < T - 1) continue;
            {   // conv2 GEMM on nbat * f2 patch rows
                int8_t *save = C.xq;
                int ld = C.ldq;
                C.xq = col; C.ldq = m->c2.kp;
                nqlin(&C, &m->c2, nbat * f2, c2out, sc);
                C.xq = save; C.ldq = ld;
            }
            for (int fb = 0; fb < nbat; fb++) {
                const int tt = t2 - nbat + 1 + fb;
                const float *co = c2out + (size_t)fb * f2 * sc;
                for (int f = 0; f < f2; f++)
                    for (int ch = 0; ch < sc; ch++) {
                        const float v = co[f * sc + ch];
                        fr[ch * f2 + f] = v > 0.f ? v : 0.f;
                    }
                NB(tq);  // quantize the row now so only int8 rows are buffered
                tasr_quant_rows(fr, 1, sc * f2, sc * f2, xq_sub + (size_t)nflat * m->sub.kp, m->sub.kp, m->sub.kp,
                                xs_sub + nflat);
                NE(tq, N_QUANT);
                nflat++;
                if (nflat == RB || tt == T - 1) {
                    int8_t *save = C.xq;
                    float *save_s = C.xs;
                    int ld = C.ldq;
                    C.xq = xq_sub; C.ldq = m->sub.kp; C.xs = xs_sub;
                    nqlin(&C, &m->sub, nflat, x + (size_t)t_flat0 * d, d);
                    C.xq = save; C.ldq = ld; C.xs = save_s;
                    t_flat0 += nflat;
                    nflat = 0;
                }
            }
            nbat = 0;
        }
        tasr_free(ring); tasr_free(c2out); tasr_free(fr); tasr_free(xs_sub); tasr_free(col); tasr_free(cmaxr); tasr_free(cmw);
    }
    tasr_free(F);
    tasr_free(xq_sub);
    const float xscale = sqrtf((float)d);
    for (int i = 0; i < T * d; i++) x[i] *= xscale;
    ndbg("sub", 0, x, (size_t)T * d);
    // ---- relative positional encodings (int8, shared by all layers before linear_pos)
    const int NP = 2 * T - 1;
    C.Tp = (T + 15) & ~15;
    float *pe = (float *)tasr_alloc(sizeof(float) * (size_t)NP * d, 0);
    NB(tpe);
    for (int i = 0; i < d / 2; i++) {
        // row r holds position (T-1-r): step the angle by -div with a rotation, re-anchored every 64 rows
        const double div = exp((2.0 * i) * -(log(10000.0) / d));
        const float cd = (float)cos(div), sd = (float)sin(div);
        float sn = 0.f, cs = 1.f;
        for (int r = 0; r < NP; r++) {
            if ((r & 63) == 0) {
                const double a = fmod((double)(T - 1 - r) * div, 2.0 * M_PI);
                sn = (float)sin(a); cs = (float)cos(a);
            }
            pe[(size_t)r * d + 2 * i] = sn;
            pe[(size_t)r * d + 2 * i + 1] = cs;
            const float s2 = sn * cd - cs * sd, c2 = cs * cd + sn * sd;  // angle - div
            sn = s2; cs = c2;
        }
    }
    // the positional rows are the same input to every layer's linear_pos: quantize them to int8 once
    const int kpp = m->L[0].pos.kp;
    int8_t *peq = (int8_t *)tasr_alloc((size_t)NP * kpp, 0);
    float *pes = (float *)tasr_alloc(sizeof(float) * NP, 0);
    ndbg("pe", 0, pe, (size_t)NP * d);
    tasr_quant_rows(pe, NP, d, d, peq, kpp, kpp, pes);
    tasr_free(pe);
    NE(tpe, N_POS);
    C.qu = (int8_t *)tasr_alloc((size_t)H * T * dhp, 0);
    C.qv = (int8_t *)tasr_alloc((size_t)H * T * dhp, 0);
    C.k8 = (int8_t *)tasr_alloc((size_t)H * T * dhp, 0);
    C.vt = (int8_t *)tasr_alloc((size_t)H * dh * C.Tp, 0);
    C.p8 = (int8_t *)tasr_alloc((size_t)H * NP * dhp, 0);
    C.squ = (float *)tasr_alloc(sizeof(float) * H * T, 0);
    C.sqv = (float *)tasr_alloc(sizeof(float) * H * T, 0);
    C.sk = (float *)tasr_alloc(sizeof(float) * H * T, 0);
    C.sv = (float *)tasr_alloc(sizeof(float) * H * T, 0);
    C.sp = (float *)tasr_alloc(sizeof(float) * H * NP, 0);
    for (int w = 0; w < NW; w++) {
        C.sc[w] = (float *)tasr_alloc(sizeof(float) * C.Tp, 1);
        C.pq[w] = (int8_t *)tasr_alloc(C.Tp, 1);
        C.iacc_n = C.Tp > 64 ? C.Tp : 64;
        C.iacc[w] = (int32_t *)tasr_alloc(sizeof(int32_t) * 2 * C.iacc_n, 1);
    }
    float *hb = (float *)tasr_alloc(sizeof(float) * RB * d, 1);           // LN outputs (row block)
    const int w2 = 3 * d > m->ff ? 3 * d : m->ff;
    float *h2 = (float *)tasr_alloc(sizeof(float) * RB * w2, 0);
    const int halo = m->k / 2;
    float *glb = (float *)tasr_alloc(sizeof(float) * (size_t)(T + 2 * halo) * d, 0);  // GLU outputs with zero halo
    float *gl = glb + (size_t)halo * d;
    float *cv = (float *)tasr_alloc(sizeof(float) * (size_t)T * d, 0);
    C.att = cv;  // attention output is consumed by linear_out before the conv module writes cv
    float *prow = (float *)tasr_alloc(sizeof(float) * RB * d, 0);
    int8_t vtmp[64];
    for (int li = 0; li < m->nl; li++) {
        const nlayer_t *L = &m->L[li];
        // FF1 (half step), row blocks
        for (int t0 = 0; t0 < T; t0 += RB) {
            const int tn = T - t0 < RB ? T - t0 : RB;
            nln_rows(x + (size_t)t0 * d, tn, d, L->n_ff1, hb);
            if (t0 == 0) ndbg("f1ln", li, hb, (size_t)tn * d);
            nquant(&C, hb, tn, d, d, L->ff1_1.kp);
            if (t0 == 0) ndbg8("f1q", li, C.xq, tn, C.ldq, L->ff1_1.kp);
            nqlin(&C, &L->ff1_1, tn, h2, m->ff);
            if (t0 == 0) ndbg("f1a", li, h2, (size_t)tn * m->ff);
            nsilu(h2, tn * m->ff);
            if (t0 == 0) ndbg("f1s", li, h2, (size_t)tn * m->ff);
            nquant(&C, h2, tn, m->ff, m->ff, L->ff1_2.kp);
            if (t0 == 0) ndbg8("f1q2", li, C.xq, tn, C.ldq, L->ff1_2.kp);
            nqlin(&C, &L->ff1_2, tn, hb, d);
            if (t0 == 0) ndbg("f1b", li, hb, (size_t)tn * d);
            if (t0 == 0 && li == 1) ndump("f1b", li, hb, tn, d);
            for (int i = 0; i < tn * d; i++) x[(size_t)t0 * d + i] += 0.5f * hb[i];
        }
        ndbg("ff1", li, x, (size_t)T * d);
        // MHSA: q/k/v for all frames (int8 per head), positional rows, then attention per head
        for (int t0 = 0; t0 < T; t0 += RB) {
            const int tn = T - t0 < RB ? T - t0 : RB;
            nln_rows(x + (size_t)t0 * d, tn, d, L->n_att, hb);
            nquant(&C, hb, tn, d, d, L->qkv.kp);
            nqlin(&C, &L->qkv, tn, h2, 3 * d);
            NB(tq8);
            for (int t = 0; t < tn; t++) {
                const int ti = t0 + t;
                const float *q = h2 + (size_t)t * 3 * d, *k = q + d, *v = q + 2 * d;
                float tmp[64];
                for (int hh = 0; hh < H; hh++) {
                    for (int e = 0; e < dh; e++) tmp[e] = q[hh * dh + e] + L->pbu[hh * dh + e];
                    C.squ[hh * T + ti] = qvec(tmp, dh, C.qu + ((size_t)hh * T + ti) * dhp, dhp);
                    for (int e = 0; e < dh; e++) tmp[e] = q[hh * dh + e] + L->pbv[hh * dh + e];
                    C.sqv[hh * T + ti] = qvec(tmp, dh, C.qv + ((size_t)hh * T + ti) * dhp, dhp);
                    C.sk[hh * T + ti] = qvec(k + hh * dh, dh, C.k8 + ((size_t)hh * T + ti) * dhp, dhp);
                    C.sv[hh * T + ti] = qvec(v + hh * dh, dh, vtmp, dh);
                    int8_t *vt = C.vt + (size_t)hh * dh * C.Tp + ti;
                    for (int e = 0; e < dh; e++) vt[(size_t)e * C.Tp] = vtmp[e];
                }
            }
            NE(tq8, N_QKV8);
        }
        for (int hh = 0; hh < H; hh++)
            for (int e = 0; e < dh; e++)
                for (int t = T; t < C.Tp; t++) C.vt[((size_t)hh * dh + e) * C.Tp + t] = 0;
        for (int r0 = 0; r0 < NP; r0 += RB) {
            const int rn = NP - r0 < RB ? NP - r0 : RB;
            {
                int8_t *save = C.xq;
                float *save_s = C.xs;
                int ld = C.ldq;
                C.xq = peq + (size_t)r0 * kpp; C.ldq = kpp; C.xs = pes + r0;
                nqlin(&C, &L->pos, rn, prow, d);
                C.xq = save; C.ldq = ld; C.xs = save_s;
            }
            NB(tpq);
            for (int r = 0; r < rn; r++)
                for (int hh = 0; hh < H; hh++)
                    C.sp[hh * NP + r0 + r] = qvec(prow + (size_t)r * d + hh * dh, dh,
                                                  C.p8 + ((size_t)hh * NP + r0 + r) * dhp, dhp);
            NE(tpq, N_POS);
        }
        {
            NB(ta);
            natt_t aj = {&C};
            NPAR(7, natt_job, &aj, H);
            NE(ta, N_ATT);
        }
        for (int t0 = 0; t0 < T; t0 += RB) {
            const int tn = T - t0 < RB ? T - t0 : RB;
            nquant(&C, C.att + (size_t)t0 * d, tn, d, d, L->out.kp);
            nqlin(&C, &L->out, tn, hb, d);
            for (int i = 0; i < tn * d; i++) x[(size_t)t0 * d + i] += hb[i];
        }
        ndbg("att", li, x, (size_t)T * d);
        // conv module: LN -> pw1 -> GLU (all frames) -> dw k=31 (+BN) -> swish -> pw2
        for (int t0 = 0; t0 < T; t0 += RB) {
            const int tn = T - t0 < RB ? T - t0 : RB;
            nln_rows(x + (size_t)t0 * d, tn, d, L->n_conv, hb);
            nquant(&C, hb, tn, d, d, L->pw1.kp);
            nqlin(&C, &L->pw1, tn, h2, 2 * d);
            for (int t = 0; t < tn; t++) {
                const float *g = h2 + (size_t)t * 2 * d;
                float *o = gl + (size_t)(t0 + t) * d;
                for (int ch = 0; ch < d; ch++) o[ch] = g[ch] * nsig(g[d + ch]);
            }
        }
        {
            NB(td);
            ndw_t j = {glb, cv, L, T, d, m->k};
            NPAR(8, ndw_job, &j, T);
            NE(td, N_DW);
        }
        for (int t0 = 0; t0 < T; t0 += RB) {
            const int tn = T - t0 < RB ? T - t0 : RB;
            nquant(&C, cv + (size_t)t0 * d, tn, d, d, L->pw2.kp);
            nqlin(&C, &L->pw2, tn, hb, d);
            for (int i = 0; i < tn * d; i++) x[(size_t)t0 * d + i] += hb[i];
        }
        ndbg("cnv", li, x, (size_t)T * d);
        // FF2 (half step) + final LN
        for (int t0 = 0; t0 < T; t0 += RB) {
            const int tn = T - t0 < RB ? T - t0 : RB;
            nln_rows(x + (size_t)t0 * d, tn, d, L->n_ff2, hb);
            nquant(&C, hb, tn, d, d, L->ff2_1.kp);
            nqlin(&C, &L->ff2_1, tn, h2, m->ff);
            nsilu(h2, tn * m->ff);
            nquant(&C, h2, tn, m->ff, m->ff, L->ff2_2.kp);
            nqlin(&C, &L->ff2_2, tn, hb, d);
            for (int t = 0; t < tn; t++) {
                float *r = x + (size_t)(t0 + t) * d;
                for (int i = 0; i < d; i++) r[i] += 0.5f * hb[(size_t)t * d + i];
                layernorm_row(r, d, L->n_out, r);
            }
        }
        ndbg("layer", li, x, (size_t)T * d);
    }
    NB(thd);
    if (m->rnnt) {
        int len = 0;
        if (maxlen) text[0] = 0;
        rnnt_greedy(m, &C, x, T, text, maxlen, &len);
        NE(thd, N_HEAD);
        if (n_frames) *n_frames = T;
        if (maxlen && text[0] == ' ') memmove(text, text + 1, strlen(text));
        tasr_free(x); tasr_free(peq); tasr_free(pes); tasr_free(C.qu); tasr_free(C.qv); tasr_free(C.k8);
        tasr_free(C.vt); tasr_free(C.p8); tasr_free(C.squ); tasr_free(C.sqv); tasr_free(C.sk); tasr_free(C.sv);
        tasr_free(C.sp); tasr_free(hb); tasr_free(h2); tasr_free(glb); tasr_free(cv); tasr_free(prow);
        tasr_free(C.xq); tasr_free(C.xs);
        for (int w = 0; w < NW; w++) {
            tasr_free(C.wtmp[w]); tasr_free(C.acc[w]); tasr_free(C.sc[w]); tasr_free(C.pq[w]); tasr_free(C.iacc[w]);
        }
        return T;
    }
    // ---- CTC head (blank = V) -> greedy or beam decoder (which expects blank at index 0)
    const int V1 = m->V + 1;
    float *lg = (float *)tasr_alloc(sizeof(float) * RB * V1, 0);
    float *lp = (float *)tasr_alloc(sizeof(float) * V1, 0);
    int prev = -1, len = 0, nf = 0;
    if (dec) tasr_decoder_reset(dec);
    if (maxlen) text[0] = 0;
    for (int t0 = 0; t0 < T; t0 += RB) {
        const int tn = T - t0 < RB ? T - t0 : RB;
        nquant(&C, x + (size_t)t0 * d, tn, d, d, m->head.kp);
        nqlin(&C, &m->head, tn, lg, V1);
        for (int t = 0; t < tn; t++) {
            const float *l = lg + (size_t)t * V1;
            if (logit_sink && nf < max_frames) memcpy(logit_sink + (size_t)nf * V1, l, sizeof(float) * V1);
            nf++;
            int best = 0;
            float bv = l[0];
            for (int v = 1; v < V1; v++) if (l[v] > bv) { bv = l[v]; best = v; }
            if (dec) {
                float z = 0.f;
                for (int v = 0; v < V1; v++) z += expf(l[v] - bv);
                const float lz = bv + logf(z);
                lp[0] = l[m->V] - lz;
                for (int v = 0; v < m->V; v++) lp[v + 1] = l[v] - lz;
                tasr_decoder_step(dec, lp);
            } else if (best != prev && best != m->V && maxlen) {
                const int nl = m->tok_len[best];
                if (len + nl + 1 < maxlen) { memcpy(text + len, m->tok[best], nl); len += nl; text[len] = 0; }
            }
            prev = best;
        }
    }
    if (dec && maxlen) {
        int toks[2048];
        const int nt = tasr_decoder_best(dec, toks, 2048);
        for (int i = 0; i < nt; i++) {
            const int nl = m->tok_len[toks[i]];
            if (len + nl + 1 < maxlen) { memcpy(text + len, m->tok[toks[i]], nl); len += nl; text[len] = 0; }
        }
    }
    NE(thd, N_HEAD);
    if (n_frames) *n_frames = nf;
    if (maxlen && text[0] == ' ') memmove(text, text + 1, strlen(text));
    // free
    tasr_free(lg); tasr_free(lp); tasr_free(x); tasr_free(peq); tasr_free(pes); tasr_free(C.qu); tasr_free(C.qv); tasr_free(C.k8);
    tasr_free(C.vt); tasr_free(C.p8); tasr_free(C.squ); tasr_free(C.sqv); tasr_free(C.sk); tasr_free(C.sv);
    tasr_free(C.sp); tasr_free(hb); tasr_free(h2); tasr_free(glb); tasr_free(cv); tasr_free(prow);
    tasr_free(C.xq); tasr_free(C.xs);
    for (int w = 0; w < NW; w++) {
        tasr_free(C.wtmp[w]); tasr_free(C.acc[w]); tasr_free(C.sc[w]); tasr_free(C.pq[w]); tasr_free(C.iacc[w]);
    }
    return T;
}

// ================================================================== streaming
// For models trained with train_nemo_stream.py --stream. Audio is fed as it arrives; every C encoder frames (40 ms
// each) run through all layers with exactly the training masks:
//   - attention over the frames of the current chunk and of the previous nl chunks (int8 keys/values cached per layer),
//   - depthwise conv over cached past frames and the current chunk, zeros after the chunk's end,
//   - fixed (global) feature normalization.
// The transcript therefore does not depend on how the audio is split into feeds, and when the speaker stops only the
// last partial chunk is left to compute. Positional projections depend only on relative offsets, which are bounded,
// so they are computed once per stream object; caches have a fixed size whatever the utterance length.
#define SRING 4096  // sample ring (feeds are consumed in slices of SSLICE samples)
#define SSLICE 1600
#define SMEL 16     // mel row ring

struct tasr_nemo_stream {
    const tasr_nemo_t *m;
    int C, nl, Lc, Lcp, OMAX, NP, halo;
    int8_t *p8;   // [layers][H][NP][dhp] positional rows, row r <-> relative offset OMAX - r
    float *sp;    // [layers][H][NP]
    int8_t *k8;   // [layers][H][Lc][dhp]
    float *sk;    // [layers][H][Lc]
    int8_t *vt;   // [layers][H][dh][Lcp]
    float *sv;    // [layers][H][Lc]
    float *gh;    // [layers][halo + C + halo][d] conv inputs: past rows, chunk rows, zeros
    int kbase, nk;
    // front end
    int16_t ring[SRING];
    long n_samp;
    int mel_t, T0, T1, T, c0_next, t2_next, finished;
    float *mel;    // [SMEL][NMEL]
    float *c0ring, *cmaxr, *cmw;
    int8_t *col;
    float *c2out, *fr, *xs_sub;
    int8_t *xq_sub;
    int nbat, nflat, t_chunk;
    float *x;      // [C][d]
    // work
    nctx_t W;
    float *hb, *h2, *att, *prow, *squ, *sqv;
    int8_t *qu, *qv;
    float *scb[NW];
    int8_t *pq[NW];
    int32_t *ia[NW];
    int ia_n;
    // decoding
    tasr_decoder_t *dec;
    float *lg, *lp;
    int prev, len, frames;
    float *sink;
    int sink_max;
    char text[2048];
};

static inline size_t sidx(const struct tasr_nemo_stream *s, int li, int hh) { return (size_t)li * s->m->h + hh; }

// ---- mel frames: same arithmetic as nfeat_job; samples outside [0, nlim) are zero
typedef struct {
    struct tasr_nemo_stream *s;
    int t0;
    long nlim;
} smel_t;
static void smel_job(void *p, int b, int e, int w)
{
    (void)w;
    smel_t *j = (smel_t *)p;
    struct tasr_nemo_stream *s = j->s;
    const tasr_nemo_t *m = s->m;
    float xw[512], re[256], im[256], pw[NBIN];
    for (int k = b; k < e; k++) {
        const int t = j->t0 + k;
        for (int i = 0; i < WIN; i++) {
            const long si = (long)HOP * t - 200 + i;
            float v = 0.f;
            if (si >= 0 && si < j->nlim) {
                const float x0 = s->ring[si & (SRING - 1)] / 32768.0f;
                v = si == 0 ? x0 : x0 - 0.97f * (s->ring[(si - 1) & (SRING - 1)] / 32768.0f);
            }
            xw[i] = v * m->window[i];
        }
        for (int i = WIN; i < 512; i++) xw[i] = 0.f;
        power_spectrum512(m, xw, re, im, pw);
        float *o = s->mel + (size_t)(t % SMEL) * NMEL;
        const float *wm = m->mel_w;
        for (int q = 0; q < NMEL; q++) {
            float acc = 0.f;
            const float *pp = pw + m->mel_start[q];
            for (int r = 0; r < m->mel_len[q]; r++) acc += pp[r] * wm[r];
            wm += m->mel_len[q];
            o[q] = (logf(acc + 5.9604644775390625e-08f) - m->nmean[q]) * (1.0f / m->nstd[q]);
        }
    }
}

// ---- attention for the n rows of a chunk (job over heads)
typedef struct {
    struct tasr_nemo_stream *s;
    int li, n;
} satt_t;
static void satt_job(void *p, int b, int e, int w)
{
    satt_t *j = (satt_t *)p;
    struct tasr_nemo_stream *s = j->s;
    const tasr_nemo_t *m = s->m;
    const int dh = m->dh, dhp = m->dhp, d = m->d, C = s->C, nkeys = s->nk;  // keys incl. this chunk
    const float scale = 1.0f / sqrtf((float)dh);
    const int kp16 = (nkeys + 15) & ~15;
    float *sc = s->scb[w];
    int8_t *pq = s->pq[w];
    int32_t *ia = s->ia[w], *ib = ia + s->ia_n;
    for (int hh = b; hh < e; hh++) {
        const size_t g = sidx(s, j->li, hh);
        const int8_t *K = s->k8 + g * s->Lc * dhp, *VT = s->vt + g * dh * s->Lcp, *P = s->p8 + g * s->NP * dhp;
        const float *sk = s->sk + g * s->Lc, *sv = s->sv + g * s->Lc, *spp = s->sp + g * s->NP;
        for (int r = 0; r < j->n; r++) {
            const int i = s->t_chunk + r;
            const int m0 = s->OMAX - i + s->kbase;  // positional row of key 0 (offset i - kbase)
            const int8_t *qu = s->qu + ((size_t)hh * C + r) * dhp, *qv = s->qv + ((size_t)hh * C + r) * dhp;
            if (dhp == 48) {
                tasr_dot48_rows(qu, K, 48, nkeys, ia);
                tasr_dot48_rows(qv, P + (size_t)m0 * 48, 48, nkeys, ib);
            } else {
                tasr_dot_rows_s8(qu, K, dhp, nkeys, dhp, ia);
                tasr_dot_rows_s8(qv, P + (size_t)m0 * dhp, dhp, nkeys, dhp, ib);
            }
            const float A = s->squ[hh * C + r] * scale, B = s->sqv[hh * C + r] * scale;
            const float *spm = spp + m0;
            float mx = -1e30f;
            for (int k = 0; k < nkeys; k++) {
                const float a = (float)ia[k] * A * sk[k] + (float)ib[k] * B * spm[k];
                sc[k] = a;
                mx = a > mx ? a : mx;
            }
            float sum = 0.f, pm = 0.f;
            const float mx64 = mx * NEXP_STEPS + 0.5f;
            for (int k = 0; k < nkeys; k++) {
                const int ix = (int)(mx64 - sc[k] * NEXP_STEPS);
                const float ex = ix < NEXP_N ? nexp_tab[ix] : 0.f;
                sum += ex;
                const float p2 = ex * sv[k];
                sc[k] = p2;
                pm = p2 > pm ? p2 : pm;
            }
            if (pm < 1e-30f) pm = 1e-30f;
            const float inv = 127.0f / pm;
            for (int k = 0; k < nkeys; k++) { float y = sc[k] * inv + 12582912.0f; pq[k] = (int8_t)(int)(y - 12582912.0f); }
            for (int k = nkeys; k < kp16; k++) pq[k] = 0;
            tasr_dot_rows_s8(pq, VT, s->Lcp, dh, kp16, ia);
            const float os = pm / 127.0f / sum;
            float *y = s->att + (size_t)r * d + hh * dh;
            for (int e2 = 0; e2 < dh; e2++) y[e2] = (float)ia[e2] * os;
        }
    }
}

static void s_qlin_rows(struct tasr_nemo_stream *s, const tasr_qlin_t *L, const float *x, int n, int K, float *y, int ldy)
{
    nquant(&s->W, x, n, K, K, L->kp);
    nqlin(&s->W, L, n, y, ldy);
}

static void s_emit(struct tasr_nemo_stream *s, const float *l)
{
    const tasr_nemo_t *m = s->m;
    const int V1 = m->V + 1;
    if (s->sink && s->frames < s->sink_max) memcpy(s->sink + (size_t)s->frames * V1, l, sizeof(float) * V1);
    int best = 0;
    float bv = l[0];
    for (int v = 1; v < V1; v++) if (l[v] > bv) { bv = l[v]; best = v; }
    if (s->dec) {
        float z = 0.f;
        for (int v = 0; v < V1; v++) z += expf(l[v] - bv);
        const float lz = bv + logf(z);
        s->lp[0] = l[m->V] - lz;
        for (int v = 0; v < m->V; v++) s->lp[v + 1] = l[v] - lz;
        tasr_decoder_step(s->dec, s->lp);
    } else if (best != s->prev && best != m->V) {
        const int nl = m->tok_len[best];
        if (s->len + nl + 1 < (int)sizeof(s->text)) { memcpy(s->text + s->len, m->tok[best], nl); s->len += nl; s->text[s->len] = 0; }
    }
    s->prev = best;
    s->frames++;
}

// ---- one chunk of n <= C encoder rows (s->x, absolute index s->t_chunk) through all layers + CTC head
static void s_encode_chunk(struct tasr_nemo_stream *s, int n)
{
    const tasr_nemo_t *m = s->m;
    const int d = m->d, H = m->h, dh = m->dh, dhp = m->dhp, C = s->C, halo = s->halo;
    // slide the caches: keep the previous nl chunks
    const int base = s->t_chunk / C > s->nl ? (s->t_chunk / C - s->nl) * C : 0;
    const int sh = base - s->kbase;
    if (sh > 0) {
        const int keep = s->nk - sh;
        for (int li = 0; li < m->nl; li++)
            for (int hh = 0; hh < H; hh++) {
                const size_t g = sidx(s, li, hh);
                memmove(s->k8 + g * s->Lc * dhp, s->k8 + (g * s->Lc + sh) * dhp, (size_t)keep * dhp);
                memmove(s->sk + g * s->Lc, s->sk + g * s->Lc + sh, sizeof(float) * keep);
                memmove(s->sv + g * s->Lc, s->sv + g * s->Lc + sh, sizeof(float) * keep);
                for (int e2 = 0; e2 < dh; e2++) {
                    int8_t *row = s->vt + (g * dh + e2) * s->Lcp;
                    memmove(row, row + sh, keep);
                }
            }
        s->kbase = base;
        s->nk = keep;
    }
    const int k0 = s->nk;  // cache row of this chunk's first frame
    float *x = s->x;
    int8_t vtmp[64];
    for (int li = 0; li < m->nl; li++) {
        const nlayer_t *L = &m->L[li];
        // FF1
        nln_rows(x, n, d, L->n_ff1, s->hb);
        s_qlin_rows(s, &L->ff1_1, s->hb, n, d, s->h2, m->ff);
        nsilu(s->h2, n * m->ff);
        s_qlin_rows(s, &L->ff1_2, s->h2, n, m->ff, s->hb, d);
        for (int i = 0; i < n * d; i++) x[i] += 0.5f * s->hb[i];
        // MHSA: q (+u/+v) for the chunk, k/v appended to the cache
        nln_rows(x, n, d, L->n_att, s->hb);
        s_qlin_rows(s, &L->qkv, s->hb, n, d, s->h2, 3 * d);
        NB(tq8);
        for (int t = 0; t < n; t++) {
            const float *q = s->h2 + (size_t)t * 3 * d, *k = q + d, *v = q + 2 * d;
            float tmp[64];
            for (int hh = 0; hh < H; hh++) {
                const size_t g = sidx(s, li, hh);
                for (int e2 = 0; e2 < dh; e2++) tmp[e2] = q[hh * dh + e2] + L->pbu[hh * dh + e2];
                s->squ[hh * C + t] = qvec(tmp, dh, s->qu + ((size_t)hh * C + t) * dhp, dhp);
                for (int e2 = 0; e2 < dh; e2++) tmp[e2] = q[hh * dh + e2] + L->pbv[hh * dh + e2];
                s->sqv[hh * C + t] = qvec(tmp, dh, s->qv + ((size_t)hh * C + t) * dhp, dhp);
                s->sk[g * s->Lc + k0 + t] = qvec(k + hh * dh, dh, s->k8 + (g * s->Lc + k0 + t) * dhp, dhp);
                s->sv[g * s->Lc + k0 + t] = qvec(v + hh * dh, dh, vtmp, dh);
                for (int e2 = 0; e2 < dh; e2++) s->vt[(g * dh + e2) * s->Lcp + k0 + t] = vtmp[e2];
            }
        }
        NE(tq8, N_QKV8);
        {
            const int nk_save = s->nk;
            s->nk = k0 + n;
            NB(ta);
            satt_t aj = {s, li, n};
            tasr_parallel(satt_job, &aj, H);
            NE(ta, N_ATT);
            s->nk = nk_save;
        }
        s_qlin_rows(s, &L->out, s->att, n, d, s->hb, d);
        for (int i = 0; i < n * d; i++) x[i] += s->hb[i];
        // conv module: GLU rows after the cached past rows, zeros after the chunk
        float *gh = s->gh + (size_t)li * (2 * halo + C) * d;
        nln_rows(x, n, d, L->n_conv, s->hb);
        s_qlin_rows(s, &L->pw1, s->hb, n, d, s->h2, 2 * d);
        for (int t = 0; t < n; t++) {
            const float *g = s->h2 + (size_t)t * 2 * d;
            float *o = gh + (size_t)(halo + t) * d;
            for (int ch = 0; ch < d; ch++) o[ch] = g[ch] * nsig(g[d + ch]);
        }
        memset(gh + (size_t)(halo + n) * d, 0, sizeof(float) * halo * d);
        {
            NB(td);
            ndw_t jd = {gh, s->att, L, n, d, m->k};  // att is free again: use it for the conv output
            tasr_parallel(ndw_job, &jd, n);
            NE(td, N_DW);
        }
        memmove(gh, gh + (size_t)n * d, sizeof(float) * halo * d);  // last halo inputs become the past rows
        s_qlin_rows(s, &L->pw2, s->att, n, d, s->hb, d);
        for (int i = 0; i < n * d; i++) x[i] += s->hb[i];
        // FF2 + final LN
        nln_rows(x, n, d, L->n_ff2, s->hb);
        s_qlin_rows(s, &L->ff2_1, s->hb, n, d, s->h2, m->ff);
        nsilu(s->h2, n * m->ff);
        s_qlin_rows(s, &L->ff2_2, s->h2, n, m->ff, s->hb, d);
        for (int t = 0; t < n; t++) {
            float *r = x + (size_t)t * d;
            for (int i = 0; i < d; i++) r[i] += 0.5f * s->hb[(size_t)t * d + i];
            layernorm_row(r, d, L->n_out, r);
        }
    }
    s->nk = k0 + n;
    NB(thd);
    s_qlin_rows(s, &m->head, x, n, d, s->lg, m->V + 1);
    for (int t = 0; t < n; t++) s_emit(s, s->lg + (size_t)t * (m->V + 1));
    NE(thd, N_HEAD);
    s->t_chunk += n;
}

// ---- front end: encoder frame t2 from conv0 rows 2t2-1..2t2+1 (conv2 batched, sub_out per chunk)
static const float *s_melrow(const struct tasr_nemo_stream *s, int t)
{
    if (t < 0 || (s->finished && t >= s->T0 - 1)) return NULL;  // row T0-1 (= n/160) is the zero frame
    return s->mel + (size_t)(t % SMEL) * NMEL;
}
static void s_conv0_row(struct tasr_nemo_stream *s, int r)
{
    const tasr_nemo_t *m = s->m;
    const int sc = m->sc, f1 = m->f1;
    NB(tc0);
    float P[3][NMEL + 2];
    for (int dt = 0; dt < 3; dt++) {
        const float *row = s_melrow(s, 2 * r - 1 + dt);
        P[dt][0] = P[dt][NMEL + 1] = 0.f;
        if (row) memcpy(&P[dt][1], row, sizeof(float) * NMEL);
        else memset(&P[dt][1], 0, sizeof(float) * NMEL);
    }
    nc0_t j = {m, (const float(*)[NMEL + 2])P, s->c0ring + (size_t)(r % 3) * sc * (f1 + 2), s->cmw};
    memset(s->cmw, 0, sizeof(float) * NW * (f1 + 2));
    tasr_parallel(nc0_job, &j, sc);
    float *cm = s->cmaxr + (size_t)(r % 3) * (f1 + 2);
    for (int f = 0; f < f1 + 2; f++) {
        float v = s->cmw[f];
        for (int w = 1; w < NW; w++) v = s->cmw[(size_t)w * (f1 + 2) + f] > v ? s->cmw[(size_t)w * (f1 + 2) + f] : v;
        cm[f] = v;
    }
    NE(tc0, N_CONV0);
}
static void s_flush_conv2(struct tasr_nemo_stream *s)
{
    const tasr_nemo_t *m = s->m;
    const int sc = m->sc, f2 = m->f2;
    if (!s->nbat) return;
    {
        int8_t *save = s->W.xq;
        int ld = s->W.ldq;
        s->W.xq = s->col; s->W.ldq = m->c2.kp;
        nqlin(&s->W, &m->c2, s->nbat * f2, s->c2out, sc);
        s->W.xq = save; s->W.ldq = ld;
    }
    for (int fb = 0; fb < s->nbat; fb++) {
        const float *co = s->c2out + (size_t)fb * f2 * sc;
        for (int f = 0; f < f2; f++)
            for (int ch = 0; ch < sc; ch++) {
                const float v = co[f * sc + ch];
                s->fr[ch * f2 + f] = v > 0.f ? v : 0.f;
            }
        NB(tq);
        tasr_quant_rows(s->fr, 1, sc * f2, sc * f2, s->xq_sub + (size_t)s->nflat * m->sub.kp, m->sub.kp, m->sub.kp,
                        s->xs_sub + s->nflat);
        NE(tq, N_QUANT);
        s->nflat++;
    }
    s->nbat = 0;
}
static void s_run_chunk(struct tasr_nemo_stream *s)
{
    const tasr_nemo_t *m = s->m;
    if (!s->nflat) return;
    int8_t *save = s->W.xq;
    float *save_s = s->W.xs;
    int ld = s->W.ldq;
    s->W.xq = s->xq_sub; s->W.ldq = m->sub.kp; s->W.xs = s->xs_sub;
    nqlin(&s->W, &m->sub, s->nflat, s->x, m->d);
    s->W.xq = save; s->W.ldq = ld; s->W.xs = save_s;
    const float xscale = sqrtf((float)m->d);
    for (int i = 0; i < s->nflat * m->d; i++) s->x[i] *= xscale;
    const int n = s->nflat;
    s->nflat = 0;
    s_encode_chunk(s, n);
}
// advance the front end as far as the available audio allows (or to the end after finish)
static void s_advance(struct tasr_nemo_stream *s)
{
    const tasr_nemo_t *m = s->m;
    const int f2 = m->f2;
    // mel frames whose samples are all present
    {
        int tend = s->finished ? s->T0 - 1 : (int)((s->n_samp - 200) / HOP) + 1;  // frames [mel_t, tend)
        if (!s->finished && s->n_samp < 200) tend = 0;
        if (tend > s->mel_t) {
            NB(tf);
            smel_t j = {s, s->mel_t, s->n_samp};
            tasr_parallel(smel_job, &j, tend - s->mel_t);
            s->mel_t = tend;
            NE(tf, N_FEAT);
        }
    }
    for (;;) {
        const int t2 = s->t2_next;
        if (s->finished ? t2 >= s->T : (4 * t2 + 3 >= s->mel_t)) break;  // needs mel rows up to 4 t2 + 3
        for (int r = 2 * t2 - 1; r <= 2 * t2 + 1; r++) {
            if (r < 0 || r < s->c0_next || (s->finished && r >= s->T1)) continue;
            s_conv0_row(s, r);
            s->c0_next = r + 1;
        }
        NB(tim);
        {
            nim_t j;
            j.m = m;
            j.col = s->col + (size_t)s->nbat * f2 * m->c2.kp;
            j.xs = s->W.xs + s->nbat * f2;
            for (int dt = 0; dt < 3; dt++) {
                const int r = 2 * t2 - 1 + dt;
                j.rows[dt] = (r >= 0 && !(s->finished && r >= s->T1)) ? s->c0ring + (size_t)(r % 3) * m->sc * (m->f1 + 2) : NULL;
                j.cmax[dt] = s->cmaxr + (size_t)((r + 3) % 3) * (m->f1 + 2);
            }
            tasr_parallel(nim_job, &j, f2);
        }
        NE(tim, N_IM2COL);
        s->t2_next++;
        const int C2B = RB / f2 < 1 ? 1 : RB / f2;
        if (++s->nbat == C2B || s->nflat + s->nbat == s->C) s_flush_conv2(s);
        if (s->nflat == s->C) s_run_chunk(s);
    }
}

tasr_nemo_stream_t *tasr_nemo_stream_new(const tasr_nemo_t *m, int chunk, int left, tasr_decoder_t *dec)
{
    if (!m || !m->nmean || m->rnnt) return NULL;  // needs a streaming-trained CTC model (fixed normalization)
    nsig_init();
    nexp_init();
    struct tasr_nemo_stream *s = (struct tasr_nemo_stream *)tasr_alloc(sizeof(*s), 0);
    if (!s) return NULL;
    memset(s, 0, sizeof(*s));
    const int d = m->d, H = m->h, dh = m->dh, dhp = m->dhp, nly = m->nl, f1 = m->f1, f2 = m->f2, sc = m->sc;
    s->m = m;
    s->C = chunk > 0 ? chunk : (m->s_chunk > 0 ? m->s_chunk : 16);
    if (s->C > RB) s->C = RB;
    const int lf = left > 0 ? left : (m->s_left > 0 ? m->s_left : 128);
    s->nl = (lf + s->C - 1) / s->C;
    s->Lc = (s->nl + 1) * s->C;
    s->Lcp = (s->Lc + 15) & ~15;
    s->OMAX = s->nl * s->C + s->C - 1;
    s->NP = s->OMAX + s->C;
    s->halo = m->k / 2;
    s->dec = dec;
    s->p8 = (int8_t *)tasr_alloc((size_t)nly * H * s->NP * dhp, 0);
    s->sp = (float *)tasr_alloc(sizeof(float) * nly * H * s->NP, 0);
    s->k8 = (int8_t *)tasr_alloc((size_t)nly * H * s->Lc * dhp, 0);
    s->sk = (float *)tasr_alloc(sizeof(float) * nly * H * s->Lc, 0);
    s->vt = (int8_t *)tasr_alloc((size_t)nly * H * dh * s->Lcp, 0);
    s->sv = (float *)tasr_alloc(sizeof(float) * nly * H * s->Lc, 0);
    s->gh = (float *)tasr_alloc(sizeof(float) * nly * (2 * s->halo + s->C) * d, 0);
    s->mel = (float *)tasr_alloc(sizeof(float) * SMEL * NMEL, 1);
    s->c0ring = (float *)tasr_alloc(sizeof(float) * 3 * sc * (f1 + 2), 0);
    s->cmaxr = (float *)tasr_alloc(sizeof(float) * 3 * (f1 + 2), 1);
    s->cmw = (float *)tasr_alloc(sizeof(float) * NW * (f1 + 2), 1);
    const int C2B = RB / f2 < 1 ? 1 : RB / f2;
    s->col = (int8_t *)tasr_alloc((size_t)C2B * f2 * m->c2.kp, 0);
    s->c2out = (float *)tasr_alloc(sizeof(float) * C2B * f2 * sc, 0);
    s->fr = (float *)tasr_alloc(sizeof(float) * sc * f2, 0);
    s->xq_sub = (int8_t *)tasr_alloc((size_t)s->C * m->sub.kp, 0);
    s->xs_sub = (float *)tasr_alloc(sizeof(float) * RB, 1);
    s->x = (float *)tasr_alloc(sizeof(float) * s->C * d, 0);
    s->W.m = m;
    s->W.ldq = m->ff > 2 * d ? m->ff : 2 * d;
    s->W.xq = (int8_t *)tasr_alloc((size_t)RB * s->W.ldq, 1);
    s->W.xs = (float *)tasr_alloc(sizeof(float) * RB, 1);
    for (int w = 0; w < NW; w++) {
        s->W.wtmp[w] = (int8_t *)tasr_alloc(16 * s->W.ldq + 16, 1);
        s->W.acc[w] = (int32_t *)tasr_alloc(sizeof(int32_t) * 64 * 16, 1);
    }
    s->hb = (float *)tasr_alloc(sizeof(float) * RB * d, 1);
    const int w2 = 3 * d > m->ff ? 3 * d : m->ff;
    s->h2 = (float *)tasr_alloc(sizeof(float) * RB * w2, 0);
    s->att = (float *)tasr_alloc(sizeof(float) * s->C * d, 0);
    s->prow = (float *)tasr_alloc(sizeof(float) * RB * d, 0);
    s->qu = (int8_t *)tasr_alloc((size_t)H * s->C * dhp, 0);
    s->qv = (int8_t *)tasr_alloc((size_t)H * s->C * dhp, 0);
    s->squ = (float *)tasr_alloc(sizeof(float) * H * s->C, 0);
    s->sqv = (float *)tasr_alloc(sizeof(float) * H * s->C, 0);
    s->ia_n = s->Lcp > 64 ? s->Lcp : 64;
    for (int w = 0; w < NW; w++) {
        s->scb[w] = (float *)tasr_alloc(sizeof(float) * s->Lcp, 1);
        s->pq[w] = (int8_t *)tasr_alloc(s->Lcp, 1);
        s->ia[w] = (int32_t *)tasr_alloc(sizeof(int32_t) * 2 * s->ia_n, 1);
    }
    s->lg = (float *)tasr_alloc(sizeof(float) * s->C * (m->V + 1), 0);
    s->lp = (float *)tasr_alloc(sizeof(float) * (m->V + 1), 0);
    if (!s->p8 || !s->k8 || !s->vt || !s->gh || !s->h2 || !s->lg || !s->W.xq) { tasr_nemo_stream_free(s); return NULL; }
    // positional rows for offsets OMAX..-(C-1), int8 once, then every layer's linear_pos per head
    const int kpp = m->L[0].pos.kp;
    float *pe = (float *)tasr_alloc(sizeof(float) * RB * d, 0);
    for (int r0 = 0; r0 < s->NP; r0 += RB) {
        const int rn = s->NP - r0 < RB ? s->NP - r0 : RB;
        for (int r = 0; r < rn; r++) {
            const double off = (double)(s->OMAX - (r0 + r));
            for (int i = 0; i < d / 2; i++) {
                const double a = off * exp((2.0 * i) * -(log(10000.0) / d));
                pe[(size_t)r * d + 2 * i] = (float)sin(a);
                pe[(size_t)r * d + 2 * i + 1] = (float)cos(a);
            }
        }
        tasr_quant_rows(pe, rn, d, d, s->W.xq, kpp, kpp, s->W.xs);
        const int ld = s->W.ldq;
        s->W.ldq = kpp;
        for (int li = 0; li < nly; li++) {
            nqlin(&s->W, &m->L[li].pos, rn, s->prow, d);
            for (int r = 0; r < rn; r++)
                for (int hh = 0; hh < H; hh++) {
                    const size_t g = sidx(s, li, hh);
                    s->sp[g * s->NP + r0 + r] = qvec(s->prow + (size_t)r * d + hh * dh, dh, s->p8 + (g * s->NP + r0 + r) * dhp, dhp);
                }
        }
        s->W.ldq = ld;
    }
    tasr_free(pe);
    tasr_nemo_stream_reset(s);
    return s;
}

void tasr_nemo_stream_reset(tasr_nemo_stream_t *s)
{
    const tasr_nemo_t *m = s->m;
    s->n_samp = 0;
    s->mel_t = s->c0_next = s->t2_next = s->finished = 0;
    s->T0 = s->T1 = s->T = 0;
    s->nbat = s->nflat = s->t_chunk = s->kbase = s->nk = 0;
    s->prev = -1; s->len = 0; s->frames = 0; s->text[0] = 0;
    memset(s->gh, 0, sizeof(float) * m->nl * (2 * s->halo + s->C) * m->d);
    memset(s->vt, 0, (size_t)m->nl * m->h * m->dh * s->Lcp);
    if (s->dec) tasr_decoder_reset(s->dec);
}

int tasr_nemo_stream_feed(tasr_nemo_stream_t *s, const int16_t *pcm, int n)
{
    if (s->finished) return s->frames;
    for (int o = 0; o < n; o += SSLICE) {
        const int k = n - o < SSLICE ? n - o : SSLICE;
        for (int i = 0; i < k; i++) s->ring[(s->n_samp + i) & (SRING - 1)] = pcm[o + i];
        s->n_samp += k;
        s_advance(s);
    }
    return s->frames;
}

static void s_text(tasr_nemo_stream_t *s, char *text, int maxlen)
{
    const tasr_nemo_t *m = s->m;
    if (!maxlen) return;
    text[0] = 0;
    int len = 0;
    if (s->dec) {
        int toks[2048];
        const int nt = tasr_decoder_best(s->dec, toks, 2048);
        for (int i = 0; i < nt; i++) {
            const int nl = m->tok_len[toks[i]];
            if (len + nl + 1 < maxlen) { memcpy(text + len, m->tok[toks[i]], nl); len += nl; text[len] = 0; }
        }
    } else {
        len = s->len < maxlen - 1 ? s->len : maxlen - 1;
        memcpy(text, s->text, len);
        text[len] = 0;
    }
    if (text[0] == ' ') memmove(text, text + 1, strlen(text));
}

int tasr_nemo_stream_text(tasr_nemo_stream_t *s, char *text, int maxlen)
{
    s_text(s, text, maxlen);
    return s->frames;
}

int tasr_nemo_stream_finish(tasr_nemo_stream_t *s, char *text, int maxlen)
{
    if (!s->finished) {
        if (s->n_samp < 2 * HOP) {  // too short for the model: no frames (like tasr_nemo_transcribe)
            s->finished = 1;
            if (maxlen) text[0] = 0;
            return 0;
        }
        s->finished = 1;
        s->T0 = (int)(s->n_samp / HOP) + 1;
        s->T1 = (s->T0 - 1) / 2 + 1;
        s->T = (s->T1 - 1) / 2 + 1;
        s_advance(s);
        s_flush_conv2(s);
        s_run_chunk(s);
    }
    s_text(s, text, maxlen);
    return s->frames;
}

void tasr_nemo_stream_free(tasr_nemo_stream_t *s)
{
    if (!s) return;
    void *p[] = {s->p8, s->sp, s->k8, s->sk, s->vt, s->sv, s->gh, s->mel, s->c0ring, s->cmaxr, s->cmw, s->col, s->c2out,
                 s->fr, s->xq_sub, s->xs_sub, s->x, s->W.xq, s->W.xs, s->hb, s->h2, s->att, s->prow, s->qu, s->qv,
                 s->squ, s->sqv, s->lg, s->lp};
    for (size_t i = 0; i < sizeof(p) / sizeof(p[0]); i++) if (p[i]) tasr_free(p[i]);
    for (int w = 0; w < NW; w++) {
        if (s->W.wtmp[w]) tasr_free(s->W.wtmp[w]);
        if (s->W.acc[w]) tasr_free(s->W.acc[w]);
        if (s->scb[w]) tasr_free(s->scb[w]);
        if (s->pq[w]) tasr_free(s->pq[w]);
        if (s->ia[w]) tasr_free(s->ia[w]);
    }
    tasr_free(s);
}

void tasr_nemo_stream_set_sink(tasr_nemo_stream_t *s, float *logits, int max_frames)
{
    s->sink = logits;
    s->sink_max = max_frames;
}
int tasr_nemo_stream_supported(const tasr_nemo_t *m) { return m && m->nmean && !m->rnnt; }
