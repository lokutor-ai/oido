#include "kernels.h"
#include <math.h>
#include <string.h>
#if defined(CONFIG_TASR_EXP_DOT_DUP)
#include "esp_cpu.h"
#endif
#if defined(CONFIG_TASR_EXP_DOT_IRAM)
#include "esp_attr.h"
#define TASR_DOT_ATTR IRAM_ATTR
#else
#define TASR_DOT_ATTR
#endif

#if defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32S3) && !defined(TASR_NO_SIMD)
#define TASR_PIE 1
#endif

#define WPF_BLK 16384  // bytes of weights fetched ahead per block
#ifdef TASR_PIE
// Weight preload: the ESP32-S3 D-cache has a ROM-driven manual preload engine that runs in the background (it reads
// external memory at the full bus rate, 89 MB/s from PSRAM, without slowing the CPU). While a block of weights is being
// multiplied, the next block is already on its way into the cache, so a layer takes max(compute, fetch) instead of
// compute + fetch. Measured on silicon in tools/gemm_bench (a 32 KB preload under a 115k-cycle kernel was free).
#include "esp32s3/rom/cache.h"
#ifdef CONFIG_TASR_EXP_NO_WPF
int tasr_wpf = 0;
#else
int tasr_wpf = 1;
#endif
static inline int wpf_ext(const void *p)
{
    const uintptr_t a = (uintptr_t)p;
    return a >= 0x3C000000u && a < 0x3E000000u;  // external memory (PSRAM, flash) on the data bus
}
static inline void wpf_start(const void *p, size_t n, int *busy)
{
    if (!tasr_wpf || !n || !wpf_ext(p)) return;
    const uintptr_t a = (uintptr_t)p & ~(uintptr_t)63;
    size_t sz = (((uintptr_t)p + n + 63) & ~(uintptr_t)63) - a;
    if (sz > 32768) sz = 32768;
    Cache_Start_DCache_Preload((uint32_t)a, (uint32_t)sz, 0);
    *busy = 1;
}
static inline void wpf_wait(int *busy)
{
    if (!*busy) return;
    while (!Cache_DCache_Preload_Done()) {}
    Cache_End_DCache_Preload(0);
    *busy = 0;
}
#else
int tasr_wpf = 0;
#define wpf_start(p, n, busy) ((void)(p), (void)(n), (void)(busy))
#define wpf_wait(busy) ((void)(busy))
#endif

// round-to-nearest-even via the 1.5*2^23 trick (matches torch.round), valid for |x| < 2^22
static inline int rne(float x)
{
    float y = x + 12582912.0f;  // not folded without -ffast-math
    return (int)(y - 12582912.0f);
}

#ifdef TASR_PIE
// ESP32-S3 PIE: 16 int8 MACs per instruction into the 40-bit ACCX accumulator.
// The asm uses the zero-overhead loop registers; the enclosing C loop must not become a hardware loop,
// hence no-branch-count-reg (disables GCC's doloop / zero-overhead loop generation for this function).
#if defined(CONFIG_TASR_EXP_DOT_DUP)
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_dot_rows_s8_core0(const int8_t *w, const int8_t *x, int ldq,
                                                                                 int T, int kp, int32_t *out)
{
    const int n = kp >> 4;
    for (int t = 0; t < T; t++) {
        const int8_t *pa = x + (size_t)t * ldq, *pb = w;
        int m = n;
        int32_t r;
        if (m & 1) {
            __asm__ volatile(
                "ee.zero.accx\n"
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "ee.vmulas.s8.accx q0, q1\n"
                : "+r"(pa), "+r"(pb) : : "memory");
            m -= 1;
        } else {
            __asm__ volatile("ee.zero.accx\n" ::: "memory");
        }
        if (m) {
            int pairs = m >> 1;
            __asm__ volatile(
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "addi %2, %2, -1\n"
                "loopnez %2, 1f\n"
                "  ee.vld.128.ip q2, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "  ee.vld.128.ip q0, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %1, 16, q2, q3\n"
                "1:\n"
                "ee.vld.128.ip q2, %0, 16\n"
                "ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "ee.vmulas.s8.accx q2, q3\n"
                : "+r"(pa), "+r"(pb), "+r"(pairs) : : "memory");
        }
        __asm__ volatile("rur.accx_0 %0\n" : "=r"(r));
        out[t] = r;
    }
}
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_dot_rows_s8_core1(const int8_t *w, const int8_t *x, int ldq,
                                                                                 int T, int kp, int32_t *out)
{
    const int n = kp >> 4;
    for (int t = 0; t < T; t++) {
        const int8_t *pa = x + (size_t)t * ldq, *pb = w;
        int m = n;
        int32_t r;
        if (m & 1) {
            __asm__ volatile(
                "ee.zero.accx\n"
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "ee.vmulas.s8.accx q0, q1\n"
                : "+r"(pa), "+r"(pb) : : "memory");
            m -= 1;
        } else {
            __asm__ volatile("ee.zero.accx\n" ::: "memory");
        }
        if (m) {
            int pairs = m >> 1;
            __asm__ volatile(
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "addi %2, %2, -1\n"
                "loopnez %2, 1f\n"
                "  ee.vld.128.ip q2, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "  ee.vld.128.ip q0, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %1, 16, q2, q3\n"
                "1:\n"
                "ee.vld.128.ip q2, %0, 16\n"
                "ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "ee.vmulas.s8.accx q2, q3\n"
                : "+r"(pa), "+r"(pb), "+r"(pairs) : : "memory");
        }
        __asm__ volatile("rur.accx_0 %0\n" : "=r"(r));
        out[t] = r;
    }
}

#if defined(CONFIG_TASR_EXP_DOT_DUP)
// experiment: each core runs its own copy of the kernel (different instruction addresses)
void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out)
{
    if (esp_cpu_get_core_id()) tasr_dot_rows_s8_core1(w, x, ldq, T, kp, out);
    else tasr_dot_rows_s8_core0(w, x, ldq, T, kp, out);
}
#endif
#else
TASR_DOT_ATTR __attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq,
                                                                                 int T, int kp, int32_t *out)
{
    const int n = kp >> 4;
    for (int t = 0; t < T; t++) {
        const int8_t *pa = x + (size_t)t * ldq, *pb = w;
        int m = n;
        int32_t r;
        if (m & 1) {
            __asm__ volatile(
                "ee.zero.accx\n"
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "ee.vmulas.s8.accx q0, q1\n"
                : "+r"(pa), "+r"(pb) : : "memory");
            m -= 1;
        } else {
            __asm__ volatile("ee.zero.accx\n" ::: "memory");
        }
        if (m) {
            int pairs = m >> 1;
            __asm__ volatile(
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "addi %2, %2, -1\n"
                "loopnez %2, 1f\n"
                "  ee.vld.128.ip q2, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "  ee.vld.128.ip q0, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %1, 16, q2, q3\n"
                "1:\n"
                "ee.vld.128.ip q2, %0, 16\n"
                "ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "ee.vmulas.s8.accx q2, q3\n"
                : "+r"(pa), "+r"(pb), "+r"(pairs) : : "memory");
        }
        __asm__ volatile("rur.accx_0 %0\n" : "=r"(r));
        out[t] = r;
    }
}
#endif
#else
void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out)
{
    for (int t = 0; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldq;
        int32_t acc = 0;
        for (int i = 0; i < kp; i++) acc += (int32_t)a[i] * (int32_t)w[i];
        out[t] = acc;
    }
}
#endif

int32_t tasr_dot_s8(const int8_t *a, const int8_t *b, int kp)
{
    int32_t r;
    tasr_dot_rows_s8(b, a, kp, 1, kp, &r);
    return r;
}

#ifdef TASR_PIE
static const uint8_t k_mask0f[16] __attribute__((aligned(16))) = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
static const int8_t k_eight[16] __attribute__((aligned(16))) = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
// 32 weights per iteration: lo nibbles -> w[j], hi nibbles -> w[j+16]; offset-binary (+8) removed with a saturating sub
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_unpack_s4(const uint8_t *src, int8_t *dst, int kp)
{
    int n = kp >> 5;
    const uint8_t *mk = k_mask0f;
    const int8_t *e8 = k_eight;
    __asm__ volatile(
        "ee.vld.128.ip q6, %3, 0\n"
        "ee.vld.128.ip q7, %4, 0\n"
        "ssai 4\n"
        "loopnez %2, 1f\n"
        "  ee.vld.128.ip q0, %0, 16\n"
        "  ee.andq q1, q0, q6\n"
        "  ee.vsr.32 q2, q0\n"
        "  ee.andq q2, q2, q6\n"
        "  ee.vsubs.s8 q1, q1, q7\n"
        "  ee.vsubs.s8 q2, q2, q7\n"
        "  ee.vst.128.ip q1, %1, 16\n"
        "  ee.vst.128.ip q2, %1, 16\n"
        "1:\n"
        : "+r"(src), "+r"(dst), "+r"(n)
        : "r"(mk), "r"(e8)
        : "memory");
}
#else
void tasr_unpack_s4(const uint8_t *src, int8_t *dst, int kp)
{
    for (int blk = 0; blk < kp; blk += 32) {
        for (int j = 0; j < 16; j++) {
            uint8_t v = src[j];
            dst[blk + j] = (int8_t)((v & 15) - 8);
            dst[blk + j + 16] = (int8_t)((v >> 4) - 8);
        }
        src += 16;
    }
}
#endif

void tasr_quant_rows(const float *x, int T, int K, int ldx, int8_t *xq, int ldq, int kp, float *xs)
{
    for (int t = 0; t < T; t++) {
        const float *r = x + (size_t)t * ldx;
        int8_t *q = xq + (size_t)t * ldq;
        float m0 = 0.f, m1 = 0.f, m2 = 0.f, m3 = 0.f;   // four running maxima (max is exact and order independent)
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            const float a0 = fabsf(r[k]), a1 = fabsf(r[k + 1]), a2 = fabsf(r[k + 2]), a3 = fabsf(r[k + 3]);
            m0 = a0 > m0 ? a0 : m0; m1 = a1 > m1 ? a1 : m1; m2 = a2 > m2 ? a2 : m2; m3 = a3 > m3 ? a3 : m3;
        }
        for (; k < K; k++) { const float a = fabsf(r[k]); m0 = a > m0 ? a : m0; }
        float m = m0 > m1 ? m0 : m1;
        const float mb = m2 > m3 ? m2 : m3;
        m = m > mb ? m : mb;
        if (m < 1e-30f) m = 1e-30f;
        const float inv = 127.0f / m;
        k = 0;
        for (; k + 4 <= K; k += 4) {   // |r*inv| <= 127 by construction
            const float y0 = r[k] * inv, y1 = r[k + 1] * inv, y2 = r[k + 2] * inv, y3 = r[k + 3] * inv;
            q[k] = (int8_t)rne(y0); q[k + 1] = (int8_t)rne(y1); q[k + 2] = (int8_t)rne(y2); q[k + 3] = (int8_t)rne(y3);
        }
        for (; k < K; k++) q[k] = (int8_t)rne(r[k] * inv);
        for (int k = K; k < kp; k++) q[k] = 0;
        xs[t] = m / 127.0f;
    }
}

// ---- table-driven exp(-u), sigmoid and tanh (language model softmax and GRU gates, log-softmax of the CTC head).
// libm costs 250-380 cycles per call on the LX7; these are ~15 cycles. 1/16-step tables with linear interpolation:
// relative error of exp(-u) < 5e-4 on [0, 24), absolute error of the sigmoid < 5e-5, of tanh < 1e-4.
#define TFAST_EXP_N 384     // u in [0, 24)
#define TFAST_SIG_N 384     // x in [-12, 12)
static float tfast_exp[TFAST_EXP_N + 2], tfast_sig[TFAST_SIG_N + 2];
static int tfast_ready;
void tasr_fast_init(void)
{
    if (tfast_ready) return;
    for (int i = 0; i < TFAST_EXP_N + 2; i++) tfast_exp[i] = (float)exp(-(double)i / 16.0);
    for (int i = 0; i < TFAST_SIG_N + 2; i++) tfast_sig[i] = (float)(1.0 / (1.0 + exp(-((double)i / 16.0 - 12.0))));
    tfast_ready = 1;
}
float tasr_fast_expneg(float u)
{
    if (u >= 23.9f) return 0.f;
    if (u < 0.f) u = 0.f;
    const float x = u * 16.0f;
    const int i = (int)x;
    const float f = x - (float)i;
    return tfast_exp[i] + f * (tfast_exp[i + 1] - tfast_exp[i]);
}
float tasr_fast_sigmoid(float x)
{
    if (x <= -11.99f) return tfast_sig[0];
    if (x >= 11.99f) return tfast_sig[TFAST_SIG_N];
    const float u = (x + 12.0f) * 16.0f;
    const int i = (int)u;
    const float f = u - (float)i;
    return tfast_sig[i] + f * (tfast_sig[i + 1] - tfast_sig[i]);
}
float tasr_fast_tanh(float x) { return 2.0f * tasr_fast_sigmoid(2.0f * x) - 1.0f; }

// float epilogue of an integer tile: y = (float)acc * xs * sw + b, evaluated in exactly this order (no fused multiply-add:
// -ffp-contract=off), four independent chains at a time so the FPU latencies overlap
void tasr_epilogue_row(float *restrict yo, const int32_t *restrict a, float xst, const float *restrict sw, const float *restrict bb, int n)
{
    int j = 0;
    for (; j + 4 <= n; j += 4) {
        float a0 = (float)a[j], a1 = (float)a[j + 1], a2 = (float)a[j + 2], a3 = (float)a[j + 3];
        const float s0 = sw[j], s1 = sw[j + 1], s2 = sw[j + 2], s3 = sw[j + 3];
        const float b0 = bb[j], b1 = bb[j + 1], b2 = bb[j + 2], b3 = bb[j + 3];
        a0 *= xst; a1 *= xst; a2 *= xst; a3 *= xst;
        a0 *= s0; a1 *= s1; a2 *= s2; a3 *= s3;
        yo[j] = a0 + b0; yo[j + 1] = a1 + b1; yo[j + 2] = a2 + b2; yo[j + 3] = a3 + b3;
    }
    for (; j < n; j++) yo[j] = (float)a[j] * xst * sw[j] + bb[j];
}
// one output channel, T rows: yo[t * ldy] = (float)acc[t] * xs[t] * sw + b
void tasr_epilogue_col(float *restrict yo, int ldy, const int32_t *restrict acc, const float *restrict xs, float sw, float b, int T)
{
    int t = 0;
    for (; t + 4 <= T; t += 4) {
        float a0 = (float)acc[t], a1 = (float)acc[t + 1], a2 = (float)acc[t + 2], a3 = (float)acc[t + 3];
        a0 *= xs[t]; a1 *= xs[t + 1]; a2 *= xs[t + 2]; a3 *= xs[t + 3];
        a0 *= sw; a1 *= sw; a2 *= sw; a3 *= sw;
        yo[(size_t)t * ldy] = a0 + b; yo[(size_t)(t + 1) * ldy] = a1 + b;
        yo[(size_t)(t + 2) * ldy] = a2 + b; yo[(size_t)(t + 3) * ldy] = a3 + b;
    }
    for (; t < T; t++) yo[(size_t)t * ldy] = (float)acc[t] * xs[t] * sw + b;
}

void tasr_qlin_range(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T, int ldq, float *y, int ldy,
                     int8_t *wtmp, int32_t *acc, int n0, int n1)
{
    const int kp = L->kp;
    if (L->blocked) {
        const size_t tb = (size_t)kp * 8;  // bytes of one 16-output tile
        const int per = tb >= WPF_BLK ? 1 : (int)(WPF_BLK / tb);
        const uint8_t *w0 = (const uint8_t *)L->w + (size_t)n0 * (kp >> 1);
        const size_t total = (size_t)((n1 - n0) / 16) * tb;
        int busy = 0, ti = 0;
        wpf_start(w0, total < per * tb ? total : per * tb, &busy);
        for (int nb = n0; nb < n1; nb += 16, ti++) {
            if (ti % per == 0) {
                wpf_wait(&busy);
                const size_t nx = (size_t)(ti + per) * tb;
                if (nx < total) wpf_start(w0 + nx, total - nx < per * tb ? total - nx : per * tb, &busy);
            }
            tasr_unpack_s4((const uint8_t *)L->w + (size_t)nb * (kp >> 1), wtmp, kp * 16);
            const float *sw = L->s + nb, *bb = L->b + nb;
            for (int t0 = 0; t0 < T; t0 += 64) {
                const int tn = T - t0 < 64 ? T - t0 : 64;
                const int8_t *x0 = xq + (size_t)t0 * ldq;
                tasr_gemm_blk16(wtmp, kp < 512 ? kp : 512, x0, ldq, tn, acc);
                for (int ks = 512; ks < kp; ks += 512) {  // K > 512: split to keep 20-bit lanes overflow-free
                    int32_t part[64 * 16];
                    const int kn = kp - ks < 512 ? kp - ks : 512;
                    tasr_gemm_blk16(wtmp + (size_t)ks * 16, kn, x0 + ks, ldq, tn, part);
                    for (int i = 0; i < tn * 16; i++) acc[i] += part[i];
                }
                for (int t = 0; t < tn; t++) {
                    float *yo = y + (size_t)(t0 + t) * ldy + nb;
                    const int32_t *a = acc + t * 16;
                    tasr_epilogue_row(yo, a, xs[t0 + t], sw, bb, 16);
                }
            }
        }
        wpf_wait(&busy);
        return;
    }
    if (L->bits == 8 && kp <= 256) {  // x-stationary kernel (wins for short rows only), 16 outputs x 64 rows per tile
        const size_t tb = (size_t)kp * 16;
        const int per = tb >= WPF_BLK ? 1 : (int)(WPF_BLK / tb);
        const int8_t *w0 = L->w + (size_t)n0 * kp;
        const size_t total = (size_t)(n1 - n0) * kp;
        int busy = 0, ti = 0;
        wpf_start(w0, total < per * tb ? total : per * tb, &busy);
        for (int nb0 = n0; nb0 < n1; nb0 += 16, ti++) {
            if (ti % per == 0) {
                wpf_wait(&busy);
                const size_t nx = (size_t)(ti + per) * tb;
                if (nx < total) wpf_start(w0 + nx, total - nx < per * tb ? total - nx : per * tb, &busy);
            }
            const int nb = n1 - nb0 < 16 ? n1 - nb0 : 16;
            const float *sw = L->s + nb0, *bb = L->b + nb0;
            for (int t0 = 0; t0 < T; t0 += 64) {
                const int tn = T - t0 < 64 ? T - t0 : 64;
                tasr_gemm_s8_xr(L->w + (size_t)nb0 * kp, kp, nb, xq + (size_t)t0 * ldq, ldq, tn, acc);
                for (int t = 0; t < tn; t++) {
                    float *yo = y + (size_t)(t0 + t) * ldy + nb0;
                    const int32_t *a = acc + t * nb;
                    tasr_epilogue_row(yo, a, xs[t0 + t], sw, bb, nb);
                }
            }
        }
        wpf_wait(&busy);
        return;
    }
    if (kp > 1024) {
        // long rows (front-end projections; activations live in PSRAM): split K so a 64 x kc activation sub-tile stays
        // cache resident, and accumulate 16 outputs x 64 rows in acc, so every weight byte is fetched once per 64 rows
        const int nk = (kp + 511) / 512;
        const int kc = ((kp + nk - 1) / nk + 31) & ~31;  // multiple of 32 (int4 unpack granularity)
        int32_t part[64];
        for (int nb0 = n0; nb0 < n1; nb0 += 16) {
            const int nb = n1 - nb0 < 16 ? n1 - nb0 : 16;
            const float *sw = L->s + nb0, *bb = L->b + nb0;
            for (int t0 = 0; t0 < T; t0 += 64) {
                const int tn = T - t0 < 64 ? T - t0 : 64;
                memset(acc, 0, sizeof(int32_t) * tn * nb);
                for (int k0 = 0; k0 < kp; k0 += kc) {
                    const int kn = kp - k0 < kc ? kp - k0 : kc;
                    for (int j = 0; j < nb; j++) {
                        const int8_t *w;
                        if (L->bits == 4) {
                            tasr_unpack_s4((const uint8_t *)L->w + ((size_t)(nb0 + j) * kp + k0) / 2, wtmp, kn);
                            w = wtmp;
                        } else {
                            w = L->w + (size_t)(nb0 + j) * kp + k0;
                        }
                        tasr_dot_rows_s8(w, xq + (size_t)t0 * ldq + k0, ldq, tn, kn, part);
                        for (int t = 0; t < tn; t++) acc[t * nb + j] += part[t];
                    }
                }
                for (int t = 0; t < tn; t++) {
                    float *yo = y + (size_t)(t0 + t) * ldy + nb0;
                    const int32_t *a = acc + t * nb;
                    tasr_epilogue_row(yo, a, xs[t0 + t], sw, bb, nb);
                }
            }
        }
        return;
    }
    // row layout, K <= 1024: activation tiles of up to 64 rows (<= 64 KB, internal RAM in both engines); each weight
    // row is fetched once per tile
    const int tb = 64;
    const size_t rb = L->bits == 8 ? (size_t)kp : (size_t)(kp >> 1);  // bytes of one weight row
    const int per = rb >= WPF_BLK ? 1 : (int)(WPF_BLK / rb);
    for (int t0 = 0; t0 < T; t0 += tb) {
        const int tn = T - t0 < tb ? T - t0 : tb;
        const uint8_t *w0 = (const uint8_t *)L->w + (size_t)n0 * rb;
        const size_t total = (size_t)(n1 - n0) * rb;
        int busy = 0;
        wpf_start(w0, total < per * rb ? total : per * rb, &busy);
        for (int n = n0; n < n1; n++) {
            if ((n - n0) % per == 0) {
                wpf_wait(&busy);
                const size_t nx = (size_t)(n - n0 + per) * rb;
                if (nx < total) wpf_start(w0 + nx, total - nx < per * rb ? total - nx : per * rb, &busy);
            }
            const int8_t *w;
            if (L->bits == 4) {
                tasr_unpack_s4((const uint8_t *)L->w + (size_t)n * (kp >> 1), wtmp, kp);
                w = wtmp;
            } else {
                w = L->w + (size_t)n * kp;
            }
            tasr_dot_rows_s8(w, xq + (size_t)t0 * ldq, ldq, tn, kp, acc);
            const float sw = L->s[n], b = L->b[n];
            float *yo = y + (size_t)t0 * ldy + n;
            tasr_epilogue_col(yo, ldy, acc, xs + t0, sw, b, tn);
        }
        wpf_wait(&busy);
    }
}

void tasr_qlin(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T, int ldq, float *y, int ldy, int8_t *wtmp)
{
    static int32_t acc[64 * 16];
    tasr_qlin_range(L, xq, xs, T, ldq, y, ldy, wtmp, acc, 0, L->n);
}

// ---------------------------------------------------------------------------------------------------------------
// int4 "blocked" GEMM: tile = [kp][16] int8 weights for 16 consecutive outputs (unpacked from the blocked int4
// layout), x rows int8. out[t*16 + j] = sum_k x[t][k] * tile[k][j].  kp multiple of 16 and <= 512 per call
// (20-bit QACC lanes: 512 * 127 * 7 < 2^19, so no overflow is possible with int4 weights).
#ifdef TASR_PIE
static inline void qacc_extract16(int32_t *o)
{
    // QACC_L -> buf[0..19], QACC_H -> buf[32..51] (128-bit stores must stay 16-byte aligned)
    uint8_t buf[64] __attribute__((aligned(16)));
    uint8_t *p = buf;
    __asm__ volatile(
        "ee.st.qacc_l.l.128.ip %0, 16\n"
        "ee.st.qacc_l.h.32.ip %0, 16\n"
        "ee.st.qacc_h.l.128.ip %0, 16\n"
        "ee.st.qacc_h.h.32.ip %0, 16\n"
        : "+r"(p) : : "memory");
    for (int h = 0; h < 2; h++) {
        const uint32_t *w = (const uint32_t *)(buf + 32 * h);
        const uint32_t w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4];
        int32_t *d = o + 8 * h;
        // 8 x 20-bit two's complement lanes packed little-endian in 160 bits
        d[0] = ((int32_t)(w0 << 12)) >> 12;
        d[1] = ((int32_t)(((w0 >> 20) | (w1 << 12)) << 12)) >> 12;
        d[2] = ((int32_t)((w1 >> 8) << 12)) >> 12;
        d[3] = ((int32_t)(((w1 >> 28) | (w2 << 4)) << 12)) >> 12;
        d[4] = ((int32_t)(((w2 >> 16) | (w3 << 16)) << 12)) >> 12;
        d[5] = ((int32_t)((w3 >> 4) << 12)) >> 12;
        d[6] = ((int32_t)(((w3 >> 24) | (w4 << 8)) << 12)) >> 12;
        d[7] = ((int32_t)((w4 >> 12) << 12)) >> 12;
    }
}

__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_gemm_blk16(const int8_t *tile, int kp, const int8_t *x,
                                                                                int ldq, int T, int32_t *out)
{
    const int nkb = kp >> 4;
    for (int t = 0; t < T; t++) {
        const int8_t *px = x + (size_t)t * ldq;
        const int8_t *pw = tile;
        __asm__ volatile("ee.zero.qacc\n" ::: "memory");
        int n = nkb;
        __asm__ volatile(
            "ee.vld.128.ip q0, %1, 16\n"            // first weight row
            "loopnez %2, 1f\n"
            "  ee.vld.128.ip q7, %0, 16\n"          // 16 activations
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 0\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 1\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 2\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 3\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 4\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 5\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 6\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 7\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 8\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 9\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 10\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 11\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 12\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 13\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 14\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 15\n"
            "1:\n"
            : "+r"(px), "+r"(pw), "+r"(n) : : "memory");
        qacc_extract16(out + (size_t)t * 16);
    }
}
#else
void tasr_gemm_blk16(const int8_t *tile, int kp, const int8_t *x, int ldq, int T, int32_t *out)
{
    for (int t = 0; t < T; t++) {
        const int8_t *px = x + (size_t)t * ldq;
        int32_t acc[16] = {0};
        for (int k = 0; k < kp; k++) {
            const int32_t xv = px[k];
            const int8_t *w = tile + (size_t)k * 16;
            for (int j = 0; j < 16; j++) acc[j] += xv * (int32_t)w[j];
        }
        for (int j = 0; j < 16; j++) out[(size_t)t * 16 + j] = acc[j];
    }
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// Specialized 48-byte dot products (attention head dim 44 padded to 48): the 48-byte query stays in q0..q2 for all
// T rows. out[t] = dot(q, x + t*ldx).
#ifdef TASR_PIE
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_dot48_rows(const int8_t *q, const int8_t *x, int ldx,
                                                                                int T, int32_t *out)
{
    const int8_t *pq = q;
    __asm__ volatile(
        "ee.vld.128.ip q0, %0, 16\n"
        "ee.vld.128.ip q1, %0, 16\n"
        "ee.vld.128.ip q2, %0, 16\n"
        : "+r"(pq) : : "memory");
    for (int t = 0; t < T; t++) {
        const int8_t *px = x + (size_t)t * ldx;
        int32_t r;
        __asm__ volatile(
            "ee.zero.accx\n"
            "ee.vld.128.ip q3, %1, 16\n"
            "ee.vld.128.ip q4, %1, 16\n"
            "ee.vmulas.s8.accx.ld.ip q5, %1, 16, q3, q0\n"
            "ee.vmulas.s8.accx q4, q1\n"
            "ee.vmulas.s8.accx q5, q2\n"
            "rur.accx_0 %0\n"
            : "=r"(r), "+r"(px) : : "memory");
        out[t] = r;
    }
}
#else
void tasr_dot48_rows(const int8_t *q, const int8_t *x, int ldx, int T, int32_t *out)
{
    for (int t = 0; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldx;
        int32_t acc = 0;
        for (int i = 0; i < 48; i++) acc += (int32_t)a[i] * (int32_t)q[i];
        out[t] = acc;
    }
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// int8 row-layout GEMM, x-stationary: acc[t * nb + j] = sum_k x[t][k] * W[j][k] for j < nb (<= 16), t < T.
// For each row t, up to 6 16-byte chunks of x live in q2..q7 while the nb weight rows stream through q0/q1, so each
// 16 MACs cost one instruction (vs. two when both operands are loaded). kp % 16 == 0; W, x 16-byte aligned.
#ifdef TASR_PIE
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_gemm_s8_xr(const int8_t *W, int kp, int nb,
                                                                                const int8_t *x, int ldq, int T,
                                                                                int32_t *acc)
{
    memset(acc, 0, sizeof(int32_t) * (size_t)T * nb);
    for (int t = 0; t < T; t++) {
        for (int k0 = 0; k0 < kp; k0 += 96) {
            int c = (kp - k0) >> 4;
            if (c > 6) c = 6;
            const int8_t *px = x + (size_t)t * ldq + k0;
            const int8_t *pw = W + k0;
            int32_t *pa = acc + (size_t)t * nb;
            int cnt = nb, r, tmp;
            const int rs = kp - 16 * c;
            switch (c) {
        case 1:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx q2, q0\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 2:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx q3, q1\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 3:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "ee.vld.128.ip q4, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q3, q1\n"
                "  ee.vmulas.s8.accx q4, q0\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 4:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "ee.vld.128.ip q4, %[px], 16\n"
                "ee.vld.128.ip q5, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q3, q1\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q4, q0\n"
                "  ee.vmulas.s8.accx q5, q1\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 5:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "ee.vld.128.ip q4, %[px], 16\n"
                "ee.vld.128.ip q5, %[px], 16\n"
                "ee.vld.128.ip q6, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q3, q1\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q4, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q5, q1\n"
                "  ee.vmulas.s8.accx q6, q0\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 6:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "ee.vld.128.ip q4, %[px], 16\n"
                "ee.vld.128.ip q5, %[px], 16\n"
                "ee.vld.128.ip q6, %[px], 16\n"
                "ee.vld.128.ip q7, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q3, q1\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q4, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q5, q1\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q6, q0\n"
                "  ee.vmulas.s8.accx q7, q1\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
            }
        }
    }
}
#else
void tasr_gemm_s8_xr(const int8_t *W, int kp, int nb, const int8_t *x, int ldq, int T, int32_t *acc)
{
    for (int t = 0; t < T; t++)
        for (int j = 0; j < nb; j++) {
            const int8_t *a = x + (size_t)t * ldq, *w = W + (size_t)j * kp;
            int32_t s = 0;
            for (int i = 0; i < kp; i++) s += (int32_t)a[i] * (int32_t)w[i];
            acc[(size_t)t * nb + j] = s;
        }
}
#endif
