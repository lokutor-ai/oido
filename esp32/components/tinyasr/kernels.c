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
        float m = 0.f;
        for (int k = 0; k < K; k++) {
            float a = fabsf(r[k]);
            m = a > m ? a : m;
        }
        if (m < 1e-30f) m = 1e-30f;
        const float inv = 127.0f / m;
        for (int k = 0; k < K; k++) q[k] = (int8_t)rne(r[k] * inv);  // |r*inv| <= 127 by construction
        for (int k = K; k < kp; k++) q[k] = 0;
        xs[t] = m / 127.0f;
    }
}

void tasr_qlin_range(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T, int ldq, float *y, int ldy,
                     int8_t *wtmp, int32_t *acc, int n0, int n1)
{
    const int kp = L->kp;
    if (L->blocked) {
        for (int nb = n0; nb < n1; nb += 16) {
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
                    const float xst = xs[t0 + t];
                    for (int j = 0; j < 16; j++) yo[j] = (float)a[j] * xst * sw[j] + bb[j];
                }
            }
        }
        return;
    }
    if (L->bits == 8 && kp <= 256) {  // x-stationary kernel (wins for short rows only), 16 outputs x 64 rows per tile
        for (int nb0 = n0; nb0 < n1; nb0 += 16) {
            const int nb = n1 - nb0 < 16 ? n1 - nb0 : 16;
            const float *sw = L->s + nb0, *bb = L->b + nb0;
            for (int t0 = 0; t0 < T; t0 += 64) {
                const int tn = T - t0 < 64 ? T - t0 : 64;
                tasr_gemm_s8_xr(L->w + (size_t)nb0 * kp, kp, nb, xq + (size_t)t0 * ldq, ldq, tn, acc);
                for (int t = 0; t < tn; t++) {
                    float *yo = y + (size_t)(t0 + t) * ldy + nb0;
                    const int32_t *a = acc + t * nb;
                    const float xst = xs[t0 + t];
                    for (int j = 0; j < nb; j++) yo[j] = (float)a[j] * xst * sw[j] + bb[j];
                }
            }
        }
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
                    const float xst = xs[t0 + t];
                    for (int j = 0; j < nb; j++) yo[j] = (float)a[j] * xst * sw[j] + bb[j];
                }
            }
        }
        return;
    }
    // row layout, K <= 1024: activation tiles of up to 64 rows (<= 64 KB, internal RAM in both engines); each weight
    // row is fetched once per tile
    const int tb = 64;
    for (int t0 = 0; t0 < T; t0 += tb) {
        const int tn = T - t0 < tb ? T - t0 : tb;
        for (int n = n0; n < n1; n++) {
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
            const float *xst = xs + t0;
            for (int t = 0; t < tn; t++) yo[(size_t)t * ldy] = (float)acc[t] * xst[t] * sw + b;
        }
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
