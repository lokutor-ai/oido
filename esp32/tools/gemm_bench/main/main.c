// GEMM micro-benchmark for the Oido int8 kernels on the ESP32-S3: where do the cycles go (compute vs. memory)?
// Weights live in internal SRAM, PSRAM or the memory-mapped flash 'model' partition; the activation tile in SRAM or PSRAM.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_attr.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "kernels.h"
#include <math.h>
#include "tasr_math.h"
#include "esp32s3/rom/cache.h"
#include "soc/extmem_reg.h"

static inline uint32_t cc(void) { return esp_cpu_get_cycle_count(); }

static void *amem(size_t n, uint32_t caps)
{
    void *p = heap_caps_aligned_alloc(64, n, caps);
    if (!p) { printf("alloc %u failed\n", (unsigned)n); abort(); }
    return p;
}

static volatile uint32_t sink;
static void evict_cache(void)  // read 512 KB of PSRAM so the 64 KB data cache holds none of the benchmark data
{
    static uint8_t *big;
    if (!big) big = amem(512 * 1024, MALLOC_CAP_SPIRAM);
    uint32_t s = 0;
    for (int i = 0; i < 512 * 1024; i += 64) s += big[i];
    sink = s;
}

static int cmp_u32(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return x < y ? -1 : x > y; }
static uint32_t median(uint32_t *v, int n) { qsort(v, n, 4, cmp_u32); return v[n / 2]; }

static void ref_dot(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out)
{
    for (int t = 0; t < T; t++) { int32_t a = 0; for (int k = 0; k < kp; k++) a += (int32_t)x[(size_t)t * ldq + k] * w[k]; out[t] = a; }
}

typedef struct { const char *name; const int8_t *w; } wloc_t;

// 16 outputs (weight rows) x T activation rows with the row-layout dot kernel
static void bench_dot(const char *wname, const int8_t *W, const char *xname, const int8_t *x, int K, int T, int warm)
{
    int32_t out[64], ref[64];
    uint32_t v[7];
    for (int rep = 0; rep < 7; rep++) {
        if (!warm) evict_cache();
        else for (int n = 0; n < 16; n++) tasr_dot_rows_s8(W + (size_t)n * K, x, K, T, K, out);  // warm both
        uint32_t t0 = cc();
        for (int n = 0; n < 16; n++) tasr_dot_rows_s8(W + (size_t)n * K, x, K, T, K, out);
        v[rep] = cc() - t0;
    }
    ref_dot(W + 15 * (size_t)K, x, K, T, K, ref);
    int ok = !memcmp(out, ref, sizeof(int32_t) * T);
    uint32_t m = median(v, 7);
    printf("dot  K=%4d T=%2d W=%-5s X=%-5s %s: %7u cyc / 16 rows = %.3f cyc/MAC (%.2f MAC/cyc) %s\n", K, T, wname, xname, warm ? "warm" : "cold",
           (unsigned)m, (double)m / (16.0 * T * K), (16.0 * T * K) / m, ok ? "ok" : "MISMATCH");
}

static void bench_xr(const char *wname, const int8_t *W, const char *xname, const int8_t *x, int K, int T, int warm)
{
    static int32_t acc[64 * 16];
    uint32_t v[7];
    for (int rep = 0; rep < 7; rep++) {
        if (!warm) evict_cache();
        else tasr_gemm_s8_xr(W, K, 16, x, K, T, acc);
        uint32_t t0 = cc();
        tasr_gemm_s8_xr(W, K, 16, x, K, T, acc);
        v[rep] = cc() - t0;
    }
    // check one output
    int32_t r[1]; ref_dot(W + 5 * (size_t)K, x, K, 1, K, r);
    int ok = acc[0 * 16 + 5] == r[0];
    uint32_t m = median(v, 7);
    printf("xr   K=%4d T=%2d W=%-5s X=%-5s %s: %7u cyc / 16 outputs = %.3f cyc/MAC (%.2f MAC/cyc) %s\n", K, T, wname, xname, warm ? "warm" : "cold",
           (unsigned)m, (double)m / (16.0 * T * K), (16.0 * T * K) / m, ok ? "ok" : "MISMATCH");
}

// the scalar float epilogue the engine runs after each int32 tile: y = acc * xs * sw + b
static void bench_epilogue(void)
{
    static int32_t acc[64 * 16];
    static float y[64 * 16], xs[64], sw[16], bb[16];
    for (int i = 0; i < 64 * 16; i++) acc[i] = i * 37 - 5000;
    for (int i = 0; i < 64; i++) xs[i] = 0.01f + i * 1e-4f;
    for (int i = 0; i < 16; i++) { sw[i] = 0.003f; bb[i] = 0.1f; }
    uint32_t v[7];
    for (int rep = 0; rep < 7; rep++) {
        uint32_t t0 = cc();
        for (int t = 0; t < 64; t++) { float *yo = y + (size_t)t * 16; const int32_t *a = acc + t * 16; const float x = xs[t];
            for (int j = 0; j < 16; j++) yo[j] = (float)a[j] * x * sw[j] + bb[j]; }
        v[rep] = cc() - t0;
    }
    uint32_t m = median(v, 7);
    printf("epilogue 64x16 outputs (old loop): %u cyc = %.2f cyc/output\n", (unsigned)m, (double)m / (64 * 16));
    static float y2[64 * 16];
    for (int rep = 0; rep < 7; rep++) {
        uint32_t t0 = cc();
        for (int t = 0; t < 64; t++) tasr_epilogue_row(y2 + (size_t)t * 16, acc + t * 16, xs[t], sw, bb, 16);
        v[rep] = cc() - t0;
    }
    m = median(v, 7);
    int same = 1;
    for (int t = 0; t < 64; t++) for (int j = 0; j < 16; j++) same &= memcmp(&y[(size_t)t * 16 + j], &y2[(size_t)t * 16 + j], 4) == 0;
    printf("epilogue 64x16 outputs (tasr_epilogue_row): %u cyc = %.2f cyc/output, identical %d\n", (unsigned)m, (double)m / (64 * 16), same);
    {   // column form (row-layout path): 16 channels x 64 rows
        static int32_t colacc[64];
        for (int i = 0; i < 64; i++) colacc[i] = i * 53 - 900;
        for (int rep = 0; rep < 7; rep++) {
            uint32_t t0 = cc();
            for (int n = 0; n < 16; n++) tasr_epilogue_col(y2 + n, 16, colacc, xs, sw[n], bb[n], 64);
            v[rep] = cc() - t0;
        }
        m = median(v, 7);
        uint32_t v0[7];
        for (int rep = 0; rep < 7; rep++) {
            uint32_t t0 = cc();
            for (int n = 0; n < 16; n++) for (int t = 0; t < 64; t++) y[(size_t)t * 16 + n] = (float)colacc[t] * xs[t] * sw[n] + bb[n];
            v0[rep] = cc() - t0;
        }
        printf("epilogue column form 16x64: old %u cyc (%.2f/output), new %u cyc (%.2f/output)\n", (unsigned)median(v0, 7), (double)median(v0, 7) / 1024,
               (unsigned)m, (double)m / 1024);
    }
}

// D-cache hardware autoload (prefetch of the next blocks after an access), configured through the ROM API
static void autoload_off(void)
{
    Cache_Disable_DCache_Autoload();
}
static void autoload_cfg(int size, int trigger)
{
    Cache_Disable_DCache_Autoload();
    struct autoload_config cfg = {.ena = 1, .order = 0, .trigger = (uint8_t)trigger, .size = (uint8_t)size};
    Cache_Config_DCache_Autoload(&cfg);
}
static void autoload_region(const void *addr, uint32_t size)
{
    struct autoload_region_config r0 = {.region = 0, .ena = 1, .addr = (uint32_t)addr, .size = size};
    int rc = Cache_Config_DCache_Region_Autoload(&r0);
    if (rc) printf("region autoload rc=%d\n", rc);
}

// sequential read bandwidth with 128-bit loads (what a weight stream does), cold cache
static void bench_bw(const char *name, const int8_t *p, size_t n)
{
    uint32_t v[5];
    for (int rep = 0; rep < 5; rep++) {
        evict_cache();
        const int8_t *q = p;
        int cnt = (int)(n >> 4);
        uint32_t t0 = cc();
        __asm__ volatile(
            "loopnez %1, 1f\n"
            "  ee.vld.128.ip q0, %0, 16\n"
            "1:\n"
            : "+r"(q), "+r"(cnt) : : "memory");
        v[rep] = cc() - t0;
    }
    uint32_t m = median(v, 5);
    printf("read %-6s %7u bytes: %9u cyc = %.2f cyc/B = %.1f MB/s at 240 MHz\n", name, (unsigned)n, (unsigned)m, (double)m / n, 240.0 * n / m);
}

// whole layer through the engine's own entry point (tile loop, unpack, epilogue included)
static uint32_t g_hash;
static int g_auto = -1;  // -1: autoload off, else size 0..3
static void bench_qlin(int N, int K, int T, int bits, const char *wname, const int8_t *wsrc)
{
    const int kp = (K + 15) & ~15;
    tasr_qlin_t L; memset(&L, 0, sizeof L);
    L.n = N; L.k = K; L.kp = kp; L.bits = bits; L.blocked = bits == 4;
    static float s_[1024], b_[1024];
    for (int i = 0; i < 1024; i++) { s_[i] = 0.002f; b_[i] = 0.01f; }
    L.s = s_; L.b = b_; L.w = wsrc;
    int8_t *xq = amem((size_t)T * kp, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    for (int i = 0; i < T * kp; i++) xq[i] = (int8_t)((i * 13 + 5) % 251 - 125);
    float xs[64]; for (int i = 0; i < 64; i++) xs[i] = 0.01f;
    float *y = amem((size_t)T * N * 4, T <= 16 ? (MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) : MALLOC_CAP_SPIRAM);
    int8_t *wtmp = amem(16 * kp + 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int32_t *acc = amem(64 * 16 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    uint32_t v[5];
    for (int rep = 0; rep < 5; rep++) {
        evict_cache();
        uint32_t t0 = cc();
        tasr_qlin_range(&L, xq, xs, T, kp, y, N, wtmp, acc, 0, N);
        v[rep] = cc() - t0;
    }
    uint32_t m = median(v, 5);
    uint32_t h = 2166136261u;
    for (int i = 0; i < T * N; i++) { uint32_t b; memcpy(&b, &y[i], 4); h = (h ^ b) * 16777619u; }
    g_hash = h;
    double macs = (double)N * K * T, wbytes = (double)N * kp * (bits == 8 ? 1.0 : 0.5);
    printf("qlin%s N=%4d K=%4d T=%2d int%d W=%-5s: %8u cyc = %.3f cyc/MAC (%.2f MAC/cyc), %.1f cyc/weight-byte, %.1f ms, hash %08x\n", tasr_wpf ? "+PF" : "   ", N, K, T, bits, wname,
           (unsigned)m, m / macs, macs / m, m / wbytes, m / 240e3, (unsigned)g_hash);
    heap_caps_free(xq); heap_caps_free(y); heap_caps_free(wtmp); heap_caps_free(acc);
}

static float g_nsig_tab[386], g_nsig_slope[386];
static inline float g_nsig(float x)
{
    float u = (x + 12.0f) * 16.0f;
    if (u <= 0.f) return 0.f;
    if (u >= 384.f) return 1.f;
    int i = (int)u;
    return g_nsig_tab[i] + (u - (float)i) * g_nsig_slope[i];
}
static inline float g_nsig2(float x)
{
    const float u = (x + 12.0f) * 16.0f;
    int i = (int)u;
    i = i < 0 ? 0 : (i > 383 ? 383 : i);
    float r = g_nsig_tab[i] + (u - (float)i) * g_nsig_slope[i];
    r = (u <= 0.f) ? 0.f : r;
    r = (u >= 384.f) ? 1.f : r;
    return r;
}
static void bench_silu(void)
{
    for (int i = 0; i <= 384; i++) g_nsig_tab[i] = (float)(1.0 / (1.0 + exp(-(-12.0 + i / 16.0))));
    for (int i = 0; i < 384; i++) g_nsig_slope[i] = g_nsig_tab[i + 1] - g_nsig_tab[i];
    static float x[5632], x0[5632];
    for (int i = 0; i < 5632; i++) x0[i] = -8.0f + 0.0028f * i;
    uint32_t v[5];
    for (int rep = 0; rep < 5; rep++) {
        memcpy(x, x0, sizeof x);
        uint32_t t0 = cc();
        for (int i = 0; i < 5632; i++) x[i] = x[i] * g_nsig(x[i]);
        v[rep] = cc() - t0;
    }
    printf("silu simple loop: %.1f cyc/element\n", (double)median(v, 5) / 5632);
    for (int rep = 0; rep < 5; rep++) {
        memcpy(x, x0, sizeof x);
        uint32_t t0 = cc();
        int i = 0;
        for (; i + 4 <= 5632; i += 4) {
            const float a0 = x[i], a1 = x[i + 1], a2 = x[i + 2], a3 = x[i + 3];
            const float s0 = g_nsig(a0), s1 = g_nsig(a1), s2 = g_nsig(a2), s3 = g_nsig(a3);
            x[i] = a0 * s0; x[i + 1] = a1 * s1; x[i + 2] = a2 * s2; x[i + 3] = a3 * s3;
        }
        v[rep] = cc() - t0;
    }
    printf("silu unrolled x4: %.1f cyc/element\n", (double)median(v, 5) / 5632);
    for (int rep = 0; rep < 5; rep++) {
        memcpy(x, x0, sizeof x);
        uint32_t t0 = cc();
        int i = 0;
        for (; i + 4 <= 5632; i += 4) {
            const float a0 = x[i], a1 = x[i + 1], a2 = x[i + 2], a3 = x[i + 3];
            const float s0 = g_nsig2(a0), s1 = g_nsig2(a1), s2 = g_nsig2(a2), s3 = g_nsig2(a3);
            x[i] = a0 * s0; x[i + 1] = a1 * s1; x[i + 2] = a2 * s2; x[i + 3] = a3 * s3;
        }
        v[rep] = cc() - t0;
    }
    {   // identical to the original on every input?
        int same = 1;
        for (int k = 0; k < 5632; k++) { float a = x0[k], b1 = g_nsig(a), b2 = g_nsig2(a); same &= memcmp(&b1, &b2, 4) == 0; }
        float ex[8] = {-12.f, -11.9999f, 0.f, 12.f, 11.99999f, -30.f, 30.f, -1e-9f};
        for (int k = 0; k < 8; k++) { float b1 = g_nsig(ex[k]), b2 = g_nsig2(ex[k]); same &= memcmp(&b1, &b2, 4) == 0; }
        printf("silu branch-free x4: %.1f cyc/element, identical to the original: %d\n", (double)median(v, 5) / 5632, same);
    }
    // layer norm row of 176 as in the engine, 8 rows
    static float r[176 * 8], o[176 * 8], gg[176], bb2[176];
    for (int i = 0; i < 176; i++) { gg[i] = 1.f; bb2[i] = 0.f; }
    for (int i = 0; i < 176 * 8; i++) r[i] = 0.01f * (i % 97) - 0.4f;
    for (int rep = 0; rep < 5; rep++) {
        uint32_t t0 = cc();
        for (int t = 0; t < 8; t++) {
            const float *q = r + t * 176; float mean = 0.f;
            for (int i = 0; i < 176; i++) mean += q[i];
            mean /= 176.f;
            float var = 0.f;
            for (int i = 0; i < 176; i++) { float d = q[i] - mean; var += d * d; }
            var /= 176.f;
            const float inv = 1.0f / sqrtf(var + 1e-5f);
            for (int i = 0; i < 176; i++) o[t * 176 + i] = (q[i] - mean) * inv * gg[i] + bb2[i];
        }
        v[rep] = cc() - t0;
    }
    printf("layernorm row 176: %.1f cyc/row (%.1f cyc/element)\n", (double)median(v, 5) / 8, (double)median(v, 5) / 8 / 176);
}

static void bench_math(void)
{
    static float in[1024], out[1024];
    for (int i = 0; i < 1024; i++) in[i] = -20.0f + 0.0391f * i;       // softmax-like arguments
    uint32_t v[5];
    static volatile float msink;
    #define TIMEIT(NAME, EXPR, N) do { for (int rep = 0; rep < 5; rep++) { float accs = 0.f; uint32_t t0 = cc(); for (int i = 0; i < 1024; i++) { float x = in[i]; accs += (EXPR); } v[rep] = cc() - t0; msink = accs; } \
        printf("math %-14s %.1f cyc/call\n", NAME, (double)median(v, 5) / 1024); (void)N; (void)out; } while (0)
    TIMEIT("expf (libm)", expf(x), 0);
    TIMEIT("tasr_expf", tasr_expf(x), 0);
    for (int i = 0; i < 1024; i++) in[i] = 1e-3f + 0.37f * i;           // mel energies
    TIMEIT("logf (libm)", logf(x), 0);
    TIMEIT("tasr_logf", tasr_logf(x), 0);
    for (int i = 0; i < 1024; i++) in[i] = -6.0f + 0.0117f * i;
    TIMEIT("tanhf (libm)", tanhf(x), 0);
    TIMEIT("tasr_tanhf", tasr_tanhf(x), 0);
    TIMEIT("sigm libm", 1.0f / (1.0f + expf(-x)), 0);
    TIMEIT("tasr_sigmoidf", tasr_sigmoidf(x), 0);
    for (int i = 0; i < 1024; i++) in[i] = 0.5f + 0.3f * i;
    TIMEIT("sqrtf", sqrtf(x), 0);
    TIMEIT("1/sqrtf", 1.0f / sqrtf(x), 0);
}

void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(500));
    printf("\nGEMM_BENCH start, CPU %d MHz\n", (int)(esp_cpu_get_cycle_count() * 0 + 240));
    const esp_partition_t *mp = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, NULL);
    const void *fptr = NULL; esp_partition_mmap_handle_t h;
    const int8_t *Wfl = NULL;
    if (mp && esp_partition_mmap(mp, 0, 2 * 1024 * 1024, ESP_PARTITION_MMAP_DATA, &fptr, &h) == ESP_OK) Wfl = (const int8_t *)fptr + 4096;
    else printf("no model partition mapped (flash rows skipped)\n");

    {
        int8_t *Ps = amem(1024 * 1024, MALLOC_CAP_SPIRAM);
        int8_t *Is = amem(32 * 1024, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        memset(Ps, 1, 1024 * 1024); memset(Is, 1, 32 * 1024);
        bench_bw("sram", Is, 32 * 1024);
        bench_bw("psram", Ps, 1024 * 1024);
        if (Wfl) bench_bw("flash", Wfl, 1024 * 1024);
        heap_caps_free(Ps); heap_caps_free(Is);
    }
    printf("autoload default state: ctrl reg 0x%08x\n", (unsigned)REG_READ(EXTMEM_DCACHE_AUTOLOAD_CTRL_REG));
    {
        int8_t *Ps = amem(1024 * 1024, MALLOC_CAP_SPIRAM);
        memset(Ps, 1, 1024 * 1024);
        for (int trig = 0; trig <= 2; trig += 2)
            for (int sz = 0; sz < 4; sz++) {
                autoload_cfg(sz, trig);
                uint32_t v[3];
                for (int rep = 0; rep < 3; rep++) {
                    evict_cache();
                    uint32_t t0 = cc();
                    for (int w = 0; w < 16; w++) {   // 64 KB windows, autoload region follows the window
                        autoload_region(Ps + (size_t)w * 65536, 65536);
                        Cache_Enable_DCache_Autoload();
                        const int8_t *q = Ps + (size_t)w * 65536;
                        int cnt = 4096;
                        __asm__ volatile("loopnez %1, 1f\n  ee.vld.128.ip q0, %0, 16\n1:\n" : "+r"(q), "+r"(cnt) : : "memory");
                    }
                    v[rep] = cc() - t0;
                }
                uint32_t m = median(v, 3);
                printf("autoload size=%d trigger=%d ctrl=0x%08x: psram 1 MB in %u cyc = %.2f cyc/B = %.1f MB/s\n", sz, trig,
                       (unsigned)REG_READ(EXTMEM_DCACHE_AUTOLOAD_CTRL_REG), (unsigned)m, m / 1048576.0, 240.0 * 1048576 / m);
            }
        autoload_off();
        heap_caps_free(Ps);
    }
    const int KS[] = {176, 704};
    for (int ki = 0; ki < 2; ki++) {
        int K = KS[ki], T = 64;
        int8_t *Wsr = amem((size_t)16 * K, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        int8_t *Wps = amem((size_t)16 * K, MALLOC_CAP_SPIRAM);
        int8_t *Xsr = amem((size_t)T * K, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        int8_t *Xps = amem((size_t)T * K, MALLOC_CAP_SPIRAM);
        for (size_t i = 0; i < (size_t)16 * K; i++) { Wsr[i] = (int8_t)((i * 31 + 7) % 251 - 125); Wps[i] = Wsr[i]; }
        for (size_t i = 0; i < (size_t)T * K; i++) { Xsr[i] = (int8_t)((i * 17 + 3) % 253 - 126); Xps[i] = Xsr[i]; }
        bench_dot("sram", Wsr, "sram", Xsr, K, T, 1);
        bench_dot("sram", Wsr, "sram", Xsr, K, T, 0);
        bench_dot("psram", Wps, "sram", Xsr, K, T, 0);
        if (Wfl) bench_dot("flash", Wfl, "sram", Xsr, K, T, 0);
        bench_dot("sram", Wsr, "psram", Xps, K, T, 0);
        if (K <= 704) {
            bench_xr("sram", Wsr, "sram", Xsr, K, T, 1);
            bench_xr("sram", Wsr, "sram", Xsr, K, T, 0);
            bench_xr("psram", Wps, "sram", Xsr, K, T, 0);
            if (Wfl) bench_xr("flash", Wfl, "sram", Xsr, K, T, 0);
        }
        heap_caps_free(Wsr); heap_caps_free(Wps); heap_caps_free(Xsr); heap_caps_free(Xps);
    }
    bench_silu();
    bench_math();
    bench_epilogue();
    {   // int4 blocked kernel pieces, all data in SRAM: unpack of one 16-output tile, the QACC GEMM per K, T rows
        const int Ks[] = {176, 352, 512};
        for (int ki = 0; ki < 3; ki++) {
            int K = Ks[ki];
            uint8_t *pk = amem((size_t)K * 8, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
            int8_t *tile = amem((size_t)K * 16 + 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
            int8_t *x = amem(64 * 512, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
            int32_t *o = amem(64 * 16 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
            for (int i = 0; i < K * 8; i++) pk[i] = (uint8_t)(i * 37 + 5);
            for (int i = 0; i < 64 * 512; i++) x[i] = (int8_t)(i * 11 + 3);
            uint32_t v[5];
            for (int rep = 0; rep < 5; rep++) { uint32_t t0 = cc(); tasr_unpack_s4(pk, tile, K * 16); v[rep] = cc() - t0; }
            uint32_t mu = median(v, 5);
            for (int rep = 0; rep < 5; rep++) { uint32_t t0 = cc(); tasr_gemm_blk16(tile, K, x, 512, 64, o); v[rep] = cc() - t0; }
            uint32_t mg = median(v, 5);
            for (int rep = 0; rep < 5; rep++) { uint32_t t0 = cc(); tasr_gemm_blk16(tile, K, x, 512, 8, o); v[rep] = cc() - t0; }
            uint32_t mg8 = median(v, 5);
            printf("int4 pieces K=%d: unpack tile %u cyc (%.2f cyc/weight); gemm_blk16 T=64: %u cyc = %.1f cyc/row (%.2f cyc per k-step incl. extract), T=8: %.1f cyc/row\n",
                   K, (unsigned)mu, (double)mu / (K * 16), (unsigned)mg, mg / 64.0, mg / 64.0 / K, mg8 / 8.0);
            heap_caps_free(pk); heap_caps_free(tile); heap_caps_free(x); heap_caps_free(o);
        }
    }
    printf("GEMM_BENCH done\n");
    for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
}
