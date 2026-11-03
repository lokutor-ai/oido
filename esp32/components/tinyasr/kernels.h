#pragma once
#include <stdint.h>

typedef struct {
    int n, k, kp, bits, blocked;
    const int8_t *w;  // int8: [n][kp]; int4: packed [n][kp/2] (per 32-block: byte j = w[j] | w[j+16] << 4, offset +8)
    const float *s;   // [n] per-row weight scale
    const float *b;   // [n] bias
} tasr_qlin_t;

// ESP32-S3: preload the next block of weights into the D-cache in the background while the current one is multiplied
// (1 = on, default; 0 = off, for comparisons). No effect on the host build.
extern int tasr_wpf;
// table-driven exp(-u) (u >= 0), sigmoid and tanh; tasr_fast_init() builds the tables once (idempotent)
void tasr_fast_init(void);
float tasr_fast_expneg(float u);
float tasr_fast_sigmoid(float x);
float tasr_fast_tanh(float x);
// float epilogue of an integer tile (see kernels.c)
void tasr_epilogue_row(float *restrict yo, const int32_t *restrict a, float xst, const float *restrict sw, const float *restrict bb, int n);
void tasr_epilogue_col(float *restrict yo, int ldy, const int32_t *restrict acc, const float *restrict xs, float sw, float b, int T);
// out[t] = dot(x + t*ldq, w) for t < T (kp multiple of 16, 16-byte aligned rows)
void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out);
// int8 dot product; kp multiple of 16, both pointers 16-byte aligned.
int32_t tasr_dot_s8(const int8_t *a, const int8_t *b, int kp);
// unpack one int4 row (kp multiple of 32) to int8
void tasr_unpack_s4(const uint8_t *src, int8_t *dst, int kp);
// dynamic symmetric per-row int8 quantization: xq rows are ldq bytes (>= kp, zero padded to kp), xs = row scale
void tasr_quant_rows(const float *x, int T, int K, int ldx, int8_t *xq, int ldq, int kp, float *xs);
// y[t][n] = sum_k xq[t][k] * w[n][k] * xs[t] * s[n] + b[n]
void tasr_qlin(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T, int ldq, float *y, int ldy, int8_t *wtmp);
// outputs [n0, n1) (multiples of 16 for blocked layers). wtmp >= 16*kp+16 bytes (blocked) or kp bytes; acc >= 64*16
void tasr_qlin_range(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T, int ldq, float *y, int ldy,
                     int8_t *wtmp, int32_t *acc, int n0, int n1);
// int4-blocked GEMM on an unpacked [kp][16] tile (kp multiple of 16, <= 512): out[t][j] for 16 outputs
void tasr_gemm_blk16(const int8_t *tile, int kp, const int8_t *x, int ldq, int T, int32_t *out);
// out[t] = dot48(q, x + t*ldx) (both 16-byte aligned)
void tasr_gemm_s8_xr(const int8_t *W, int kp, int nb, const int8_t *x, int ldq, int T, int32_t *acc);
void tasr_dot48_rows(const int8_t *q, const int8_t *x, int ldx, int T, int32_t *out);
