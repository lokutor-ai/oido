// PATCH (not applied): deterministic exp / log / tanh so the host build and the firmware agree bit for bit.
// Without it, libm differs in the last bit between a laptop and the ESP32's newlib; on ~20% of hard (noisy) clips that
// changes a word (22/27 identical with libm, 27/27 with this header, tested 2026-10-02 with beam search + LM).
// Applying it moves published LibriSpeech numbers by ~0.05-0.1 (int8 greedy 3.70/8.23 -> 3.63/8.24; int8 + PD LM
// 3.33/7.15 -> 3.35/7.25; int4 4.61/9.98 -> 4.60/9.96; transducer 2.99/6.67 -> 3.01/6.69), so apply it together with
// the board-measurement update of the README, model cards, blog and paper. Replace in tasr_nemo.c / tinyasr_lm.c /
// tasr_seg.c: expf->tasr_expf, logf->tasr_logf, tanhf->tasr_tanhf, 1/(1+expf(-x))->tasr_sigmoidf, log1pf(expf(d)) ->
// tasr_logf(1 + tasr_expf(d)), powf(a, 0.02f) -> tasr_expf(0.02f * tasr_logf(a)). Max rel. error ~2e-7.
#pragma once
#include <stdint.h>
#include <string.h>

static inline float tasr_expf(float x)
{
    if (x < -87.0f) return 0.f;
    if (x > 88.0f) x = 88.0f;
    const float fn = x * 1.44269504088896341f;
    const int n = (int)(fn + (fn >= 0.f ? 0.5f : -0.5f));
    const float r = (x - (float)n * 0.693145751953125f) - (float)n * 1.428606765330187e-06f;
    float p = 1.f + r * (1.f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * (0.0083333338f + r * 0.0013888889f)))));
    uint32_t b;
    memcpy(&b, &p, 4);
    b += (uint32_t)n << 23;
    memcpy(&p, &b, 4);
    return p;
}

static inline float tasr_logf(float x)
{
    if (!(x > 1.17549435e-38f)) return -87.3365f;
    uint32_t b;
    memcpy(&b, &x, 4);
    int e = (int)((b >> 23) & 0xff) - 127;
    b = (b & 0x007fffffu) | 0x3f800000u;
    float m;
    memcpy(&m, &b, 4);
    if (m > 1.41421356f) { m *= 0.5f; e += 1; }
    const float s = (m - 1.f) / (m + 1.f), s2 = s * s;
    const float lm = 2.f * s * (1.f + s2 * (0.33333334f + s2 * (0.2f + s2 * (0.14285715f + s2 * (0.11111111f + s2 * 0.09090909f)))));
    return lm + (float)e * 0.693147180559945f;
}

static inline float tasr_tanhf(float x)
{
    if (x > 9.f) return 1.f;
    if (x < -9.f) return -1.f;
    return 1.f - 2.f / (tasr_expf(2.f * x) + 1.f);
}

static inline float tasr_sigmoidf(float x) { return 1.f / (1.f + tasr_expf(-x)); }
