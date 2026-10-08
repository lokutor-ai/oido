// tinyasr firmware for ESP32-S3 (N16R8): transcribes the utterances stored in the "audio" partition
// and reports transcript + real-time factor measured with the CPU cycle counter.
#include <stdio.h>
#include <string.h>
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "sdkconfig.h"
#include "tinyasr.h"
#include "tinyasr_lm.h"
#include "tasr_nemo.h"

static const char *TAG = "tinyasr";

// ---- dual-core parallel-for: worker task pinned to core 1 runs the upper half of each job
static struct {
    tasr_job_fn fn;
    void *ctx;
    int b, e;
} g_job;
static SemaphoreHandle_t g_start, g_done;
static void worker_task(void *arg)
{
    (void)arg;
    for (;;) {
        xSemaphoreTake(g_start, portMAX_DELAY);
        g_job.fn(g_job.ctx, g_job.b, g_job.e, 1);
        xSemaphoreGive(g_done);
    }
}
static void par_for(tasr_job_fn fn, void *ctx, int n)
{
    if (n < 2) { fn(ctx, 0, n, 0); return; }
    int mid = n / 2;
    g_job.fn = fn; g_job.ctx = ctx; g_job.b = mid; g_job.e = n;
    xSemaphoreGive(g_start);
    fn(ctx, 0, mid, 0);
    xSemaphoreTake(g_done, portMAX_DELAY);
}
#define CPU_HZ 240000000.0

#define INTERNAL_RESERVE (72 * 1024)  // keep internal RAM for I2S DMA, Wi-Fi/BLE stacks, task stacks
static void *fw_alloc(size_t n, int kind)
{
    void *p = NULL;
    if (kind == 1 && n <= 48 * 1024 &&
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL) > n + INTERNAL_RESERVE)
        p = heap_caps_aligned_alloc(16, n ? n : 16, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) p = heap_caps_aligned_alloc(16, n ? n : 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (p) memset(p, 0, n);
    else ESP_LOGE(TAG, "alloc %u kind %d failed", (unsigned)n, kind);
    return p;
}
static void fw_free(void *p) { heap_caps_free(p); }

static const void *map_partition(uint8_t subtype, const char *name, size_t *size)
{
    const esp_partition_t *p = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, subtype, name);
    if (!p) return NULL;
    const void *ptr;
    esp_partition_mmap_handle_t h;
    if (esp_partition_mmap(p, 0, p->size, ESP_PARTITION_MMAP_DATA, &ptr, &h) != ESP_OK) return NULL;
    *size = p->size;
    return ptr;
}

static void calibrate(void)
{
    // 4M single-cycle NOPs (+ loop setup): on silicon cycles ~= 4.0M; under QEMU -icount this gives cycles/instruction
    uint32_t c0 = esp_cpu_get_cycle_count();
    int64_t t0 = esp_timer_get_time();
    __asm__ volatile("movi a8, 1000000\n loop a8, 1f\n nop\n nop\n nop\n nop\n 1:\n" ::: "a8");
    uint32_t dc = esp_cpu_get_cycle_count() - c0;
    printf("CALIB 4M nops: cycles %u, time %lld us\n", (unsigned)dc, esp_timer_get_time() - t0);
}

// ---- NeMo Conformer-CTC small (utterance mode): transcribe each stored utterance, report RTF
static void run_nemo(const uint8_t *blob, uint32_t blob_size)
{
    tasr_nemo_t *m = tasr_nemo_load(blob, blob_size);
    if (!m) { ESP_LOGE(TAG, "nemo model load failed"); return; }
    tasr_decoder_t *dec = NULL;
    size_t lsize = 0;
    const uint8_t *lmp = map_partition(0x42, "lm", &lsize);
    uint32_t lbytes = 0;
    if (lmp) memcpy(&lbytes, lmp, 4);
    if (lmp && lbytes > 0 && lbytes < lsize) {
        tasr_lm_t *lm = tasr_lm_load(lmp + 16, lbytes);
        if (lm) {
            size_t mv = tasr_lm_to_ram(lm);
            dec = tasr_decoder_create(lm, 1025, CONFIG_TASR_BEAM, 6, tasr_lm_weight(lm, CONFIG_TASR_LM_WEIGHT_X100 / 100.0f),
                                      tasr_lm_bonus(lm, CONFIG_TASR_TOKEN_BONUS_X100 / 100.0f));
            ESP_LOGI(TAG, "LM %u bytes in PSRAM, beam %d", (unsigned)mv, CONFIG_TASR_BEAM);
        }
    }
    size_t free_ps = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
#ifdef CONFIG_TASR_MODE_MIC
    const size_t reserve = 3600000;  // 20 s utterance buffers (~2.5 MB) + mic ring and recording buffer (0.9 MB)
#else
    const size_t reserve = 2700000;  // per-utterance buffers for clips up to 20 s (1.9 MB measured at 15 s)
#endif
    const int streaming = tasr_nemo_stream_supported(m);
    // a streaming model also keeps its attention / conv caches and positional rows (~2 MB at chunk 16, left 128)
    size_t budget = free_ps > reserve + (streaming ? 2200000 : 0) ? free_ps - reserve - (streaming ? 2200000 : 0) : 0;
    size_t moved = tasr_nemo_place_weights(m, budget);
    ESP_LOGI(TAG, "NeMo conformer-ctc-small: %u weight bytes, %u moved to PSRAM | free PSRAM %u internal %u",
             (unsigned)tasr_nemo_weight_bytes(m), (unsigned)moved, (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
#ifdef CONFIG_TASR_MODE_MIC
    extern void run_mic_nemo(const tasr_nemo_t *m, struct tasr_decoder *dec);
    run_mic_nemo(m, dec);
#endif
    size_t asize = 0;
    const uint8_t *aud = map_partition(0x41, "audio", &asize);
    if (!aud || memcmp(aud, "AUD0", 4)) { ESP_LOGE(TAG, "no audio partition"); return; }
    uint32_t n_utt;
    memcpy(&n_utt, aud + 4, 4);
    const uint8_t *p = aud + 8;
    double tot_audio = 0, tot_cycles = 0, tot_fin = 0;
    static char text[4096];
    tasr_nemo_stream_t *st = streaming ? tasr_nemo_stream_new(m, 0, 0, dec) : NULL;
    if (streaming) printf("STREAM model: chunk/left defaults, stream %s\n", st ? "ok" : "ALLOCATION FAILED");
    for (uint32_t u = 0; u < n_utt; u++) {
        uint32_t ns, tl;
        memcpy(&ns, p, 4); memcpy(&tl, p + 4, 4);
        const char *ref = (const char *)(p + 8);
        const int16_t *pcm = (const int16_t *)(p + 8 + ((tl + 3) & ~3u));
        p = (const uint8_t *)(pcm + ns + (ns & 1));
        uint32_t c0 = esp_cpu_get_cycle_count();
        tasr_nemo_transcribe(m, pcm, ns, dec, text, sizeof(text), NULL, 0, NULL);
        uint32_t cyc = esp_cpu_get_cycle_count() - c0;
        double sec = ns / 16000.0;
        tot_audio += sec;
        tot_cycles += cyc;
        printf("UTT %u | audio %.2fs | cycles %u | RTF %.3f\nREF: %.*s\nHYP: %s\n", (unsigned)u, sec, (unsigned)cyc,
               cyc / CPU_HZ / sec, (int)tl, ref, text);
        if (st) {  // the same audio arriving in 20 ms blocks; what matters is the compute left after the last block
            tasr_nemo_stream_reset(st);
            uint32_t s0 = esp_cpu_get_cycle_count();
            for (uint32_t o = 0; o < ns; o += 320) tasr_nemo_stream_feed(st, pcm + o, ns - o < 320 ? ns - o : 320);
            uint32_t s1 = esp_cpu_get_cycle_count();
            tasr_nemo_stream_finish(st, text, sizeof(text));
            uint32_t s2 = esp_cpu_get_cycle_count();
            tot_fin += s2 - s1;
            printf("STREAM %u | feed cycles %u | finish cycles %u (%.0f ms @240MHz) | RTF %.3f\nSHYP: %s\n", (unsigned)u,
                   (unsigned)(s1 - s0), (unsigned)(s2 - s1), (s2 - s1) / CPU_HZ * 1000.0, (s2 - s0) / CPU_HZ / sec, text);
        }
    }
    if (st) printf("STREAM mean finish %.0f ms @240MHz\n", tot_fin / n_utt / CPU_HZ * 1000.0);
    printf("TOTAL audio %.1fs cycles %.0f RTF@240MHz %.3f\n", tot_audio, tot_cycles, tot_cycles / CPU_HZ / tot_audio);
    printf("MEM min free PSRAM %u internal %u\n", (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
    for (int i = 0; tasr_nemo_profile_name(i); i++)
        printf("PROF %-10s %12llu cycles %5.1f%%\n", tasr_nemo_profile_name(i), (unsigned long long)tasr_nemo_profile_value(i),
               100.0 * tasr_nemo_profile_value(i) / tot_cycles);
    printf("DONE\n");
}

void app_main(void)
{
    calibrate();
    tasr_alloc = fw_alloc;
    tasr_free = fw_free;
#ifndef TASR_SINGLE_CORE
    g_start = xSemaphoreCreateBinary();
    g_done = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(worker_task, "asr_w1", 16384, NULL, configMAX_PRIORITIES - 2, NULL, 1);
    tasr_parallel = par_for;
    ESP_LOGI(TAG, "dual-core enabled");
#endif
    size_t msize = 0, asize = 0;
    const uint8_t *mflash = map_partition(0x40, "model", &msize);
    const uint8_t *aud = map_partition(0x41, "audio", &asize);
    if (!mflash) { ESP_LOGE(TAG, "model partition missing"); return; }
    uint32_t blob_size;
    memcpy(&blob_size, mflash, 4);  // image = [u32 size][pad to 16][TASR blob]
    const uint8_t *blob = mflash + 16;
    ESP_LOGI(TAG, "model blob %u bytes; free PSRAM %u, internal %u", (unsigned)blob_size,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    // weights stay memory-mapped in flash; hot matrices are moved to PSRAM after the stream is allocated
    if (!memcmp(blob, "TNM1", 4)) { run_nemo(blob, blob_size); return; }
    tasr_model_t *m = tasr_model_load(blob, blob_size);
    if (!m) { ESP_LOGE(TAG, "model load failed"); return; }
    tasr_stream_t *s = tasr_stream_create(m, CONFIG_TASR_CHUNK, CONFIG_TASR_LEFT);
    if (!s) { ESP_LOGE(TAG, "stream create failed"); return; }
    {   // optional LM + beam search (partition "lm": [u32 size][pad16][TLM1 blob])
        size_t lsize = 0;
        const uint8_t *lmp = map_partition(0x42, "lm", &lsize);
        uint32_t lbytes = 0;
        if (lmp) memcpy(&lbytes, lmp, 4);
        if (lmp && lbytes > 0 && lbytes < lsize) {
            tasr_lm_t *lm = tasr_lm_load(lmp + 16, lbytes);
            if (lm) {
                size_t mv = tasr_lm_to_ram(lm);  // LM weights are re-read every emitting frame: keep them in PSRAM
                ESP_LOGI(TAG, "LM weights in PSRAM: %u bytes", (unsigned)mv);
                tasr_decoder_t *dec = tasr_decoder_create(lm, 257, CONFIG_TASR_BEAM, 6, CONFIG_TASR_LM_WEIGHT_X100 / 100.0f,
                                                          CONFIG_TASR_TOKEN_BONUS_X100 / 100.0f);
                tasr_stream_set_decoder(s, dec);
                ESP_LOGI(TAG, "LM beam search: beam %d, lm %u bytes", CONFIG_TASR_BEAM,
                         (unsigned)lbytes);
            }
        }
    }
#ifndef TASR_WEIGHTS_IN_FLASH
    {
        size_t free_ps = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
        size_t budget = free_ps > 600000 ? free_ps - 600000 : 0;  // keep headroom for the logit sink / app
        int64_t t0 = esp_timer_get_time();
        size_t moved = tasr_model_place_weights(m, budget);
        ESP_LOGI(TAG, "moved %u of %u weight bytes to PSRAM in %lld ms", (unsigned)moved,
                 (unsigned)tasr_model_weight_bytes(m), (esp_timer_get_time() - t0) / 1000);
    }
#endif
#ifdef CONFIG_TASR_MODE_MIC
    extern void run_mic(tasr_stream_t *s);
    run_mic(s);
#endif
    const int max_frames = 400, V1 = 257;
    float *sink = heap_caps_malloc(sizeof(float) * max_frames * V1, MALLOC_CAP_SPIRAM);
    int n_frames = 0;

    ESP_LOGI(TAG, "weights %u bytes | after alloc: free PSRAM %u, internal %u", (unsigned)tasr_model_weight_bytes(m),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    if (memcmp(aud, "AUD0", 4)) { ESP_LOGE(TAG, "no audio"); return; }
    uint32_t n_utt;
    memcpy(&n_utt, aud + 4, 4);
    const uint8_t *p = aud + 8;
    double tot_audio = 0, tot_cycles = 0;
    uint32_t worst_block = 0;
    for (uint32_t u = 0; u < n_utt; u++) {
        uint32_t ns, tl;
        memcpy(&ns, p, 4); memcpy(&tl, p + 4, 4);
        const char *ref = (const char *)(p + 8);
        const int16_t *pcm = (const int16_t *)(p + 8 + ((tl + 3) & ~3u));
        p = (const uint8_t *)(pcm + ns + (ns & 1));
        tasr_stream_reset(s);
        if (sink) tasr_stream_set_logit_sink(s, sink, max_frames, &n_frames);
        uint64_t cyc = 0;
        for (uint32_t i = 0; i < ns; i += 320) {  // 20 ms blocks as from an I2S mic
            int n = ns - i < 320 ? ns - i : 320;
            uint32_t c0 = esp_cpu_get_cycle_count();
            tasr_stream_feed(s, pcm + i, n);
            uint32_t dc = esp_cpu_get_cycle_count() - c0;
            cyc += dc;
            if (dc > worst_block) worst_block = dc;
        }
        uint32_t c0 = esp_cpu_get_cycle_count();
        tasr_stream_finish(s);
        cyc += esp_cpu_get_cycle_count() - c0;
        double sec = ns / 16000.0;
        tot_audio += sec;
        tot_cycles += cyc;
        double cs1 = 0, cs2 = 0;
        for (int i = 0; i < n_frames * V1; i++) { cs1 += sink[i]; cs2 += (double)sink[i] * sink[i]; }
        printf("UTT %u | audio %.2fs | cycles %llu | RTF %.3f | frames %d logit-sum %.4f logit-sq %.4f\nREF: %.*s\nHYP: %s\n",
               (unsigned)u, sec, (unsigned long long)cyc, cyc / CPU_HZ / sec, n_frames, cs1, cs2, (int)tl, ref,
               tasr_stream_text(s));
    }
    printf("TOTAL audio %.1fs cycles %.0f RTF@240MHz %.3f worst-block %.1f ms\n", tot_audio, tot_cycles,
           tot_cycles / CPU_HZ / tot_audio, worst_block / CPU_HZ * 1000);
    double ptot = 0;
    for (int i = 0; tasr_profile_name(i); i++) ptot += tasr_profile_value(i);
    for (int i = 0; tasr_profile_name(i); i++)
        printf("PROF %-10s %12llu cycles %5.1f%%\n", tasr_profile_name(i), (unsigned long long)tasr_profile_value(i),
               100.0 * tasr_profile_value(i) / (ptot > 0 ? ptot : 1));
    printf("DONE\n");
}
