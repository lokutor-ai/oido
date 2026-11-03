// Live microphone mode: I2S MEMS mic -> capture task -> PSRAM ring buffer -> AGC + energy VAD -> streaming ASR.
// The capture task runs at high priority so no audio is lost while the ASR task is busy with an encoder chunk.
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/stream_buffer.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "sdkconfig.h"
#include "tinyasr.h"
#include "tasr_nemo.h"
#include "tasr_seg.h"
#include "oled.h"

#ifdef CONFIG_TASR_MODE_MIC
#define RING_SAMPLES (16000 * 8)  // 8 s of audio headroom (speech continuing while an utterance is transcribed)
static StreamBufferHandle_t g_ring;
static i2s_chan_handle_t g_rx;

static void capture_task(void *arg)
{
    (void)arg;
    static int32_t raw[320];
    static int16_t pcm[320];
    float dc = 0.f;
    for (;;) {
        size_t got = 0;
        i2s_channel_read(g_rx, raw, sizeof(raw), &got, portMAX_DELAY);
        int n = got / 4;
        for (int i = 0; i < n; i++) {  // 24-bit left-justified -> 16-bit with DC removal
            float x = (float)(raw[i] >> 8) / 256.0f;
            dc += 0.001f * (x - dc);
            float y = x - dc;
            pcm[i] = (int16_t)(y > 32767.f ? 32767.f : (y < -32768.f ? -32768.f : y));
        }
        if (xStreamBufferSend(g_ring, pcm, n * 2, 0) != (size_t)n * 2) ESP_LOGW("mic", "ring overflow");
    }
}
#endif

#ifdef CONFIG_TASR_MODE_MIC
static void mic_start(void)
{
    static int started;
    if (started) return;
    started = 1;
    static const char *TAG = "mic";
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_desc_num = 8;
    cc.dma_frame_num = 320;
    ESP_ERROR_CHECK(i2s_new_channel(&cc, NULL, &g_rx));
    i2s_std_config_t sc = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {.mclk = I2S_GPIO_UNUSED, .bclk = CONFIG_TASR_I2S_BCLK, .ws = CONFIG_TASR_I2S_WS,
                     .dout = I2S_GPIO_UNUSED, .din = CONFIG_TASR_I2S_DIN},
    };
    sc.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;  // INMP441 with L/R tied to GND
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(g_rx, &sc));
    ESP_ERROR_CHECK(i2s_channel_enable(g_rx));
    uint8_t *storage = heap_caps_malloc(RING_SAMPLES * 2 + 1, MALLOC_CAP_SPIRAM);
    static StaticStreamBuffer_t sb;
    g_ring = xStreamBufferCreateStatic(RING_SAMPLES * 2, 2, storage, &sb);
    xTaskCreatePinnedToCore(capture_task, "mic_cap", 4096, NULL, configMAX_PRIORITIES - 1, NULL, 0);
    ESP_LOGI(TAG, "listening (BCLK %d WS %d DIN %d)", CONFIG_TASR_I2S_BCLK, CONFIG_TASR_I2S_WS, CONFIG_TASR_I2S_DIN);
}

#endif

// Utterance mode for the NeMo engine: record while speech is detected (max 20 s), transcribe after 0.6 s of silence.
void run_mic_nemo(const tasr_nemo_t *m, struct tasr_decoder *dec, tasr_nemo_stream_t *st_pre)
{
#ifdef CONFIG_TASR_MODE_MIC
    mic_start();
    const int cap = 16000 * 20;
    int16_t *utt = heap_caps_malloc(sizeof(int16_t) * cap, MALLOC_CAP_SPIRAM);
    static int16_t in[320];
    static char text[4096];
    static tasr_seg_t seg;
    tasr_seg_init(&seg, utt, cap);
    seg.hang = CONFIG_TASR_SEG_HANG_MS / 20;  // 320-sample blocks
    oled_start();
    char idle[32] = "* listening";
    oled_status(idle);
    int hearing = 0;
    tasr_nemo_stream_t *st = st_pre ? st_pre : tasr_nemo_stream_new(m, CONFIG_TASR_STREAM_CHUNK, CONFIG_TASR_STREAM_LEFT, dec);   // st_pre: created before the weights were placed
    if (st) {  // streaming model: the encoder runs while you speak; only the last chunk is left when you stop
        printf("ready (streaming): speak, partial text follows you, the final line prints after each pause\n");
        int fed = 0;
        for (;;) {
            size_t got = xStreamBufferReceive(g_ring, in, sizeof(in), portMAX_DELAY);
            const int n = tasr_seg_feed(&seg, in, (int)(got / 2));
            if (hearing && !seg.speech && !n) {  // a blip the segmenter discarded: forget what the stream heard
                tasr_nemo_stream_reset(st);
                fed = 0;
                oled_status(idle);
            }
            hearing = seg.speech;
            if (!seg.speech && !n) continue;
            const int have = n ? n : seg.n;
            if (have > fed) {
                const int before = tasr_nemo_stream_text(st, NULL, 0);
                if (tasr_nemo_stream_feed(st, utt + fed, have - fed) > before) {
                    tasr_nemo_stream_text(st, text, sizeof(text));
                    printf("\r... %s", text);
                    fflush(stdout);
                    oled_partial(text);
                }
                fed = have;
            }
            if (!n) continue;
            int64_t t0 = esp_timer_get_time();
            tasr_nemo_stream_finish(st, text, sizeof(text));
            const double dt = (esp_timer_get_time() - t0) / 1e6;
            printf("\r%s   [%.1f s audio, final text %.0f ms after the pause was detected]\n", text, n / 16000.0, dt * 1000);
            oled_push(text[0] ? text : "(nothing recognized)");
            snprintf(idle, sizeof(idle), "* listening  +%.0f ms", dt * 1000);
            oled_status(idle);
            tasr_nemo_stream_reset(st);
            fed = 0;
            hearing = 0;
            tasr_seg_next(&seg);
        }
    }
    printf("ready: speak, the transcript prints after each pause\n");
    for (;;) {
        size_t got = xStreamBufferReceive(g_ring, in, sizeof(in), portMAX_DELAY);
        const int n = tasr_seg_feed(&seg, in, (int)(got / 2));
        if (seg.speech != hearing) {  // speech started, or a blip was discarded
            hearing = seg.speech;
            oled_status(hearing ? "* hearing you..." : idle);
        }
        if (!n) continue;
        oled_status("transcribing...");
        int64_t t0 = esp_timer_get_time();
        tasr_nemo_transcribe(m, utt, n, dec, text, sizeof(text), NULL, 0, NULL);
        const double dt = (esp_timer_get_time() - t0) / 1e6;
        printf("%s   [%.1f s audio, %.2f s compute, RTF %.2f]\n", text, n / 16000.0, dt, dt / (n / 16000.0));
        oled_push(text[0] ? text : "(nothing recognized)");
        snprintf(idle, sizeof(idle), "* listening  rtf %.2f", dt / (n / 16000.0));
        oled_status(idle);
        hearing = 0;
        tasr_seg_next(&seg);
    }
#else
    (void)m; (void)dec;
#endif
}

void run_mic(tasr_stream_t *s)
{
#ifdef CONFIG_TASR_MODE_MIC
    mic_start();

    static int16_t in[320], pcm[320];
    static tasr_seg_t vad;  // shared VAD/AGC; the streaming model consumes audio as it arrives
    tasr_seg_init(&vad, NULL, 0);
    oled_start();
    oled_status("* listening (stream)");
    for (;;) {
        size_t got = xStreamBufferReceive(g_ring, in, sizeof(in), portMAX_DELAY);
        const int n = (int)(got / 2);
        const int voiced = tasr_seg_vad(&vad, in, pcm, n);
        if (voiced) { vad.speech = 1; vad.silence = 0; }
        else vad.silence++;
        if (!vad.speech) continue;
        if (tasr_stream_feed(s, pcm, n) > 0) {
            printf("\r%s", tasr_stream_text(s));
            fflush(stdout);
            oled_partial(tasr_stream_text(s));
        }
        if (vad.silence > TASR_SEG_HANG_BLOCKS) {  // 0.8 s of silence ends the utterance
            tasr_stream_finish(s);
            printf("\r%s\n", tasr_stream_text(s));
            oled_push(tasr_stream_text(s));
            tasr_stream_reset(s);
            vad.speech = 0;
        }
    }
#else
    (void)s;
#endif
}
