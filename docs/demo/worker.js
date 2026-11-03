// Runs the Oído engine (WebAssembly build of the ESP32 firmware's C code) off the main thread.
importScripts('oido.js');
let M = null, loaded = false, audioSec = 0, engineMs = 0;
const ready = OidoModule().then(m => { M = m; });
let bufPtr = 0, bufCap = 0, modelPtr = 0, lmPtr = 0, pcmPtr = 0, pcmCap = 0;

function put(bytes) {                       // copy into 16-byte aligned wasm memory that stays alive while the model is loaded
  const p = M._oido_alloc(bytes.byteLength);
  M.HEAPU8.set(new Uint8Array(bytes), p);
  return p;
}
function pcmBuffer(n) {
  if (n > pcmCap) { if (pcmPtr) M._oido_free(pcmPtr); pcmCap = Math.max(n, 16000); pcmPtr = M._oido_alloc(pcmCap * 2); }
  return pcmPtr;
}

onmessage = async (e) => {
  const d = e.data;
  await ready;
  try {
    if (d.type === 'load') {
      M._oido_unload();
      if (modelPtr) M._oido_free(modelPtr);
      if (lmPtr) M._oido_free(lmPtr);
      modelPtr = put(d.model); lmPtr = put(d.lm);
      const rc = M._oido_load(modelPtr, d.model.byteLength, lmPtr, d.lm.byteLength, d.beam, d.topk);
      loaded = rc === 0; audioSec = 0; engineMs = 0;
      postMessage({ type: 'loaded', ok: loaded, rc });
    } else if (d.type === 'reset') {
      M._oido_reset(); audioSec = 0; engineMs = 0;
      postMessage({ type: 'reset' });
    } else if (d.type === 'feed') {
      const pcm = d.pcm, p = pcmBuffer(pcm.length);
      M.HEAP16.set(pcm, p >> 1);
      const t0 = performance.now();
      M._oido_feed(p, pcm.length);
      const text = M.UTF8ToString(M._oido_partial());
      engineMs += performance.now() - t0; audioSec += pcm.length / 16000;
      postMessage({ type: 'text', text, audioSec, engineMs });
    } else if (d.type === 'finish') {
      const t0 = performance.now();
      const text = M.UTF8ToString(M._oido_finish());
      engineMs += performance.now() - t0;
      postMessage({ type: 'final', text, audioSec, engineMs });
    }
  } catch (err) {
    postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
