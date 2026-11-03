'use strict';
const BASE = 'https://raw.githubusercontent.com/lokutor-ai/oido/main/models/';
const VARIANTS = {
  A: { file: 'oido_v2a.tnm', beam: 4, topk: 6, rtf: 0.72, delay: 430 },
  B: { file: 'oido_v2b.tnm', beam: 2, topk: 3, rtf: 0.54, delay: 260 },
};
const EXAMPLES = [
  { file: 'examples/b14.wav', ref: 'as for the ichthyosaurus has he returned to his submarine cavern', who: 'LibriSpeech' },
  { file: 'examples/b16.wav', ref: 'for ned was so absorbed in business that he ignored the whole bailey question and left them in utter darkness', who: 'LibriSpeech' },
  { file: 'examples/b05.wav', ref: 'this is best done in the spring when disturbed plants can recover', who: 'Common Voice' },
];
const $ = (id) => document.getElementById(id);
const out = $('out'), stats = $('stats'), status = $('status'), bar = $('bar'), micBtn = $('mic');
const worker = new Worker('worker.js');
const cache = {};
let lmBuf = null, current = null, recording = null, busy = false, waiters = [];

worker.onmessage = (e) => {
  const d = e.data;
  if (d.type === 'text') { show(d.text, true); showStats(d); }
  else if (d.type === 'final') { show(d.text, false); showStats(d); }
  else if (d.type === 'error') { status.innerHTML = '<span class="err">Engine error: ' + d.message + '</span>'; }
  waiters = waiters.filter((w) => !(w.type === d.type && (w.resolve(d), true)));
};
const waitFor = (type) => new Promise((resolve) => waiters.push({ type, resolve }));

function show(text, partial) {
  out.textContent = text || (partial ? '…' : '(nothing recognized)');
  out.className = partial ? 'partial' : '';
}
function showStats(d) {
  const speed = d.engineMs > 0 ? (d.audioSec * 1000 / d.engineMs) : 0;
  stats.textContent = d.audioSec.toFixed(1) + ' s of audio · ' + d.engineMs.toFixed(0) + ' ms of engine time in this browser (' + speed.toFixed(0) + '× faster than speech)';
}
function selected() { return document.querySelector('input[name=variant]:checked').value; }

async function download(url, label) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('download failed (' + r.status + ')');
  const total = +r.headers.get('content-length') || 0, reader = r.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    if (total) { bar.firstElementChild.style.width = (100 * got / total).toFixed(0) + '%'; status.textContent = label + ' ' + (got / 1e6).toFixed(1) + ' / ' + (total / 1e6).toFixed(1) + ' MB'; }
  }
  const buf = new Uint8Array(got); let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  return buf.buffer;
}

async function ensureModel(key) {
  if (current === key) return true;
  bar.style.display = 'block'; bar.firstElementChild.style.width = '0';
  try {
    if (!lmBuf) lmBuf = await download(BASE + 'oido_v2.tlm', 'Downloading the language model');
    if (!cache[key]) cache[key] = await download(BASE + VARIANTS[key].file, 'Downloading model ' + key);
    status.textContent = 'Loading the engine…';
    const v = VARIANTS[key];
    worker.postMessage({ type: 'load', model: cache[key].slice(0), lm: lmBuf.slice(0), beam: v.beam, topk: v.topk });
    const r = await waitFor('loaded');
    if (!r.ok) throw new Error('model load failed (' + r.rc + ')');
    current = key;
    $('chiprtf').textContent = v.rtf.toFixed(2); $('chipdelay').textContent = v.delay;
    status.textContent = 'Model ' + key + ' ready.';
    return true;
  } catch (err) {
    status.innerHTML = '<span class="err">' + err.message + '</span>';
    return false;
  } finally { bar.style.display = 'none'; }
}

const toInt16 = (f) => { const o = new Int16Array(f.length); for (let i = 0; i < f.length; i++) o[i] = Math.max(-32768, Math.min(32767, Math.round(f[i] * 32768))); return o; };   // exact for audio that was 16-bit

async function startMic() {
  if (busy) return;
  if (!(window.AudioWorklet && window.WebAssembly && navigator.mediaDevices)) { status.innerHTML = '<span class="err">This browser lacks AudioWorklet or microphone access.</span>'; return; }
  busy = true; micBtn.disabled = true;
  if (!(await ensureModel(selected()))) { busy = false; micBtn.disabled = false; return; }
  try {
    worker.postMessage({ type: 'reset' }); await waitFor('reset');
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    const ctx = new AudioContext({ latencyHint: 'interactive' });
    await ctx.audioWorklet.addModule('worklet.js');
    const src = ctx.createMediaStreamSource(stream), node = new AudioWorkletNode(ctx, 'capture', { numberOfInputs: 1, numberOfOutputs: 0 });
    node.port.onmessage = (e) => { const pcm = toInt16(e.data); worker.postMessage({ type: 'feed', pcm }, [pcm.buffer]); };
    src.connect(node);
    recording = { stream, ctx, node };
    micBtn.textContent = 'Stop'; micBtn.classList.add('rec'); status.textContent = 'Listening… speak now.'; show('', true);
  } catch (err) {
    status.innerHTML = '<span class="err">Microphone: ' + (err.message || err) + '</span>';
  }
  busy = false; micBtn.disabled = false;
}
async function stopMic() {
  if (!recording) return;
  const r = recording; recording = null; micBtn.disabled = true;
  r.stream.getTracks().forEach((t) => t.stop());
  r.node.disconnect(); await r.ctx.close();
  worker.postMessage({ type: 'finish' }); await waitFor('final');
  micBtn.textContent = 'Start microphone'; micBtn.classList.remove('rec'); micBtn.disabled = false; status.textContent = 'Done.';
}
micBtn.onclick = () => (recording ? stopMic() : startMic());

async function transcribeBuffer(arrayBuf) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });   // decoding resamples straight to 16 kHz (no resampling for 16 kHz files)
  const dec = await ctx.decodeAudioData(arrayBuf); ctx.close();
  let f = dec.getChannelData(0);
  if (dec.numberOfChannels > 1) {                          // mix to mono
    f = new Float32Array(dec.length);
    for (let c = 0; c < dec.numberOfChannels; c++) { const x = dec.getChannelData(c); for (let i = 0; i < f.length; i++) f[i] += x[i] / dec.numberOfChannels; }
  }
  let pcm = toInt16(f);
  if (Math.abs(dec.sampleRate - 16000) > 1) {              // a browser that ignored the requested rate: resample offline
    const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(dec.duration * 16000)), 16000);
    const s = off.createBufferSource(); const b = off.createBuffer(1, f.length, dec.sampleRate); b.copyToChannel(f, 0); s.buffer = b; s.connect(off.destination); s.start();
    pcm = toInt16((await off.startRendering()).getChannelData(0));
  }
  worker.postMessage({ type: 'reset' }); await waitFor('reset');
  for (let i = 0; i < pcm.length; i += 8000) {            // 0.5 s blocks, as a live stream would arrive
    const blk = pcm.slice(i, i + 8000); worker.postMessage({ type: 'feed', pcm: blk }, [blk.buffer]);
  }
  worker.postMessage({ type: 'finish' }); await waitFor('final');
}
async function runFile(getBuf) {
  if (busy || recording) return;
  busy = true; micBtn.disabled = true; $('pick').disabled = true;
  try {
    if (await ensureModel(selected())) { status.textContent = 'Transcribing…'; show('', true); await transcribeBuffer(await getBuf()); status.textContent = 'Done.'; }
  } catch (err) { status.innerHTML = '<span class="err">' + (err.message || err) + '</span>'; }
  busy = false; micBtn.disabled = false; $('pick').disabled = false;
}
$('pick').onclick = () => $('file').click();
$('file').onchange = (e) => { const f = e.target.files[0]; if (f) runFile(() => f.arrayBuffer()); e.target.value = ''; };
for (const ex of EXAMPLES) {
  const b = document.createElement('button'); b.innerHTML = '▶ Try an example <small>(' + ex.who + ': “' + ex.ref + '”)</small>';
  b.onclick = () => runFile(async () => (await fetch(ex.file)).arrayBuffer());
  $('examples').appendChild(b);
}
document.querySelectorAll('input[name=variant]').forEach((r) => r.addEventListener('change', () => { const v = VARIANTS[selected()]; $('chiprtf').textContent = v.rtf.toFixed(2); $('chipdelay').textContent = v.delay; }));
