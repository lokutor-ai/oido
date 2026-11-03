// Microphone capture: resamples the audio context's rate to 16 kHz (windowed sinc) and posts 100 ms blocks of float samples.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.fc = Math.min(1, 0.94 / this.ratio);          // cutoff as a fraction of the input Nyquist
    this.N = Math.max(8, Math.ceil(8 * this.ratio));   // taps on each side of the output position
    this.hist = new Float32Array(0);
    this.base = 0;                                      // absolute input index of hist[0]
    this.pos = 0;                                       // absolute input position of the next output sample
    this.out = new Float32Array(1600);
    this.nout = 0;
  }
  kernel(d) {                                           // windowed sinc at distance d (input samples)
    const x = Math.PI * this.fc * d;
    const s = Math.abs(d) < 1e-9 ? 1 : Math.sin(x) / x;
    const w = 0.5 + 0.5 * Math.cos(Math.PI * d / this.N);
    return this.fc * s * w;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    const h = new Float32Array(this.hist.length + ch.length);
    h.set(this.hist); h.set(ch, this.hist.length);
    this.hist = h;
    const N = this.N;
    while (this.pos + N < this.base + this.hist.length) {
      const c = Math.floor(this.pos);
      let acc = 0;
      for (let n = c - N + 1; n <= c + N; n++) {
        const i = n - this.base;
        if (i >= 0) acc += this.hist[i] * this.kernel(n - this.pos);
      }
      this.out[this.nout++] = acc;
      if (this.nout === this.out.length) { this.port.postMessage(this.out.slice(0)); this.nout = 0; }
      this.pos += this.ratio;
    }
    const keep = Math.floor(this.pos) - N - 1;         // drop what no later output needs
    if (keep > this.base) { this.hist = this.hist.slice(keep - this.base); this.base = keep; }
    return true;
  }
}
registerProcessor('capture', Capture);
