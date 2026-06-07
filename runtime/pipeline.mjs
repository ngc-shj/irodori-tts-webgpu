// Environment-agnostic Irodori-TTS inference core (Node + browser).
// The caller injects: an `ort` module, created InferenceSessions, a tokenizer
// (transformers.js), and the text normalizer. All tensor glue (RF Euler loop,
// CFG, duration->seqlen, bundle assembly) lives here — mirrors irodori_tts/rf.py
// and inference_runtime.py. Verified against PyTorch (corr=1.0).

const HOP = 1920;
const SR = 48000;
const LATENT_DIM = 32;
const BOS = 1;

// ---- text normalization (port of irodori_tts/text_normalization.py) ----
const SIMPLE = [
  ["\t", ""], ["[n]", ""], ["\\[n\\]", ""], ["　", ""], ["？", "?"], ["！", "!"],
  ["♥", "♡"], ["●", "○"], ["◯", "○"], ["〇", "○"],
];
function stripOuterBrackets(text) {
  const pairs = { "「": "」", "『": "』", "（": "）", "【": "】", "(": ")" };
  while (text.length >= 2) {
    const s = text[0], e = text[text.length - 1];
    if (pairs[s] === e) {
      let depth = 0, all = true;
      for (let i = 0; i < text.length; i++) {
        if (text[i] === s) depth++; else if (text[i] === e) depth--;
        if (depth === 0 && i < text.length - 1) { all = false; break; }
      }
      if (all && depth === 0) { text = text.slice(1, -1); continue; }
    }
    break;
  }
  return text;
}
export function normalizeText(t) {
  for (const [a, b] of SIMPLE) t = t.split(a).join(b);
  t = t.replace(/[;▼♀♂《》≪≫①②③④⑤⑥]/g, "");
  t = t.replace(/[˗‐-―⁃−⎯⏤─━⸺⸻]/g, "");
  t = t.replace(/[～〜]/g, "ー");
  t = t.replace(/…{3,}/g, "……");
  t = stripOuterBrackets(t);
  t = t.normalize("NFKC");
  t = t.split("...").join("…").split("..").join("…");
  return t;
}

// ---- seeded Gaussian noise (browser default; Node verify injects x0) ----
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussianNoise(n, seed) {
  const rng = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const u1 = Math.max(rng(), 1e-12), u2 = rng();
    out[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  return out;
}

export class IrodoriTTS {
  constructor({ ort, sessions, tokenizer }) {
    this.ort = ort;
    this.s = sessions; // {text, speaker, duration, dit, dac, enc}
    this.tok = tokenizer;
  }

  _t(data, shape, type = "float32") { return new this.ort.Tensor(type, data, shape); }

  tokenize(text) {
    const norm = normalizeText(text);
    const ids = this.tok.encode(norm, { add_special_tokens: false });
    return Int32Array.from([BOS, ...ids].map(Number));
  }

  async encodeText(text) {
    const ids = this.tokenize(text);
    const S = ids.length;
    const idsBig = BigInt64Array.from(ids, (x) => BigInt(x));
    const mask = new Uint8Array(S).fill(1);
    const out = await this.s.text.run({
      input_ids: this._t(idsBig, [1, S], "int64"),
      mask: this._t(mask, [1, S], "bool"),
    });
    return { state: out.text_state.data, S, dim: out.text_state.dims[2], mask };
  }

  async encodeRefLatent(refLatent, T, refMask) {
    const mask = refMask || new Uint8Array(T).fill(1);
    const out = await this.s.speaker.run({
      ref_latent: this._t(refLatent, [1, T, LATENT_DIM]),
      ref_mask: this._t(mask, [1, T], "bool"),
    });
    return {
      state: out.speaker_state.data, Tspk: out.speaker_state.dims[1],
      dim: out.speaker_state.dims[2], mask: out.speaker_mask.data,
    };
  }

  // waveform (Float32, mono) -> reference latent (B=1,T,32). Pads to HOP multiple.
  async wavToRefLatent(wav, sr, { ensureMax = true } = {}) {
    if (sr !== SR) throw new Error(`expected ${SR} Hz reference, got ${sr}`);
    let x = wav;
    if (ensureMax) {
      let peak = 0; for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
      if (peak > 1) { x = Float32Array.from(x, (v) => v / peak); }
    }
    const padded = Math.ceil(x.length / HOP) * HOP;
    const buf = new Float32Array(padded); buf.set(x);
    const out = await this.s.enc.run({ wav: this._t(buf, [1, 1, padded]) });
    return { latent: out.latent.data, T: out.latent.dims[1] };
  }

  async predictDuration(text, spk, { durationScale = 1.0, minSeconds = 0.5, maxSeconds = 30.0 } = {}) {
    const aux = new Float32Array(14);
    const out = await this.s.duration.run({
      text_state: this._t(text.state, [1, text.S, text.dim]),
      text_mask: this._t(text.mask, [1, text.S], "bool"),
      aux: this._t(aux, [1, 14]),
      speaker_state: this._t(spk.state, [1, spk.Tspk, spk.dim]),
      speaker_mask: this._t(spk.mask, [1, spk.Tspk], "bool"),
      has_speaker: this._t(new Uint8Array([1]), [1], "bool"),
    });
    const logFrames = out.log_frames.data[0];
    const predFrames = Math.expm1(logFrames) * durationScale;
    const minF = Math.ceil(minSeconds * SR / HOP);
    const maxF = Math.floor(maxSeconds * SR / HOP);
    return Math.max(minF, Math.min(maxF, Math.round(predFrames)));
  }

  // Rectified-flow Euler + independent CFG (text + speaker). Verified vs PyTorch.
  async rfLoop(text, spk, seqLen, {
    numSteps = 40, cfgText = 3.0, cfgSpk = 5.0, cfgMinT = 0.5, cfgMaxT = 1.0,
    initScale = 0.999, seed = 0, x0 = null,
  } = {}) {
    const S = seqLen, D = LATENT_DIM, SD = S * D;
    let xt = x0 ? Float32Array.from(x0) : gaussianNoise(SD, seed);

    const St = text.S, Dt = text.dim, Ts = spk.Tspk, Ds = spk.dim;
    const zerosT = new Float32Array(St * Dt), zerosTm = new Uint8Array(St);
    const zerosS = new Float32Array(Ts * Ds), zerosSm = new Uint8Array(Ts);
    // batch-3 bundles: [cond, text-uncond, speaker-uncond]
    const cat3 = (a, b, c, n) => { const o = new Float32Array(3 * n); o.set(a, 0); o.set(b, n); o.set(c, 2 * n); return o; };
    const cat3b = (a, b, c, n) => { const o = new Uint8Array(3 * n); o.set(a, 0); o.set(b, n); o.set(c, 2 * n); return o; };
    const textB = cat3(text.state, zerosT, text.state, St * Dt);
    const textMB = cat3b(text.mask, zerosTm, text.mask, St);
    const spkB = cat3(spk.state, spk.state, zerosS, Ts * Ds);
    const spkMB = cat3b(spk.mask, spk.mask, zerosSm, Ts);

    const tSched = new Float32Array(numSteps + 1);
    for (let i = 0; i <= numSteps; i++) tSched[i] = (1 - i / numSteps) * initScale;

    for (let i = 0; i < numSteps; i++) {
      const t = tSched[i], dt = tSched[i + 1] - t;
      let v;
      if (t >= cfgMinT && t <= cfgMaxT) {
        const xc = new Float32Array(3 * SD); xc.set(xt, 0); xc.set(xt, SD); xc.set(xt, 2 * SD);
        const o = await this.s.dit.run({
          x_t: this._t(xc, [3, S, D]), t: this._t(new Float32Array([t, t, t]), [3]),
          text_state: this._t(textB, [3, St, Dt]), text_mask: this._t(textMB, [3, St], "bool"),
          speaker_state: this._t(spkB, [3, Ts, Ds]), speaker_mask: this._t(spkMB, [3, Ts], "bool"),
        });
        const v3 = o.v.data;
        v = new Float32Array(SD);
        for (let j = 0; j < SD; j++) {
          const vc = v3[j];
          v[j] = vc + cfgText * (vc - v3[SD + j]) + cfgSpk * (vc - v3[2 * SD + j]);
        }
      } else {
        const o = await this.s.dit.run({
          x_t: this._t(xt, [1, S, D]), t: this._t(new Float32Array([t]), [1]),
          text_state: this._t(text.state, [1, St, Dt]), text_mask: this._t(text.mask, [1, St], "bool"),
          speaker_state: this._t(spk.state, [1, Ts, Ds]), speaker_mask: this._t(spk.mask, [1, Ts], "bool"),
        });
        v = o.v.data;
      }
      const nxt = new Float32Array(SD);
      for (let j = 0; j < SD; j++) nxt[j] = xt[j] + v[j] * dt;
      xt = nxt;
    }
    return xt; // (1, S, 32)
  }

  async decode(latent, S) {
    const D = LATENT_DIM;
    const z = new Float32Array(D * S);
    for (let s = 0; s < S; s++) for (let d = 0; d < D; d++) z[d * S + s] = latent[s * D + d];
    const out = await this.s.dac.run({ z: this._t(z, [1, D, S]) });
    return out.audio.data; // Float32
  }

  async synthesize(text, refWav, sr, opts = {}) {
    const t = await this.encodeText(text);
    const ref = await this.wavToRefLatent(refWav, sr);
    const spk = await this.encodeRefLatent(ref.latent, ref.T);
    const seqLen = opts.seconds
      ? Math.max(1, Math.round(opts.seconds * SR / HOP))
      : await this.predictDuration(t, spk, opts);
    const latent = await this.rfLoop(t, spk, seqLen, opts);
    const audio = await this.decode(latent, seqLen);
    return { audio, sampleRate: SR, seqLen };
  }
}
