// Verify the JS rectified-flow Euler+CFG loop (+DAC decode) against the
// PyTorch sampler capture. This is the riskiest reimplementation for the port.
import * as ort from "onnxruntime-node";
import { readFileSync } from "node:fs";

const REF = new URL("../artifacts/ref/", import.meta.url);
const man = JSON.parse(readFileSync(new URL("sampler_manifest.json", REF), "utf8"));
const P = man.params;

function load(name) {
  const m = man[name];
  const buf = readFileSync(new URL(`${name}.bin`, REF));
  if (m.dtype === "bool") return { data: Uint8Array.from(buf), shape: m.shape };
  return { data: new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4), shape: m.shape };
}

const x0 = load("x0");
const textState = load("text_state");
const textMask = load("text_mask");
const spkState = load("speaker_state");
const spkMask = load("speaker_mask");
const finalRef = load("final_latent");
const audioRef = load("audio");

const [B3, Ttext, Dtext] = textState.shape;
const [, Tspk, Dspk] = spkState.shape;
const [, S, D] = x0.shape; // (1, seq, 32)
const N = P.num_steps;

// Identify which batch chunk is text-uncond vs speaker-uncond (zeroed state).
function chunkIsZero(t, chunk, Tn, Dn) {
  const off = chunk * Tn * Dn;
  let mx = 0;
  for (let i = 0; i < Tn * Dn; i++) mx = Math.max(mx, Math.abs(t.data[off + i]));
  return mx < 1e-6;
}
let textUncond = -1, spkUncond = -1;
for (let c = 1; c < 3; c++) {
  if (chunkIsZero(textState, c, Ttext, Dtext)) textUncond = c;
  if (chunkIsZero(spkState, c, Tspk, Dspk)) spkUncond = c;
}
console.log(`[map] textUncondChunk=${textUncond} spkUncondChunk=${spkUncond} `
  + `s_text=${P.cfg_scale_text} s_spk=${P.cfg_scale_speaker}`);

const dit = await ort.InferenceSession.create(
  new URL("../artifacts/onnx/dit.onnx", import.meta.url).pathname, { executionProviders: ["cpu"] });
const dac = await ort.InferenceSession.create(
  new URL("../artifacts/onnx/dacvae_decoder.onnx", import.meta.url).pathname, { executionProviders: ["cpu"] });

const T = (data, shape, type = "float32") => new ort.Tensor(type, data, shape);

// t-schedule: (1 - linspace(0,1,N+1)) * init_scale
const tSched = new Float32Array(N + 1);
for (let i = 0; i <= N; i++) tSched[i] = (1 - i / N) * P.init_scale;

let xt = Float32Array.from(x0.data); // (1,S,D)

for (let i = 0; i < N; i++) {
  const t = tSched[i], tNext = tSched[i + 1];
  const useCfg = t >= P.cfg_min_t && t <= P.cfg_max_t;
  let v; // Float32Array (S*D)
  if (useCfg) {
    // batch=3: repeat xt across 3, feed full bundles
    const xc = new Float32Array(3 * S * D);
    xc.set(xt, 0); xc.set(xt, S * D); xc.set(xt, 2 * S * D);
    const feeds = {
      x_t: T(xc, [3, S, D]), t: T(new Float32Array([t, t, t]), [3]),
      text_state: T(textState.data, [3, Ttext, Dtext]), text_mask: T(textMask.data, [3, Ttext], "bool"),
      speaker_state: T(spkState.data, [3, Tspk, Dspk]), speaker_mask: T(spkMask.data, [3, Tspk], "bool"),
    };
    const v3 = (await dit.run(feeds)).v.data; // (3,S,D)
    v = new Float32Array(S * D);
    const sc = { [textUncond]: P.cfg_scale_text, [spkUncond]: P.cfg_scale_speaker };
    for (let j = 0; j < S * D; j++) {
      const vc = v3[j];
      v[j] = vc + sc[textUncond] * (vc - v3[textUncond * S * D + j])
                + sc[spkUncond] * (vc - v3[spkUncond * S * D + j]);
    }
  } else {
    const feeds = {
      x_t: T(xt, [1, S, D]), t: T(new Float32Array([t]), [1]),
      text_state: T(textState.data.slice(0, Ttext * Dtext), [1, Ttext, Dtext]),
      text_mask: T(textMask.data.slice(0, Ttext), [1, Ttext], "bool"),
      speaker_state: T(spkState.data.slice(0, Tspk * Dspk), [1, Tspk, Dspk]),
      speaker_mask: T(spkMask.data.slice(0, Tspk), [1, Tspk], "bool"),
    };
    v = (await dit.run(feeds)).v.data;
  }
  const dt = tNext - t;
  const nxt = new Float32Array(S * D);
  for (let j = 0; j < S * D; j++) nxt[j] = xt[j] + v[j] * dt;
  xt = nxt;
}

// latent parity
let maxd = 0;
for (let j = 0; j < S * D; j++) maxd = Math.max(maxd, Math.abs(xt[j] - finalRef.data[j]));
const fmax = finalRef.data.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
console.log(`[latent] max|Δ|=${maxd.toExponential(3)} rel=${(maxd / fmax).toExponential(3)}`);

// decode: z = latent transposed (1,S,D)->(1,D,S)
const z = new Float32Array(D * S);
for (let s = 0; s < S; s++) for (let d = 0; d < D; d++) z[d * S + s] = xt[s * D + d];
const audio = (await dac.run({ z: T(z, [1, D, S]) })).audio.data;
const n = Math.min(audio.length, audioRef.data.length);
let amax = 0, corrNum = 0, a2 = 0, b2 = 0;
for (let i = 0; i < n; i++) {
  amax = Math.max(amax, Math.abs(audio[i] - audioRef.data[i]));
  corrNum += audio[i] * audioRef.data[i]; a2 += audio[i] ** 2; b2 += audioRef.data[i] ** 2;
}
const corr = corrNum / Math.sqrt(a2 * b2);
console.log(`[audio ] len js=${audio.length} ref=${audioRef.data.length} `
  + `max|Δ|=${amax.toExponential(3)} corr=${corr.toFixed(6)}`);
const ok = maxd / fmax < 1e-2 && corr > 0.999;
console.log(ok ? "JS RF-LOOP + DECODE PARITY: PASS" : "JS RF-LOOP + DECODE PARITY: FAIL");
process.exit(ok ? 0 : 1);
