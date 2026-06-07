// Quick check: run the captured RF loop with the fp16 DiT (+ fp32 decoder) and
// compare audio to the PyTorch capture. fp16 lowers precision, so expect high
// (not perfect) correlation.
import * as ort from "onnxruntime-node";
import { readFileSync } from "node:fs";

const REF = new URL("../artifacts/ref/", import.meta.url);
const man = JSON.parse(readFileSync(new URL("sampler_manifest.json", REF), "utf8"));
const P = man.params;
const load = (n) => { const m = man[n]; const b = readFileSync(new URL(`${n}.bin`, REF));
  return m.dtype === "bool" ? { data: Uint8Array.from(b), shape: m.shape } : { data: new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4), shape: m.shape }; };

const x0 = load("x0"), ts = load("text_state"), tm = load("text_mask"), ss = load("speaker_state"), sm = load("speaker_mask"), audioRef = load("audio");
const [, Ttext, Dt] = ts.shape, [, Tsp, Ds] = ss.shape, [, S, D] = x0.shape, N = P.num_steps;
const T = (d, s, t = "float32") => new ort.Tensor(t, d, s);
const dit = await ort.InferenceSession.create(new URL("../artifacts/onnx_fp16/dit.onnx", import.meta.url).pathname, { executionProviders: ["cpu"] });
const dac = await ort.InferenceSession.create(new URL("../artifacts/onnx/dacvae_decoder.onnx", import.meta.url).pathname, { executionProviders: ["cpu"] });

const tSched = new Float32Array(N + 1);
for (let i = 0; i <= N; i++) tSched[i] = (1 - i / N) * P.init_scale;
let xt = Float32Array.from(x0.data);
for (let i = 0; i < N; i++) {
  const t = tSched[i], dt = tSched[i + 1] - t; let v;
  if (t >= P.cfg_min_t && t <= P.cfg_max_t) {
    const xc = new Float32Array(3 * S * D); xc.set(xt, 0); xc.set(xt, S * D); xc.set(xt, 2 * S * D);
    const v3 = (await dit.run({ x_t: T(xc, [3, S, D]), t: T(new Float32Array([t, t, t]), [3]),
      text_state: T(ts.data, [3, Ttext, Dt]), text_mask: T(tm.data, [3, Ttext], "bool"),
      speaker_state: T(ss.data, [3, Tsp, Ds]), speaker_mask: T(sm.data, [3, Tsp], "bool") })).v.data;
    v = new Float32Array(S * D);
    for (let j = 0; j < S * D; j++) v[j] = v3[j] + P.cfg_scale_text * (v3[j] - v3[S * D + j]) + P.cfg_scale_speaker * (v3[j] - v3[2 * S * D + j]);
  } else {
    v = (await dit.run({ x_t: T(xt, [1, S, D]), t: T(new Float32Array([t]), [1]),
      text_state: T(ts.data.slice(0, Ttext * Dt), [1, Ttext, Dt]), text_mask: T(tm.data.slice(0, Ttext), [1, Ttext], "bool"),
      speaker_state: T(ss.data.slice(0, Tsp * Ds), [1, Tsp, Ds]), speaker_mask: T(sm.data.slice(0, Tsp), [1, Tsp], "bool") })).v.data;
  }
  const nx = new Float32Array(S * D); for (let j = 0; j < S * D; j++) nx[j] = xt[j] + v[j] * dt; xt = nx;
}
const z = new Float32Array(D * S); for (let s = 0; s < S; s++) for (let d = 0; d < D; d++) z[d * S + s] = xt[s * D + d];
const audio = (await dac.run({ z: T(z, [1, D, S]) })).audio.data;
let ab = 0, a2 = 0, b2 = 0; const n = Math.min(audio.length, audioRef.data.length);
for (let i = 0; i < n; i++) { ab += audio[i] * audioRef.data[i]; a2 += audio[i] ** 2; b2 += audioRef.data[i] ** 2; }
const corr = ab / Math.sqrt(a2 * b2);
console.log(`[fp16 dit] audio corr vs fp32 capture = ${corr.toFixed(5)}`);
console.log(corr > 0.99 ? "FP16 DIT AUDIO: OK (>0.99)" : "FP16 DIT AUDIO: CHECK");
process.exit(corr > 0.99 ? 0 : 1);
