// Full-chain headless verification in Node:
//   tokenize -> text_encoder -> (captured ref_latent) -> speaker_encoder
//   -> duration -> RF loop (injected capture x0) -> DAC decode
// compared against the PyTorch sampler capture.
import * as ort from "onnxruntime-node";
import { AutoTokenizer, env } from "@huggingface/transformers";
import { readFileSync } from "node:fs";
import { IrodoriTTS } from "../runtime/pipeline.mjs";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL("../tokenizer/", import.meta.url).pathname;

const REF = new URL("../artifacts/ref/", import.meta.url);
const man = JSON.parse(readFileSync(new URL("sampler_manifest.json", REF), "utf8"));
const P = man.params;
const load = (name) => {
  const m = man[name];
  const buf = readFileSync(new URL(`${name}.bin`, REF));
  if (m.dtype === "bool") return { data: Uint8Array.from(buf), shape: m.shape };
  return { data: new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4), shape: m.shape };
};
const corr = (a, b) => {
  const n = Math.min(a.length, b.length); let ab = 0, a2 = 0, b2 = 0;
  for (let i = 0; i < n; i++) { ab += a[i] * b[i]; a2 += a[i] ** 2; b2 += b[i] ** 2; }
  return ab / Math.sqrt(a2 * b2);
};
const maxAbs = (a, b) => { let m = 0; const n = Math.min(a.length, b.length); for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

const mk = (name) => ort.InferenceSession.create(
  new URL(`../artifacts/onnx/${name}.onnx`, import.meta.url).pathname, { executionProviders: ["cpu"] });

const sessions = {
  text: await mk("text_encoder"), speaker: await mk("speaker_encoder"),
  duration: await mk("duration"), dit: await mk("dit"),
  dac: await mk("dacvae_decoder"), enc: await mk("dacvae_encoder"),
};
const tokenizer = await AutoTokenizer.from_pretrained("llmjp_tok");
const tts = new IrodoriTTS({ ort, sessions, tokenizer });

// 1. text encoder (from scratch tokenization)
const text = await tts.encodeText(man.text);
console.log(`[text] S=${text.S} dim=${text.dim} (capture text_state padded len=${man.text_state.shape[1]})`);

// 2. speaker encoder from captured ref_latent
const refLat = load("ref_latent");
const refMask = load("ref_mask");
const [, Tref] = [refLat.shape[0], refLat.shape[1]];
const spk = await tts.encodeRefLatent(refLat.data, Tref, refMask.data);
const spkCap = load("speaker_state");
const spkChunk0 = spkCap.data.subarray(0, spk.Tspk * spk.dim);
console.log(`[speaker] Tspk=${spk.Tspk} dim=${spk.dim} max|Δ|vs_capture_chunk0=${maxAbs(spk.state, spkChunk0).toExponential(3)}`);

// 3. duration
const seqLen = await tts.predictDuration(text, spk);
console.log(`[duration] predicted seqLen=${seqLen} (capture seq=${load("x0").shape[1]})`);

// 4. RF loop with injected capture x0 (force seq=139 to match capture), then decode
const x0 = load("x0");
const capSeq = x0.shape[1];
const latent = await tts.rfLoop(text, spk, capSeq, {
  numSteps: P.num_steps, cfgText: P.cfg_scale_text, cfgSpk: P.cfg_scale_speaker,
  cfgMinT: P.cfg_min_t, cfgMaxT: P.cfg_max_t, initScale: P.init_scale, x0: x0.data,
});
const finalRef = load("final_latent");
console.log(`[latent] max|Δ|vs_capture=${maxAbs(latent, finalRef.data).toExponential(3)}`);

const audio = await tts.decode(latent, capSeq);
const audioRef = load("audio");
const c = corr(audio, audioRef.data);
console.log(`[audio] len=${audio.length} corr_vs_capture=${c.toFixed(6)} max|Δ|=${maxAbs(audio, audioRef.data).toExponential(3)}`);

const ok = c > 0.999;
console.log(ok ? "FULL-CHAIN (JS) PARITY: PASS" : "FULL-CHAIN (JS) PARITY: FAIL");
process.exit(ok ? 0 : 1);
