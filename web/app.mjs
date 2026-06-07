// Browser glue for the Irodori-TTS WebGPU runtime.
// Reuses the environment-agnostic core (pipeline.mjs) verified headless in Node.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.webgpu.mjs";
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";
import { IrodoriTTS } from "../runtime/pipeline.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/";
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "/tokenizer/"; // serves /tokenizer/llmjp_tok/*

const logEl = document.getElementById("log");
const log = (m, c) => { logEl.innerHTML += (c ? `<span class="${c}">${m}</span>` : m) + "\n"; logEl.scrollTop = logEl.scrollHeight; };

const MODELS = {
  text: "text_encoder", speaker: "speaker_encoder", duration: "duration",
  dit: "dit", dac: "dacvae_decoder", enc: "dacvae_encoder",
};
// Only these have fp16 variants; the small encoders stay fp32.
const FP16_AVAIL = new Set(["dit", "dacvae_decoder", "dacvae_encoder"]);
const baseFor = (name, fp16) => (fp16 && FP16_AVAIL.has(name)) ? "/artifacts/onnx_fp16" : "/artifacts/onnx";

const sessionOpt = (name, fp16) => {
  const base = baseFor(name, fp16);
  return {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "all",
    externalData: [{ path: `${name}.onnx.data`, data: `${base}/${name}.onnx.data` }],
  };
};

const getPrecision = () => document.querySelector('input[name="prec"]:checked').value; // "fp32" | "fp16"

const cache = new Map();        // precision -> IrodoriTTS
let tokenizer = null;
let adapterLogged = false;

async function loadModels(precision) {
  if (cache.has(precision)) return cache.get(precision);
  if (!navigator.gpu) throw new Error("WebGPU unavailable (navigator.gpu undefined).");
  if (!adapterLogged) {
    const a = await navigator.gpu.requestAdapter();
    log(`WebGPU adapter: ${a?.info?.vendor || "?"} / ${a?.info?.architecture || "?"}`);
    adapterLogged = true;
  }
  const fp16 = precision === "fp16";
  log(`loading models (${precision})…`);
  const sessions = {};
  for (const [key, name] of Object.entries(MODELS)) {
    const t0 = performance.now();
    sessions[key] = await ort.InferenceSession.create(`${baseFor(name, fp16)}/${name}.onnx`, sessionOpt(name, fp16));
    log(`  ${name} [${baseFor(name, fp16).split("/").pop()}] ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  }
  if (!tokenizer) tokenizer = await AutoTokenizer.from_pretrained("llmjp_tok");
  const inst = new IrodoriTTS({ ort, sessions, tokenizer });
  cache.set(precision, inst);
  log(`models ready (${precision}).`, "ok");
  return inst;
}

async function fileToMono48k(file) {
  const arr = await file.arrayBuffer();
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(arr);
  await ctx.close();
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 48000), 48000);
  const src = off.createBufferSource();
  src.buffer = decoded; src.connect(off.destination); src.start();
  return (await off.startRendering()).getChannelData(0).slice();
}

function encodeWav(f32, sr) {
  const n = f32.length, buf = new ArrayBuffer(44 + n * 2), dv = new DataView(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); ws(8, "WAVE"); ws(12, "fmt ");
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true); ws(36, "data"); dv.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) dv.setInt16(44 + i * 2, Math.max(-1, Math.min(1, f32[i])) * 32767, true);
  return new Blob([buf], { type: "audio/wav" });
}

const rand = (n) => Float32Array.from({ length: n }, () => Math.random() * 2 - 1);
const ones = (n) => new Uint8Array(n).fill(1);
const T = (d, s, t = "float32") => new ort.Tensor(t, d, s);

async function timeRuns(fn, iters) {
  await fn(); await fn(); // warmup (first run compiles WebGPU shaders)
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) await fn();
  return (performance.now() - t0) / iters;
}

// Benchmark the DiT step (batch=3 CFG) + DAC decode at the selected precision.
async function measure() {
  const btn = document.getElementById("measure");
  try {
    btn.disabled = true;
    const precision = getPrecision();
    const steps = parseInt(document.getElementById("steps").value, 10) || 16;
    const m = await loadModels(precision);
    const S = 139, St = 14, Tsp = 93, D = 32;       // representative ~3.6s utterance
    const ditFeed = (B) => ({
      x_t: T(rand(B * S * D), [B, S, D]), t: T(rand(B), [B]),
      text_state: T(rand(B * St * 512), [B, St, 512]), text_mask: T(ones(B * St), [B, St], "bool"),
      speaker_state: T(rand(B * Tsp * 768), [B, Tsp, 768]), speaker_mask: T(ones(B * Tsp), [B, Tsp], "bool"),
    });
    log(`measuring DiT step (${precision}, batch=3 CFG, S=${S})…`);
    const cfgFeed = ditFeed(3);
    const stepMs = await timeRuns(() => m.s.dit.run(cfgFeed), 10);
    log(`measuring DAC decode (${precision})…`);
    const z = T(rand(D * S), [1, D, S]);
    const decMs = await timeRuns(() => m.s.dac.run({ z }), 5);
    const audioSec = S * 1920 / 48000;
    // Steps in the CFG window use batch=3; later steps batch=1 (~2x cheaper). Use
    // the measured batch-3 step as a conservative upper bound for all steps.
    const total = stepMs * steps + decMs;
    log(`[${precision}] DiT ${stepMs.toFixed(1)} ms/step · decode ${decMs.toFixed(0)} ms`, "ok");
    log(`[${precision}] ~${steps} steps + decode ≈ ${(total / 1000).toFixed(2)} s for ${audioSec.toFixed(2)} s audio `
      + `(RTF ${(total / 1000 / audioSec).toFixed(2)}×)`, "ok");
  } catch (e) {
    log(`ERROR: ${e.message || e}`, "err"); console.error(e);
  } finally {
    btn.disabled = false;
  }
}

async function run() {
  const btn = document.getElementById("run");
  try {
    const text = document.getElementById("text").value;
    const file = document.getElementById("ref").files[0];
    if (!file) { log("Please choose a reference .wav first.", "err"); return; }
    const steps = parseInt(document.getElementById("steps").value, 10) || 16;
    const seed = parseInt(document.getElementById("seed").value, 10) || 0;
    const precision = getPrecision();
    btn.disabled = true;

    const m = await loadModels(precision);
    log("decoding reference audio…");
    const ref = await fileToMono48k(file);
    log(`reference: ${(ref.length / 48000).toFixed(2)} s @48k`);

    log(`synthesizing (${precision}, ${steps} steps)…`);
    const t0 = performance.now();
    const { audio, sampleRate, seqLen } = await m.synthesize(text, ref, 48000, { numSteps: steps, seed });
    const dt = (performance.now() - t0) / 1000;
    const dur = audio.length / sampleRate;
    log(`done: ${dur.toFixed(2)} s audio (seqLen=${seqLen}) in ${dt.toFixed(1)} s (RTF ${(dt / dur).toFixed(2)}×)`, "ok");

    const url = URL.createObjectURL(encodeWav(audio, sampleRate));
    const audioEl = document.getElementById("out");
    audioEl.src = url; audioEl.style.display = "block";
    const dl = document.getElementById("dl");
    dl.href = url; dl.style.display = "inline"; dl.download = `irodori_${precision}.wav`;
  } catch (e) {
    log(`ERROR: ${e.message || e}`, "err"); console.error(e);
  } finally {
    btn.disabled = false;
  }
}

document.getElementById("run").addEventListener("click", run);
document.getElementById("measure").addEventListener("click", measure);
