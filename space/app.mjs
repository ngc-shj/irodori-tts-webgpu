// Browser glue for the Irodori-TTS WebGPU runtime.
// Reuses the environment-agnostic core (pipeline.mjs) verified headless in Node.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.webgpu.mjs";
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";
import { IrodoriTTS } from "./runtime/pipeline.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/";
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "/tokenizer/"; // serves /tokenizer/llmjp_tok/*

// ONNX artifacts are hosted in the model repo (resolve URLs, CORS-enabled), so
// this Space stays tiny and does not duplicate the ~3.7 GB of weights.
const MODEL_REPO = "https://huggingface.co/noguchis/irodori-tts-onnx/resolve/main";

const logEl = document.getElementById("log");
const log = (m, c) => { logEl.innerHTML += (c ? `<span class="${c}">${m}</span>` : m) + "\n"; logEl.scrollTop = logEl.scrollHeight; };

const MODELS = {
  text: "text_encoder", speaker: "speaker_encoder", duration: "duration",
  dit: "dit", dac: "dacvae_decoder", enc: "dacvae_encoder",
};
// Components with an fp16 variant (keyed by MODELS key).
const FP16_KEYS = new Set(["dit", "dac", "enc", "text", "speaker", "duration"]);

// Which components run fp16, read from the checkboxes. text/speaker/duration share
// one "cond" toggle — fp16 only shrinks their download (~half), same speed.
const getFp16 = () => {
  const cond = document.getElementById("fp16-cond").checked;
  return {
    dit: document.getElementById("fp16-dit").checked,
    dac: document.getElementById("fp16-dac").checked,
    enc: document.getElementById("fp16-enc").checked,
    text: cond, speaker: cond, duration: cond,
  };
};
const fp16Label = (s) => {
  const on = ["dit", "dac", "enc"].filter((k) => s[k]);
  if (s.text) on.push("cond");
  return on.length ? `fp16:${on.join("+")}` : "fp32";
};
const baseFor = (key, s) => (FP16_KEYS.has(key) && s[key]) ? `${MODEL_REPO}/onnx_fp16` : `${MODEL_REPO}/onnx`;

// Persist the model files (~1.2 GB at fp16) in the Cache Storage API so returning
// visitors download the set once instead of re-fetching 1+ GB from the model repo
// every load. Skipped on localhost previews. Bump CACHE_NAME when artifacts change.
const CACHE_NAME = "irodori-tts-models-v1";
const onLocalhost = ["localhost", "127.0.0.1", ""].includes(location.hostname);
const useModelCache = !onLocalhost && "caches" in globalThis;

async function fetchModelFile(url) {
  if (!useModelCache) return new Uint8Array(await (await fetch(url)).arrayBuffer());
  const store = await caches.open(CACHE_NAME);
  let res = await store.match(url);
  if (!res) {
    const net = await fetch(url);
    if (!net.ok) throw new Error(`fetch ${url}: ${net.status}`);
    await store.put(url, net.clone());
    res = net;
  }
  return new Uint8Array(await res.arrayBuffer());
}

async function dropStaleModelCaches() {
  for (const n of await caches.keys())
    if (n.startsWith("irodori-tts-models-") && n !== CACHE_NAME) await caches.delete(n);
}

const sessionOpt = (name, data) => ({
  executionProviders: ["webgpu"],
  graphOptimizationLevel: "all",
  externalData: [{ path: `${name}.onnx.data`, data }],
});

const cache = new Map();        // fp16-combo label -> IrodoriTTS
let tokenizer = null;
let adapterLogged = false;

async function loadModels(s) {
  const key = fp16Label(s);
  if (cache.has(key)) return cache.get(key);
  if (!navigator.gpu) throw new Error("WebGPU unavailable (navigator.gpu undefined). Use Chrome/Edge.");
  if (!adapterLogged) {
    const a = await navigator.gpu.requestAdapter();
    log(`WebGPU adapter: ${a?.info?.vendor || "?"} / ${a?.info?.architecture || "?"}`);
    adapterLogged = true;
  }
  if (useModelCache) await dropStaleModelCaches();
  log(`loading models (${key})… ${useModelCache ? "cached after first load" : "downloading"} from the model repo`);
  const sessions = {};
  for (const [k, name] of Object.entries(MODELS)) {
    const base = baseFor(k, s);
    const t0 = performance.now();
    const [model, data] = await Promise.all([
      fetchModelFile(`${base}/${name}.onnx`),
      fetchModelFile(`${base}/${name}.onnx.data`),
    ]);
    sessions[k] = await ort.InferenceSession.create(model, sessionOpt(name, data));
    log(`  ${name} [${base.split("/").pop()}] ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  }
  if (!tokenizer) tokenizer = await AutoTokenizer.from_pretrained("llmjp_tok");
  const inst = new IrodoriTTS({ ort, sessions, tokenizer });
  cache.set(key, inst);
  log(`models ready (${key}).`, "ok");
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

// Benchmark the DiT step (batch=3 CFG and batch=1) + DAC decode, then estimate
// the real per-utterance time using rfLoop's actual batch=3/batch=1 step split.
async function measure() {
  const btn = document.getElementById("measure");
  try {
    btn.disabled = true;
    const s = getFp16(); const label = fp16Label(s);
    const steps = parseInt(document.getElementById("steps").value, 10) || 16;
    const m = await loadModels(s);
    const S = 139, St = 14, Tsp = 93, D = 32;       // representative ~3.6s utterance
    const ditFeed = (B) => ({
      x_t: T(rand(B * S * D), [B, S, D]), t: T(rand(B), [B]),
      text_state: T(rand(B * St * 512), [B, St, 512]), text_mask: T(ones(B * St), [B, St], "bool"),
      speaker_state: T(rand(B * Tsp * 768), [B, Tsp, 768]), speaker_mask: T(ones(B * Tsp), [B, Tsp], "bool"),
    });
    log(`measuring DiT step (${label}, batch=3 / batch=1, S=${S})…`);
    const cfg3Ms = await timeRuns(() => m.s.dit.run(ditFeed(3)), 10);
    const cfg1Ms = await timeRuns(() => m.s.dit.run(ditFeed(1)), 10);
    log(`measuring DAC decode (${label})…`);
    const z = T(rand(D * S), [1, D, S]);
    const decMs = await timeRuns(() => m.s.dac.run({ z }), 5);
    const audioSec = S * 1920 / 48000;
    // Mirror rfLoop's schedule: t_i = (1 - i/steps) * initScale; steps with
    // t in [cfgMinT, cfgMaxT] run batch=3 (CFG), the rest batch=1.
    const initScale = 0.999, cfgMinT = 0.5, cfgMaxT = 1.0;
    let nCfg = 0;
    for (let i = 0; i < steps; i++) {
      const t = (1 - i / steps) * initScale;
      if (t >= cfgMinT && t <= cfgMaxT) nCfg++;
    }
    const total = cfg3Ms * nCfg + cfg1Ms * (steps - nCfg) + decMs;
    log(`[${label}] DiT batch=3 ${cfg3Ms.toFixed(1)} · batch=1 ${cfg1Ms.toFixed(1)} ms/step · decode ${decMs.toFixed(0)} ms`, "ok");
    log(`[${label}] ${steps} steps (${nCfg}×b3 + ${steps - nCfg}×b1) + decode ≈ ${(total / 1000).toFixed(2)} s `
      + `for ${audioSec.toFixed(2)} s audio (RTF ${(total / 1000 / audioSec).toFixed(2)}×)`, "ok");
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
    const s = getFp16(); const label = fp16Label(s);
    btn.disabled = true;

    const m = await loadModels(s);
    log("decoding reference audio…");
    const ref = await fileToMono48k(file);
    log(`reference: ${(ref.length / 48000).toFixed(2)} s @48k`);

    log(`synthesizing (${label}, ${steps} steps)…`);
    const t0 = performance.now();
    const { audio, sampleRate, seqLen } = await m.synthesize(text, ref, 48000, { numSteps: steps, seed });
    const dt = (performance.now() - t0) / 1000;
    const dur = audio.length / sampleRate;
    log(`done: ${dur.toFixed(2)} s audio (seqLen=${seqLen}) in ${dt.toFixed(1)} s (RTF ${(dt / dur).toFixed(2)}×)`, "ok");

    const url = URL.createObjectURL(encodeWav(audio, sampleRate));
    const audioEl = document.getElementById("out");
    audioEl.src = url; audioEl.style.display = "block";
    const dl = document.getElementById("dl");
    dl.href = url; dl.style.display = "inline"; dl.download = `irodori_${label}.wav`;
  } catch (e) {
    log(`ERROR: ${e.message || e}`, "err"); console.error(e);
  } finally {
    btn.disabled = false;
  }
}

document.getElementById("run").addEventListener("click", run);
document.getElementById("measure").addEventListener("click", measure);
