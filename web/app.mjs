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

const ONNX = "/artifacts/onnx";
const MODELS = {
  text: "text_encoder", speaker: "speaker_encoder", duration: "duration",
  dit: "dit", dac: "dacvae_decoder", enc: "dacvae_encoder",
};

const sessionOpt = (name) => ({
  executionProviders: ["webgpu"],
  graphOptimizationLevel: "all",
  // ORT-Web requires explicit external-data declarations; `path` must equal the
  // location string baked into the .onnx (here "<name>.onnx.data").
  externalData: [{ path: `${name}.onnx.data`, data: `${ONNX}/${name}.onnx.data` }],
});

let tts = null;

async function loadModels() {
  if (tts) return tts;
  if (!navigator.gpu) throw new Error("WebGPU unavailable (navigator.gpu undefined).");
  const adapter = await navigator.gpu.requestAdapter();
  log(`WebGPU adapter: ${adapter?.info?.vendor || "?"} / ${adapter?.info?.architecture || "?"}`);

  const sessions = {};
  for (const [key, name] of Object.entries(MODELS)) {
    const t0 = performance.now();
    sessions[key] = await ort.InferenceSession.create(`${ONNX}/${name}.onnx`, sessionOpt(name));
    log(`  loaded ${name} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  }
  const tokenizer = await AutoTokenizer.from_pretrained("llmjp_tok");
  tts = new IrodoriTTS({ ort, sessions, tokenizer });
  log("all models ready.", "ok");
  return tts;
}

// Decode an uploaded audio file to mono Float32 @48kHz.
async function fileToMono48k(file) {
  const arr = await file.arrayBuffer();
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(arr);
  await ctx.close();
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 48000), 48000);
  const src = off.createBufferSource();
  src.buffer = decoded; src.connect(off.destination); src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0).slice();
}

function encodeWav(float32, sr) {
  const n = float32.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); ws(8, "WAVE"); ws(12, "fmt ");
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true); ws(36, "data"); dv.setUint32(40, n * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) { const s = Math.max(-1, Math.min(1, float32[i])); dv.setInt16(o, s * 32767, true); o += 2; }
  return new Blob([buf], { type: "audio/wav" });
}

async function run() {
  try {
    const text = document.getElementById("text").value;
    const file = document.getElementById("ref").files[0];
    if (!file) { log("Please choose a reference .wav first.", "err"); return; }
    const steps = parseInt(document.getElementById("steps").value, 10) || 16;
    const seed = parseInt(document.getElementById("seed").value, 10) || 0;
    document.getElementById("run").disabled = true;

    const m = await loadModels();
    log("decoding reference audio…");
    const ref = await fileToMono48k(file);
    log(`reference: ${(ref.length / 48000).toFixed(2)}s @48k`);

    log(`synthesizing "${text.slice(0, 24)}…" (${steps} steps)…`);
    const t0 = performance.now();
    const { audio, sampleRate, seqLen } = await m.synthesize(text, ref, 48000, {
      numSteps: steps, seed,
    });
    const dt = (performance.now() - t0) / 1000;
    const dur = audio.length / sampleRate;
    log(`done: ${dur.toFixed(2)}s audio (seqLen=${seqLen}) in ${dt.toFixed(1)}s `
      + `(RTF ${(dt / dur).toFixed(2)}×)`, "ok");

    const blob = encodeWav(audio, sampleRate);
    const url = URL.createObjectURL(blob);
    const audioEl = document.getElementById("out");
    audioEl.src = url; audioEl.style.display = "block";
    const dl = document.getElementById("dl");
    dl.href = url; dl.style.display = "inline"; dl.download = "irodori_webgpu.wav";
  } catch (e) {
    log(`ERROR: ${e.message || e}`, "err"); console.error(e);
  } finally {
    document.getElementById("run").disabled = false;
  }
}

document.getElementById("run").addEventListener("click", run);
