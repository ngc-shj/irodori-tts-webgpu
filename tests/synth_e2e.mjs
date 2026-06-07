// Exercise the exact entrypoint the browser calls — tts.synthesize(text, wav, sr)
// — end to end in Node: WAV decode -> dacvae_encoder -> speaker_encoder ->
// duration -> RF loop -> dacvae_decoder. Validates the full path runs and
// produces sane (finite, correctly-sized, non-silent) audio.
//
//   node tests/synth_e2e.mjs /path/to/reference.wav "テキスト"
import * as ort from "onnxruntime-node";
import { AutoTokenizer, env } from "@huggingface/transformers";
import { readFileSync, writeFileSync } from "node:fs";
import { IrodoriTTS } from "../runtime/pipeline.mjs";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL("../tokenizer/", import.meta.url).pathname;

const refPath = process.argv[2];
const text = process.argv[3] || "こんにちは、これは音声合成のテストです。";
if (!refPath) { console.error("usage: node tests/synth_e2e.mjs <ref.wav> [text]"); process.exit(2); }

// Minimal WAV reader: PCM16 or float32, mono or first channel, asserts 48kHz.
function readWavMono48k(path) {
  const b = readFileSync(path);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  if (dv.getUint32(0, false) !== 0x52494646) throw new Error("not RIFF");
  let off = 12, fmt = null, dataOff = -1, dataLen = 0;
  while (off + 8 <= b.byteLength) {
    const id = dv.getUint32(off, false), sz = dv.getUint32(off + 4, true);
    if (id === 0x666d7420) { // "fmt "
      fmt = { audioFormat: dv.getUint16(off + 8, true), channels: dv.getUint16(off + 10, true),
              sampleRate: dv.getUint32(off + 12, true), bits: dv.getUint16(off + 22, true) };
    } else if (id === 0x64617461) { dataOff = off + 8; dataLen = sz; }
    off += 8 + sz + (sz & 1);
  }
  if (!fmt || dataOff < 0) throw new Error("missing fmt/data");
  if (fmt.sampleRate !== 48000) throw new Error(`expected 48kHz, got ${fmt.sampleRate}`);
  const ch = fmt.channels;
  let samples;
  if (fmt.audioFormat === 3 && fmt.bits === 32) { // float32
    const f = new Float32Array(b.buffer, b.byteOffset + dataOff, dataLen / 4);
    samples = ch === 1 ? f : f.filter((_, i) => i % ch === 0);
  } else if (fmt.audioFormat === 1 && fmt.bits === 16) { // PCM16
    const n = dataLen / 2, out = new Float32Array(Math.floor(n / ch));
    for (let i = 0, j = 0; i < n; i += ch, j++) out[j] = dv.getInt16(dataOff + i * 2, true) / 32768;
    samples = out;
  } else throw new Error(`unsupported wav: fmt=${fmt.audioFormat} bits=${fmt.bits}`);
  return samples;
}

function writeWav(path, f32, sr) {
  const n = f32.length, buf = Buffer.alloc(44 + n * 2), dv = new DataView(buf.buffer);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); ws(8, "WAVE"); ws(12, "fmt ");
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true); ws(36, "data"); dv.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) dv.setInt16(44 + i * 2, Math.max(-1, Math.min(1, f32[i])) * 32767, true);
  writeFileSync(path, buf);
}

const mk = (name) => ort.InferenceSession.create(
  new URL(`../artifacts/onnx/${name}.onnx`, import.meta.url).pathname, { executionProviders: ["cpu"] });
const sessions = {
  text: await mk("text_encoder"), speaker: await mk("speaker_encoder"),
  duration: await mk("duration"), dit: await mk("dit"),
  dac: await mk("dacvae_decoder"), enc: await mk("dacvae_encoder"),
};
const tokenizer = await AutoTokenizer.from_pretrained("llmjp_tok");
const tts = new IrodoriTTS({ ort, sessions, tokenizer });

const ref = readWavMono48k(refPath);
console.log(`[ref] ${(ref.length / 48000).toFixed(2)}s @48k`);
const t0 = Date.now();
const { audio, sampleRate, seqLen } = await tts.synthesize(text, ref, 48000, { numSteps: 16, seed: 0 });
const dt = (Date.now() - t0) / 1000;

let sumsq = 0, finite = true;
for (const x of audio) { if (!Number.isFinite(x)) finite = false; sumsq += x * x; }
const rms = Math.sqrt(sumsq / audio.length);
const expected = seqLen * 1920;
const out = new URL("../artifacts/synth_e2e.wav", import.meta.url).pathname;
writeWav(out, audio, sampleRate);

console.log(`[synth] seqLen=${seqLen} samples=${audio.length} (expected ${expected}) `
  + `dur=${(audio.length / sampleRate).toFixed(2)}s in ${dt.toFixed(1)}s`);
console.log(`[check] finite=${finite} rms=${rms.toFixed(4)} -> ${out}`);
const ok = finite && audio.length === expected && rms > 0.005;
console.log(ok ? "SYNTHESIZE E2E PATH: PASS" : "SYNTHESIZE E2E PATH: FAIL");
process.exit(ok ? 0 : 1);
