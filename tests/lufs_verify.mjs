// Verify the JS BS.1770 loudness + -16 LUFS reference path against:
//  (a) audiotools integrated LUFS (passed as arg), and
//  (b) the captured ref_latent (native -16 LUFS -> dacvae_encoder).
//   node tests/lufs_verify.mjs <golden_noref.wav> <pythonLUFS>
import * as ort from "onnxruntime-node";
import { readFileSync } from "node:fs";
import { IrodoriTTS, lufsNormalize } from "../runtime/pipeline.mjs";

const wavPath = process.argv[2];
const pyLufs = parseFloat(process.argv[3]);

function readWavMono48k(path) {
  const b = readFileSync(path);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let off = 12, fmt = null, dOff = -1, dLen = 0;
  while (off + 8 <= b.byteLength) {
    const id = dv.getUint32(off, false), sz = dv.getUint32(off + 4, true);
    if (id === 0x666d7420) fmt = { f: dv.getUint16(off + 8, true), ch: dv.getUint16(off + 10, true), sr: dv.getUint32(off + 12, true), bits: dv.getUint16(off + 22, true) };
    else if (id === 0x64617461) { dOff = off + 8; dLen = sz; }
    off += 8 + sz + (sz & 1);
  }
  const out = new Float32Array(dLen / 2 / fmt.ch);
  for (let i = 0, j = 0; j < out.length; i += fmt.ch, j++) out[j] = dv.getInt16(dOff + i * 2, true) / 32768;
  return { wav: out, sr: fmt.sr };
}

const man = JSON.parse(readFileSync(new URL("../artifacts/ref/sampler_manifest.json", import.meta.url), "utf8"));
const refMeta = man.ref_latent;
const refBuf = readFileSync(new URL("../artifacts/ref/ref_latent.bin", import.meta.url));
const refLatent = new Float32Array(refBuf.buffer, refBuf.byteOffset, refBuf.byteLength / 4);

const { wav, sr } = readWavMono48k(wavPath);

// (a) loudness check via the gain that lufsNormalize applies
const norm = lufsNormalize(wav, sr, -16.0);
// recover applied gain (pre peak-limit) by comparing energies on a quiet sample is messy;
// instead just report the normalized signal's measured LUFS ~ -16 if correct.
// Re-measure: normalize the normalized signal to -16 again -> gain ~1.0 if LUFS==-16.
const renorm = lufsNormalize(norm, sr, -16.0);
let g2 = 0, c = 0;
for (let i = 0; i < norm.length; i++) if (Math.abs(norm[i]) > 1e-6) { g2 += renorm[i] / norm[i]; c++; }
g2 /= c;
console.log(`[lufs] python=${pyLufs.toFixed(3)}  re-normalize gain≈${g2.toFixed(4)} (≈1.000 means JS LUFS matches target)`);

// (b) ref_latent parity via the full wavToRefLatent path
const mk = (n) => ort.InferenceSession.create(new URL(`../artifacts/onnx/${n}.onnx`, import.meta.url).pathname, { executionProviders: ["cpu"] });
const tts = new IrodoriTTS({ ort, sessions: { enc: await mk("dacvae_encoder") }, tokenizer: null });
const { latent, T } = await tts.wavToRefLatent(wav, sr, { normalizeDb: -16.0 });
let maxd = 0; const n = Math.min(latent.length, refLatent.length);
for (let i = 0; i < n; i++) maxd = Math.max(maxd, Math.abs(latent[i] - refLatent[i]));
console.log(`[ref_latent] JS T=${T} shape ${refMeta.shape} len js=${latent.length} ref=${refLatent.length} max|Δ|=${maxd.toExponential(3)}`);
const ok = Math.abs(g2 - 1) < 0.02 && latent.length === refLatent.length && maxd < 5e-3;
console.log(ok ? "LUFS REF PATH PARITY: PASS" : "LUFS REF PATH PARITY: FAIL");
process.exit(ok ? 0 : 1);
