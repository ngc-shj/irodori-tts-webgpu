// Smoke test: onnxruntime-node loads every exported graph (external .data) and
// the DiT step runs on random inputs with the expected dynamic shape + finite
// output. Numerical parity vs PyTorch is covered by loop_verify / full_verify.
import * as ort from "onnxruntime-node";

const mk = (name) => ort.InferenceSession.create(
  new URL(`../artifacts/onnx/${name}.onnx`, import.meta.url).pathname, { executionProviders: ["cpu"] });

const names = ["text_encoder", "speaker_encoder", "duration", "dit", "dacvae_decoder", "dacvae_encoder"];
const sessions = {};
for (const n of names) { sessions[n] = await mk(n); console.log(`loaded ${n}`); }

const B = 2, S = 30, St = 12, Tsp = 20, D = 32;
const rand = (n) => Float32Array.from({ length: n }, () => Math.random() * 2 - 1);
const ones = (n) => new Uint8Array(n).fill(1);
const T = (d, s, t = "float32") => new ort.Tensor(t, d, s);

const out = await sessions.dit.run({
  x_t: T(rand(B * S * D), [B, S, D]), t: T(rand(B), [B]),
  text_state: T(rand(B * St * 512), [B, St, 512]), text_mask: T(ones(B * St), [B, St], "bool"),
  speaker_state: T(rand(B * Tsp * 768), [B, Tsp, 768]), speaker_mask: T(ones(B * Tsp), [B, Tsp], "bool"),
});
const v = out.v;
const finite = v.data.every(Number.isFinite);
const shapeOk = JSON.stringify(v.dims) === JSON.stringify([B, S, D]);
console.log(`dit v dims=[${v.dims}] finite=${finite} shapeOk=${shapeOk}`);
const ok = finite && shapeOk;
console.log(ok ? "ORT-NODE SMOKE: PASS" : "ORT-NODE SMOKE: FAIL");
process.exit(ok ? 0 : 1);
