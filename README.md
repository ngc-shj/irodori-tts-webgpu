# irodori-tts-webgpu

English | [日本語](README.ja.md)

Run [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) (a Japanese flow-matching
TTS with zero-shot voice cloning) **fully in the browser on WebGPU** — no server-side
inference. The PyTorch model is exported to ONNX; the rectified-flow sampling loop,
CFG, tokenization and duration logic are reimplemented in JavaScript and run via
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) (WebGPU EP).

The same runtime core (`runtime/pipeline.mjs`) runs in Node (onnxruntime-node) for
headless verification and in the browser (onnxruntime-web WebGPU) unchanged.

## How it runs — offline export vs in-browser inference

There is **no inference server**. Work splits into three phases:

| phase | where / when | cost |
| --- | --- | --- |
| **1. Export** (`export/`, Python) | dev machine or CI, **once** (only when the model changes) | PyTorch → ONNX; produces the static `artifacts/onnx/*.onnx`. All heavy deps live here. |
| **2. Serve** (`web/serve.py` or any static host / CDN / HF Hub) | deploy time | **none** — just hands over the `.onnx` files. No computation. |
| **3. Inference** (browser, WebGPU) | every generation, **client-side** | the full TTS runs on the user's GPU. Text & reference audio never leave the device. |

So "server-side" applies only to the one-time **build** (phase 1) and to **static
file hosting** (phase 2). The actual speech synthesis is 100% in the browser.

```text
[once]   export (Python) ──▶ artifacts/onnx/*.onnx ──▶ put on any static host
[runtime] browser fetches *.onnx ──▶ WebGPU generates audio   (zero server compute)
```

## Status — verified end-to-end

The full JS pipeline reproduces the official PyTorch runtime **bit-faithfully**
(headless, onnxruntime-node CPU vs PyTorch on the same inputs):

| stage | metric |
| --- | --- |
| tokenizer (llm-jp-3-150m via transformers.js) | exact id match |
| text encoder | max\|Δ\| 4.5e-6 |
| speaker encoder | max\|Δ\| 2.0e-5 |
| duration → seqLen | exact (139 == 139) |
| RF Euler loop + CFG (DiT step) | latent max\|Δ\| 1.2e-4 |
| DAC decode | **waveform corr = 1.000000** |

## Architecture

```text
text ──tokenize(llm-jp)──▶ text_encoder.onnx ─┐
ref.wav ─decode/normalize─▶ dacvae_encoder.onnx ─▶ speaker_encoder.onnx ─┤
                                                  duration.onnx ─▶ seqLen │
                                                                          ▼
        seeded noise x0 ─▶  RF Euler loop ×N  (per step: dit.onnx, batch=3 CFG)
                                    │  v = v_cond + s_text·(v_cond−v_text⁻) + s_spk·(v_cond−v_spk⁻)
                                    ▼
                              latent ─▶ dacvae_decoder.onnx ─▶ 48kHz waveform
```

Only the **forward graphs** are ONNX (`artifacts/onnx/*.onnx`). All control flow
(sampling loop, CFG combine, schedules, tokenization, duration→length) lives in
`runtime/pipeline.mjs`, mirroring `irodori_tts/rf.py` + `inference_runtime.py`.

### Two blockers solved during export

- **Complex RoPE** (`view_as_complex`) is not ONNX-exportable → replaced with a
  mathematically identical real-valued, fully-symbolic implementation
  (`export/rope_patch.py`), applied as a monkeypatch (upstream untouched).
- **Data-dependent guards** in the duration predictor → bypassed at export time.

Exports use `torch.onnx.export(dynamo=True)` with `dynamic_shapes` so batch and all
sequence lengths stay symbolic (required for variable text + batched CFG).

## Layout

```text
export/      Python: ONNX export + parity + capture (depends on Irodori-TTS)
runtime/     pipeline.mjs — environment-agnostic inference core
web/         index.html + app.mjs (ORT-web WebGPU) + serve.py
tokenizer/   llmjp_tok/ — fast tokenizer.json for llm-jp/llm-jp-3-150m
tests/       Node headless parity tests (onnxruntime-node)
artifacts/   ONNX models + capture data (gitignored; generate via export/)
```

> **Tokenizer note:** Irodori-TTS-500M-v3's checkpoint config sets
> `text_tokenizer_repo = llm-jp/llm-jp-3-150m` (vocab 99574), **not** the
> `sbintuitions/sarashina2.2-0.5b` default in `config.py`. Use llm-jp.

## Run the browser app (local macOS)

```bash
python3 web/serve.py    # stdlib only — no venv needed (macOS: use python3, not python)
# open http://127.0.0.1:8137/web/  in a WebGPU browser (Chrome/Edge; Safari needs the flag)
# pick a reference voice .wav, enter text, press 生成
```

The UI has an **fp32 / fp16** selector and a **計測 (Measure)** button that benchmarks
the DiT step + DAC decode on the real WebGPU device and reports ms/step and the
extrapolated real-time factor — switch precision and re-measure to compare.
(fp16 needs the fp16 artifacts; see below.)

Models are served from localhost (fp32, ~2.3 GB total) so there is no download
cost; everything runs on the Metal-backed GPU.

## Regenerate artifacts (self-contained — no separate Irodori-TTS clone)

`setup_env.sh` builds a local `.venv` with the curated deps and installs
`irodori_tts`, `dacvae`, and `descript-audiotools` with `--no-deps` (so they do not
drag in CUDA torch / torchcodec / silentcipher, nor the `tensorboard → protobuf<4`
pin that would otherwise make the whole onnx/onnxscript install unsatisfiable).
Requires [`uv`](https://docs.astral.sh/uv/). It prints the commands below on success.

```bash
bash export/setup_env.sh   # .venv + deps + tokenizer/llmjp_tok
```

### fp32 artifacts (required) → `artifacts/onnx/`

```bash
.venv/bin/python export/export_dacvae_decoder.py
.venv/bin/python export/export_text_encoder.py
.venv/bin/python export/export_dit.py
.venv/bin/python export/export_rest.py        # speaker, duration, dacvae encoder
# optional, for the parity capture used by tests/:
.venv/bin/python export/golden.py --no-ref --out outputs/golden_noref   # a reference wav
.venv/bin/python export/capture_sampler.py                              # -> artifacts/ref/
```

### fp16 artifacts (optional, faster in-browser) → `artifacts/onnx_fp16/`

Each fp16 component has a dedicated step (see *fp16 — per component* below for the
why). Only `dacvae_encoder` survives a post-hoc `convert_fp16.py`; DiT, the
conditioning encoders, and the decoder each need their own script.

```bash
.venv/bin/python export/export_dit_fp16.py        # DiT — exported from a half() model
.venv/bin/python export/export_encoders_fp16.py   # text/speaker/duration — half() model (~half the download)
.venv/bin/python export/convert_fp16.py           # dacvae encoder (post-hoc fp16)
.venv/bin/python export/rewrite_convtranspose.py  # decoder: ConvTranspose -> Conv (-> dacvae_decoder_subpix.onnx)
.venv/bin/python export/convert_fp16_decoder_mixed.py \
    --in artifacts/onnx/dacvae_decoder_subpix.onnx \
    --out artifacts/onnx_fp16/dacvae_decoder.onnx # decoder fp16 (Snake kept fp32)
```

The export scripts import `irodori_tts` (the installed package) for the model
definition and monkeypatch it for ONNX (RoPE, guards) without touching upstream.
No `PYTHONPATH` and no external clone are required.

## Headless verification (Node)

```bash
npm install
node tests/ort_test.mjs       # onnxruntime-node bit-exact vs Python ORT
node tests/loop_verify.mjs    # RF loop + decode vs PyTorch capture (corr 1.0)
node tests/full_verify.mjs    # full chain vs PyTorch capture (corr 1.0)
```

`loop_verify`/`full_verify` need `artifacts/ref/` from
`export/capture_sampler.py` (a seeded PyTorch reference run).

## Embed in your own app (git submodule)

The inference core [`runtime/pipeline.mjs`](runtime/pipeline.mjs) is a single,
dependency-free ES module: it imports nothing and you inject everything
(`ort`, the created sessions, a tokenizer). That makes it easy to vendor as a
submodule and drive from your own UI/server — `web/app.mjs` is just one such
caller you can copy from.

**1. Add the submodule** (pin it to a commit; the API is plain JS, no build step):

```bash
git submodule add https://github.com/ngc-shj/irodori-tts-webgpu vendor/irodori-tts-webgpu
git -C vendor/irodori-tts-webgpu checkout <commit>     # pin
# later: git submodule update --remote   # to advance the pin
```

**2. Provide the artifacts and tokenizer** — these are **not** in the repo
(`artifacts/` is gitignored, ~0.65 GB fp16 / ~2.3 GB fp32). Generate them with
`export/` (see *Regenerate artifacts* above) or host your own copy, then serve the
six `*.onnx` (+ `*.onnx.data`) files and `tokenizer/llmjp_tok/` as static files.
For fp16, the decoder **must** be the Conv-rewritten one — produce it with the
two-step pipeline in the *Decoder fp16* section, not a naive `convert_fp16.py`.

**3. Wire it up** (browser, WebGPU). The constructor takes `{ ort, sessions,
tokenizer }`; `sessions` needs all six keys:

```js
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.webgpu.mjs";
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";
import { IrodoriTTS } from "./vendor/irodori-tts-webgpu/runtime/pipeline.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/";
env.allowRemoteModels = false; env.allowLocalModels = true;
env.localModelPath = "/tokenizer/";          // serves /tokenizer/llmjp_tok/*

const base = "/models";                       // where you host the .onnx (+ .onnx.data)
const names = { text:"text_encoder", speaker:"speaker_encoder", duration:"duration",
                dit:"dit", dac:"dacvae_decoder", enc:"dacvae_encoder" };
const opt = (n) => ({ executionProviders:["webgpu"], graphOptimizationLevel:"all",
  externalData:[{ path:`${n}.onnx.data`, data:`${base}/${n}.onnx.data` }] });
const sessions = {};
for (const [k, n] of Object.entries(names))
  sessions[k] = await ort.InferenceSession.create(`${base}/${n}.onnx`, opt(n));

const tokenizer = await AutoTokenizer.from_pretrained("llmjp_tok");
const tts = new IrodoriTTS({ ort, sessions, tokenizer });

// refWav: Float32Array, mono, 48 kHz (resample yourself; see web/app.mjs fileToMono48k)
const { audio, sampleRate, seqLen } = await tts.synthesize(text, refWav, 48000,
  { numSteps: 16, seed: 0 });                 // audio: Float32Array @48 kHz
```

Mix fp16/fp32 by pointing `base` per-model at your fp16 vs fp32 folders (see
`baseFor` in `web/app.mjs`). For **Node / headless**, the same code works with
`onnxruntime-node` and `executionProviders:["cpu"]`. The module also re-exports
`normalizeText`, `lufsNormalize`, and `integratedLoudness` if you need them.

**Cache the models when you deploy.** Re-fetching the ~1.2 GB (fp16) set on every
visit is wasteful on a shared host (HF Spaces, etc.). `web/app.mjs` already stores
each model file in the **Cache Storage API** on non-localhost origins, so a
returning visitor downloads it once (bump `CACHE_NAME` when you re-export). If you
write your own loader, do the same — or serve the `*.onnx`/`*.onnx.data` with
`Cache-Control: public, max-age=31536000, immutable` and content-hashed names.
(localhost keeps `no-store` via `web/serve.py` so regenerated artifacts stay fresh.)

**Browser note:** prefer **Chrome** for deployed pages. On the same M3 Pro, Chrome
matches localhost (~0.5× RTF), but Edge's efficiency/throttling slowed the DiT
sampling step ~2–3× on the hosted page (decode unaffected) — same GPU, same model.
Edge users can disable efficiency mode if needed.

## fp16 — per component (measured on WebGPU, M3 Pro)

fp16 is selected per component (UI checkboxes; `export/export_dit_fp16.py` +
`export/convert_fp16.py`). Measured in-browser by isolating each:

| component | fp16 on WebGPU | speed vs fp32 |
| --- | --- | --- |
| **DiT** | ✅ stable | ~168 vs ~234 ms/step (**1.4×**) |
| **encoder** (dacvae) | ✅ stable | negligible (runs once) |
| **decoder** | ✅ stable (Conv-rewritten) | decode ~530 vs ~997 ms (**1.9×**) |
| **text/spk/dur** | ✅ stable | none — **download only** (~649 → ~328 MB) |

**Recommended: all fp16** (the UI default) — **audibly identical to fp32**
and faster: a 3.6 s clip generates in ~2.3 s (RTF 0.65×), well under real-time.
(The 計測 estimate is conservative — it times every step at batch=3 CFG, but the
second half of the schedule runs batch=1.)

### Decoder fp16 needed a ConvTranspose rewrite

Naively converting the decoder to fp16 produces **noise on WebGPU** (CPU/fp32 are
fine). It is **not** a numerical problem — the model is fp16-sound every way we
emulated it offline (fp16 weights, fp16 activations, even fp16 in-conv
accumulation: all corr ≈ 1.0; max intermediate magnitude ~13, nowhere near the
65504 fp16 ceiling). The cause is **onnxruntime-web's WebGPU fp16 `ConvTranspose`
kernel**, which is broken (its fp16 `Conv` is fine). Isolated by per-op mixed
precision and matching public issues
[microsoft/onnxruntime#26367](https://github.com/microsoft/onnxruntime/issues/26367)
/ [#26732](https://github.com/microsoft/onnxruntime/issues/26732); still broken in
ort-web 1.26.0. Since the decode cost is dominated by the upsampling
ConvTranspose, simply keeping it fp32 gives no speedup.

Fix: `export/rewrite_convtranspose.py` replaces the 4 ConvTranspose layers with a
mathematically identical **sub-pixel (polyphase) form built from `Conv` only**
(`Conv` → reshape/transpose/reshape pixel-shuffle). All four have the regular shape
`kernel = 2·stride, pad = stride/2`, so each becomes a kernel-3 Conv emitting
`Cout·s` channels, shuffled to length `Lin·s` — verified exact vs the original
(corr 1.0). The whole decoder is then `Conv`-only and runs fp16 cleanly.

Regenerate it with the two decoder steps under *fp16 artifacts* in *Regenerate
artifacts* above (`convert_fp16.py` no longer touches the decoder).

> **Verification:** the rewrite is checked offline (two fp32 graphs, corr 1.0 —
> CPU comparison is valid there). But **onnxruntime CPU upcasts fp16→fp32**, so it
> cannot judge the WebGPU fp16 path; confirm fp16 audio in-browser (計測/生成).

## Known limitations / TODO

- **VoiceDesign (caption/emoji style)** and the no-ref path are not wired into the
  browser app yet (base 500M-v3 voice-clone only).
- **fp16 conditioning encoders** (text/speaker/duration) are available via
  `export_encoders_fp16.py` and the *text/spk/dur* UI toggle — they halve that
  download (~649 → ~328 MB) with no quality change, but no speed gain (run once).

## License

Inference/runtime code in this repository is released under the [MIT License](./LICENSE).

Model weights and the Irodori-TTS architecture are subject to their respective
upstream licenses:

| Component | ONNX files | Source | License |
| --- | --- | --- | --- |
| Irodori-TTS (500M-v3 weights) | `dit`, `text_encoder`, `speaker_encoder`, `duration` | [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) · [model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) | MIT |
| Semantic-DACVAE (codec) | `dacvae_encoder`, `dacvae_decoder` | [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) (derived from [facebook/dacvae-watermarked](https://huggingface.co/facebook/dacvae-watermarked)) | MIT |
| llm-jp-3-150m tokenizer | `tokenizer/llmjp_tok/` | [llm-jp/llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m) | Apache-2.0 |

### Redistributing the ONNX artifacts

The exported `*.onnx` files are derivative weights. All upstream licenses (MIT /
MIT / Apache-2.0) permit redistribution and commercial use, so the artifacts may
be re-hosted (e.g. on HF Hub) provided you:

- include the upstream MIT copyright notices (Irodori-TTS, Semantic-DACVAE) and
  the Apache-2.0 license text for the llm-jp tokenizer;
- state the modifications made here (PyTorch → ONNX conversion, fp16 conversion,
  and the decoder ConvTranspose → Conv rewrite) — required by Apache-2.0.

**Ethical use** (carried over from the Irodori-TTS model card): do not use this
model to clone or impersonate anyone's voice without their explicit consent, and
do not generate deepfakes or speech intended to mislead.
