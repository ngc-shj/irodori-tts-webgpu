---
license: other
license_name: mit-and-apache-2.0
license_link: LICENSE
language:
  - ja
library_name: onnxruntime
pipeline_tag: text-to-speech
tags:
  - tts
  - text-to-speech
  - voice-cloning
  - zero-shot
  - onnx
  - webgpu
  - flow-matching
  - japanese
base_model:
  - Aratako/Irodori-TTS-500M-v3
  - Aratako/Semantic-DACVAE-Japanese-32dim
  - llm-jp/llm-jp-3-150m
---

# Irodori-TTS — ONNX artifacts for in-browser WebGPU inference

ONNX exports of [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) (a Japanese
flow-matching TTS with zero-shot voice cloning), packaged to run **fully in the
browser on WebGPU** with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/).
There is **no inference server** — these files are served as static assets and the
full TTS runs client-side on the user's GPU. Text and reference audio never leave
the device.

Runtime/export code, headless verification, and the browser app:
👉 **[ngc-shj/irodori-tts-webgpu](https://github.com/ngc-shj/irodori-tts-webgpu)**

## What's in here

| File(s) | Component | Source weights |
| --- | --- | --- |
| `dit.onnx` (+ `.data`) | Rectified-flow DiT | Irodori-TTS-500M-v3 |
| `text_encoder.onnx` | Text encoder | Irodori-TTS-500M-v3 |
| `speaker_encoder.onnx` | Speaker encoder | Irodori-TTS-500M-v3 |
| `duration.onnx` | Duration predictor | Irodori-TTS-500M-v3 |
| `dacvae_encoder.onnx` | Codec encoder | Semantic-DACVAE-Japanese-32dim |
| `dacvae_decoder.onnx` | Codec decoder (48 kHz) | Semantic-DACVAE-Japanese-32dim |
| `tokenizer/llmjp_tok/` | Fast tokenizer | llm-jp/llm-jp-3-150m |

Two precision variants are provided:

- **fp32** (`onnx/`, ~2.3 GB) — reference precision.
- **fp16** (`onnx_fp16/`, ~0.65 GB) — **audibly identical to fp32** and faster on
  WebGPU (DiT ~1.4×, decode ~1.9×). The fp16 decoder is the **ConvTranspose → Conv
  rewritten** form (a naive fp16 decoder produces noise on onnxruntime-web's WebGPU
  `ConvTranspose` kernel — see the repo for details).

## Modifications from upstream

These artifacts are **derivative weights**. The following changes were made during
export (no upstream training; weights are numerically faithful):

- PyTorch → ONNX export (`torch.onnx.export(dynamo=True)`, symbolic shapes).
- Complex RoPE (`view_as_complex`, not ONNX-exportable) replaced with a
  mathematically identical real-valued implementation.
- Data-dependent guards in the duration predictor bypassed at export time.
- fp16 conversion of all components.
- Decoder `ConvTranspose` layers rewritten to a sub-pixel (polyphase) `Conv`-only
  form (verified corr 1.0 vs the original).

Faithfulness vs the official PyTorch runtime (headless, same inputs): text encoder
max|Δ| 4.5e-6, speaker encoder 2.0e-5, DiT-step latent 1.2e-4, **DAC decode
waveform corr = 1.000000**.

## Usage

Browser (WebGPU) and Node (onnxruntime-node) both drive the same dependency-free
inference core, `runtime/pipeline.mjs`:

```js
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.webgpu.mjs";
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";
import { IrodoriTTS } from "./runtime/pipeline.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/";
env.allowRemoteModels = false; env.allowLocalModels = true;
env.localModelPath = "/tokenizer/";

const base = "/onnx";   // or /onnx_fp16
const names = { text:"text_encoder", speaker:"speaker_encoder", duration:"duration",
                dit:"dit", dac:"dacvae_decoder", enc:"dacvae_encoder" };
const opt = (n) => ({ executionProviders:["webgpu"], graphOptimizationLevel:"all",
  externalData:[{ path:`${n}.onnx.data`, data:`${base}/${n}.onnx.data` }] });
const sessions = {};
for (const [k, n] of Object.entries(names))
  sessions[k] = await ort.InferenceSession.create(`${base}/${n}.onnx`, opt(n));

const tokenizer = await AutoTokenizer.from_pretrained("llmjp_tok");
const tts = new IrodoriTTS({ ort, sessions, tokenizer });

// refWav: Float32Array, mono, 48 kHz
const { audio, sampleRate } = await tts.synthesize(text, refWav, 48000,
  { numSteps: 16, seed: 0 });
```

Full setup, the WebGPU demo app, and headless parity tests are in the
[GitHub repo](https://github.com/ngc-shj/irodori-tts-webgpu).

## License

These artifacts combine weights under two permissive licenses, all of which permit
redistribution and commercial use:

| Component | Source | License |
| --- | --- | --- |
| Irodori-TTS (500M-v3 weights) | [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) · [model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) | MIT |
| Semantic-DACVAE (codec) | [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) (derived from [facebook/dacvae-watermarked](https://huggingface.co/facebook/dacvae-watermarked)) | MIT |
| llm-jp-3-150m tokenizer | [llm-jp/llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m) | Apache-2.0 |

The upstream MIT copyright notices and the Apache-2.0 license text are included in
this repository under [`LICENSES/`](LICENSES/). Per Apache-2.0, the modifications
listed above (ONNX/fp16 conversion, decoder rewrite) are stated as required.

## Ethical use

Carried over from the Irodori-TTS model card:

- **Do not** use this model to clone or impersonate anyone's voice (voice actors,
  celebrities, public figures, private individuals) without their explicit consent.
- **Do not** generate deepfakes or synthetic speech intended to mislead others or
  spread misinformation.

Generated voices may coincidentally resemble real people; this is a probabilistic
artifact of the latent space. The developers assume no liability for misuse.

## Citation

If you use these artifacts, please credit the upstream models above and link to the
[irodori-tts-webgpu](https://github.com/ngc-shj/irodori-tts-webgpu) project.
