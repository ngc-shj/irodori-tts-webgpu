# irodori-tts-webgpu

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
python web/serve.py
# open http://127.0.0.1:8137/web/  in a WebGPU browser (Chrome/Edge; Safari needs the flag)
# pick a reference voice .wav, enter text, press 生成
```

Models are served from localhost (fp32, ~2.3 GB total) so there is no download
cost; everything runs on the Metal-backed GPU.

## Regenerate artifacts (self-contained — no separate Irodori-TTS clone)

`setup_env.sh` builds a local `.venv` with the curated deps and installs the
`irodori_tts` package itself with `--no-deps` (so it does not drag in CUDA torch /
torchcodec / silentcipher). Requires [`uv`](https://docs.astral.sh/uv/).

```bash
bash export/setup_env.sh        # .venv + deps + irodori_tts (--no-deps) + tokenizer/llmjp_tok
.venv/bin/python export/export_dacvae_decoder.py
.venv/bin/python export/export_text_encoder.py
.venv/bin/python export/export_dit.py
.venv/bin/python export/export_rest.py        # speaker, duration, dacvae encoder
# optional, for the parity capture used by tests/:
.venv/bin/python export/golden.py --no-ref --out outputs/golden_noref   # a reference wav
.venv/bin/python export/capture_sampler.py                              # -> artifacts/ref/
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

## Known limitations / TODO
- **fp16**: `dacvae_{decoder,encoder}` convert cleanly; the dynamic-shape DiT hits
  an onnxconverter-common Cast error. fp32 is fine when served locally; fp16 (or
  int8) matters for remote/CDN hosting.
- **VoiceDesign (caption/emoji style)** and the no-ref path are not wired into the
  browser app yet (base 500M-v3 voice-clone only).

## License
Inference/runtime code here is MIT. Model weights and the Irodori-TTS architecture
are subject to their respective upstream licenses (Irodori-TTS, llm-jp-3, DACVAE).
