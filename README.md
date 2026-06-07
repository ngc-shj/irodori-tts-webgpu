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

## fp16 — per component (measured on WebGPU, M3 Pro)

fp16 is selected per component (UI checkboxes; `export/export_dit_fp16.py` +
`export/convert_fp16.py`). Measured in-browser by isolating each:

| component | fp16 on WebGPU | speed vs fp32 |
| --- | --- | --- |
| **DiT** | ✅ stable | ~168 vs ~234 ms/step (**1.4×**) |
| **encoder** | ✅ stable | negligible (runs once) |
| **decoder** | ❌ **noise** — keep fp32 | (~470 vs ~997 ms) |

**Recommended: DiT + encoder fp16, decoder fp32** (the UI default) — **audibly
identical to fp32** (confirmed by listening) and ~1.25× faster: real generation
≈ 2.4 s vs 3.0 s (fp32) for a 3.6 s clip, both faster than real-time. (The 計測
estimate of RTF 0.66 is conservative — it times every step at batch=3 CFG, but
the second half of the schedule runs batch=1.) Full-fp16 is unusable: the decoder
produces noise.

The DAC decoder breaks in fp16 (Snake activation `x + sin²(ax)/a` + large 1536-ch
convs overflow/lose precision); stabilizing it would need mixed precision and it
only runs once, so fp32 is the right call. The headless `loop_fp16_check.mjs`
reported corr 0.99999 only because **onnxruntime-node (CPU) upcasts fp16→fp32** —
not representative of the real GPU path; trust the in-browser 計測/生成 instead.

## Known limitations / TODO

- **VoiceDesign (caption/emoji style)** and the no-ref path are not wired into the
  browser app yet (base 500M-v3 voice-clone only).
- **fp16 encoders**: text/speaker/duration graphs are still fp32 (small; ~650 MB total).

## License

Inference/runtime code in this repository is released under the [MIT License](./LICENSE).

Model weights and the Irodori-TTS architecture are subject to their respective
upstream licenses:

| Component | Source | License |
| --- | --- | --- |
| Irodori-TTS (code + 500M-v3 weights) | [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) · [model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) | MIT |
| llm-jp-3-150m tokenizer | [llm-jp/llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m) | Apache-2.0 |
| DACVAE | [facebookresearch/dacvae](https://github.com/facebookresearch/dacvae) | Apache-2.0 |
