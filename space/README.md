---
title: Irodori-TTS WebGPU
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: other
models:
  - noguchis/irodori-tts-onnx
short_description: Japanese zero-shot voice cloning, fully in-browser on WebGPU
---

# Irodori-TTS — WebGPU (in-browser)

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) (Japanese flow-matching TTS
with zero-shot voice cloning) running **entirely in the browser on WebGPU** — no
inference server. The rectified-flow sampling loop, CFG, tokenization and duration
logic run in JavaScript via [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/);
text and reference audio never leave the device.

- **Models:** ONNX weights stream from
  [noguchis/irodori-tts-onnx](https://huggingface.co/noguchis/irodori-tts-onnx)
  (fp16 ~0.65 GB / fp32 ~2.3 GB), cached by the browser after first load.
- **Source / export pipeline:**
  [ngc-shj/irodori-tts-webgpu](https://github.com/ngc-shj/irodori-tts-webgpu)
- **Requires** a WebGPU browser. **Chrome recommended** (~0.5× RTF on an M3 Pro —
  faster than real-time). Edge works but its efficiency/throttling can slow the
  sampling step several-fold on this hosted page; if it feels slow, use Chrome or
  turn off Edge's efficiency mode. Safari needs the WebGPU flag.

## Usage

1. Pick a reference voice `.wav` (any audio file works; resampled to 48 kHz).
2. Enter Japanese text.
3. Press **生成** to synthesize, or **計測** to benchmark the DiT step + decode.

## License & attribution

This demo bundles only UI + tokenizer; the model weights are licensed per their
upstream sources (Irodori-TTS / Semantic-DACVAE: MIT; llm-jp tokenizer: Apache-2.0).
See the [model card](https://huggingface.co/noguchis/irodori-tts-onnx) for full
attribution and the ethical-use notice (no non-consensual voice cloning, no
misleading deepfakes).
