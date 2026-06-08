#!/usr/bin/env python3
"""Convert exported fp32 ONNX models to fp16 (web-ready). Keeps IO types fp32
so the JS runtime feeds/reads float32 while internal compute is fp16.

Only the dacvae_encoder is converted here — it is the only graph whose post-hoc
fp16 conversion is valid. Everything else needs a dedicated script because
onnxconverter-common mis-handles the dynamo Cast nodes (val_* type errors):
- text_encoder / speaker_encoder / duration: export_encoders_fp16.py (half() model;
  fp16 only shrinks their download ~half — same speed, they run once).
- DiT: export_dit_fp16.py (half() model).
- DACVAE decoder: rewrite_convtranspose.py (ConvTranspose -> Conv, which ORT-web's
  fp16 kernel handles; its fp16 ConvTranspose is broken) + convert_fp16_decoder_mixed.py."""
from __future__ import annotations

import os
from pathlib import Path

import onnx
from onnxconverter_common import float16

MODELS = ["dacvae_encoder"]


def main() -> None:
    src = Path("artifacts/onnx")
    dst = Path("artifacts/onnx_fp16")
    dst.mkdir(exist_ok=True)
    for name in MODELS:
        in_path = src / f"{name}.onnx"
        if not in_path.exists():
            print(f"[skip] {name}: not found")
            continue
        model = onnx.load(str(in_path))  # auto-loads external .data
        fp16_model = float16.convert_float_to_float16(
            model, keep_io_types=True, disable_shape_infer=True,
        )
        out_path = dst / f"{name}.onnx"
        onnx.save(
            fp16_model, str(out_path),
            save_as_external_data=True, all_tensors_to_one_file=True,
            location=f"{name}.onnx.data",
        )
        total = sum(os.path.getsize(dst / f) for f in os.listdir(dst) if f.startswith(name))
        print(f"[fp16] {name}: -> {out_path} ({total/1e6:.1f} MB)")
        # verify it loads
        import onnxruntime as ort
        ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        print(f"       loads OK in onnxruntime")


if __name__ == "__main__":
    main()
