#!/usr/bin/env python3
"""Mixed-precision fp16 conversion for the DACVAE decoder.

Run this on the ConvTranspose-rewritten decoder (rewrite_convtranspose.py output),
NOT the raw export: ORT-web's WebGPU fp16 ConvTranspose kernel is broken (noise),
so the decoder must be Conv-only before fp16 conversion. With ConvTranspose gone,
the whole decoder runs fp16 cleanly and decode drops ~997ms -> ~530ms on M3 Pro.

This keeps the Snake elementwise chain (alpha*x -> Sin -> Pow -> (1/a)*..) and its
alpha / 1-over-alpha constants in fp32 via node_block_list. That is numerical
insurance, not the noise fix: for small alpha, 1/a reaches ~45000 (near the fp16
ceiling 65504) and the cancellation x + (1/a)*sin^2(a*x) is precision-sensitive.
Offline emulation showed fp16 Snake is actually fine, but keeping it fp32 is cheap
(elementwise, no overflow) so we err on the safe side. Heavy Conv stays fp16 (the
speedup); weight-norm (max sum-of-squares ~134) stays fp16 too. IO stays fp32.

NOTE: onnxruntime CPU upcasts fp16->fp32, so a CPU parity check cannot reproduce
the GPU behaviour. The only real test is in-browser on WebGPU (web/ Measure + 生成).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from onnxconverter_common import float16


def snake_node_names(graph: onnx.GraphProto) -> list[str]:
    """Snake chain per activation: Mul(alpha,x), Sin, Pow(,2), Mul(1/a,sin^2)."""
    names = []
    for n in graph.node:
        if n.op_type in ("Sin", "Pow"):
            names.append(n.name)
        elif n.op_type == "Mul" and any(
            ("alpha" in i) or ("reciprocal" in i) for i in n.input
        ):
            names.append(n.name)
    return names


def weightnorm_node_names(graph: onnx.GraphProto) -> list[str]:
    """ReduceL2/Div of the weight-norm reparam (w = g * v / ||v||)."""
    return [n.name for n in graph.node if n.op_type in ("ReduceL2", "Div")]


def is_snake_init(name: str) -> bool:
    return ("alpha" in name) or ("reciprocal" in name)


def restore_fp32_snake_initializers(
    converted: onnx.ModelProto, original: onnx.ModelProto
) -> int:
    """Re-instate the Snake alpha / 1-over-alpha constants at fp32.

    convert_float_to_float16 fp16-casts every initializer (and clamps anything
    above max_finite_val), even those feeding fp32-blocked nodes — it just casts
    them back up. For Snake's reciprocal this is lossy: 1/a reaches ~45000 and the
    fp16 round-trip (plus the cancellation needing a and 1/a consistent) degrades
    the small-alpha channels. Keep these constants exactly fp32; their consumer
    Cast(->fp32) becomes an identity, so the graph stays valid.
    """
    orig = {i.name: i for i in original.graph.initializer}
    inits = converted.graph.initializer
    restored = 0
    for idx, t in enumerate(list(inits)):
        if is_snake_init(t.name) and t.name in orig:
            inits[idx].CopyFrom(orig[t.name])
            restored += 1
    return restored


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="artifacts/onnx/dacvae_decoder.onnx")
    p.add_argument("--out", default="artifacts/onnx_fp16/dacvae_decoder.onnx")
    p.add_argument("--block-weightnorm", action="store_true",
                   help="also keep weight-norm ReduceL2/Div in fp32 (negligible cost; "
                        "default off — it does not overflow)")
    p.add_argument("--block-ops", default="",
                   help="comma-separated op_types to also keep in fp32, e.g. "
                        "'ConvTranspose' to dodge ORT-web's weak fp16 ConvTranspose")
    args = p.parse_args()

    model = onnx.load(args.inp)  # auto-loads external .data
    original = onnx.load(args.inp)  # keep an unconverted copy for fp32 restore
    block = snake_node_names(model.graph)
    if args.block_weightnorm:
        block += weightnorm_node_names(model.graph)
    extra_ops = {o.strip() for o in args.block_ops.split(",") if o.strip()}
    if extra_ops:
        block += [n.name for n in model.graph.node if n.op_type in extra_ops]
    tags = ["snake"] + (["weightnorm"] if args.block_weightnorm else []) + sorted(extra_ops)
    print(f"[mixed] keeping {len(block)} nodes in fp32 ({'+'.join(tags)})")

    fp16_model = float16.convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=True,
        node_block_list=block, max_finite_val=65504.0,
    )
    n_restored = restore_fp32_snake_initializers(fp16_model, original)
    print(f"[mixed] restored {n_restored} Snake constants (alpha, 1/alpha) to fp32")

    # Drop stale external-data refs left by the load/convert round-trip; tensors
    # are already in raw_data, so a fresh external write would otherwise bloat 3x.
    for t in fp16_model.graph.initializer:
        if t.HasField("data_location") and t.data_location == onnx.TensorProto.EXTERNAL:
            t.ClearField("data_location")
            del t.external_data[:]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    name = out_path.stem
    data_path = out_path.parent / f"{name}.onnx.data"
    if data_path.exists():  # onnx external writer appends to an existing file
        data_path.unlink()
    onnx.save(
        fp16_model, str(out_path),
        save_as_external_data=True, all_tensors_to_one_file=True,
        location=f"{name}.onnx.data",
    )
    total = sum(os.path.getsize(out_path.parent / f)
                for f in os.listdir(out_path.parent) if f.startswith(name))
    print(f"[mixed] {name}: -> {out_path} ({total/1e6:.1f} MB)")

    # CPU sanity (upcasts fp16->fp32; confirms the graph is valid, NOT GPU parity).
    import onnxruntime as ort
    sess32 = ort.InferenceSession(args.inp, providers=["CPUExecutionProvider"])
    sess16 = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    z = np.random.randn(1, 32, 92).astype(np.float32)
    a32 = sess32.run(["audio"], {"z": z})[0]
    a16 = sess16.run(["audio"], {"z": z})[0]
    n = min(a32.size, a16.size)
    a, b = a32.flatten()[:n], a16.flatten()[:n]
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"[cpu-sanity] corr(fp32, mixed)={corr:.6f}  "
          f"max|Δ|={np.abs(a - b).max():.3e}  (CPU upcasts; real test = WebGPU)")


if __name__ == "__main__":
    main()
