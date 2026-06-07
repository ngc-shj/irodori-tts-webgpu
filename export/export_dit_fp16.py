#!/usr/bin/env python3
"""Export the DiT step with fp16 weights/compute but fp32 inputs/outputs.

Done directly from a half() PyTorch model (cast at the graph boundary in the
wrapper) instead of post-hoc onnxconverter-common — which mis-handles the
dynamo-inserted Cast nodes on the dynamic-shape DiT (val_53 type error).

Halves the DiT download (~1.4GB -> ~0.7GB) for remote/CDN hosting and is faster
on the WebGPU (Metal) fp16 path. The JS runtime needs no change (fp32 IO).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.export import Dim

from common import load_model
from rope_patch import apply_patch, reset_rope_caches


class DiTStepFp16(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # already .half()

    def forward(self, x_t, t, text_state, text_mask, speaker_state, speaker_mask):
        h = torch.float16
        v = self.model.forward_with_encoded_conditions(
            x_t=x_t.to(h), t=t.to(h),
            text_state=text_state.to(h), text_mask=text_mask,
            speaker_state=speaker_state.to(h), speaker_mask=speaker_mask,
            caption_state=None, caption_mask=None, latent_mask=None, context_kv_cache=None,
        )
        return v.float()


def main() -> None:
    import onnxruntime as ort
    model, cfg = load_model()
    apply_patch(); reset_rope_caches(model)
    model = model.half()
    wrapper = DiTStepFp16(model).eval()

    B, S, St, Tsp = 2, 92, 24, 40
    ld = cfg.patched_latent_dim
    args = (
        torch.randn(B, S, ld), torch.rand(B),
        torch.randn(B, St, cfg.text_dim), torch.ones(B, St, dtype=torch.bool),
        torch.randn(B, Tsp, cfg.speaker_dim), torch.ones(B, Tsp, dtype=torch.bool),
    )
    with torch.inference_mode():
        ref = wrapper(*args)
    print(f"[fp16] torch v shape={tuple(ref.shape)} dtype={ref.dtype}")

    out = Path("artifacts/onnx_fp16/dit.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    b = Dim("b"); s = Dim("s"); st = Dim("st"); tsp = Dim("tsp")
    torch.onnx.export(
        wrapper, args, str(out),
        input_names=["x_t", "t", "text_state", "text_mask", "speaker_state", "speaker_mask"],
        output_names=["v"],
        dynamic_shapes={
            "x_t": {0: b, 1: s}, "t": {0: b},
            "text_state": {0: b, 1: st}, "text_mask": {0: b, 1: st},
            "speaker_state": {0: b, 1: tsp}, "speaker_mask": {0: b, 1: tsp},
        },
        opset_version=18, dynamo=True,
    )
    sz = sum(f.stat().st_size for f in out.parent.glob("dit.onnx*")) / 1e6
    print(f"[fp16] exported -> {out} (~{sz:.0f} MB)")

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    feed = {"x_t": args[0].numpy(), "t": args[1].numpy(),
            "text_state": args[2].numpy(), "text_mask": args[3].numpy(),
            "speaker_state": args[4].numpy(), "speaker_mask": args[5].numpy()}
    got = sess.run(["v"], feed)[0]
    print(f"[fp16] ORT loads OK; max|Δ| vs torch-fp16 = {np.abs(ref.numpy()-got).max():.3e}")


if __name__ == "__main__":
    main()
