#!/usr/bin/env python3
"""Export the DiT velocity step (forward_with_encoded_conditions) to ONNX.

Base 500M-v3: speaker conditioning ON, caption OFF. Exported WITHOUT the
context KV cache so it is a single self-contained graph the JS RF loop calls
once per Euler step (batched over CFG bundles via dynamic batch dim).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import load_model
from rope_patch import apply_patch, reset_rope_caches


class DiTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_t, t, text_state, text_mask, speaker_state, speaker_mask):
        return self.model.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=None,
            caption_mask=None,
            latent_mask=None,
            context_kv_cache=None,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/onnx/dit.onnx")
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()

    model, cfg = load_model()
    wrapper = DiTWrapper(model).eval()

    # Export with B=2 so the batch Dim stays symbolic (torch.export specializes size-1 dims).
    B, S = 2, 92
    St, Tsp = 24, 40
    ld = cfg.patched_latent_dim
    x_t = torch.randn(B, S, ld)
    t = torch.rand(B)
    text_state = torch.randn(B, St, cfg.text_dim)
    text_mask = torch.ones(B, St, dtype=torch.bool)
    speaker_state = torch.randn(B, Tsp, cfg.speaker_dim)
    speaker_mask = torch.ones(B, Tsp, dtype=torch.bool)

    inputs = (x_t, t, text_state, text_mask, speaker_state, speaker_mask)

    with torch.inference_mode():
        ref_complex = wrapper(*inputs).clone()
    apply_patch()
    reset_rope_caches(model)
    with torch.inference_mode():
        ref = wrapper(*inputs)
    eq = (ref_complex - ref).abs()
    print(f"[rope] complex-vs-real max_abs_diff={eq.max():.3e}")
    print(f"[torch] v shape={tuple(ref.shape)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from torch.export import Dim
    b = Dim("b"); s = Dim("s"); st = Dim("st"); tsp = Dim("tsp")
    dynamic_shapes = {
        "x_t": {0: b, 1: s},
        "t": {0: b},
        "text_state": {0: b, 1: st},
        "text_mask": {0: b, 1: st},
        "speaker_state": {0: b, 1: tsp},
        "speaker_mask": {0: b, 1: tsp},
    }
    torch.onnx.export(
        wrapper,
        inputs,
        str(out),
        input_names=["x_t", "t", "text_state", "text_mask", "speaker_state", "speaker_mask"],
        output_names=["v"],
        dynamic_shapes=dynamic_shapes,
        opset_version=args.opset,
        dynamo=True,
    )
    sz = sum(f.stat().st_size for f in out.parent.glob(out.name + "*")) / 1e6
    print(f"[onnx] exported -> {out} (~{sz:.0f} MB total)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    feed = {
        "x_t": x_t.numpy(), "t": t.numpy(),
        "text_state": text_state.numpy(), "text_mask": text_mask.numpy(),
        "speaker_state": speaker_state.numpy(), "speaker_mask": speaker_mask.numpy(),
    }
    got = sess.run(["v"], feed)[0]
    print(f"[parity] max_abs_diff={np.abs(ref.numpy()-got).max():.3e} mean={np.abs(ref.numpy()-got).mean():.3e}")

    # different shapes + batch=2 (CFG bundle)
    B2, S2, St2, Tsp2 = 2, 60, 30, 33
    x2 = torch.randn(B2, S2, ld); t2 = torch.rand(B2)
    ts2 = torch.randn(B2, St2, cfg.text_dim); tm2 = torch.ones(B2, St2, dtype=torch.bool)
    sp2 = torch.randn(B2, Tsp2, cfg.speaker_dim); sm2 = torch.ones(B2, Tsp2, dtype=torch.bool)
    with torch.inference_mode():
        r2 = wrapper(x2, t2, ts2, tm2, sp2, sm2).numpy()
    g2 = sess.run(["v"], {"x_t": x2.numpy(), "t": t2.numpy(), "text_state": ts2.numpy(),
                          "text_mask": tm2.numpy(), "speaker_state": sp2.numpy(),
                          "speaker_mask": sm2.numpy()})[0]
    print(f"[parity b=2,s=60] max_abs_diff={np.abs(r2-g2).max():.3e}")


if __name__ == "__main__":
    main()
