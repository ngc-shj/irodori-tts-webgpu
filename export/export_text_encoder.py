#!/usr/bin/env python3
"""Export TextEncoder(+text_norm) to ONNX. Probes whether RoPE (view_as_complex)
exports under the torch 2.10 dynamo exporter."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import load_model
from rope_patch import apply_patch, reset_rope_caches


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_encoder = model.text_encoder
        self.text_norm = model.text_norm

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.text_norm(self.text_encoder(input_ids, mask))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/onnx/text_encoder.onnx")
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()

    model, cfg = load_model()
    wrapper = TextEncoderWrapper(model).eval()

    S = 24
    ids = torch.randint(0, cfg.text_vocab_size, (1, S), dtype=torch.long)
    mask = torch.ones(1, S, dtype=torch.bool)

    # Reference with the original complex RoPE.
    with torch.inference_mode():
        ref_complex = wrapper(ids, mask).clone()

    # Switch to real-valued RoPE and confirm equivalence before exporting.
    apply_patch()
    reset_rope_caches(model)
    with torch.inference_mode():
        ref = wrapper(ids, mask)
    eqdiff = (ref_complex - ref).abs()
    print(f"[rope] complex-vs-real max_abs_diff={eqdiff.max():.3e} mean={eqdiff.mean():.3e}")
    print(f"[torch] text_state shape={tuple(ref.shape)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (ids, mask),
        str(out),
        input_names=["input_ids", "mask"],
        output_names=["text_state"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "mask": {0: "b", 1: "s"},
                      "text_state": {0: "b", 1: "s"}},
        opset_version=args.opset,
    )
    print(f"[onnx] exported -> {out}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(["text_state"], {"input_ids": ids.numpy(), "mask": mask.numpy()})[0]
    diff = np.abs(ref.numpy() - got)
    print(f"[parity] max_abs_diff={diff.max():.3e} mean={diff.mean():.3e}")

    # different length
    S2 = 40
    ids2 = torch.randint(0, cfg.text_vocab_size, (1, S2), dtype=torch.long)
    mask2 = torch.ones(1, S2, dtype=torch.bool)
    with torch.inference_mode():
        ref2 = wrapper(ids2, mask2).numpy()
    got2 = sess.run(["text_state"], {"input_ids": ids2.numpy(), "mask": mask2.numpy()})[0]
    print(f"[parity@40] max_abs_diff={np.abs(ref2-got2).max():.3e}")


if __name__ == "__main__":
    main()
