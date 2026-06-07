#!/usr/bin/env python3
"""Export the DACVAE decoder (latent -> waveform) to ONNX and verify parity.

Highest-risk component: convolutional codec with weight-norm and a bypassed
watermark branch. We export the exact callable used by DACVAECodec.decode_latent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from irodori_tts.codec import DACVAECodec


class DecoderWrapper(torch.nn.Module):
    def __init__(self, dac_model: torch.nn.Module):
        super().__init__()
        self.dac = dac_model

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, D, T_latent) -> audio (B, 1, samples)
        return self.dac.decode(z)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    p.add_argument("--out", default="artifacts/onnx/dacvae_decoder.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--latent-frames", type=int, default=92)
    args = p.parse_args()

    codec = DACVAECodec.load(repo_id=args.codec_repo, device="cpu", dtype=torch.float32)
    print(f"[dacvae] latent_dim={codec.latent_dim} sample_rate={codec.sample_rate}")

    model = codec.model.eval()
    wrapper = DecoderWrapper(model).eval()

    B, D, T = 1, codec.latent_dim, args.latent_frames
    z = torch.randn(B, D, T, dtype=torch.float32)

    with torch.inference_mode():
        ref = wrapper(z)
    print(f"[torch] decode out shape={tuple(ref.shape)} dtype={ref.dtype}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (z,),
        str(out_path),
        input_names=["z"],
        output_names=["audio"],
        dynamic_axes={"z": {0: "batch", 2: "frames"}, "audio": {0: "batch", 2: "samples"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"[onnx] exported -> {out_path} ({size_mb:.1f} MB)")

    # Parity check with onnxruntime
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["audio"], {"z": z.numpy()})[0]
    ref_np = ref.cpu().numpy()
    diff = np.abs(ref_np - ort_out)
    print(f"[parity] max_abs_diff={diff.max():.3e} mean_abs_diff={diff.mean():.3e} "
          f"ort_shape={ort_out.shape}")

    # Also test a different frame count to confirm dynamic axis works
    z2 = torch.randn(1, D, 50, dtype=torch.float32)
    with torch.inference_mode():
        ref2 = wrapper(z2).cpu().numpy()
    ort2 = sess.run(["audio"], {"z": z2.numpy()})[0]
    d2 = np.abs(ref2 - ort2)
    print(f"[parity@50] max_abs_diff={d2.max():.3e} shape={ort2.shape}")


if __name__ == "__main__":
    main()
