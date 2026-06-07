#!/usr/bin/env python3
"""Generate golden reference output with the official PyTorch runtime.

Saves WAV via soundfile (avoids the torchcodec dependency) and the raw audio
array as .npy for later ONNX parity comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-500M-v3")
    p.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    p.add_argument("--text", default="こんにちは、これは音声合成のテストです。")
    p.add_argument("--ref-wav", default=None)
    p.add_argument("--no-ref", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/golden")
    args = p.parse_args()

    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(repo_id=args.hf_checkpoint, filename="model.safetensors")

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=ckpt,
            model_device=args.device,
            codec_repo=args.codec_repo,
            model_precision="fp32",
            codec_device=args.device,
            codec_precision="fp32",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        )
    )

    result = runtime.synthesize(
        SamplingRequest(
            text=args.text,
            ref_wav=args.ref_wav,
            no_ref=bool(args.no_ref),
            num_steps=args.num_steps,
            seed=args.seed,
            cfg_guidance_mode="independent",
        ),
        log_fn=print,
    )

    audio = result.audio.detach().cpu().float().squeeze(0).numpy()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out.with_suffix(".wav")), audio, result.sample_rate)
    np.save(str(out.with_suffix(".npy")), audio)
    print(f"[golden] seed={result.used_seed} sr={result.sample_rate} "
          f"samples={audio.shape} -> {out}.wav/.npy")


if __name__ == "__main__":
    main()
