#!/usr/bin/env python3
"""End-to-end ONNX parity test.

Run the official runtime twice with the same seed:
  1. stock PyTorch model  -> golden audio
  2. model/codec forwards swapped for ONNX sessions -> candidate audio
Both reuse the exact same glue (noise, t-schedule, CFG, duration->seqlen,
trimming), so any difference is pure ONNX numerical error.

Voice-clone path: reference = a provided wav (defaults to the no-ref golden).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torchaudio
import onnxruntime as ort


def _sf_load(path, *a, **k):
    # Replace torchaudio.load (needs torchcodec on this build) with soundfile.
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
    return torch.from_numpy(data.T).contiguous(), sr


torchaudio.load = _sf_load

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
from huggingface_hub import hf_hub_download


def _sess(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="こんにちは、これは音声合成のテストです。")
    ap.add_argument("--ref-wav", default="outputs/golden_noref.wav")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = hf_hub_download(repo_id="Aratako/Irodori-TTS-500M-v3", filename="model.safetensors")
    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision="fp32", codec_device="cpu", codec_precision="fp32",
        codec_deterministic_encode=True, codec_deterministic_decode=True,
        compile_model=False, compile_dynamic=False,
    ))

    def make_req():
        return SamplingRequest(
            text=args.text, ref_wav=args.ref_wav, no_ref=False,
            num_steps=args.steps, seed=args.seed, cfg_guidance_mode="independent",
            context_kv_cache=False,  # force per-step recompute (matches exported DiT graph)
        )

    # ---- 1. torch golden ----
    res_torch = runtime.synthesize(make_req(), log_fn=None)
    audio_torch = res_torch.audio.detach().cpu().float().squeeze(0).numpy()
    print(f"[torch] audio samples={audio_torch.shape} seed={res_torch.used_seed}")

    # ---- 2. monkeypatch model/codec forwards to ONNX ----
    model = runtime.model
    cfg = runtime.model_cfg
    se = _sess("artifacts/onnx/text_encoder.onnx")
    sp = _sess("artifacts/onnx/speaker_encoder.onnx")
    sd = _sess("artifacts/onnx/duration.onnx")
    sdit = _sess("artifacts/onnx/dit.onnx")
    sdec = _sess("artifacts/onnx/dacvae_decoder.onnx")

    def np_(x):
        return x.detach().cpu().numpy()

    def onnx_encode_conditions(text_input_ids, text_mask, ref_latent, ref_mask,
                               caption_input_ids=None, caption_mask=None,
                               speaker_state_override=None, speaker_mask_override=None,
                               speaker_uncond_mode="mask", **kw):
        ts = se.run(["text_state"], {"input_ids": np_(text_input_ids).astype(np.int64),
                                     "mask": np_(text_mask).astype(bool)})[0]
        text_state = torch.from_numpy(ts)
        # speaker branch (wav-ref path; patch_size=1 -> no extra patching)
        rs, rm = sp.run(["speaker_state", "speaker_mask"],
                        {"ref_latent": np_(ref_latent).astype(np.float32),
                         "ref_mask": np_(ref_mask).astype(bool)})
        return (text_state, text_mask.bool(), torch.from_numpy(rs),
                torch.from_numpy(rm), None, None)

    def onnx_forward(x_t, t, text_state, text_mask, speaker_state, speaker_mask,
                     caption_state=None, caption_mask=None, latent_mask=None,
                     context_kv_cache=None, **kw):
        v = sdit.run(["v"], {
            "x_t": np_(x_t).astype(np.float32), "t": np_(t).astype(np.float32),
            "text_state": np_(text_state).astype(np.float32), "text_mask": np_(text_mask).astype(bool),
            "speaker_state": np_(speaker_state).astype(np.float32),
            "speaker_mask": np_(speaker_mask).astype(bool),
        })[0]
        return torch.from_numpy(v).to(x_t.dtype)

    def onnx_duration(text_state, text_mask, speaker_state, speaker_mask,
                      duration_features, has_speaker, caption_state=None,
                      caption_mask=None, has_caption=None, **kw):
        lf = sd.run(["log_frames"], {
            "text_state": np_(text_state).astype(np.float32), "text_mask": np_(text_mask).astype(bool),
            "aux": np_(duration_features).astype(np.float32),
            "speaker_state": np_(speaker_state).astype(np.float32),
            "speaker_mask": np_(speaker_mask).astype(bool),
            "has_speaker": np_(has_speaker).astype(bool),
        })[0]
        return torch.from_numpy(lf).float()

    orig_decode = runtime.codec.decode_latent

    def onnx_decode(latent):
        z = latent.transpose(1, 2).contiguous()  # (B,D,T)
        audio = sdec.run(["audio"], {"z": np_(z).astype(np.float32)})[0]
        return torch.from_numpy(audio)

    model.encode_conditions = onnx_encode_conditions
    model.forward_with_encoded_conditions = onnx_forward
    model.predict_duration_log_frames = onnx_duration
    runtime.codec.decode_latent = onnx_decode

    res_onnx = runtime.synthesize(make_req(), log_fn=None)
    audio_onnx = res_onnx.audio.detach().cpu().float().squeeze(0).numpy()
    print(f"[onnx ] audio samples={audio_onnx.shape} seed={res_onnx.used_seed}")

    n = min(audio_torch.shape[-1], audio_onnx.shape[-1])
    at, ao = audio_torch[..., :n], audio_onnx[..., :n]
    diff = np.abs(at - ao)
    denom = np.abs(at).max() + 1e-9
    print(f"[E2E parity] len_torch={audio_torch.shape[-1]} len_onnx={audio_onnx.shape[-1]}")
    print(f"[E2E parity] max_abs_diff={diff.max():.3e} mean_abs_diff={diff.mean():.3e} "
          f"rel_to_peak={diff.max()/denom:.3e}")
    # correlation as a perceptual-ish sanity metric
    corr = float(np.corrcoef(at.ravel(), ao.ravel())[0, 1])
    print(f"[E2E parity] waveform_correlation={corr:.6f}")

    import soundfile as sf
    sf.write("outputs/e2e_onnx.wav", audio_onnx, res_onnx.sample_rate)
    sf.write("outputs/e2e_torch.wav", audio_torch, res_torch.sample_rate)
    print("[saved] outputs/e2e_onnx.wav, outputs/e2e_torch.wav")


if __name__ == "__main__":
    main()
