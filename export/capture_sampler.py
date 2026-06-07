#!/usr/bin/env python3
"""Capture rectified-flow sampler ground truth for JS-loop verification.

Adapted from spike/tts-webgpu/capture_sampler.py to this machine: uses the
no-ref golden wav as the reference, forces the no-kv-cache path (matches the
exported DiT), and dumps both an .npz and raw .bin tensors for the JS harness.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchaudio


def _sf_load(path, *a, **k):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T).contiguous(), sr


torchaudio.load = _sf_load

OUT = Path("artifacts/ref")
REF_WAV = "outputs/golden_noref.wav"
TEXT = "この提案、結論は分かりました。ただ前提が一つ怪しい。"
CKPT = "Aratako/Irodori-TTS-500M-v3"
STEPS = 16
SEED = 1234


def main() -> None:
    from huggingface_hub import hf_hub_download
    from irodori_tts import rf as rf_mod
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    from irodori_tts.model import TextToLatentRFDiT

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = hf_hub_download(repo_id=CKPT, filename="model.safetensors")
    rt = InferenceRuntime.from_key(
        RuntimeKey(checkpoint=ckpt, model_device="cpu", codec_device="cpu")
    )

    trace, const, final = [], {}, {}

    # Capture the reference latent fed into the speaker encoder (cond pass).
    ref_capture = {}
    sp_enc = rt.model.speaker_encoder
    sp_fwd_orig = sp_enc.forward

    def sp_hook(latent, mask):
        if "ref_latent" not in ref_capture:
            ref_capture["ref_latent"] = latent.detach().cpu().numpy()
            ref_capture["ref_mask"] = mask.detach().cpu().numpy().astype(np.uint8)
        return sp_fwd_orig(latent, mask)

    sp_enc.forward = sp_hook

    fwd_orig = TextToLatentRFDiT.forward_with_encoded_conditions

    def fwd_hook(self, x_t, t, text_state, text_mask, speaker_state, speaker_mask,
                 caption_state=None, caption_mask=None, latent_mask=None,
                 context_kv_cache=None):
        b = x_t.shape[0]
        if not const and b == 3:
            const["text_state"] = text_state.detach().cpu().numpy()
            const["text_mask"] = text_mask.detach().cpu().numpy().astype(np.uint8)
            const["speaker_state"] = speaker_state.detach().cpu().numpy()
            const["speaker_mask"] = speaker_mask.detach().cpu().numpy().astype(np.uint8)
            const["x0"] = x_t[:1].detach().cpu().numpy()
        trace.append((float(t[0].item()), int(b)))
        return fwd_orig(self, x_t, t, text_state, text_mask, speaker_state,
                        speaker_mask, caption_state, caption_mask, latent_mask,
                        context_kv_cache=None)

    sampler_orig = rf_mod.sample_euler_rf_cfg

    def sampler_hook(*a, **kw):
        kw["use_context_kv_cache"] = False
        out = sampler_orig(*a, **kw)
        final["latent"] = out.detach().cpu().numpy()
        return out

    TextToLatentRFDiT.forward_with_encoded_conditions = fwd_hook
    rf_mod.sample_euler_rf_cfg = sampler_hook
    import irodori_tts.inference_runtime as ir
    ir.sample_euler_rf_cfg = sampler_hook

    res = rt.synthesize(SamplingRequest(
        text=TEXT, ref_wav=REF_WAV, num_steps=STEPS, seed=SEED,
        cfg_scale_text=3.0, cfg_scale_speaker=5.0, cfg_guidance_mode="independent",
        t_schedule_mode="linear", trim_tail=True, context_kv_cache=False,
    ))
    audio = res.audio.detach().cpu().float().numpy().reshape(-1)

    np.savez(OUT / "sampler_capture.npz",
             text_state=const["text_state"], text_mask=const["text_mask"],
             speaker_state=const["speaker_state"], speaker_mask=const["speaker_mask"],
             x0=const["x0"], final_latent=final["latent"], audio=audio,
             sample_rate=np.int64(res.sample_rate),
             cfg_scale_text=np.float32(3.0), cfg_scale_speaker=np.float32(5.0),
             num_steps=np.int64(STEPS))

    # Raw bins + manifest for the JS harness.
    meta = {}
    def dump(name, arr, dtype):
        a = arr.astype(np.float32 if dtype == "float32" else np.uint8)
        (OUT / f"{name}.bin").write_bytes(a.tobytes())
        meta[name] = {"shape": list(arr.shape), "dtype": dtype}
    dump("text_state", const["text_state"], "float32")
    dump("text_mask", const["text_mask"], "bool")
    dump("speaker_state", const["speaker_state"], "float32")
    dump("speaker_mask", const["speaker_mask"], "bool")
    dump("x0", const["x0"], "float32")
    dump("final_latent", final["latent"], "float32")
    dump("audio", audio, "float32")
    dump("ref_latent", ref_capture["ref_latent"], "float32")
    dump("ref_mask", ref_capture["ref_mask"], "bool")
    meta["text"] = TEXT
    meta["params"] = {"num_steps": STEPS, "cfg_scale_text": 3.0,
                      "cfg_scale_speaker": 5.0, "sample_rate": int(res.sample_rate),
                      "init_scale": 0.999, "cfg_min_t": 0.5, "cfg_max_t": 1.0}
    (OUT / "sampler_manifest.json").write_text(json.dumps(meta, indent=2))

    print(f"[capture] steps={len(trace)} seq_len={const['x0'].shape[1]} "
          f"final_latent={final['latent'].shape} audio={audio.shape} sr={res.sample_rate}")
    print(f"[capture] b_trace={[b for _, b in trace]}")


if __name__ == "__main__":
    main()
