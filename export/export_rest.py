#!/usr/bin/env python3
"""Export the remaining model graphs: speaker encoder, duration predictor,
and the deterministic DACVAE encoder. All with symbolic batch + length."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.export import Dim

from common import load_model
from rope_patch import apply_patch, reset_rope_caches
from irodori_tts.codec import DACVAECodec
import irodori_tts.model as M


def _safe_attention_mask_noguard(x, mask):
    # Export-time variant without the data-dependent `has_any.all()` Python branch.
    # Safe for inference, where conditioning sequences always have >=1 valid token.
    return x, mask.to(device=x.device, dtype=torch.bool)


M._safe_attention_mask = _safe_attention_mask_noguard


def _parity(name, ref, got):
    ref = ref if isinstance(ref, np.ndarray) else ref.numpy()
    print(f"[parity:{name}] max_abs_diff={np.abs(ref - got).max():.3e}")


class SpeakerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ref_latent, ref_mask):
        rs = self.model.speaker_encoder(ref_latent, ref_mask)
        rs = self.model.speaker_norm(rs)
        rs, rm = self.model._prepend_masked_mean_token(rs, ref_mask)
        return rs, rm


class DurationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text_state, text_mask, aux, speaker_state, speaker_mask, has_speaker):
        return self.model.predict_duration_log_frames(
            text_state=text_state, text_mask=text_mask,
            speaker_state=speaker_state, speaker_mask=speaker_mask,
            duration_features=aux, has_speaker=has_speaker,
        )


class EncoderWrapper(torch.nn.Module):
    def __init__(self, dac):
        super().__init__()
        self.dac = dac

    def forward(self, wav):  # wav: (B,1,T) pre-padded to hop multiple
        z = self.dac.encoder(wav)
        mean, _ = self.dac.quantizer.in_proj(z).chunk(2, dim=1)
        return mean.transpose(1, 2)  # (B, T_latent, 32)


def main() -> None:
    import onnxruntime as ort
    Path("artifacts/onnx").mkdir(exist_ok=True)
    model, cfg = load_model()
    apply_patch(); reset_rope_caches(model)

    # ---- speaker encoder ----
    sp = SpeakerWrapper(model).eval()
    B, T = 2, 40
    rl = torch.randn(B, T, cfg.speaker_patched_latent_dim)
    rm = torch.ones(B, T, dtype=torch.bool)
    with torch.inference_mode():
        rs, rmo = sp(rl, rm)
    b = Dim("b"); t = Dim("t")
    torch.onnx.export(
        sp, (rl, rm), "artifacts/onnx/speaker_encoder.onnx",
        input_names=["ref_latent", "ref_mask"], output_names=["speaker_state", "speaker_mask"],
        dynamic_shapes={"ref_latent": {0: b, 1: t}, "ref_mask": {0: b, 1: t}},
        opset_version=18, dynamo=True,
    )
    s = ort.InferenceSession("artifacts/onnx/speaker_encoder.onnx", providers=["CPUExecutionProvider"])
    g = s.run(None, {"ref_latent": rl.numpy(), "ref_mask": rm.numpy()})
    _parity("speaker_state", rs, g[0])
    print(f"  speaker_state shape={g[0].shape} mask shape={g[1].shape}")

    # ---- duration predictor ----
    dur = DurationWrapper(model).eval()
    St, Tsp = 24, 41
    ts = torch.randn(B, St, cfg.text_dim); tm = torch.ones(B, St, dtype=torch.bool)
    aux = torch.zeros(B, cfg.duration_aux_dim)
    sps = torch.randn(B, Tsp, cfg.speaker_dim); spm = torch.ones(B, Tsp, dtype=torch.bool)
    hs = torch.ones(B, dtype=torch.bool)
    with torch.inference_mode():
        rd = dur(ts, tm, aux, sps, spm, hs)
    bb = Dim("b"); st = Dim("st"); tsp = Dim("tsp")
    torch.onnx.export(
        dur, (ts, tm, aux, sps, spm, hs), "artifacts/onnx/duration.onnx",
        input_names=["text_state", "text_mask", "aux", "speaker_state", "speaker_mask", "has_speaker"],
        output_names=["log_frames"],
        dynamic_shapes={"text_state": {0: bb, 1: st}, "text_mask": {0: bb, 1: st},
                        "aux": {0: bb}, "speaker_state": {0: bb, 1: tsp},
                        "speaker_mask": {0: bb, 1: tsp}, "has_speaker": {0: bb}},
        opset_version=18, dynamo=True,
    )
    s = ort.InferenceSession("artifacts/onnx/duration.onnx", providers=["CPUExecutionProvider"])
    g = s.run(None, {"text_state": ts.numpy(), "text_mask": tm.numpy(), "aux": aux.numpy(),
                     "speaker_state": sps.numpy(), "speaker_mask": spm.numpy(),
                     "has_speaker": hs.numpy()})[0]
    _parity("log_frames", rd, g)
    print(f"  log_frames={rd.numpy().tolist()} vs onnx={g.tolist()}")

    # ---- DACVAE encoder (deterministic) ----
    codec = DACVAECodec.load(device="cpu", dtype=torch.float32)
    hop = int(codec.model.hop_length)
    print(f"[codec] hop_length={hop}")
    enc = EncoderWrapper(codec.model).eval()
    T_aud = hop * 60
    wav = torch.randn(2, 1, T_aud)
    with torch.inference_mode():
        re = enc(wav)
    bbb = Dim("b"); frames = Dim("frames", max=4_000_000)
    torch.onnx.export(
        enc, (wav,), "artifacts/onnx/dacvae_encoder.onnx",
        input_names=["wav"], output_names=["latent"],
        dynamic_shapes={"wav": {0: bbb, 2: frames}},
        opset_version=18, dynamo=True,
    )
    s = ort.InferenceSession("artifacts/onnx/dacvae_encoder.onnx", providers=["CPUExecutionProvider"])
    g = s.run(None, {"wav": wav.numpy()})[0]
    _parity("dacvae_encoder", re, g)
    print(f"  latent shape={g.shape}")


if __name__ == "__main__":
    main()
