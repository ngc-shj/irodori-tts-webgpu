#!/usr/bin/env python3
"""Export the conditioning encoders (text, speaker, duration) with fp16
weights/compute but fp32 inputs/outputs.

Same reason as export_dit_fp16.py: post-hoc onnxconverter-common mis-handles the
dynamo Cast nodes (val_*/Cast type errors), so we export from a half() model and
cast at the graph boundary instead. These graphs are MatMul/attention only (no
Conv/ConvTranspose), so fp16 is correct on WebGPU; the win is download size
(~649 MB fp32 -> ~325 MB fp16), not speed (they run once on short sequences).

Outputs -> artifacts/onnx_fp16/{text_encoder,speaker_encoder,duration}.onnx
The JS runtime is unchanged (fp32 IO); web/app.mjs loads these when their fp16
checkbox is on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.export import Dim

from common import load_model
from rope_patch import apply_patch, reset_rope_caches
import irodori_tts.model as M

H = torch.float16


def _safe_attention_mask_noguard(x, mask):
    return x, mask.to(device=x.device, dtype=torch.bool)


M._safe_attention_mask = _safe_attention_mask_noguard


class TextEncoderFp16(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_encoder = model.text_encoder
        self.text_norm = model.text_norm

    def forward(self, input_ids, mask):
        return self.text_norm(self.text_encoder(input_ids, mask)).float()


class SpeakerFp16(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ref_latent, ref_mask):
        rs = self.model.speaker_encoder(ref_latent.to(H), ref_mask)
        rs = self.model.speaker_norm(rs)
        rs, rm = self.model._prepend_masked_mean_token(rs, ref_mask)
        return rs.float(), rm


class DurationFp16(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text_state, text_mask, aux, speaker_state, speaker_mask, has_speaker):
        return self.model.predict_duration_log_frames(
            text_state=text_state.to(H), text_mask=text_mask,
            speaker_state=speaker_state.to(H), speaker_mask=speaker_mask,
            duration_features=aux.to(H), has_speaker=has_speaker,
        ).float()


def _check(name, sess_path, feed, ref):
    import onnxruntime as ort
    sess = ort.InferenceSession(sess_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, feed)[0]
    ref = ref if isinstance(ref, np.ndarray) else ref.detach().numpy()
    sz = sum(f.stat().st_size for f in Path(sess_path).parent.glob(Path(sess_path).name + "*")) / 1e6
    print(f"[fp16] {name}: ORT loads OK (~{sz:.0f} MB)  max|Δ| vs torch-fp16 = {np.abs(ref-got).max():.3e}")


def main() -> None:
    out_dir = Path("artifacts/onnx_fp16")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model()
    apply_patch(); reset_rope_caches(model)
    model = model.half()
    B = 2

    # ---- text encoder (legacy exporter, matching export_text_encoder.py) ----
    te = TextEncoderFp16(model).eval()
    S = 24
    ids = torch.randint(0, cfg.text_vocab_size, (1, S), dtype=torch.long)
    tmask = torch.ones(1, S, dtype=torch.bool)
    with torch.inference_mode():
        ref = te(ids, tmask)
    p = str(out_dir / "text_encoder.onnx")
    torch.onnx.export(
        te, (ids, tmask), p,
        input_names=["input_ids", "mask"], output_names=["text_state"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "mask": {0: "b", 1: "s"},
                      "text_state": {0: "b", 1: "s"}},
        opset_version=18,
    )
    _check("text_encoder", p, {"input_ids": ids.numpy(), "mask": tmask.numpy()}, ref)

    # ---- speaker encoder ----
    sp = SpeakerFp16(model).eval()
    T = 40
    rl = torch.randn(B, T, cfg.speaker_patched_latent_dim)
    rm = torch.ones(B, T, dtype=torch.bool)
    with torch.inference_mode():
        rs, _ = sp(rl, rm)
    b = Dim("b"); t = Dim("t")
    p = str(out_dir / "speaker_encoder.onnx")
    torch.onnx.export(
        sp, (rl, rm), p,
        input_names=["ref_latent", "ref_mask"], output_names=["speaker_state", "speaker_mask"],
        dynamic_shapes={"ref_latent": {0: b, 1: t}, "ref_mask": {0: b, 1: t}},
        opset_version=18, dynamo=True,
    )
    _check("speaker_encoder", p, {"ref_latent": rl.numpy(), "ref_mask": rm.numpy()}, rs)

    # ---- duration predictor ----
    dur = DurationFp16(model).eval()
    St, Tsp = 24, 41
    ts = torch.randn(B, St, cfg.text_dim); tm = torch.ones(B, St, dtype=torch.bool)
    aux = torch.zeros(B, cfg.duration_aux_dim)
    sps = torch.randn(B, Tsp, cfg.speaker_dim); spm = torch.ones(B, Tsp, dtype=torch.bool)
    hs = torch.ones(B, dtype=torch.bool)
    with torch.inference_mode():
        rd = dur(ts, tm, aux, sps, spm, hs)
    bb = Dim("b"); st = Dim("st"); tsp = Dim("tsp")
    p = str(out_dir / "duration.onnx")
    torch.onnx.export(
        dur, (ts, tm, aux, sps, spm, hs), p,
        input_names=["text_state", "text_mask", "aux", "speaker_state", "speaker_mask", "has_speaker"],
        output_names=["log_frames"],
        dynamic_shapes={"text_state": {0: bb, 1: st}, "text_mask": {0: bb, 1: st},
                        "aux": {0: bb}, "speaker_state": {0: bb, 1: tsp},
                        "speaker_mask": {0: bb, 1: tsp}, "has_speaker": {0: bb}},
        opset_version=18, dynamo=True,
    )
    _check("duration", p, {"text_state": ts.numpy(), "text_mask": tm.numpy(), "aux": aux.numpy(),
                           "speaker_state": sps.numpy(), "speaker_mask": spm.numpy(),
                           "has_speaker": hs.numpy()}, rd)


if __name__ == "__main__":
    main()
