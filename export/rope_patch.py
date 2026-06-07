"""ONNX-exportable real-valued RoPE.

The stock implementation in irodori_tts.model uses torch.view_as_complex /
view_as_real, which the ONNX exporter cannot lower ("No decompositions
registered for the complex-valued input"). We monkeypatch the two module-level
functions with mathematically equivalent real arithmetic.

Complex form:  (a + b i) * (cos + i sin) = (a cos - b sin) + i (a sin + b cos)
"""
from __future__ import annotations

import torch

import irodori_tts.model as M


def precompute_freqs_cis_real(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    # Returns rotation angles of shape (end, dim/2) instead of complex exponentials.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    return torch.outer(t, freqs)  # (end, dim/2)


def apply_rotary_emb_real(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: (B, S, H, Dh); freqs: (S, Dh/2) angles. Interleaved (GPT-J) convention:
    # consecutive element pairs (x[2i], x[2i+1]) are (real, imag).
    # Use strided slicing (no -1 reshape) so the exporter keeps every dim symbolic.
    cos = torch.cos(freqs)[None, :, None, :]
    sin = torch.sin(freqs)[None, :, None, :]
    xf = x.float()
    a = xf[..., 0::2]
    b = xf[..., 1::2]
    stacked = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1)
    out = torch.flatten(stacked, start_dim=-2)
    return out.type_as(x)


_PATCHED = False


def _rope_freqs_dynamic(self, seq_len, device):
    # No cache / no data-dependent branch, so the exporter keeps seq_len symbolic.
    return precompute_freqs_cis_real(self.head_dim, seq_len).to(device)


def apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    M.precompute_freqs_cis = precompute_freqs_cis_real
    M.apply_rotary_emb = apply_rotary_emb_real
    M.TextEncoder._rope_freqs = _rope_freqs_dynamic
    M.ReferenceLatentEncoder._rope_freqs = _rope_freqs_dynamic
    M.TextToLatentRFDiT._rope_freqs = _rope_freqs_dynamic
    _PATCHED = True


def reset_rope_caches(model: torch.nn.Module) -> None:
    # Drop any complex-valued RoPE caches so the patched real precompute runs.
    for module in model.modules():
        if hasattr(module, "_freqs_cis_cache"):
            module._freqs_cis_cache = torch.empty(0, 0, dtype=torch.float32)
