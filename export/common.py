"""Shared loaders for the ONNX export PoC."""
from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download

from irodori_tts.config import ModelConfig
from irodori_tts.inference_runtime import _load_checkpoint_from_safetensors
from irodori_tts.model import TextToLatentRFDiT


def load_model(
    repo_id: str = "Aratako/Irodori-TTS-500M-v3",
    device: str = "cpu",
) -> tuple[TextToLatentRFDiT, ModelConfig]:
    ckpt = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    model_state, model_cfg_dict, _ = _load_checkpoint_from_safetensors(__import__("pathlib").Path(ckpt))
    cfg = ModelConfig(**model_cfg_dict)
    model = TextToLatentRFDiT(cfg).to(device)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.eval()
    return model, cfg
