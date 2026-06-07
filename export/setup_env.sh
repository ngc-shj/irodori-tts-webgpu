#!/usr/bin/env bash
# Self-contained Python env for the ONNX export step — NO separate Irodori-TTS
# clone needed. Installs curated deps + the irodori_tts package (--no-deps) +
# the llm-jp tokenizer.json. Requires `uv`.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

uv venv --python 3.10 .venv
uv pip install --python .venv -r export/requirements.txt
# onnx needs protobuf>=4; dacvae's tensorboard pulls 3.19, so re-pin AFTER.
uv pip install --python .venv "protobuf>=4.25"
# the model definition only — without its CUDA/torchcodec/silentcipher deps.
uv pip install --python .venv --no-deps "git+https://github.com/Aratako/Irodori-TTS"

# fast tokenizer.json for runtime/browser (llm-jp-3-150m has no tokenizer.json on the Hub).
.venv/bin/python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-150m", use_fast=True).save_pretrained("tokenizer/llmjp_tok")
print("tokenizer/llmjp_tok ready")
PY

echo
echo "Setup complete. Generate artifacts (from repo root):"
echo "  .venv/bin/python export/export_dacvae_decoder.py"
echo "  .venv/bin/python export/export_text_encoder.py"
echo "  .venv/bin/python export/export_dit.py"
echo "  .venv/bin/python export/export_rest.py"
