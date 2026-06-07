#!/usr/bin/env bash
# Self-contained Python env for the ONNX export step — NO separate Irodori-TTS
# clone needed. Installs curated deps + the irodori_tts package (--no-deps) +
# the llm-jp tokenizer.json. Requires `uv`.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

uv venv --python 3.10 .venv

# 1. The onnx/export stack + torch. Resolves cleanly only because dacvae and
#    descript-audiotools are NOT in requirements.txt (their tensorboard ->
#    protobuf<4 pin is unsatisfiable with onnx/onnxscript; see requirements.txt).
uv pip install --python .venv -r export/requirements.txt
uv pip install --python .venv "protobuf>=4.25"   # onnx needs >=4

# 2. Runtime deps of dacvae + descript-audiotools, installed normally. None of
#    these pin protobuf, so they coexist with onnx; a recent tensorboard is
#    protobuf>=4 compatible (unlike the old one descript-audiotools's metadata
#    pins — which is why descript-audiotools itself goes in with --no-deps below).
uv pip install --python .venv \
  argbind einops numba \
  flatten_dict julius librosa ffmpy importlib_resources \
  rich markdown2 pyloudnorm randomname torch_stoi scipy tensorboard

# 3. Model definitions + descript-audiotools, --no-deps — their dependency
#    metadata pins protobuf<4 / onnx<1.14, which would break the resolve; the real
#    runtime deps are supplied above. dacvae = the codec used for export.
uv pip install --python .venv --no-deps "descript-audiotools>=0.7.2"
uv pip install --python .venv --no-deps "git+https://github.com/facebookresearch/dacvae"
uv pip install --python .venv --no-deps "git+https://github.com/Aratako/Irodori-TTS"

# fast tokenizer.json for runtime/browser (llm-jp-3-150m has no tokenizer.json on the Hub).
.venv/bin/python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-150m", use_fast=True).save_pretrained("tokenizer/llmjp_tok")
print("tokenizer/llmjp_tok ready")
PY

echo
echo "Setup complete. Generate artifacts (from repo root):"
echo
echo "  # fp32 (required) -> artifacts/onnx/"
echo "  .venv/bin/python export/export_dacvae_decoder.py"
echo "  .venv/bin/python export/export_text_encoder.py"
echo "  .venv/bin/python export/export_dit.py"
echo "  .venv/bin/python export/export_rest.py            # speaker, duration, dacvae encoder"
echo
echo "  # fp16 (optional, faster in-browser) -> artifacts/onnx_fp16/"
echo "  .venv/bin/python export/export_dit_fp16.py        # DiT (special fp16 export)"
echo "  .venv/bin/python export/convert_fp16.py           # dacvae encoder (+ others)"
echo "  .venv/bin/python export/rewrite_convtranspose.py  # decoder: ConvTranspose -> Conv"
echo "  .venv/bin/python export/convert_fp16_decoder_mixed.py \\"
echo "      --in artifacts/onnx/dacvae_decoder_subpix.onnx \\"
echo "      --out artifacts/onnx_fp16/dacvae_decoder.onnx  # decoder fp16 (Snake kept fp32)"
