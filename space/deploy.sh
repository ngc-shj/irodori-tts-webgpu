#!/usr/bin/env bash
# Deploy the static HF Space. Stages the Space-specific files (this dir) together
# with runtime/pipeline.mjs and tokenizer/llmjp_tok/ from the repo, then uploads.
# The ~3.7 GB of ONNX weights are NOT bundled — the app fetches them at runtime
# from the model repo (see MODEL_REPO in app.mjs). Requires a write-scoped HF login.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

SPACE_ID="${SPACE_ID:-noguchis/irodori-tts-webgpu}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp space/README.md space/index.html space/app.mjs "$STAGE/"
mkdir -p "$STAGE/runtime" "$STAGE/tokenizer/llmjp_tok"
cp runtime/pipeline.mjs "$STAGE/runtime/"
cp tokenizer/llmjp_tok/* "$STAGE/tokenizer/llmjp_tok/"

.venv/bin/python - "$SPACE_ID" "$STAGE" <<'PY'
import sys
from huggingface_hub import HfApi
space_id, stage = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(space_id, repo_type="space", space_sdk="static", private=False, exist_ok=True)
api.upload_folder(folder_path=stage, repo_id=space_id, repo_type="space",
                  commit_message="Deploy WebGPU demo")
print("deployed: https://huggingface.co/spaces/" + space_id)
PY
