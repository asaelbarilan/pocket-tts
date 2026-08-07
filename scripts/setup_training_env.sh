#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url "${TORCH_INDEX_URL}"
python -m pip install -e '.[training]'
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA PyTorch is not available. Check TORCH_INDEX_URL and the GPU driver.")
print("gpu:", torch.cuda.get_device_name(0))
print("vram GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY
