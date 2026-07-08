#!/usr/bin/env bash
# Linux NVIDIA GPU setup for Mesh-main fall-detection pipeline.
# Usage:  bash setup_gpu.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="4D-humans"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This script needs an NVIDIA GPU + driver."
  exit 1
fi

# --- conda ---------------------------------------------------------------
if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "Install Miniforge/Miniconda first: https://github.com/conda-forge/miniforge"
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Env ${ENV_NAME} exists — activating (remove with: conda env remove -n ${ENV_NAME} -y)"
else
  echo "==> Creating conda env from environment.yml (PyTorch + CUDA 11.8)..."
  conda env create -f "${PROJECT_DIR}/environment.yml"
fi
conda activate "${ENV_NAME}"

# --- project package + extras --------------------------------------------
pip install -e "${PROJECT_DIR}" --no-deps
pip install gdown opencv-python-headless yt-dlp

# --- verify GPU stack ----------------------------------------------------
python - <<'PY'
import torch, detectron2
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("detectron2", detectron2.__version__)
from hmr2.models.hmr2 import HMR2
print("HMR2 import OK")
PY

# --- HMR2 / Detectron2 weights (~5 GB, one-time) ------------------------
if [[ ! -f "${HOME}/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt" ]]; then
  echo "==> Downloading HMR2 model weights..."
  python -c "from hmr2.models import download_models; from hmr2.configs import CACHE_DIR_4DHUMANS; download_models(CACHE_DIR_4DHUMANS)"
fi

# --- system tools --------------------------------------------------------
for cmd in ffmpeg ffprobe; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "WARNING: $cmd not found — install with: sudo apt install ffmpeg"
  fi
done

echo ""
echo "GPU setup complete."
echo "  conda activate ${ENV_NAME}"
echo "  cd ${PROJECT_DIR}"
echo "  python detect_fall_clips_human_v2.py data/fall_dataset_clips/016_V17"
