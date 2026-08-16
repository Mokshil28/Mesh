#!/usr/bin/env bash
# Create the mmAP / binary fall-classification conda environment.
# Works on Apple Silicon (MPS) and Linux CUDA hosts.
set -euo pipefail

MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-mmap}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f "${MINIFORGE_DIR}/etc/profile.d/conda.sh" ]]; then
  echo "Miniforge not found at ${MINIFORGE_DIR}."
  echo "Install it first with: bash ../setup_miniforge.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Removing existing env ${ENV_NAME}"
  conda env remove -n "${ENV_NAME}" -y
fi

conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y
conda activate "${ENV_NAME}"

# Prefer a current PyTorch build; CUDA hosts can override TORCH_INDEX.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  pip install torch torchvision
else
  pip install torch torchvision --index-url "${TORCH_INDEX}"
fi

pip install \
  'timm==0.4.12' \
  'einops==0.3.2' \
  'pandas==1.3.4' \
  'albumentations==1.1.0' \
  'wandb==0.12.11' \
  'numpy==1.26.4' \
  'PyYAML' \
  'scikit-learn' \
  'matplotlib' \
  'tqdm' \
  'opencv-python-headless==4.8.1.78'

python - <<'PY'
import torch
import timm
import einops
print('torch', torch.__version__)
print('mps', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
print('cuda', torch.cuda.is_available())
print('timm', timm.__version__)
print('einops OK')
PY

echo ""
echo "Environment '${ENV_NAME}' ready."
echo "Activate with:"
echo "  source ${MINIFORGE_DIR}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
echo "Project: ${PROJECT_DIR}"
