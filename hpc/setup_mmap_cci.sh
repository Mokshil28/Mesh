#!/usr/bin/env bash
# One-time mmAP binary-classification env on CCI CS GPU servers.
set -euo pipefail

echo "=== GPU check ==="
nvidia-smi || { echo "No GPU visible — are you on cci-csgpu1/3?"; exit 1; }

module purge 2>/dev/null || true
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "Installing Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# Newer conda requires explicit ToS acceptance before create/install.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

ENV_NAME="mmap-gpu"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" -c conda-forge python=3.10 pip -y
fi
conda activate "$ENV_NAME"

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "Env $ENV_NAME already has torch+CUDA."
else
  pip install --upgrade pip
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
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
import torch, timm
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('timm', timm.__version__)
PY

echo "Done. Activate with: conda activate $ENV_NAME"
