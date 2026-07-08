#!/usr/bin/env bash
# Pin-compatible macOS setup for Mesh-main (4D-Humans fall pipeline).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MINIFORGE_DIR="${HOME}/miniforge3"
ENV_NAME="4D-humans"

source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env remove -n "${ENV_NAME}" -y
fi

conda create -n "${ENV_NAME}" python=3.10 pip -y
conda activate "${ENV_NAME}"

# detectron2 + torch 1.11 + torchvision 0.12 (prebuilt, no compile).
conda install -y -c conda-forge \
  detectron2=0.6=py310he2379ec_2_cpu \
  numpy=1.26.4

# Python deps — use --no-deps where pip would upgrade torch.
pip install \
  'setuptools==69.5.1' \
  'scipy==1.11.4' \
  'opencv-python==4.8.1.78' \
  'scikit-image==0.22.0' \
  smplx==0.1.28 pyrender einops timm dill pandas gdown rich \
  hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset \
  'pytorch-lightning==1.9.5' --no-deps \
  'torchmetrics==0.11.4' --no-deps \
  'lightning-utilities>=0.7.0' fsspec tqdm \
  PyOpenGL==3.1.0 pyglet trimesh freetype-py imageio \
  networkx lazy_loader tifffile

pip install --no-build-isolation \
  "https://github.com/mattloper/chumpy/archive/refs/heads/master.zip"

pip install -e "${PROJECT_DIR}" --no-deps

# HMR2 weights (skip if already cached).
if [[ ! -f "${HOME}/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt" ]]; then
  python -c "from hmr2.models import download_models; from hmr2.configs import CACHE_DIR_4DHUMANS; download_models(CACHE_DIR_4DHUMANS)"
fi

mkdir -p "${PROJECT_DIR}/data/fall_dataset/clips"

python - <<'PY'
import torch, detectron2, pytorch_lightning as pl
from hmr2.models.hmr2 import HMR2
print('torch', torch.__version__)
print('detectron2', detectron2.__version__)
print('pytorch_lightning', pl.__version__)
print('HMR2 import OK')
PY

echo ""
echo "Setup finished. If SMPL is missing, see RUNNING.md"
