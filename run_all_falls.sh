#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate 4D-humans
cd "${PROJECT_DIR}"

unset PYOPENGL_PLATFORM
python video_demo.py \
  --clips_dir data/fall_dataset/clips \
  --out_folder fall_out \
  --detector regnety \
  --frame_stride 3
