#!/usr/bin/env bash
set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate 4D-humans
cd /Users/mshah76/4D-Humans

unset PYOPENGL_PLATFORM
python video_demo.py \
  --clips_dir data/fall_dataset/clips \
  --out_folder fall_out \
  --detector regnety \
  --frame_stride 3
