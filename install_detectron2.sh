#!/usr/bin/env bash
set -euo pipefail

ENV_PYTHON="/Users/mshah76/miniforge3/envs/4D-humans/bin/python"
ENV_PIP="/Users/mshah76/miniforge3/envs/4D-humans/bin/pip"

if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools are required. Run: xcode-select --install"
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

curl -fsSL -o "$TMP_DIR/detectron2.zip" https://github.com/facebookresearch/detectron2/archive/refs/heads/main.zip
unzip -q "$TMP_DIR/detectron2.zip" -d "$TMP_DIR"

export CC="${CONDA_PREFIX:-/Users/mshah76/miniforge3/envs/4D-humans}/bin/clang"
export CXX="${CONDA_PREFIX:-/Users/mshah76/miniforge3/envs/4D-humans}/bin/clang++"
export FORCE_CUDA=0

"$ENV_PIP" install --no-build-isolation "$TMP_DIR/detectron2-main"
"$ENV_PYTHON" -c "import detectron2; print('detectron2', detectron2.__version__)"
