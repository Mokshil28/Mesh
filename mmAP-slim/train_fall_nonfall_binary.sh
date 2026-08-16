#!/usr/bin/env bash
# Train binary fall vs non-fall on the balanced SMPL heatmap dataset (no HMR).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
# shellcheck disable=SC1091
source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
conda activate mmap
cd "${ROOT}"

DATA_ROOT="/Volumes/data/fall down/radar_data/fall_nonfall_binary_balanced/dataset"
if [[ ! -d "${DATA_ROOT}/train" ]]; then
  echo "Missing dataset at ${DATA_ROOT}"
  exit 1
fi

DEVICE="${DEVICE:-mps}"
if [[ "${DEVICE}" == "mps" ]] && ! python -c 'import torch; assert torch.backends.mps.is_available()'; then
  echo "MPS unavailable; falling back to CPU"
  DEVICE=cpu
fi

mkdir -p output/finetune/fall_nonfall_binary_smpl
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
python -u run_finetuning_heatmap_wholemodel.py \
  --config cfgs/finetune/fall_nonfall_binary.yaml \
  --device "${DEVICE}" \
  --data_path "${DATA_ROOT}/train" \
  --eval_data_path "${DATA_ROOT}/val" \
  --nb_classes 2 \
  --num_workers 0 \
  --no_pin_mem \
  --output_dir output/finetune/fall_nonfall_binary_smpl \
  "$@"
