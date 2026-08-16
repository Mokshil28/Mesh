#!/usr/bin/env bash
# Upload mmAP-slim code + balanced binary dataset to CCI GPU, then train.
set -euo pipefail

HOST="${HPC_HOST:-mshah76@cci-csgpu1.charlotte.edu}"
REMOTE_DIR="${HPC_DIR:-~/Mesh-main}"
DATA_LOCAL="${DATA_LOCAL:-/Volumes/data/fall down/radar_data/fall_nonfall_binary_balanced}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"

echo "=== 1) Sync mmAP-slim code to ${HOST}:${REMOTE_DIR} ==="
rsync -avz --progress \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'output/' \
  --exclude '.venv/' \
  --exclude 'finetune_dataaug/' \
  mmAP-slim/ "${HOST}:${REMOTE_DIR}/mmAP-slim/"

rsync -avz --progress \
  hpc/setup_mmap_cci.sh \
  hpc/run_mmap_binary_train_cci.sh \
  "${HOST}:${REMOTE_DIR}/hpc/"

if [[ ! -d "${DATA_LOCAL}/dataset/train" ]]; then
  echo "Missing local dataset: ${DATA_LOCAL}/dataset/train"
  exit 1
fi

echo "=== 2) Sync balanced binary dataset (~22GB) ==="
echo "Remote path: ${REMOTE_DIR}/data/fall_nonfall_binary_balanced/"
rsync -avz --progress \
  --exclude '._*' \
  "${DATA_LOCAL}/" \
  "${HOST}:${REMOTE_DIR}/data/fall_nonfall_binary_balanced/"

echo ""
echo "Upload done. Next on the GPU server:"
echo "  ssh ${HOST}"
echo "  bash ~/Mesh-main/hpc/setup_mmap_cci.sh"
echo "  bash ~/Mesh-main/hpc/run_mmap_binary_train_cci.sh"
