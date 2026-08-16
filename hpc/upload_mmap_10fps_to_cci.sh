#!/usr/bin/env bash
# Upload 10fps binary dataset + mmAP code to CCI.
set -euo pipefail

HOST="${HPC_HOST:-mshah76@cci-csgpu1.charlotte.edu}"
REMOTE_DIR="${HPC_DIR:-~/Mesh-main}"
DATA_LOCAL="${DATA_LOCAL:-/Volumes/data/fall down/radar_data/fall_nonfall_binary_10fps}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"

echo "=== Sync mmAP-slim code ==="
rsync -avz --progress \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude 'output/' --exclude '.venv/' \
  mmAP-slim/ "${HOST}:${REMOTE_DIR}/mmAP-slim/"

rsync -avz --progress \
  hpc/setup_mmap_cci.sh \
  hpc/setup_10fps_mmap_pipeline.sh \
  hpc/run_mmap_10fps_train_cci.sh \
  hpc/upload_mmap_10fps_to_cci.sh \
  "${HOST}:${REMOTE_DIR}/hpc/"

[[ -d "${DATA_LOCAL}/dataset/train" ]] || { echo "Missing ${DATA_LOCAL}/dataset/train — run setup_10fps_mmap_pipeline.sh first"; exit 1; }

echo "=== Sync 10fps binary dataset ==="
# Prefer tar stream for many small files
ssh "$HOST" "mkdir -p ${REMOTE_DIR}/data && rm -rf ${REMOTE_DIR}/data/fall_nonfall_binary_10fps"
tar --exclude='._*' --exclude='.DS_Store' -C "$(dirname "$DATA_LOCAL")" -cf - "$(basename "$DATA_LOCAL")" \
  | ssh "$HOST" "tar -xf - -C ${REMOTE_DIR}/data && du -sh ${REMOTE_DIR}/data/fall_nonfall_binary_10fps && find ${REMOTE_DIR}/data/fall_nonfall_binary_10fps/dataset -name '*.npy' | wc -l"

echo "Upload done. On GPU:"
echo "  bash ~/Mesh-main/hpc/run_mmap_10fps_train_cci.sh"
