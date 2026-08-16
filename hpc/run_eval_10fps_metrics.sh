#!/usr/bin/env bash
# Full binary metrics (Acc / balanced / F1 / precision / recall / confusion)
# for 10fps fall/non-fall checkpoints.
#
# Usage:
#   bash ~/Mesh-main/hpc/run_eval_10fps_metrics.sh bs32 val
#   bash ~/Mesh-main/hpc/run_eval_10fps_metrics.sh bs64 val
#   bash ~/Mesh-main/hpc/run_eval_10fps_metrics.sh bs32 test   # sealed test — final only
#   bash ~/Mesh-main/hpc/run_eval_10fps_metrics.sh all val      # both models, val only
set -euo pipefail

ROOT="${HOME}/Mesh-main"
DATA="${ROOT}/data/fall_nonfall_binary_10fps/dataset"
WHICH="${1:-all}"
SPLIT="${2:-val}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || source /data/swap/mshah76_sam_body4d_gpu3/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null \
  || { echo "conda not found"; exit 1; }
conda activate mmap-gpu

cd "${ROOT}/mmAP-slim"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

run_one() {
  local tag="$1"
  local ckpt="$2"
  local out="$3"
  [[ -f "${ckpt}" ]] || { echo "Missing checkpoint: ${ckpt}"; return 1; }
  [[ -d "${DATA}/${SPLIT}" ]] || { echo "Missing split: ${DATA}/${SPLIT}"; return 1; }
  mkdir -p "${out}"
  echo "=== Evaluating ${tag} on ${SPLIT} ==="
  python -u eval_binary_fall_metrics.py \
    --checkpoint "${ckpt}" \
    --data_root "${DATA}" \
    --split "${SPLIT}" \
    --output_dir "${out}" \
    --batch_size 48 \
    --num_workers 4 \
    --device cuda
  echo "=== Done ${tag}/${SPLIT} -> ${out}/metrics_${SPLIT}.txt ==="
}

case "${WHICH}" in
  bs32)
    run_one bs32 \
      "${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_10fps/checkpoint-best.pth" \
      "${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_10fps/eval_${SPLIT}"
    ;;
  bs64)
    run_one bs64 \
      "${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_10fps_bs64/checkpoint-best.pth" \
      "${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_10fps_bs64/eval_${SPLIT}"
    ;;
  all)
    run_one bs32 \
      "${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_10fps/checkpoint-best.pth" \
      "${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_10fps/eval_${SPLIT}" || true
    run_one bs64 \
      "${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_10fps_bs64/checkpoint-best.pth" \
      "${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_10fps_bs64/eval_${SPLIT}" || true
    ;;
  *)
    echo "Usage: $0 {bs32|bs64|all} {val|test|train}"
    exit 1
    ;;
esac
