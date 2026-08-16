#!/usr/bin/env bash
# Detach binary fall/non-fall training so Cursor/agent shell exit cannot kill it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${ROOT}/output/finetune/fall_nonfall_binary_smpl"
PIDFILE="${OUT}/train.pid"
LOG="${OUT}/train.log"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"

mkdir -p "${OUT}"

if [[ -f "${PIDFILE}" ]]; then
  old="$(cat "${PIDFILE}" || true)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "Training already running as PID ${old}"
    exit 0
  fi
  rm -f "${PIDFILE}"
fi

# shellcheck disable=SC1091
source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
conda activate mmap

DATA_ROOT="/Volumes/data/fall down/radar_data/fall_nonfall_binary_balanced/dataset"
if [[ ! -d "${DATA_ROOT}/train" ]]; then
  echo "Missing dataset at ${DATA_ROOT}"
  exit 1
fi

DEVICE="${DEVICE:-mps}"
if [[ "${DEVICE}" == "mps" ]] && ! python -c 'import torch; assert torch.backends.mps.is_available()'; then
  DEVICE=cpu
fi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Double-detach on macOS: nohup + background from a subshell that exits immediately.
(
  cd "${ROOT}"
  # caffeinate keeps the Mac from sleeping mid-train; -i ignores idle sleep.
  exec caffeinate -dims nohup python -u run_finetuning_heatmap_wholemodel.py \
    --config cfgs/finetune/fall_nonfall_binary.yaml \
    --device "${DEVICE}" \
    --data_path "${DATA_ROOT}/train" \
    --eval_data_path "${DATA_ROOT}/val" \
    --nb_classes 2 \
    --num_workers 0 \
    --no_pin_mem \
    --save_ckpt_freq 1 \
    --output_dir "${OUT}" \
    >> "${LOG}" 2>&1
) &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PIDFILE}"
# Give it a moment to start writing
sleep 3
if kill -0 "${TRAIN_PID}" 2>/dev/null; then
  echo "Started detached training PID ${TRAIN_PID}"
  echo "Log: ${LOG}"
  echo "PID file: ${PIDFILE}"
  tail -n 5 "${LOG}" || true
else
  echo "Training failed to stay up; see ${LOG}"
  exit 1
fi
