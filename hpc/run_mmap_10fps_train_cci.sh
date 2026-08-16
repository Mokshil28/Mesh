#!/usr/bin/env bash
# Train mmAP on 10fps fall/non-fall with class balance.
# Phase A (default): batch 64
# Phase B: --phase32  (resume from best, batch 32)
set -euo pipefail

ROOT="${HOME}/Mesh-main"
DATA="${ROOT}/data/fall_nonfall_binary_10fps/dataset"
OUT="${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_10fps"
GPU="${CUDA_VISIBLE_DEVICES:-4}"   # pick a less-busy GPU by default
SESSION="${SESSION:-mmap_10fps}"
EPOCHS="${EPOCHS:-30}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mmap-gpu

PHASE="64"
BATCH="${BATCH:-64}"
RESUME_FLAGS=(--no_auto_resume)
EXTRA_TITLE="batch${BATCH}"

case "${1:-}" in
  --phase32)
    PHASE="32"
    BATCH=32
    SESSION="${SESSION}_b32"
    OUT="${OUT}_b32"
    RESUME_FLAGS=(--finetune "${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_10fps/checkpoint-best.pth" --no_auto_resume)
    EXTRA_TITLE="batch32_from_best"
    ;;
  --phase16)
    # Safer when GPUs are shared / partially occupied
    PHASE="16"
    BATCH=16
    SESSION="${SESSION}_b16"
    OUT="${OUT}_b16"
    RESUME_FLAGS=(--no_auto_resume)
    EXTRA_TITLE="batch16"
    ;;
  --status)
    tmux ls 2>/dev/null | grep -E "mmap_10fps" || echo "(no mmap_10fps session)"
    echo "=== batch64 log ==="; tail -n 30 "${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_10fps/train.log" 2>/dev/null || true
    echo "=== batch32 log ==="; tail -n 30 "${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_10fps_b32/train.log" 2>/dev/null || true
    exit 0
    ;;
  --attach)
    tmux attach -t "${SESSION}"
    exit 0
    ;;
  --kill)
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    echo killed
    exit 0
    ;;
esac

[[ -d "${DATA}/train" ]] || { echo "Missing dataset ${DATA}/train"; exit 1; }
mkdir -p "${OUT}"
nvidia-smi -L || true

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux '${SESSION}' already exists"
  exit 0
fi

RUNNER="${OUT}/_tmux_train.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "\$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mmap-gpu
cd "\$HOME/Mesh-main/mmAP-slim"
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "Starting 10fps train (${EXTRA_TITLE}) on GPU ${GPU} at \$(date)" | tee -a "${OUT}/train.log"
python -u run_finetuning_heatmap_wholemodel.py \\
  --config cfgs/finetune/fall_nonfall_binary_10fps.yaml \\
  --device cuda \\
  --data_path ${DATA}/train \\
  --eval_data_path ${DATA}/val \\
  --nb_classes 2 \\
  --batch_size ${BATCH} \\
  --num_workers 4 \\
  --epochs ${EPOCHS} \\
  --save_ckpt_freq 1 \\
  --output_dir ${OUT} \\
  --class_balance \\
  --class_weight \\
  ${RESUME_FLAGS[*]} \\
  2>&1 | tee -a ${OUT}/train.log
echo "Finished ${EXTRA_TITLE} at \$(date)" | tee -a "${OUT}/train.log"
EOF
chmod +x "${RUNNER}"

tmux new-session -d -s "${SESSION}" "${RUNNER}"
echo "Started tmux '${SESSION}'  phase=${PHASE} batch=${BATCH} gpu=${GPU}"
echo "Status: bash ~/Mesh-main/hpc/run_mmap_10fps_train_cci.sh --status"
echo "Attach: tmux attach -t ${SESSION}"
sleep 3
tail -n 40 "${OUT}/train.log" 2>/dev/null || true
