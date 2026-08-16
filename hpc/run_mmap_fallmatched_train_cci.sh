#!/usr/bin/env bash
# Calibrate non-fall windows to fall dB range, then retrain with class balance.
set -euo pipefail

ROOT="${HOME}/Mesh-main"
SRC_DATA="${ROOT}/data/fall_nonfall_binary_balanced"
OUT_DATA="${ROOT}/data/fall_nonfall_binary_fallmatched"
OUT="${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_fallmatched"
SESSION="${SESSION:-mmap_balanced}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
RUNNER="${OUT}/_tmux_train.sh"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mmap-gpu

case "${1:-}" in
  --status)
    tmux ls 2>/dev/null | grep "${SESSION}" || echo "(no session)"
    tail -n 40 "${OUT}/train.log" 2>/dev/null || echo "no log yet"
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

mkdir -p "${OUT}"
cd "${ROOT}/mmAP-slim"

if [[ ! -f "${OUT_DATA}/fall_norm_refs.json" ]]; then
  echo "=== Calibrating non-fall windows to fall dB distribution ==="
  python -u heatmap-prep/calibrate_nonfall_to_fall.py \
    --fall-root "${SRC_DATA}/dataset" \
    --binary-dataset "${SRC_DATA}" \
    --out "${OUT_DATA}"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session '${SESSION}' already exists"
  exit 0
fi

cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "\$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mmap-gpu
cd "\$HOME/Mesh-main/mmAP-slim"
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "Starting fallmatched train on GPU ${GPU} at \$(date)" | tee -a "${OUT}/train.log"
python -u run_finetuning_heatmap_wholemodel.py \\
  --config cfgs/finetune/fall_nonfall_binary.yaml \\
  --device cuda \\
  --data_path ${OUT_DATA}/dataset/train \\
  --eval_data_path ${OUT_DATA}/dataset/val \\
  --nb_classes 2 \\
  --batch_size 32 \\
  --num_workers 4 \\
  --epochs 30 \\
  --save_ckpt_freq 1 \\
  --output_dir ${OUT} \\
  --norm_refs ${OUT_DATA}/fall_norm_refs.json \\
  --class_balance \\
  --class_weight \\
  --no_auto_resume \\
  2>&1 | tee -a ${OUT}/train.log
echo "Finished at \$(date)" | tee -a "${OUT}/train.log"
EOF
chmod +x "${RUNNER}"

tmux new-session -d -s "${SESSION}" "${RUNNER}"
echo "Started tmux session '${SESSION}' on GPU ${GPU}"
echo "Status: bash ~/Mesh-main/hpc/run_mmap_fallmatched_train_cci.sh --status"
sleep 4
tail -n 40 "${OUT}/train.log" 2>/dev/null || true
