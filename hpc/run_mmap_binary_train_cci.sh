#!/usr/bin/env bash
# Launch binary fall/non-fall mmAP training on CCI GPU inside tmux.
set -euo pipefail

ROOT="${HOME}/Mesh-main"
OUT="${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_smpl"
DATA="${ROOT}/data/fall_nonfall_binary_balanced/dataset"
SESSION="${SESSION:-mmap_binary}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
RUNNER="${OUT}/_tmux_train.sh"

# shellcheck disable=SC1091
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

case "${1:-}" in
  --status)
    echo "=== tmux ==="
    tmux ls 2>/dev/null | grep "${SESSION}" || echo "(no session)"
    echo "=== log ==="
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

if [[ ! -d "${DATA}/train" ]]; then
  echo "Missing dataset at ${DATA}/train"
  exit 1
fi

nvidia-smi || exit 1
mkdir -p "${OUT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session '${SESSION}' already exists. Use --attach / --status / --kill"
  exit 0
fi

cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "\$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "\$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate mmap-gpu
cd "\$HOME/Mesh-main/mmAP-slim"
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "Starting train on GPU ${GPU} at \$(date)" | tee -a "${OUT}/train.log"
python -u run_finetuning_heatmap_wholemodel.py \\
  --config cfgs/finetune/fall_nonfall_binary.yaml \\
  --device cuda \\
  --data_path ${DATA}/train \\
  --eval_data_path ${DATA}/val \\
  --nb_classes 2 \\
  --batch_size 32 \\
  --num_workers 4 \\
  --epochs 30 \\
  --save_ckpt_freq 1 \\
  --output_dir ${OUT} \\
  2>&1 | tee -a ${OUT}/train.log
echo "Finished at \$(date)" | tee -a "${OUT}/train.log"
EOF
chmod +x "${RUNNER}"

tmux new-session -d -s "${SESSION}" "${RUNNER}"
echo "Started tmux session '${SESSION}' on GPU ${GPU}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Status: bash ~/Mesh-main/hpc/run_mmap_binary_train_cci.sh --status"
echo "Log:    tail -f ${OUT}/train.log"
sleep 3
tail -n 30 "${OUT}/train.log" 2>/dev/null || echo "(log starting...)"
