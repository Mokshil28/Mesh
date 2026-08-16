#!/usr/bin/env bash
# Batch-64 / 64-epoch 10fps train on cci-csgpu3 (separate from csgpu1 batch-32).
# Usage on csgpu3:
#   CUDA_VISIBLE_DEVICES=3 bash ~/Mesh-main/hpc/run_mmap_10fps_bs64_csgpu3.sh
set -euo pipefail

ROOT="${HOME}/Mesh-main"
DATA="${ROOT}/data/fall_nonfall_binary_10fps/dataset"
OUT="${ROOT}/mmAP-slim/output/finetune/fall_nonfall_binary_10fps_bs64"
GPU="${CUDA_VISIBLE_DEVICES:-3}"
SESSION="${SESSION:-mmap_10fps_bs64}"
EPOCHS="${EPOCHS:-64}"
BATCH="${BATCH:-64}"

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || { echo "No conda found"; exit 1; }

if ! conda env list | grep -qE '^mmap-gpu\s'; then
  echo "Missing conda env mmap-gpu. Run: bash ~/Mesh-main/hpc/setup_mmap_cci.sh"
  exit 1
fi
conda activate mmap-gpu

[[ -d "${DATA}/train" ]] || { echo "Missing dataset ${DATA}/train"; exit 1; }
mkdir -p "${OUT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux '${SESSION}' already exists — attach with: tmux attach -t ${SESSION}"
  exit 0
fi

nvidia-smi -i "${GPU}" --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv || true

RUNNER="${OUT}/_tmux_train.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "\$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \\
  || source "\$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate mmap-gpu
cd "\$HOME/Mesh-main/mmAP-slim"
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "Starting 10fps train batch${BATCH} epochs${EPOCHS} on GPU ${GPU} at \$(date)" | tee -a "${OUT}/train.log"
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
  --no_auto_resume \\
  2>&1 | tee -a ${OUT}/train.log
echo "Finished batch${BATCH} at \$(date)" | tee -a "${OUT}/train.log"
EOF
chmod +x "${RUNNER}"

tmux new-session -d -s "${SESSION}" "${RUNNER}"
echo "Started tmux '${SESSION}'  batch=${BATCH} epochs=${EPOCHS} gpu=${GPU}"
echo "  Attach: tmux attach -t ${SESSION}"
echo "  Log:    tail -f ${OUT}/train.log"
sleep 5
tail -n 40 "${OUT}/train.log" 2>/dev/null || true
