#!/usr/bin/env bash
# Setup 10fps fall/non-fall mmAP training environment (local prep + CCI launchers).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="/Volumes/data/fall down/radar_data"
FALL10="${DATA}/fall_detection_sim_10fps"
FALL_HM="${DATA}/mmAP_heatmap_fall_10fps"
NONFALL_HM="${DATA}/mmAP_heatmap_nonfall_sim"
OUT_DS="${DATA}/fall_nonfall_binary_10fps"
HOST="${HPC_HOST:-mshah76@cci-csgpu1.charlotte.edu}"

echo "=== 0) Checks ==="
[[ -d "$FALL10" ]] || { echo "Missing $FALL10"; exit 1; }
[[ -d "$NONFALL_HM" ]] || { echo "Missing non-fall heatmaps: $NONFALL_HM"; exit 1; }

# Prefer local mmap env
if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate mmap 2>/dev/null || conda activate mmap-gpu 2>/dev/null || true
fi

echo "=== 1) Materialize 10fps fall spectrograms -> mmAP .npy ==="
python "$ROOT/mmAP-slim/heatmap-prep/materialize_nonfall_spectrograms.py" \
  --input-root "$FALL10" \
  --out "$FALL_HM" \
  --min-time 128

echo "=== 2) Build balanced binary dataset (group-safe splits + windows) ==="
python "$ROOT/mmAP-slim/heatmap-prep/prepare_binary_fall_dataset.py" \
  --fall-root "$FALL_HM" \
  --nonfall-root "$NONFALL_HM" \
  --out "$OUT_DS" \
  --balance-to-fall \
  --write-windows \
  --window 128 \
  --stride 128

echo "=== 3) Verify ==="
python "$ROOT/mmAP-slim/heatmap-prep/verify_binary_dataset.py" --out "$OUT_DS" || true
ls -la "$OUT_DS"
cat "$OUT_DS/manifests/summary.json" 2>/dev/null || true

echo ""
echo "Local prep done."
echo "Next:"
echo "  bash $ROOT/hpc/upload_mmap_10fps_to_cci.sh"
echo "  ssh $HOST"
echo "  bash ~/Mesh-main/hpc/run_mmap_10fps_train_cci.sh   # batch 64 first"
echo "  bash ~/Mesh-main/hpc/run_mmap_10fps_train_cci.sh --phase32  # then batch 32"
