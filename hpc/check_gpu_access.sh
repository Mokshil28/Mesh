#!/usr/bin/env bash
# Run on GPU server — report what compute you have. Paste output back to share with your team/AI.
#   bash hpc/check_gpu_access.sh | tee ~/gpu_report.txt
set -euo pipefail

echo "=============================================="
echo "  GPU ACCESS REPORT — $(date -Iseconds)"
echo "=============================================="
echo ""
echo "=== Host ==="
hostname -f 2>/dev/null || hostname
echo "User: $(whoami)"
echo ""

echo "=== GPU list ==="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L
  echo ""
  echo "=== GPU memory / utilization ==="
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
else
  echo "nvidia-smi NOT FOUND — no NVIDIA GPU on this node"
fi
echo ""

echo "=== Who is logged in ==="
who 2>/dev/null || true
echo ""

echo "=== Disk (home) ==="
df -h ~ 2>/dev/null || df -h .
echo ""

echo "=== RAM ==="
free -h 2>/dev/null || vm_stat 2>/dev/null | head -5
echo ""

echo "=== Conda env ==="
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda env list 2>/dev/null | grep -E '4d-humans|base' || true
fi
echo ""

echo "=== Mesh project ==="
if [[ -d ~/Mesh-main ]]; then
  du -sh ~/Mesh-main 2>/dev/null
  du -sh ~/Mesh-main/fall_out 2>/dev/null || true
  du -sh ~/Mesh-main/data/fall_dataset_clips 2>/dev/null || true
fi
echo ""

echo "=== Active mesh tmux sessions ==="
tmux ls 2>/dev/null | grep -E 'mesh_gpu|mesh_v01|v04' || echo "(none)"
echo ""

echo "=============================================="
echo "  Paste this file when asked about GPU access"
echo "=============================================="
