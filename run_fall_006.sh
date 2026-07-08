#!/usr/bin/env bash
# V01 fall_006 — mesh only (falling person replaced by 3D mesh, no video).
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/Users/mshah76/miniforge3/envs/4D-humans/bin/python}"

"$PYTHON" export_fall_mesh_only.py \
  --video data/fall_dataset_clips/001_V01/fall_006.mp4 \
  --out_dir fall_out/fall_006_mesh \
  --out_html fall_out/V001_fall_006_viewer.html \
  --title "V001 fall_006 — Mesh Only Fall" \
  "$@"

echo ""
echo "Open viewer:"
echo "  open fall_out/V001_fall_006_viewer.html"
