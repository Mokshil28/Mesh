#!/usr/bin/env bash
# Re-run fall_002: tight per-frame mesh tracking + interactive viewer.
#   ./run_fall_002.sh              full export (~12 min CPU)
#   ./run_fall_002.sh --build-only rebuild viewer HTML only
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/Users/mshah76/miniforge3/envs/4D-humans/bin/python}"
VIDEO="data/fall_dataset_clips/001_V01/fall_002.mp4"
OUT_DIR="fall_out/fall_002_mesh"
OUT_HTML="fall_out/V001_fall_002_viewer.html"

"$PYTHON" export_interactive_fall.py \
  --video "$VIDEO" \
  --out_dir "$OUT_DIR" \
  --out_html "$OUT_HTML" \
  --title "V001 fall_002 — Falling Person Mesh" \
  "$@"

echo ""
echo "Open viewer:"
echo "  open $OUT_HTML"
