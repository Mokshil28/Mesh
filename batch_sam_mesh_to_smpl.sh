#!/usr/bin/env bash
# Incrementally convert completed SAM mesh tracks into SMPL parameters and
# centred 360° inspection viewers. By default, each output is written directly
# into its own clip directory, beside mesh_4d_individual (for example,
# 001_V01/fall_006/smpl_params.npz). Safe to re-run while rsync is still adding
# clips: it skips existing outputs and ignores files modified very recently.
set -euo pipefail

INPUT_ROOT="${1:?Usage: bash batch_sam_mesh_to_smpl.sh <sam_gpu_batch_raw> [separate_output_root]}"
# Keep each clip's fitted SMPL parameters and 360° viewer with that clip unless
# an explicit separate output root is requested for a special export.
OUTPUT_ROOT="${2:-}"
USER_HOME_DIR="${HOME:-/Users/mshah76}"
PYTHON_BIN="${PYTHON_BIN:-$USER_HOME_DIR/miniforge3/envs/4D-humans/bin/python}"
SETTLE_SECONDS="${SETTLE_SECONDS:-180}"
ITERATIONS="${ITERATIONS:-80}"
POINTS="${POINTS:-384}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ -z "$OUTPUT_ROOT" ]] || mkdir -p "$OUTPUT_ROOT"

# Do not sort here: macOS `sort` spills its temporary files to the internal
# disk.  Streaming is safer when the final output lives on an external drive.
find "$INPUT_ROOT" -type d -name mesh_4d_individual -print0 | while IFS= read -r -d '' mesh_root; do
  clip_dir="$(dirname "$mesh_root")"
  rel="${clip_dir#"$INPUT_ROOT"/}"
  # SAM normally assigns the faller as person 1.  Do not silently use another
  # identity; multi-person exceptions should be selected in QC.
  person_dir="$mesh_root/1"
  [[ -d "$person_dir" ]] || { echo "[skip no-person-1] $rel"; continue; }
  # Ignore macOS resource-fork sidecars (._*.ply) created by Finder copies.
  ply_count=$(find "$person_dir" -maxdepth 1 -type f -name '*.ply' ! -name '._*' | wc -l | tr -d ' ')
  [[ "$ply_count" -ge 2 ]] || { echo "[skip incomplete] $rel"; continue; }
  # Avoid starting on a clip whose rsync transfer is still changing.
  if find "$person_dir" -maxdepth 1 -type f -name '*.ply' ! -name '._*' -mmin "-$(( (SETTLE_SECONDS + 59) / 60 ))" -print -quit | grep -q .; then
    echo "[wait copying] $rel"
    continue
  fi
  if [[ -n "$OUTPUT_ROOT" ]]; then
    out="$OUTPUT_ROOT/$rel"
  else
    out="$clip_dir"
  fi
  [[ -f "$out/smpl_params.npz" && -f "$out/sam_mesh_360_centered.html" ]] && { echo "[skip done] $rel"; continue; }
  echo "[fit] $rel ($ply_count frames)"
  OMP_NUM_THREADS=1 "$PYTHON_BIN" "$ROOT/sam_mesh_to_smpl.py" "$person_dir" \
    --out "$out" --points "$POINTS" --iterations "$ITERATIONS" --center-at-start
  # The main converter writes sam_mesh_360.html. Name it explicitly as the
  # centred viewer expected by this batch without duplicating the 13 MB file.
  mv "$out/sam_mesh_360.html" "$out/sam_mesh_360_centered.html"
done
