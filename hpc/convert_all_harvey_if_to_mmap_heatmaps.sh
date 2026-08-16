#!/usr/bin/env bash
# Convert Harvey's complete radar IF dataset into mmAP angle/doppler/range
# arrays.  This creates only .npy data files, not PNG previews, to keep the
# batch compact.  It is resumable: completed clips are skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${INPUT_ROOT:-/Volumes/data/fall down/radar_data/fall_detection_sim}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/Volumes/data/fall down/radar_data/mmAP_heatmap_gpu_clips}"

echo "Source IF signals: $(find "$INPUT_ROOT" -type f -name if_signal.npy ! -name '._*' | wc -l | tr -d ' ')"
echo "Output folder: $OUTPUT_ROOT"
PREVIEWS=0 bash "$ROOT/hpc/convert_if_folder_to_heatmaps.sh" "$INPUT_ROOT" "$OUTPUT_ROOT"
