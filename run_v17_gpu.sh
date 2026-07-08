#!/usr/bin/env bash
# Run V17 human-v2 fall detection on GPU server.
set -euo pipefail
cd "$(dirname "$0")"

FOLDER="data/fall_dataset_clips/016_V17"
SOURCE="${FOLDER}/source.mp4"
YOUTUBE="https://www.youtube.com/watch?v=9LGqSGk-PXs"

source "${HOME}/miniforge3/etc/profile.d/conda.sh" 2>/dev/null \
  || source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate 4D-humans

mkdir -p "${FOLDER}"

# Download source if missing
if [[ ! -f "${SOURCE}" ]]; then
  echo "==> Downloading V17 source..."
  yt-dlp -f "bv*+ba/b" --merge-output-format mp4 -o "${FOLDER}/source.%(ext)s" "${YOUTUBE}"
fi

# human_v2 reads scene_cut_manifest.json to skip slow ffmpeg scene pass
if [[ ! -f "${FOLDER}/scene_cut_manifest.json" && -f "${FOLDER}/manifest.json" ]]; then
  cp "${FOLDER}/manifest.json" "${FOLDER}/scene_cut_manifest.json"
  echo "==> Copied manifest.json -> scene_cut_manifest.json"
fi

echo "==> Running detect_fall_clips_human_v2.py on GPU..."
python -u detect_fall_clips_human_v2.py "${FOLDER}"

echo ""
echo "Done. Clips in ${FOLDER}/"
ls -1 "${FOLDER}"/fall_*.mp4 2>/dev/null | wc -l | xargs -I{} echo "{} clips generated"
