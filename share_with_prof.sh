#!/usr/bin/env bash
# Zip fall mesh overlay videos for your professor (no SMPL / raw dataset).
set -eo pipefail
cd "$(dirname "$0")"
OUT="professor_deliverables.zip"
rm -f "$OUT"

shopt -s nullglob
videos=(fall_out/FALL_*/overlay.mp4)
if [ ${#videos[@]} -eq 0 ]; then
  echo "No fall videos found in fall_out/FALL_*/overlay.mp4"
  exit 1
fi

zip -r "$OUT" "${videos[@]}"

echo "Created: $(pwd)/$OUT (${#videos[@]} videos)"
echo "Send this file to your professor (email, Drive, etc.)"
ls -lh "$OUT"
