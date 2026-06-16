#!/usr/bin/env bash
# Zip key deliverables for your professor (no SMPL / raw dataset).
set -eo pipefail
cd "$(dirname "$0")"
OUT="professor_deliverables.zip"
rm -f "$OUT"

zip -r "$OUT" \
  my_images/dancing.png \
  my_results/dancing_all.png \
  fall_out/FALL_*/overlay.mp4 \
  -x '*/.*' '*/FALL_*_mesh/*' 2>/dev/null || true

# Fallback if dancing_all.png is elsewhere
if ! unzip -l "$OUT" 2>/dev/null | grep -q dancing_all; then
  zip -r "$OUT" my_results/dancing_*.png fall_out/*/overlay.mp4 -x '*/.*' 2>/dev/null || true
fi

echo "Created: $(pwd)/$OUT"
echo "Send this file to your professor (email, Drive, etc.)"
ls -lh "$OUT"
