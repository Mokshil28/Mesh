#!/usr/bin/env bash
# Convert every simulator IF file under one subject folder into mmAP's three
# full-sequence heatmaps.  Existing completed clips are skipped safely.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${1:?Usage: $0 INPUT_ROOT OUTPUT_ROOT}"
OUTPUT_ROOT="${2:?Usage: $0 INPUT_ROOT OUTPUT_ROOT}"
# Prefer the lightweight local environment used by the IF-to-heatmap
# converter.  Set PYTHON_BIN explicitly when running on another machine.
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-heatmap/bin/python}"
CONVERTER="$ROOT/mmAP-slim/heatmap-prep/if_signal_to_heatmaps.py"
PREVIEWS="${PREVIEWS:-0}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"
count=0
while IFS= read -r if_signal; do
  clip_dir="$(dirname "$if_signal")"
  rel="${clip_dir#"$INPUT_ROOT"/}"
  out="$OUTPUT_ROOT/$rel"
  if [[ -f "$out/angle.npy" && -f "$out/doppler.npy" && -f "$out/range.npy" ]]; then
    echo "[skip done] $rel"
    continue
  fi
  echo "[convert] $rel"
  args=(--input "$if_signal" --out "$out")
  [[ "$PREVIEWS" == "1" ]] || args+=(--no-preview)
  MPLCONFIGDIR=/tmp/matplotlib-mmap "$PYTHON_BIN" "$CONVERTER" "${args[@]}"
  count=$((count + 1))
done < <(find "$INPUT_ROOT" -type f -name if_signal.npy ! -name '._*' | sort)
echo "Converted $count clips into $OUTPUT_ROOT"
