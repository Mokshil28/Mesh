#!/usr/bin/env python3
"""Materialize simulated non-fall ``spectrogram.npz`` archives as mmAP arrays.

Each archive must contain finite two-dimensional ``angle``, ``doppler``, and
``range`` arrays.  The output retains the source hierarchy and writes the
three ``.npy`` files consumed by the binary fall-data preparer.  Completed
clips are skipped, making interruption and re-runs safe.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np

EXPECTED_HEIGHTS = {"angle": 128, "doppler": 256, "range": 150}


def load_checked(path: Path, minimum_time: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in EXPECTED_HEIGHTS if name not in archive]
        if missing:
            raise ValueError(f"missing arrays: {', '.join(missing)}")
        arrays = {name: np.asarray(archive[name], dtype=np.float32) for name in EXPECTED_HEIGHTS}
    widths = set()
    for name, values in arrays.items():
        height = EXPECTED_HEIGHTS[name]
        if values.ndim != 2 or values.shape[0] != height:
            raise ValueError(f"{name} has shape {values.shape}; expected ({height}, time)")
        if values.shape[1] < minimum_time:
            raise ValueError(f"{name} has only {values.shape[1]} time bins; need {minimum_time}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        widths.add(values.shape[1])
    if len(widths) != 1:
        raise ValueError(f"modalities have mismatched time widths: {sorted(widths)}")
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-time", type=int, default=128)
    parser.add_argument("--max-clips", type=int, help="Validation limit; omit for all clips")
    args = parser.parse_args()
    if args.min_time <= 0:
        parser.error("--min-time must be positive")
    sources = sorted(path for path in args.input_root.rglob("spectrogram.npz") if not path.name.startswith("._"))
    if args.max_clips is not None:
        sources = sources[:args.max_clips]
    if not sources:
        raise RuntimeError("No spectrogram.npz files found")

    converted = skipped = invalid = 0
    for index, source in enumerate(sources, start=1):
        relative = source.parent.relative_to(args.input_root)
        destination = args.out / relative
        outputs = [destination / f"{name}.npy" for name in EXPECTED_HEIGHTS]
        if all(path.is_file() for path in outputs):
            skipped += 1
            continue
        try:
            arrays = load_checked(source, args.min_time)
        # A small number of copied AMASS archives are truncated/corrupt.  They
        # are unusable training examples, so record and skip them rather than
        # abandoning a long resumable conversion.
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            invalid += 1
            print(f"[invalid] {relative}: {exc}")
            continue
        destination.mkdir(parents=True, exist_ok=True)
        for name, values in arrays.items():
            np.save(destination / f"{name}.npy", values)
        converted += 1
        if index % 100 == 0 or index == len(sources):
            print(f"processed {index}/{len(sources)} (converted={converted}, skipped={skipped}, invalid={invalid})", flush=True)
    print(f"Done: converted={converted}, skipped={skipped}, invalid={invalid}, scanned={len(sources)}")


if __name__ == "__main__":
    main()
