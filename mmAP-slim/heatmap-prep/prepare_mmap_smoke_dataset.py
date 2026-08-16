#!/usr/bin/env python3
"""Create a small mmAP-format dataset from complete three-modality heatmaps.

This is a pipeline smoke test, not a valid binary classifier dataset: all
examples are labelled ``fall``.  Clips, rather than individual windows, are
split into train/eval to avoid leakage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


MODALITIES = {"angle": 128, "doppler": 256, "range": 100}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=128)
    args = ap.parse_args()

    clips = sorted(path for path in args.input.iterdir()
                   if path.is_dir() and all((path / f"{m}.npy").is_file() for m in MODALITIES))
    if len(clips) < 2:
        raise RuntimeError("Need at least two complete clips")
    split = max(1, round(len(clips) * 2 / 3))
    groups = {"train": clips[:split], "eval": clips[split:]}
    written = {name: 0 for name in groups}

    for group, group_clips in groups.items():
        for clip in group_clips:
            values = {m: np.load(clip / f"{m}.npy", allow_pickle=False) for m in MODALITIES}
            widths = {array.shape[1] for array in values.values()}
            if len(widths) != 1:
                raise ValueError(f"Mismatched time widths for {clip.name}: {widths}")
            for modality, expected_height in MODALITIES.items():
                if values[modality].shape[0] < expected_height:
                    raise ValueError(f"{clip.name}/{modality} is too short: {values[modality].shape}")
                values[modality] = values[modality][:expected_height]
            width = widths.pop()
            for start in range(0, width - args.window + 1, args.stride):
                name = f"{clip.name}_w{start:04d}.npy"
                for modality, array in values.items():
                    dst = args.out / group / modality / "fall" / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    np.save(dst, array[:, start:start + args.window].astype(np.float32))
                written[group] += 1

    print(f"train fall samples: {written['train']}")
    print(f"eval fall samples: {written['eval']}")
    print(f"dataset: {args.out}")


if __name__ == "__main__":
    main()
