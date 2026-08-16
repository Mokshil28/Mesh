#!/usr/bin/env python3
"""Render one or more 2-D NumPy heatmap arrays as JPEG images."""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib import colormaps
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing .npy heatmaps")
    parser.add_argument("--out", type=Path, help="Output directory (defaults to input_dir)")
    args = parser.parse_args()

    output_dir = args.out or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.input_dir.glob("*.npy"))
    if not sources:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")
    for source in sources:
        values = np.load(source, allow_pickle=False)
        if values.ndim != 2:
            raise ValueError(f"Expected a 2-D array in {source}, got {values.shape}")
        low, high = np.percentile(values, [1, 99])
        normalized = np.clip((values - low) / (high - low), 0, 1)
        rgb = (colormaps["magma"](normalized)[..., :3] * 255).astype(np.uint8)
        image = Image.fromarray(rgb).resize(
            (values.shape[1] * 2, values.shape[0] * 2)
        )
        destination = output_dir / f"{source.stem}.jpg"
        image.save(destination, quality=95)
        print(destination)


if __name__ == "__main__":
    main()
