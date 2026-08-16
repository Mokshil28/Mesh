#!/usr/bin/env python3
"""Check filename, shape, and finite-value consistency of an mmAP dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    args = ap.parse_args()
    expected = {"angle": 128, "doppler": 256, "range": 100}
    for split in ("train", "eval"):
        names: dict[str, set[str]] = {}
        for modality, height in expected.items():
            folder = args.dataset / split / modality / "fall"
            files = sorted(path for path in folder.glob("*.npy") if not path.name.startswith("._"))
            if not files:
                raise RuntimeError(f"No files in {folder}")
            names[modality] = {path.name for path in files}
            for path in files:
                x = np.load(path, allow_pickle=False)
                if x.shape != (height, 128) or not np.isfinite(x).all():
                    raise RuntimeError(f"Invalid {path}: shape={x.shape}, finite={np.isfinite(x).all()}")
        if len({frozenset(v) for v in names.values()}) != 1:
            raise RuntimeError(f"Mismatched modality filenames in {split}")
        print(f"{split}: {len(names['angle'])} matching fall samples; shapes valid")


if __name__ == "__main__":
    main()
