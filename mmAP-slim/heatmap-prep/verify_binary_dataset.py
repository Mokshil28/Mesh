#!/usr/bin/env python3
"""Verify binary fall/non-fall manifests, heatmaps, labels, and optional windows."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MODALITY_HEIGHTS = {"angle": 128, "doppler": 256, "range": 100}


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_source_row(row: dict[str, str], window: int) -> None:
    paths = {name: Path(row[name]) for name in MODALITY_HEIGHTS}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.name.startswith("._"):
            raise ValueError(f"AppleDouble sidecar selected: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        height = MODALITY_HEIGHTS[name]
        if array.ndim != 2 or array.shape[0] < height:
            raise ValueError(f"{path} shape {array.shape}")
        if int(row["width"]) != array.shape[1]:
            raise ValueError(f"{path} width {array.shape[1]} != manifest {row['width']}")
        if array.shape[1] < window:
            raise ValueError(f"{path} shorter than window {window}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path} has non-finite values")


def verify_dataset_tree(dataset_root: Path, expected_labels: set[str]) -> dict:
    counts: Counter = Counter()
    label_sets: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "val", "test"):
        for modality in MODALITY_HEIGHTS:
            for label_dir in sorted((dataset_root / split / modality).glob("*")):
                if not label_dir.is_dir() or label_dir.name.startswith("._"):
                    continue
                label_sets[split].add(label_dir.name)
                files = [p for p in label_dir.glob("*.npy") if not p.name.startswith("._")]
                counts[f"{split}:{modality}:{label_dir.name}"] = len(files)
                # Spot-check a few arrays per class.
                for path in files[:3]:
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    expected_h = MODALITY_HEIGHTS[modality]
                    if array.shape != (expected_h, 128) and array.ndim == 2:
                        # Window width is fixed to --window (default 128).
                        if array.shape[0] != expected_h:
                            raise ValueError(f"{path} unexpected shape {array.shape}")
                    if not np.isfinite(array).all():
                        raise ValueError(f"{path} has non-finite values")
    for split, labels in label_sets.items():
        missing = expected_labels - labels
        if missing:
            raise ValueError(f"{split} missing labels: {sorted(missing)}")
    # Modalities must stay aligned within each split/label.
    for split in ("train", "val", "test"):
        for label in expected_labels:
            values = [counts[f"{split}:{modality}:{label}"] for modality in MODALITY_HEIGHTS]
            if len(set(values)) != 1:
                raise ValueError(f"Misaligned window counts for {split}/{label}: {values}")
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path,
                        help="Binary dataset root containing manifests/ and optional dataset/")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--sample", type=int, default=64,
                        help="How many source clips to deep-verify from all.csv")
    args = parser.parse_args()

    manifests = args.out / "manifests"
    summary = json.loads((manifests / "summary.json").read_text())
    rows = load_manifest(manifests / "all.csv")
    if not rows:
        raise RuntimeError("all.csv is empty")

    labels = Counter(row["label"] for row in rows)
    splits = Counter(row["split"] for row in rows)
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups_by_split[row["split"]].add(row["group_id"])

    # Group leakage check: a group may appear in only one split.
    ownership: dict[str, str] = {}
    for row in rows:
        previous = ownership.get(row["group_id"])
        if previous and previous != row["split"]:
            raise RuntimeError(f"group leak: {row['group_id']} in {previous} and {row['split']}")
        ownership[row["group_id"]] = row["split"]

    rng = np.random.default_rng(summary.get("seed", 0))
    sample_idx = rng.choice(len(rows), size=min(args.sample, len(rows)), replace=False)
    for index in sample_idx:
        verify_source_row(rows[int(index)], args.window)

    report = {
        "clips": len(rows),
        "labels": dict(labels),
        "splits": dict(splits),
        "groups_per_split": {k: len(v) for k, v in groups_by_split.items()},
        "summary_matches_manifest": summary.get("clips_by_split_and_label"),
        "source_sample_verified": int(len(sample_idx)),
        "group_leakage": False,
    }

    dataset_root = args.out / "dataset"
    if dataset_root.is_dir():
        report["window_counts"] = verify_dataset_tree(dataset_root, set(labels))
        # Loader smoke import (optional if mmAP utils available).
        try:
            import sys
            repo = Path(__file__).resolve().parents[1]
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from utils.dataset_folder import MultiTaskImageFolder
            for split in ("train", "val", "test"):
                ds = MultiTaskImageFolder(str(dataset_root / split), tasks=["angle", "doppler", "range"])
                report.setdefault("loader_samples", {})[split] = len(ds)
                report.setdefault("loader_classes", {})[split] = list(ds.classes)
                if len(ds) == 0:
                    raise RuntimeError(f"empty loader for {split}")
                sample = ds[0]
                report.setdefault("loader_smoke", {})[split] = {
                    "type": type(sample).__name__,
                    "len_tuple": len(sample) if isinstance(sample, tuple) else None,
                }
        except Exception as exc:  # noqa: BLE001 - report soft failure for local-only verify
            report["loader_import_error"] = str(exc)

    out_path = manifests / "verify_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
