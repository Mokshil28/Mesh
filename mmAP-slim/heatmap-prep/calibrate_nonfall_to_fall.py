#!/usr/bin/env python3
"""Calibrate non-fall mmAP heatmaps into the fall dB distribution.

Fall heatmaps come from Harvey IF → ``if_signal_to_heatmaps`` (true dB maps with
large dynamic range). Non-fall AMASS spectrograms land in a very different
numeric range (Doppler means ~+23 vs fall ~−32). Per-sample min-max then
destroys absolute scale and stretches flat non-fall maps into noise.

This script:
1. Estimates robust fall reference percentiles from fall windows/clips
2. Maps every non-fall array into that fall dB range (per-array percentile affine)
3. Writes a new dataset tree (does not modify the source)

Works on either:
- source clip trees (angle/doppler/range.npy next to each other), or
- binary window trees (dataset/{split}/{mod}/{fall,non_fall}/*.npy)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

MODALITIES = ("angle", "doppler", "range")


def is_appledouble(path: Path) -> bool:
    return path.name.startswith("._")


def collect_fall_arrays(root: Path, limit: int) -> dict[str, list[np.ndarray]]:
    bags = {m: [] for m in MODALITIES}
    # Prefer binary-window layout if present
    window_hits = list(root.glob("**/fall/*.npy")) + list(root.glob("**/fall/**/*.npy"))
    window_hits = [p for p in window_hits if not is_appledouble(p) and p.parent.name == "fall"]
    if window_hits:
        # group by modality folder name
        by_mod = {m: [] for m in MODALITIES}
        for p in window_hits:
            mod = p.parent.parent.name
            if mod in by_mod:
                by_mod[mod].append(p)
        rng = np.random.default_rng(0)
        for m, paths in by_mod.items():
            if not paths:
                continue
            choose = paths if len(paths) <= limit else list(rng.choice(paths, size=limit, replace=False))
            for p in choose:
                bags[m].append(np.load(p, allow_pickle=False))
        return bags

    # Source clip layout
    angles = [p for p in root.rglob("angle.npy") if not is_appledouble(p)]
    rng = np.random.default_rng(0)
    if len(angles) > limit:
        angles = list(rng.choice(angles, size=limit, replace=False))
    for angle in angles:
        clip = angle.parent
        if "non_fall" in str(clip) or "nonfall" in str(clip).lower():
            continue
        for m in MODALITIES:
            path = clip / f"{m}.npy"
            if path.is_file():
                bags[m].append(np.load(path, allow_pickle=False))
    return bags


def estimate_refs(bags: dict[str, list[np.ndarray]], lo: float, hi: float) -> dict[str, dict[str, float]]:
    refs = {}
    for m, arrays in bags.items():
        if not arrays:
            raise RuntimeError(f"No fall arrays found for modality {m}")
        # subsample pixels for speed
        samples = []
        for a in arrays:
            flat = a.reshape(-1)
            if flat.size > 20000:
                idx = np.linspace(0, flat.size - 1, 20000, dtype=int)
                flat = flat[idx]
            samples.append(flat)
        cat = np.concatenate(samples)
        refs[m] = {
            "p_lo": float(np.percentile(cat, lo)),
            "p_hi": float(np.percentile(cat, hi)),
            "mean": float(cat.mean()),
            "std": float(cat.std() + 1e-6),
        }
        if refs[m]["p_hi"] - refs[m]["p_lo"] < 1e-3:
            raise RuntimeError(f"Degenerate fall percentiles for {m}: {refs[m]}")
    return refs


def calibrate_array(arr: np.ndarray, ref: dict[str, float], lo: float, hi: float) -> np.ndarray:
    src_lo = float(np.percentile(arr, lo))
    src_hi = float(np.percentile(arr, hi))
    if abs(src_hi - src_lo) < 1e-6:
        # nearly constant map: place at fall mean
        return np.full_like(arr, ref["mean"], dtype=np.float32)
    scale = (ref["p_hi"] - ref["p_lo"]) / (src_hi - src_lo)
    out = (arr - src_lo) * scale + ref["p_lo"]
    return out.astype(np.float32)


def calibrate_binary_tree(src: Path, dst: Path, refs: dict, lo: float, hi: float) -> dict:
    """Copy dataset/ and calibrate only non_fall windows."""
    src_ds = src / "dataset" if (src / "dataset").is_dir() else src
    dst_ds = dst / "dataset"
    counts = {"copied_fall": 0, "calibrated_nonfall": 0}
    for split in ("train", "val", "test"):
        for mod in MODALITIES:
            for label in ("fall", "non_fall"):
                sdir = src_ds / split / mod / label
                if not sdir.is_dir():
                    continue
                ddir = dst_ds / split / mod / label
                ddir.mkdir(parents=True, exist_ok=True)
                files = [p for p in sdir.glob("*.npy") if not is_appledouble(p)]
                for i, path in enumerate(files, 1):
                    out = ddir / path.name
                    if label == "fall":
                        if not out.exists():
                            shutil.copy2(path, out)
                        counts["copied_fall"] += 1
                    else:
                        arr = np.load(path, allow_pickle=False)
                        cal = calibrate_array(arr, refs[mod], lo, hi)
                        np.save(out, cal)
                        counts["calibrated_nonfall"] += 1
                    if i % 2000 == 0:
                        print(f"  {split}/{mod}/{label}: {i}/{len(files)}", flush=True)
                print(f"done {split}/{mod}/{label}: {len(files)}", flush=True)
    # copy manifests if present
    man = src / "manifests"
    if man.is_dir():
        shutil.copytree(man, dst / "manifests", dirs_exist_ok=True)
    return counts


def calibrate_clip_tree(src: Path, dst: Path, refs: dict, lo: float, hi: float) -> dict:
    counts = {"clips": 0}
    angles = [p for p in src.rglob("angle.npy") if not is_appledouble(p)]
    for i, angle in enumerate(angles, 1):
        clip = angle.parent
        rel = clip.relative_to(src)
        out_dir = dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for m in MODALITIES:
            arr = np.load(clip / f"{m}.npy", allow_pickle=False)
            cal = calibrate_array(arr, refs[m], lo, hi)
            np.save(out_dir / f"{m}.npy", cal)
        counts["clips"] += 1
        if i % 200 == 0 or i == len(angles):
            print(f"calibrated clips {i}/{len(angles)}", flush=True)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fall-root", required=True, type=Path,
                    help="Fall heatmaps OR binary dataset root used to estimate fall dB refs")
    ap.add_argument("--nonfall-root", type=Path,
                    help="Source non-fall clip tree (if not calibrating a binary dataset)")
    ap.add_argument("--binary-dataset", type=Path,
                    help="Existing fall_nonfall_binary_* root with dataset/{split}/...")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--percentile-lo", type=float, default=5.0)
    ap.add_argument("--percentile-hi", type=float, default=95.0)
    ap.add_argument("--fall-sample-limit", type=int, default=400)
    args = ap.parse_args()

    print("Estimating fall reference percentiles...", flush=True)
    bags = collect_fall_arrays(args.fall_root, args.fall_sample_limit)
    refs = estimate_refs(bags, args.percentile_lo, args.percentile_hi)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fall_norm_refs.json").write_text(json.dumps({
        "percentile_lo": args.percentile_lo,
        "percentile_hi": args.percentile_hi,
        "modalities": refs,
        "note": "Shared normalization for training: x' = 2*(x-p_lo)/(p_hi-p_lo)-1",
    }, indent=2))
    print(json.dumps(refs, indent=2))

    if args.binary_dataset:
        print(f"Calibrating binary dataset {args.binary_dataset} -> {args.out}", flush=True)
        counts = calibrate_binary_tree(
            args.binary_dataset, args.out, refs, args.percentile_lo, args.percentile_hi
        )
    elif args.nonfall_root:
        print(f"Calibrating non-fall clips {args.nonfall_root} -> {args.out}", flush=True)
        # also copy fall tree reference pointer in summary only
        counts = calibrate_clip_tree(
            args.nonfall_root, args.out, refs, args.percentile_lo, args.percentile_hi
        )
    else:
        raise SystemExit("Provide --binary-dataset or --nonfall-root")

    (args.out / "calibration_summary.json").write_text(json.dumps(counts, indent=2))
    print("Done", counts)


if __name__ == "__main__":
    main()
