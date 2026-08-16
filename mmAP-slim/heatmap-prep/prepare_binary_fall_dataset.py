#!/usr/bin/env python3
"""Prepare validated, group-safe mmAP splits for fall vs non-fall classification.

The input roots contain one directory per clip, with matching ``angle.npy``,
``doppler.npy``, and ``range.npy`` arrays.  The script writes CSV manifests
immediately.  With ``--write-windows`` it additionally creates the folder
layout consumed by mmAP's ``MultiTaskImageFolder`` loader.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


MODALITY_HEIGHTS = {"angle": 128, "doppler": 256, "range": 100}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Clip:
    clip_id: str
    group_id: str
    label: str
    angle: str
    doppler: str
    range: str
    width: int


def split_for_group(group_id: str, seed: int) -> str:
    """Stable group split: train 70%, validation 15%, test 15%."""
    value = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 70 else "val" if value < 85 else "test"


def verify_clip(clip_dir: Path, *, strict: bool = False) -> tuple[dict[str, Path], int]:
    files = {name: clip_dir / f"{name}.npy" for name in MODALITY_HEIGHTS}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing modality files: {', '.join(missing)}")
    arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in files.items()}
    widths: set[int] = set()
    for name, array in arrays.items():
        height = MODALITY_HEIGHTS[name]
        if array.ndim != 2 or array.shape[0] < height:
            raise ValueError(f"{files[name]} has shape {array.shape}; expected at least ({height}, time)")
        if strict:
            # Full USB scans are too slow; sample a few columns across the clip.
            cols = np.linspace(0, array.shape[1] - 1, num=min(32, array.shape[1]), dtype=int)
            sample = np.asarray(array[:, cols])
            if not np.isfinite(sample).all():
                raise ValueError(f"{files[name]} contains non-finite values")
        widths.add(array.shape[1])
    if len(widths) != 1:
        raise ValueError(f"Mismatched time widths in {clip_dir}: {widths}")
    return files, widths.pop()


def discover(root: Path, label: str, window: int, *, strict: bool = False) -> tuple[list[Clip], list[str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input root is not mounted or does not exist: {root}")
    clips: list[Clip] = []
    errors: list[str] = []
    angles = [p for p in sorted(root.rglob("angle.npy")) if not p.name.startswith("._")]
    print(f"discovering {label}: {len(angles)} angle.npy candidates under {root}", flush=True)
    for index, angle in enumerate(angles, start=1):
        clip_dir = angle.parent
        try:
            files, width = verify_clip(clip_dir, strict=strict)
            if width < window:
                raise ValueError(f"time width {width} is shorter than window {window}")
            # Layout is .../<batch>/<subject-or-session>/<clip>/angle.npy.
            group_id = clip_dir.parent.name
            relative = clip_dir.relative_to(root).as_posix()
            clips.append(Clip(
                clip_id=f"{label}:{relative}", group_id=group_id, label=label,
                angle=str(files["angle"]), doppler=str(files["doppler"]),
                range=str(files["range"]), width=width,
            ))
        except (OSError, ValueError) as exc:
            errors.append(f"{clip_dir}: {exc}")
        if index % 500 == 0 or index == len(angles):
            print(f"  {label}: scanned {index}/{len(angles)} (ok={len(clips)} err={len(errors)})", flush=True)
    return clips, errors


def strict_verify_selected(clips: list[Clip]) -> tuple[list[Clip], list[str]]:
    """Deep finite-value check only for clips that will enter the dataset."""
    kept: list[Clip] = []
    errors: list[str] = []
    for index, clip in enumerate(clips, start=1):
        try:
            verify_clip(Path(clip.angle).parent, strict=True)
            kept.append(clip)
        except (OSError, ValueError) as exc:
            errors.append(f"{clip.clip_id}: {exc}")
        if index % 200 == 0 or index == len(clips):
            print(f"strict verify {index}/{len(clips)} (kept={len(kept)} err={len(errors)})", flush=True)
    return kept, errors


def corpus_key(clip: Clip) -> str:
    """Top-level corpus/session folder used for stratified non-fall sampling."""
    relative = clip.clip_id.split(":", 1)[-1]
    return relative.split("/", 1)[0]


def subsample_clips(clips: list[Clip], max_clips: int, seed: int) -> list[Clip]:
    """Keep up to ``max_clips`` examples with corpus-stratified sampling."""
    if max_clips <= 0 or len(clips) <= max_clips:
        return list(clips)

    by_corpus: dict[str, list[Clip]] = {}
    for clip in clips:
        by_corpus.setdefault(corpus_key(clip), []).append(clip)

    rng = np.random.default_rng(seed)
    corpora = sorted(by_corpus)
    for corpus in corpora:
        rng.shuffle(by_corpus[corpus])

    # Proportional quota, then fill remainders round-robin so small corpora survive.
    total = len(clips)
    quotas = {c: max(1, int(round(max_clips * len(by_corpus[c]) / total))) for c in corpora}
    while sum(quotas.values()) > max_clips:
        donor = max(corpora, key=lambda c: (quotas[c], len(by_corpus[c])))
        if quotas[donor] <= 1:
            break
        quotas[donor] -= 1
    selected: list[Clip] = []
    leftovers: list[Clip] = []
    for corpus in corpora:
        take = min(quotas[corpus], len(by_corpus[corpus]))
        selected.extend(by_corpus[corpus][:take])
        leftovers.extend(by_corpus[corpus][take:])
    if len(selected) < max_clips:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_clips - len(selected)])
    elif len(selected) > max_clips:
        rng.shuffle(selected)
        selected = selected[:max_clips]
    selected.sort(key=lambda clip: clip.clip_id)
    return selected


def write_manifest(path: Path, clips: list[Clip], assignments: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*Clip.__dataclass_fields__, "split"])
        writer.writeheader()
        for clip in clips:
            row = asdict(clip)
            row["split"] = assignments[clip.group_id]
            writer.writerow(row)


def write_windows(clips: list[Clip], assignments: dict[str, str], destination: Path,
                  window: int, stride: int) -> Counter:
    written: Counter = Counter()
    for index, clip in enumerate(clips, start=1):
        split = assignments[clip.group_id]
        arrays = {
            "angle": np.load(clip.angle, mmap_mode="r", allow_pickle=False),
            "doppler": np.load(clip.doppler, mmap_mode="r", allow_pickle=False),
            "range": np.load(clip.range, mmap_mode="r", allow_pickle=False),
        }
        safe_name = clip.clip_id.replace(":", "_").replace("/", "_")
        for start in range(0, clip.width - window + 1, stride):
            for modality, height in MODALITY_HEIGHTS.items():
                out = destination / split / modality / clip.label / f"{safe_name}_w{start:04d}.npy"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, np.asarray(arrays[modality][:height, start:start + window], dtype=np.float32))
            written[f"{split}:{clip.label}"] += 1
        if index % 50 == 0 or index == len(clips):
            print(f"materialized {index}/{len(clips)} clips")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fall-root", required=True, type=Path)
    parser.add_argument("--nonfall-root", type=Path,
                        help="Add this when Harvey's non-fall heatmaps arrive")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--nonfall-max", type=int, default=None,
                        help="Cap non-fall clips (recommended: match fall count for ~1:1 balance)")
    parser.add_argument("--balance-to-fall", action="store_true",
                        help="Set --nonfall-max to the number of valid fall clips")
    parser.add_argument("--write-windows", action="store_true",
                        help="Create mmAP-ready train/val/test modality folders")
    args = parser.parse_args()
    if args.window <= 0 or args.stride <= 0:
        parser.error("--window and --stride must be positive")
    if args.write_windows and not args.nonfall_root:
        parser.error("Refusing to create a training dataset without --nonfall-root")
    if args.nonfall_max is not None and args.nonfall_max <= 0:
        parser.error("--nonfall-max must be positive")

    fall_clips, errors = discover(args.fall_root, "fall", args.window)
    clips: list[Clip] = list(fall_clips)
    nonfall_before = 0
    nonfall_after = 0
    if args.nonfall_root:
        nonfall_clips, nonfall_errors = discover(args.nonfall_root, "non_fall", args.window)
        errors.extend(nonfall_errors)
        nonfall_before = len(nonfall_clips)
        max_nonfall = len(fall_clips) if args.balance_to_fall else args.nonfall_max
        if max_nonfall is not None:
            nonfall_clips = subsample_clips(nonfall_clips, max_nonfall, args.seed)
        nonfall_after = len(nonfall_clips)
        clips.extend(nonfall_clips)
    if not clips:
        raise RuntimeError("No valid clips found")

    strict_kept, strict_errors = strict_verify_selected(clips)
    if strict_errors:
        clips = strict_kept
        errors.extend(strict_errors)
        print(f"removed {len(strict_errors)} clips failing strict verification", flush=True)
    if not clips:
        raise RuntimeError("No valid clips remain after strict verification")
    fall_clips = [c for c in clips if c.label == "fall"]
    nonfall_after = sum(1 for c in clips if c.label == "non_fall")
    assignments = {group: split_for_group(group, args.seed) for group in {clip.group_id for clip in clips}}
    manifests = args.out / "manifests"
    write_manifest(manifests / "all.csv", clips, assignments)
    for split in SPLITS:
        write_manifest(manifests / f"{split}.csv", [c for c in clips if assignments[c.group_id] == split], assignments)
    summary = {
        "window": args.window,
        "stride": args.stride,
        "seed": args.seed,
        "balance_to_fall": bool(args.balance_to_fall),
        "nonfall_max": args.nonfall_max if not args.balance_to_fall else len(fall_clips),
        "nonfall_clips_before_sample": nonfall_before,
        "nonfall_clips_after_sample": nonfall_after,
        "fall_clips_valid": len(fall_clips),
        "classes": sorted({clip.label for clip in clips}),
        "clips_by_split_and_label": {
            split: dict(Counter(c.label for c in clips if assignments[c.group_id] == split)) for split in SPLITS
        },
        "groups_by_split": {split: sorted(g for g, value in assignments.items() if value == split) for split in SPLITS},
        "invalid_clip_count": len(errors),
        "invalid_clips": errors[:200],
        "invalid_clips_truncated": len(errors) > 200,
    }
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["clips_by_split_and_label"], indent=2))
    print(f"valid clips: {len(clips)}; invalid clips: {len(errors)}")
    if args.nonfall_root:
        print(f"non-fall kept: {nonfall_after}/{nonfall_before}; fall: {len(fall_clips)}")
    if not args.nonfall_root:
        print("Non-fall root not supplied: manifests are provisional and training is intentionally blocked.")
    if args.write_windows:
        dataset_root = args.out / "dataset"
        if dataset_root.exists():
            raise RuntimeError(f"Refusing to overwrite existing dataset directory: {dataset_root}")
        written = write_windows(clips, assignments, dataset_root, args.window, args.stride)
        print(json.dumps(dict(written), indent=2))
        print(f"mmAP dataset: {dataset_root}")


if __name__ == "__main__":
    main()
