#!/usr/bin/env python3
"""Trim interior scene cuts only — keep user 3–4s window, max 4s, no expansion."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

MAX_CLIP_SEC = 4.0
MIN_CLIP_SEC = 1.5
INTERIOR_MARGIN = 0.08
LEADING_SCENE_MAX = 1.0  # drop leading scene if <1s before cut to fall


def parse_time(value: str) -> float:
    value = str(value).strip()
    if re.match(r"^\d+(\.\d+)?$", value):
        return float(value)
    parts = value.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Bad time: {value}")


def fmt_time(sec: float) -> str:
    m, s = divmod(sec, 60)
    if m >= 60:
        h, m = divmod(int(m), 60)
        if abs(s - round(s)) < 0.05:
            return f"{h}:{int(m)}:{int(round(s)):02d}"
        return f"{h}:{int(m)}:{s:05.2f}".rstrip("0").rstrip(".")
    if abs(s - round(s)) < 0.05:
        return f"{int(m)}:{int(round(s)):02d}"
    return f"{int(m)}:{s:04.1f}".rstrip("0").rstrip(".")


def overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def scenes_in_window(scenes: list[dict], start: float, end: float) -> list[dict]:
    hits = [
        s for s in scenes
        if s["segment_start_sec"] < end - INTERIOR_MARGIN
        and s["segment_end_sec"] > start + INTERIOR_MARGIN
    ]
    return sorted(hits, key=lambda s: s["segment_start_sec"])


def interior_cuts(scenes: list[dict], start: float, end: float) -> list[float]:
    cuts = []
    for s in scenes:
        t = s["segment_start_sec"]
        if start + INTERIOR_MARGIN < t < end - INTERIOR_MARGIN:
            cuts.append(t)
    return sorted(set(cuts))


def trim_interior_only(user_start: float, user_end: float, scenes: list[dict]) -> tuple[float, float, str | None]:
    """Trim only when a scene boundary falls inside the user's window."""
    start, end = user_start, user_end
    note_parts = []

    # Cut to fall scene right after user_start (e.g. wrong first frame then switch)
    for s in scenes:
        t = s["segment_start_sec"]
        if user_start < t < user_end - INTERIOR_MARGIN:
            if t - user_start < LEADING_SCENE_MAX:
                start = max(start, t)
                note_parts.append(f"start at {s['clip']}")

    overlapping = scenes_in_window(scenes, start, end)
    cuts = interior_cuts(scenes, start, end)

    if len(overlapping) >= 2:
        first = overlapping[0]
        lead = overlap_len(start, end, first["segment_start_sec"], first["segment_end_sec"])
        if lead < LEADING_SCENE_MAX and first["segment_start_sec"] <= start + INTERIOR_MARGIN:
            fall = overlapping[1]
            if fall["segment_start_sec"] > start + INTERIOR_MARGIN:
                start = max(start, fall["segment_start_sec"])
                note_parts.append(f"drop leading scene ({first['clip']})")
        elif not note_parts:
            fall = max(
                overlapping,
                key=lambda s: overlap_len(start, end, s["segment_start_sec"], s["segment_end_sec"]),
            )
            if fall["segment_start_sec"] > start + INTERIOR_MARGIN:
                start = fall["segment_start_sec"]
                note_parts.append(f"start at {fall['clip']}")

        for s in overlapping[1:]:
            if s["segment_start_sec"] > start + MIN_CLIP_SEC and s["segment_start_sec"] < end - INTERIOR_MARGIN:
                end = min(end, s["segment_start_sec"])
                note_parts.append(f"end before {s['clip']}")
                break

        fall_scene = None
        for s in overlapping:
            if s["segment_start_sec"] <= start + INTERIOR_MARGIN and s["segment_end_sec"] > start:
                fall_scene = s
        if fall_scene:
            end = min(end, fall_scene["segment_end_sec"])

    elif cuts and not note_parts:
        start = max(start, cuts[0])
        note_parts.append("start after interior cut")
        for t in cuts[1:]:
            if t > start + MIN_CLIP_SEC:
                end = min(end, t)
                note_parts.append("end before interior cut")
                break

    # Never expand beyond user window
    start = max(user_start, start)
    end = min(user_end, end)
    end = min(end, start + MAX_CLIP_SEC)

    if end - start < MIN_CLIP_SEC:
        return user_start, user_end, None

    if not note_parts:
        return user_start, user_end, None

    changed = abs(start - user_start) > 0.04 or abs(end - user_end) > 0.04
    if not changed:
        return user_start, user_end, None

    return round(start, 3), round(end, 3), "; ".join(dict.fromkeys(note_parts))


def process_folder(folder: Path, source_manual: Path, dry_run: bool = False) -> list[dict]:
    scene_path = folder / "scene_cut_manifest.json"
    if not source_manual.exists() or not scene_path.exists():
        print(f"SKIP {folder.name}")
        return []

    scenes = json.loads(scene_path.read_text())
    manual = json.loads(source_manual.read_text())
    changes = []
    updated = []

    for entry in manual:
        us = parse_time(entry["start"])
        ue = parse_time(entry["end"])
        ns, ne, trim_note = trim_interior_only(us, ue, scenes)
        note = re.sub(r";?\s*scene-trimmed.*", "", entry.get("note", "")).strip("; ")
        if trim_note:
            note = (note + f"; interior scene trim ({trim_note})").strip("; ")

        updated.append({
            **{k: v for k, v in entry.items() if k not in ("start_sec", "end_sec", "duration_sec")},
            "start": fmt_time(ns),
            "end": fmt_time(ne),
            "note": note,
            "start_sec": ns,
            "end_sec": ne,
        })
        if abs(ns - us) > 0.04 or abs(ne - ue) > 0.04:
            changes.append({
                "clip": entry["clip"],
                "before": f"{entry['start']} -> {entry['end']} ({ue-us:.1f}s)",
                "after": f"{fmt_time(ns)} -> {fmt_time(ne)} ({ne-ns:.1f}s)",
                "reason": trim_note,
            })

    if not dry_run:
        manual_out = folder / "manual_timings.json"
        manual_out.write_text(json.dumps(
            [{k: v for k, v in e.items() if k not in ("start_sec", "end_sec")} for e in updated],
            indent=2,
        ) + "\n")
        manifest = [{
            **{k: v for k, v in e.items()},
            "duration_sec": round(e["end_sec"] - e["start_sec"], 3),
            "mode": "user_manual",
        } for e in updated]
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (folder / "scene_trim_report.json").write_text(json.dumps(changes, indent=2) + "\n")

    print(f"{folder.name}: {len(changes)}/{len(manual)} clips trimmed (unchanged: {len(manual)-len(changes)})")
    return changes


def recut_folder(folder: Path) -> None:
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "cut_clips_from_manual.py"), str(folder)],
        check=True,
    )


def main():
    folders = sys.argv[1:] if len(sys.argv) > 1 else ["030_V31", "031_V32", "032_V33", "033_V34"]
    base = Path(__file__).parent / "data" / "fall_dataset_clips"
    dry_run = "--dry-run" in folders
    if dry_run:
        folders = [f for f in folders if f != "--dry-run"]

    total = 0
    for name in folders:
        folder = base / name
        src = folder / "manual_timings_pre_scene_trim.json"
        if not src.exists():
            src = folder / "manual_timings.json"
        changes = process_folder(folder, src, dry_run=dry_run)
        total += len(changes)
        for c in changes[:8]:
            print(f"  {c['clip']}: {c['before']} => {c['after']}")
            if c.get("reason"):
                print(f"    ({c['reason']})")

    if dry_run:
        print(f"\nDry run: would trim {total} clips")
        return

    print("\nRe-cutting...")
    for name in folders:
        recut_folder(base / name)
    print(f"Done: trimmed {total} clips (max {MAX_CLIP_SEC}s, no expansion)")


if __name__ == "__main__":
    main()
