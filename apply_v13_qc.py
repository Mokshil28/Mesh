#!/usr/bin/env python3
"""Apply V13 QC: deletes, 3s cap, +1s extend, scene+3s window for clip 45."""
import json
import subprocess
import sys
from pathlib import Path

FPS = 23.976

DELETE = {
    1, 5, 6, 7, 15, 16, 17, 18, 19, 20, 22, 34, 38, 39, 41, 42, 43, 47, 48,
    52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    73, 74, 75, 77, 79, 80, 81, 82, 83, 84, 86, 89, 90, 94, 95, 96, 99, 100,
    103, 104, 106, 107, 108, 110, 111, 112, 114, 115, 117,
}

CAP_FROM_CLIP_START = {3.0: {11}}

EXTEND_END_SEC = {35, 40, 49, 51, 88, 92, 97, 98, 102}

# Start offset from scene_start_frame; end uses scene_end - margin
SCENE_START_OFFSET = {45: 3.0}

SCENE_END_MARGIN = 5


def sec_to_frames(sec: float) -> int:
    return int(round(sec * FPS))


def cut_clip(source: Path, out: Path, start_f: int, end_f: int) -> None:
    start_sec = start_f / FPS
    duration = max((end_f - start_f) / FPS, 0.1)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_sec:.6f}", "-i", str(source),
        "-t", f"{duration:.6f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def clip_num(name: str) -> int:
    return int(name.replace("fall_", "").replace(".mp4", ""))


def apply_edit(num: int, e: dict) -> tuple[int, int, str]:
    ss = e["scene_start_frame"]
    se = e["scene_end_frame"]
    start = e["start_frame"]
    end = e["end_frame"]

    if num in SCENE_START_OFFSET:
        offset = SCENE_START_OFFSET[num]
        start = ss + sec_to_frames(offset)
        end = se - SCENE_END_MARGIN
        return start, end, f"qc_v13 scene +{offset}s through fall+rest"

    for dur, nums in CAP_FROM_CLIP_START.items():
        if num in nums:
            end = start + sec_to_frames(dur)
            end = min(end, se - SCENE_END_MARGIN)
            return start, end, f"qc_v13 cap {dur}s from clip start"

    if num in EXTEND_END_SEC:
        new_end = min(end + sec_to_frames(1.0), se - SCENE_END_MARGIN)
        if new_end > end:
            return start, new_end, "qc_v13 extend +1s at end"
        return start, end, "qc_v13 extend +1s (no room in scene)"

    return start, end, e.get("note", "")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/fall_dataset_clips/012_V13"
    )
    folder = folder.resolve()
    source = folder / "source.mp4"
    manifest = json.loads((folder / "manifest.json").read_text())

    by_num = {clip_num(e["clip"]): e for e in manifest}
    rejected = []
    kept = []

    for num in sorted(by_num):
        e = by_num[num]
        if num in DELETE:
            rejected.append({
                "clip": e["clip"],
                "source_clip": e.get("source_clip", e["clip"]),
                "reason": "user_qc_delete_v13",
            })
            continue

        start, end, note = apply_edit(num, e)
        kept.append({
            **e,
            "start_frame": start,
            "end_frame": end,
            "duration_sec": round((end - start) / FPS, 2),
            "note": note,
            "mode": "fall_trim_manual_qc",
            "_old_num": num,
        })

    manual = []
    new_manifest = []
    for i, e in enumerate(kept, 1):
        name = f"fall_{i:03d}.mp4"
        old = e.pop("_old_num")
        manual.append({
            "clip": name,
            "start_frame": e["start_frame"],
            "end_frame": e["end_frame"],
            "note": e["note"],
        })
        entry = {k: v for k, v in e.items() if k != "clip"}
        entry["clip"] = name
        entry["source_clip_v13"] = f"fall_{old:03d}.mp4"
        new_manifest.append(entry)

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(new_manifest, indent=2) + "\n")

    rej_path = folder / "rejected_clips.json"
    prev = json.loads(rej_path.read_text()) if rej_path.exists() else []
    rej_path.write_text(json.dumps(prev + rejected, indent=2) + "\n")

    print(f"Deleted {len(DELETE)} | Kept {len(manual)}")

    ok = 0
    for entry in manual:
        out = folder / entry["clip"]
        try:
            cut_clip(source, out, entry["start_frame"], entry["end_frame"])
            ok += 1
            dur = (entry["end_frame"] - entry["start_frame"]) / FPS
            print(f"  OK {entry['clip']}  {dur:.2f}s  {entry['note']}")
        except subprocess.CalledProcessError as exc:
            print(f"  FAIL {entry['clip']}: {exc}")

    print(f"Done: {ok}/{len(manual)}")


if __name__ == "__main__":
    main()
