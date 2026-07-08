#!/usr/bin/env python3
"""Apply V14 QC round 2."""
import json
import subprocess
import sys
from pathlib import Path

FPS = 25.0
MARGIN = 5

DELETE = {
    60, 7, 16, 22, 23, 26, 27, 32, 33, 38, 39, 45,
    91, 90, 89, 86, 85, 84, 82, 78, 77, 48, 49, 42, 52, 55, 62, 68, 72, 76,
}

# (start_sec, end_sec) from scene_start_frame; end None = scene_end - margin
SCENE_WINDOW = {
    3: (3, 6),
    5: (3, 7),
    8: (2, 6),
    17: (4, 8),
    20: (11, 15),
    29: (1, 5),
    34: (5, 9),
    36: (2, 6),
    37: (10, 15),
    41: (1, 5),
    50: (9, 12),
    51: (10, 14),
    53: (2, 6),
    57: (7, 11),
    64: (4, 8),
    65: (3, 7),
    66: (4, 8),
    71: (3, 7),
    75: (2, 6),
    79: (2, 6),
    83: (3, 7),
}

SCENE_START_ONLY = {1: 1.0, 44: 3.0}
SCENE_START_PLUS_EXTEND_END = {18: 1.0}  # scene +1s start, +1s on prior end
EXTEND_END_1 = {80, 92}

MERGE_3SEC = ([46, 47], "qc_v14r2 merge 46+47 3s fall")


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


def window_bounds(ss: int, se: int, t0: float, t1: float | None) -> tuple[int, int]:
    start = ss + sec_to_frames(t0)
    if t1 is None:
        end = se - MARGIN
    else:
        end = ss + sec_to_frames(t1)
    end = min(end, se - MARGIN)
    start = max(start, ss + MARGIN)
    return start, end


def merge_3sec_fall(entries: list[dict]) -> tuple[int, int]:
    ss = min(e["scene_start_frame"] for e in entries)
    se = max(e["scene_end_frame"] for e in entries)
    onset = min(e.get("onset_frame", ss) for e in entries)
    start = max(ss + MARGIN, onset - sec_to_frames(1.2))
    end = start + sec_to_frames(3.0)
    end = min(end, se - MARGIN)
    if end - start < sec_to_frames(2.5):
        start = max(ss + MARGIN, end - sec_to_frames(3.0))
    return start, end


def apply_edit(num: int, e: dict) -> tuple[int, int, str]:
    ss = e["scene_start_frame"]
    se = e["scene_end_frame"]
    start = e["start_frame"]
    end = e["end_frame"]

    if num in SCENE_WINDOW:
        t0, t1 = SCENE_WINDOW[num]
        s, en = window_bounds(ss, se, t0, t1)
        return s, en, f"qc_v14r2 scene {t0}s-{t1}s"

    if num in SCENE_START_ONLY:
        off = SCENE_START_ONLY[num]
        s = ss + sec_to_frames(off)
        return s, se - MARGIN, f"qc_v14r2 scene +{off}s start"

    if num in SCENE_START_PLUS_EXTEND_END:
        off = SCENE_START_PLUS_EXTEND_END[num]
        s = ss + sec_to_frames(off)
        en = min(se - MARGIN, end + sec_to_frames(1.0))
        return s, en, f"qc_v14r2 scene +{off}s start, +1s end"

    if num in EXTEND_END_1:
        en = min(se - MARGIN, end + sec_to_frames(1.0))
        return start, en, "qc_v14r2 extend +1s end"

    return start, end, e.get("note", "")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/fall_dataset_clips/013_V14"
    )
    folder = folder.resolve()
    source = folder / "source.mp4"
    manifest = json.loads((folder / "manifest.json").read_text())
    by_num = {clip_num(e["clip"]): e for e in manifest}

    merge_nums, merge_note = MERGE_3SEC
    merged_nums = set(merge_nums)

    rejected = []
    kept: list[dict] = []

    if not any(n in DELETE for n in merge_nums):
        entries = [by_num[n] for n in merge_nums]
        start, end = merge_3sec_fall(entries)
        base = {**entries[0]}
        base["scene_start_frame"] = min(e["scene_start_frame"] for e in entries)
        base["scene_end_frame"] = max(e["scene_end_frame"] for e in entries)
        base["source_scene"] = "+".join(e["source_scene"] for e in entries)
        base["start_frame"] = start
        base["end_frame"] = end
        base["duration_sec"] = round((end - start) / FPS, 2)
        base["note"] = merge_note
        base["mode"] = "fall_trim_manual_qc"
        base["_old_num"] = merge_nums[0]
        base["_merged_from"] = merge_nums
        kept.append(base)

    for num in sorted(by_num):
        if num in merged_nums or num in DELETE:
            if num in DELETE:
                rejected.append({
                    "clip": by_num[num]["clip"],
                    "source_clip": f"fall_{num:03d}.mp4",
                    "reason": "user_qc_delete_v14r2",
                })
            continue

        e = by_num[num]
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

    kept.sort(key=lambda x: x["_old_num"])

    manual = []
    new_manifest = []
    for i, e in enumerate(kept, 1):
        name = f"fall_{i:03d}.mp4"
        old = e.pop("_old_num")
        merged = e.pop("_merged_from", None)
        manual.append({
            "clip": name,
            "start_frame": e["start_frame"],
            "end_frame": e["end_frame"],
            "note": e["note"],
        })
        entry = {k: v for k, v in e.items() if k != "clip"}
        entry["clip"] = name
        if merged:
            entry["source_clip_v14r2"] = [f"fall_{n:03d}.mp4" for n in merged]
        else:
            entry["source_clip_v14r2"] = f"fall_{old:03d}.mp4"
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
