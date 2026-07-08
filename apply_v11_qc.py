#!/usr/bin/env python3
"""Apply V11 QC round 2: deletes, duration caps, scene-relative windows."""
import json
import subprocess
import sys
from pathlib import Path

FPS = 29.970

DELETE = {13, 32, 90, 104, 63, 67, 85, 87, 96, 97, 103, 107, 110}

CAP_FROM_CLIP_START = {
    4.0: {26, 48, 49, 50, 57, 69, 71, 72, 79, 81, 91, 94, 99, 100, 101, 105, 118, 121},
    5.0: {29, 31, 42, 60, 66, 102, 112, 115, 116},
    6.0: {77},
}

# scene-relative windows in seconds: (start_offset, end_offset) from scene_start_frame
SCENE_WINDOW = {
    36: (2.0, 6.0),
    64: (2.0, 7.0),
    65: (2.0, 7.0),
    93: (2.0, 7.0),
    109: (3.0, 7.0),
    111: (4.0, 8.0),
}


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

    if num in SCENE_WINDOW:
        t0, t1 = SCENE_WINDOW[num]
        start = ss + sec_to_frames(t0)
        end = ss + sec_to_frames(t1)
        note = f"qc_v4 scene {t0}s-{t1}s"
    else:
        for dur, nums in CAP_FROM_CLIP_START.items():
            if num in nums:
                end = start + sec_to_frames(dur)
                end = min(end, se - 5)
                note = f"qc_v4 cap {dur}s from clip start"
                return start, end, note

    return start, end, note if num in SCENE_WINDOW else e.get("note", "")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/fall_dataset_clips/011_V11"
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
                "reason": "user_qc_delete_v4",
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
        entry["source_clip_v4"] = f"fall_{old:03d}.mp4"
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
