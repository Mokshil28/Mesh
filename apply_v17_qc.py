#!/usr/bin/env python3
"""Apply V17 manual QC: cut clips from user-provided mm:ss windows."""
import json
import re
import subprocess
import sys
from pathlib import Path

# User manual timings for 016_V17 (37 clips)
MANUAL_CLIPS = [
    ("0:00", "0:03", "qc_v17 manual"),
    ("0:09", "0:13", "qc_v17 manual"),
    ("0:20", "0:23", "qc_v17 manual"),
    ("0:26", "0:28", "qc_v17 manual"),
    ("0:30", "0:32", "qc_v17 manual"),
    ("0:36", "0:39", "qc_v17 manual"),
    ("2:33", "2:37", "qc_v17 manual"),
    ("2:51", "2:54", "qc_v17 manual"),
    ("3:00", "3:04", "qc_v17 manual"),
    ("4:15", "4:19", "qc_v17 manual"),
    ("5:24", "5:29", "qc_v17 manual"),
    ("5:42", "5:46", "qc_v17 manual"),
    ("5:50", "5:54", "qc_v17 manual"),
    ("6:29", "6:33", "qc_v17 manual"),
    ("6:50", "6:54", "qc_v17 manual"),
    ("6:55", "6:59", "qc_v17 manual"),
    # scene_055 @ frame 13822 — +1s after cut to skip camera change
    ("7:42", "7:46", "qc_v17 scene_055 lady about to fall, no camera change"),
    ("8:05", "8:09", "qc_v17 manual"),
    ("9:32", "9:36", "qc_v17 manual"),
    ("10:12", "10:16", "qc_v17 manual"),
    ("10:58", "11:01", "qc_v17 manual"),
    ("11:06", "11:09", "qc_v17 manual"),
    ("12:11", "12:14", "qc_v17 manual"),
    ("12:34", "12:38", "qc_v17 manual"),
    ("13:06", "13:09", "qc_v17 manual"),
    ("13:24", "13:27", "qc_v17 manual"),
    ("13:34", "13:38", "qc_v17 manual"),
    ("13:52", "13:56", "qc_v17 manual"),
    ("14:38", "14:42", "qc_v17 manual"),
    ("15:09", "15:12", "qc_v17 manual"),
    ("16:33", "16:36", "qc_v17 manual"),
    ("16:41", "16:44", "qc_v17 manual"),
    ("17:29", "17:33", "qc_v17 manual"),
    ("18:18", "18:22", "qc_v17 manual"),
    ("18:54", "18:57", "qc_v17 manual"),
    ("19:01", "19:05", "qc_v17 manual"),
    ("20:48", "20:51", "qc_v17 manual"),
]


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


def get_fps(video: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
            str(video),
        ],
        text=True,
    ).strip()
    if "/" in out:
        num, den = out.split("/")
        return float(num) / float(den)
    return float(out)


def cut_clip(source: Path, out: Path, start_f: int, end_f: int, fps: float) -> None:
    start_sec = start_f / fps
    duration = max((end_f - start_f) / fps, 0.1)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_sec:.6f}", "-i", str(source),
        "-t", f"{duration:.6f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/fall_dataset_clips/016_V17"
    )
    folder = folder.resolve()
    source = folder / "source.mp4"
    if not source.exists():
        print(f"Missing source: {source}")
        sys.exit(1)

    fps = get_fps(source)
    manual = []
    manifest = []

    for i, (start_s, end_s, note) in enumerate(MANUAL_CLIPS, 1):
        name = f"fall_{i:03d}.mp4"
        start_f = int(round(parse_time(start_s) * fps))
        end_f = int(round(parse_time(end_s) * fps))
        manual.append({
            "clip": name,
            "start": start_s,
            "end": end_s,
            "start_frame": start_f,
            "end_frame": end_f,
            "note": note,
        })
        manifest.append({
            "clip": name,
            "mode": "manual_qc_v17",
            "start_frame": start_f,
            "end_frame": end_f,
            "duration_sec": round((end_f - start_f) / fps, 2),
            "note": note,
            "source_clip_v17": "user_manual",
        })

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Cutting {len(manual)} V17 manual clips @ {fps:.3f} fps")
    ok = 0
    for entry in manual:
        out = folder / entry["clip"]
        try:
            cut_clip(source, out, entry["start_frame"], entry["end_frame"], fps)
            ok += 1
            dur = (entry["end_frame"] - entry["start_frame"]) / fps
            print(f"  OK {entry['clip']}  {entry['start']}-{entry['end']}  {dur:.2f}s")
        except subprocess.CalledProcessError as exc:
            print(f"  FAIL {entry['clip']}: {exc}")

    print(f"Done: {ok}/{len(manual)}")


if __name__ == "__main__":
    main()
