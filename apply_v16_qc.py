#!/usr/bin/env python3
"""Apply V16 manual QC: replace auto clips with user-provided mm:ss windows."""
import json
import re
import subprocess
import sys
from pathlib import Path

# User manual timings for 015_V16 (27 clips)
MANUAL_CLIPS = [
    ("2:11", "2:16", "qc_v16 manual"),
    ("3:06", "3:10", "qc_v16 manual"),
    ("3:46", "3:50", "qc_v16 manual"),
    ("4:56", "5:00", "qc_v16 manual"),
    ("5:02", "5:05", "qc_v16 manual"),
    ("5:22", "5:26", "qc_v16 manual"),
    ("6:14", "6:18", "qc_v16 manual"),
    ("7:24", "7:28", "qc_v16 manual"),
    ("8:10", "8:14", "qc_v16 scene change to pre-fall"),
    ("8:31", "8:35", "qc_v16 manual"),
    ("8:38", "8:41", "qc_v16 cut before clip change"),
    ("9:46", "9:50", "qc_v16 runner about to fall"),
    ("10:51", "10:55", "qc_v16 manual"),
    ("11:05", "11:08", "qc_v16 manual"),
    ("12:13", "12:16", "qc_v16 manual"),
    ("12:44", "12:47", "qc_v16 manual"),
    ("13:01", "13:05", "qc_v16 manual"),
    ("14:05", "14:09", "qc_v16 manual"),
    ("15:29", "15:32", "qc_v16 manual"),
    ("15:41", "15:44", "qc_v16 manual"),
    ("16:26", "16:29", "qc_v16 manual"),
    ("19:48", "19:52", "qc_v16 manual"),
    ("19:59", "20:02", "qc_v16 manual"),
    ("20:02", "20:06", "qc_v16 manual"),
    ("20:35", "20:38", "qc_v16 manual"),
    ("20:39", "20:43", "qc_v16 lady about to fall"),
    ("20:53", "20:56", "qc_v16 manual"),
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
        "data/fall_dataset_clips/015_V16"
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
            "mode": "manual_qc_v16",
            "start_frame": start_f,
            "end_frame": end_f,
            "duration_sec": round((end_f - start_f) / fps, 2),
            "note": note,
            "source_clip_v16": "user_manual",
        })

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Cutting {len(manual)} V16 manual clips @ {fps:.3f} fps")
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
