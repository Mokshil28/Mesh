#!/usr/bin/env python3
"""Apply V49 manual QC: cut fall clips from user-provided timestamps."""
import json
import re
import subprocess
import sys
from pathlib import Path

MANUAL_CLIPS = [
    ("0:04", "0:08", "v49 user manual"),
    ("0:39", "0:42", "v49 user manual"),
    ("1:05", "1:10", "v49 user manual"),
    ("3:08", "3:11", "v49 user manual"),
    ("4:07", "4:11", "v49 user manual"),
    ("5:04", "5:08", "v49 user manual"),
    ("6:07", "6:11", "v49 user manual"),
    ("6:30", "6:34", "v49 user manual"),
    ("6:40", "6:44", "v49 user manual"),
    ("7:00", "7:03.3", "v49 user manual"),
    ("7:39", "7:42", "v49 user manual"),
    ("7:56", "7:59", "v49 user manual"),
    ("8:24", "8:28", "v49 user manual"),
    ("8:58", "9:02", "v49 user manual"),
    ("9:26", "9:30", "v49 user manual"),
    ("16:45", "16:49", "v49 user manual"),
    ("19:31", "19:35", "v49 user manual"),
]

YOUTUBE_URL = "https://www.youtube.com/watch?v=a_EnhZEatmg"


def parse_time(value: str) -> float:
    value = str(value).strip().replace("sec", "")
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
        "data/fall_dataset_clips/049_V49"
    )
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "source.mp4"

    fps = get_fps(source) if source.exists() else 30.0
    manual, manifest = [], []

    for i, (start_s, end_s, note) in enumerate(MANUAL_CLIPS, 1):
        name = f"fall_{i:03d}.mp4"
        start_f = int(round(parse_time(start_s) * fps))
        end_f = int(round(parse_time(end_s) * fps))
        if end_f <= start_f:
            end_f = start_f + int(round(2.5 * fps))
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
            "mode": "manual_qc_v49",
            "start_frame": start_f,
            "end_frame": end_f,
            "duration_sec": round((end_f - start_f) / fps, 2),
            "note": note,
            "source_clip_v49": "user_manual",
        })

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "video_info.json").write_text(json.dumps({
        "video_id": "V49",
        "folder": "049_V49",
        "youtube_url": YOUTUBE_URL,
        "note": f"{len(manual)} user manual clips",
    }, indent=2) + "\n")
    print(f"Wrote {len(manual)} entries -> {folder / 'manual_timings.json'}")

    if not source.exists():
        print(f"Missing {source} — add source video then re-run to cut clips.")
        return

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    print(f"Cutting {len(manual)} V49 clips @ {fps:.3f} fps")
    ok = 0
    for entry in manual:
        outp = folder / entry["clip"]
        try:
            cut_clip(source, outp, entry["start_frame"], entry["end_frame"], fps)
            ok += 1
            dur = (entry["end_frame"] - entry["start_frame"]) / fps
            print(f"  OK {entry['clip']}  {entry['start']}-{entry['end']}  {dur:.2f}s")
        except subprocess.CalledProcessError as exc:
            print(f"  FAIL {entry['clip']}: {exc}")
    print(f"Done: {ok}/{len(manual)}")


if __name__ == "__main__":
    main()
