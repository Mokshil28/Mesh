#!/usr/bin/env python3
"""Apply V51 manual QC: cut fall clips from user-provided timestamps."""
import json
import re
import subprocess
import sys
from pathlib import Path

MANUAL_CLIPS = [
    ("0:06", "0:09", "v51 user manual"),
    ("0:12", "0:15", "v51 user manual"),
    ("0:26", "0:29", "v51 user manual"),
    ("0:43", "0:45", "v51 user manual"),
    ("0:53", "0:57", "v51 user manual"),
    ("1:27", "1:30", "v51 user manual"),
    ("3:01", "3:03", "v51 user manual"),
    ("3:25", "3:28", "v51 user manual"),
    ("3:40", "3:43", "v51 user manual"),
    ("4:06", "4:10", "v51 user manual"),
    ("4:46.5", "4:48.5", "v51 user manual"),
    ("6:27", "6:31", "v51 user manual"),
    ("7:41", "7:44", "v51 user manual"),
    ("8:02", "8:06", "v51 user manual"),
    ("8:18", "8:21", "v51 user manual"),
    ("8:25", "8:27", "v51 user manual"),
    ("8:42", "8:45", "v51 user manual"),
    ("9:08", "9:11", "v51 user manual"),
    ("9:43", "9:46", "v51 user manual"),
    ("10:01", "10:05", "v51 user manual"),
    ("12:13", "12:17", "v51 user manual"),
    ("12:22", "12:26", "v51 user manual"),
    ("13:49", "13:52", "v51 user manual"),
    ("14:08", "14:11", "v51 user manual"),
    ("14:33", "14:36", "v51 user manual"),
    ("14:52", "14:55", "v51 user manual"),
    ("16:25", "16:29", "v51 user manual"),
    ("16:59", "17:03", "v51 user manual"),
    ("17:22", "17:26", "v51 user manual"),
    ("17:40", "17:44", "v51 user manual"),
    ("19:50", "19:54", "v51 user manual"),
    ("20:06", "20:10", "v51 user manual"),
    ("20:23", "20:27", "v51 user manual"),
]

YOUTUBE_URL = "https://www.youtube.com/watch?v=SskNzUn2_xs"


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
        "data/fall_dataset_clips/051_V51"
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
            "mode": "manual_qc_v51",
            "start_frame": start_f,
            "end_frame": end_f,
            "duration_sec": round((end_f - start_f) / fps, 2),
            "note": note,
            "source_clip_v51": "user_manual",
        })

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "video_info.json").write_text(json.dumps({
        "video_id": "V51",
        "folder": "051_V51",
        "youtube_url": YOUTUBE_URL,
        "note": f"{len(manual)} user manual clips",
    }, indent=2) + "\n")
    print(f"Wrote {len(manual)} entries -> {folder / 'manual_timings.json'}")

    if not source.exists():
        print(f"Missing {source} — add source video then re-run to cut clips.")
        return

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    print(f"Cutting {len(manual)} V51 clips @ {fps:.3f} fps")
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
