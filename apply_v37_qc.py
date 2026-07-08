#!/usr/bin/env python3
"""Apply V37 manual QC: cut fall clips from user-provided timestamps."""
import json
import re
import subprocess
import sys
from pathlib import Path

# User V37 manual list (mm:ss or seconds) — QC pass 2
REJECTED_CLIPS = [
    {"clip": "was_fall_061.mp4", "start": "22:46", "end": "22:50", "reason": "v37 QC delete"},
    {"clip": "was_fall_073.mp4", "start": "26:56.9", "end": "27:00", "reason": "v37 QC delete"},
]

MANUAL_CLIPS = [
    ("0:02", "0:05", "v37 user manual"),
    ("0:31.3", "0:34", "v37 QC — start after camera cut to falling woman"),
    ("0:57", "1:01", "v37 user manual"),
    ("1:23.3", "1:26", "v37 QC — start after camera cut to falling woman"),
    ("2:03", "2:06", "v37 user manual"),
    ("2:33", "2:37", "v37 user manual"),
    ("2:38.5", "2:42", "v37 QC — start after camera cut to falling woman"),
    ("2:44", "2:46", "v37 user manual"),
    ("2:58", "3:02", "v37 user manual"),
    ("3:44", "3:47", "v37 user manual"),
    ("4:20", "4:23", "v37 user manual"),
    ("4:25", "4:29", "v37 user manual"),
    ("4:36", "4:39", "v37 user manual"),
    ("4:54", "4:57", "v37 user manual"),
    ("6:37", "6:38.5", "v37 user manual"),
    ("6:40", "6:43", "v37 QC — start +1 sec"),
    ("6:55", "6:58", "v37 user manual"),
    ("7:00", "7:04", "v37 user manual"),
    ("7:59", "8:03", "v37 user manual"),
    ("8:15", "8:18", "v37 user manual"),
    ("8:29", "8:33", "v37 user manual"),
    ("8:40", "8:44", "v37 user manual"),
    ("8:51", "8:55", "v37 user manual"),
    ("8:59", "9:03", "v37 user manual"),
    ("9:06", "9:09", "v37 user manual"),
    ("9:19", "9:24", "v37 user manual"),
    ("9:40", "9:44", "v37 user manual"),
    ("9:49", "9:53", "v37 user manual"),
    ("10:01", "10:05", "v37 user manual"),
    ("11:04", "11:07", "v37 QC — start +1 sec"),
    ("11:10.3", "11:14", "v37 QC — start after camera cut to falling woman"),
    ("11:15", "11:19", "v37 user manual"),
    ("11:47", "11:51", "v37 user manual"),
    ("12:46", "12:50", "v37 user manual"),
    ("12:56", "13:00", "v37 user manual"),
    ("13:06", "13:09.5", "v37 user manual"),
    ("13:13", "13:18", "v37 user manual"),
    ("13:35", "13:38", "v37 user manual"),
    ("13:46", "13:49", "v37 user manual — user wrote end 13:39, corrected"),
    ("13:51", "13:55", "v37 user manual"),
    ("13:59", "14:01", "v37 user manual"),
    ("14:07", "14:10", "v37 user manual"),
    ("14:14", "14:17", "v37 user manual"),
    ("14:24", "14:28", "v37 user manual"),
    ("14:34", "14:37.5", "v37 user manual"),
    ("14:43", "14:47", "v37 user manual"),
    ("14:54", "14:58", "v37 user manual"),
    ("15:03", "15:05", "v37 user manual"),
    ("16:24", "16:27", "v37 user manual"),
    ("17:26", "17:29", "v37 user manual"),
    ("17:33", "17:37", "v37 user manual"),
    ("17:41", "17:45", "v37 user manual"),
    ("18:03", "18:07", "v37 user manual"),
    ("19:03", "19:07", "v37 user manual"),
    ("19:15", "19:19", "v37 user manual"),
    ("19:23", "19:27", "v37 user manual"),
    ("19:45", "19:49", "v37 user manual"),
    ("19:59", "20:02", "v37 user manual"),
    ("22:05", "22:09", "v37 user manual"),
    ("22:35", "22:39", "v37 user manual"),
    ("23:14", "23:18", "v37 user manual"),
    ("23:26", "23:28", "v37 user manual"),
    ("23:33", "23:38", "v37 user manual"),
    ("24:02", "24:06", "v37 user manual"),
    ("24:36", "24:38", "v37 user manual"),
    ("25:05", "25:08", "v37 user manual"),
    ("25:22", "25:26", "v37 user manual"),
    ("25:57", "26:00", "v37 user manual — user wrote end 25:29, corrected"),
    ("26:23", "26:27", "v37 user manual"),
    ("26:43", "26:47", "v37 user manual"),
    ("26:50.5", "26:54", "v37 QC — start after camera cut to falling woman"),
    ("27:06", "27:10", "v37 user manual"),
    ("27:16", "27:20", "v37 user manual"),
    ("27:30", "27:34", "v37 user manual"),
    ("27:39", "27:42", "v37 user manual"),
    ("27:55", "27:59", "v37 user manual"),
    ("28:02.3", "28:06", "v37 QC — start after camera cut to falling woman"),
    ("30:06", "30:10", "v37 user manual"),
    ("30:19", "30:22", "v37 user manual"),
    ("30:34.8", "30:37", "v37 user manual"),
    ("30:54", "30:58", "v37 user manual"),
    ("31:06", "31:09", "v37 user manual"),
    ("31:12", "31:14", "v37 user manual"),
]


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


def mmss(frame: int, fps: float) -> str:
    s = frame / fps
    m = int(s // 60)
    sec = s - m * 60
    if abs(sec - round(sec)) < 0.05:
        return f"{m}:{int(round(sec)):02d}"
    return f"{m}:{sec:.1f}"


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/fall_dataset_clips/037_V37"
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
            "mode": "manual_qc_v37",
            "start_frame": start_f,
            "end_frame": end_f,
            "duration_sec": round((end_f - start_f) / fps, 2),
            "note": note,
            "source_clip_v37": "user_manual",
        })

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "rejected_clips.json").write_text(
        json.dumps(REJECTED_CLIPS, indent=2) + "\n"
    )
    print(f"Wrote {len(manual)} entries -> {folder / 'manual_timings.json'}")
    print(f"Rejected {len(REJECTED_CLIPS)} clips -> {folder / 'rejected_clips.json'}")

    if not source.exists():
        print(f"Missing {source} — add source video then re-run to cut clips.")
        return

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    print(f"Cutting {len(manual)} V37 clips @ {fps:.3f} fps")
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
