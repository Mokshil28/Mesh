#!/usr/bin/env python3
"""Cut fall clips from a source video using manual_timings.json (CPU/ffmpeg)."""
import json
import re
import subprocess
import sys
from pathlib import Path


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


def clip_range(entry: dict, fps: float) -> tuple[float, float]:
    if "start_frame" in entry:
        return entry["start_frame"] / fps, entry["end_frame"] / fps
    return parse_time(entry["start"]), parse_time(entry["end"])


def cut_clip(source: Path, out: Path, start: float, end: float) -> None:
    duration = max(end - start, 0.1)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.6f}", "-i", str(source),
        "-t", f"{duration:.6f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: cut_clips_from_manual.py <clip_folder> [source_video.mp4]")
        sys.exit(1)

    folder = Path(sys.argv[1]).resolve()
    manual = folder / "manual_timings.json"
    if not manual.exists():
        print(f"Missing {manual}")
        sys.exit(1)

    source = Path(sys.argv[2]) if len(sys.argv) > 2 else folder / "source.mp4"
    if not source.exists():
        print(f"Missing source video: {source}")
        sys.exit(1)

    entries = json.loads(manual.read_text())
    fps = get_fps(source)
    print(f"Source: {source} ({fps:.3f} fps), cutting {len(entries)} clips -> {folder}")

    ok = 0
    for entry in entries:
        name = entry["clip"]
        start, end = clip_range(entry, fps)
        out = folder / name
        try:
            cut_clip(source, out, start, end)
            ok += 1
            print(f"  OK {name}  {start:.2f}s -> {end:.2f}s")
        except subprocess.CalledProcessError as exc:
            print(f"  FAIL {name}: {exc}")

    print(f"Done: {ok}/{len(entries)} clips in {folder}")


if __name__ == "__main__":
    main()
