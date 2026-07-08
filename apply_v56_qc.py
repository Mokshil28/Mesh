#!/usr/bin/env python3
"""Apply V56 manual QC: cut fall clips from user-provided timestamps."""
import json
import re
import subprocess
import sys
from pathlib import Path

MANUAL_CLIPS = [
    ("0:04", "0:08", "v56 user manual - cut before camera cut"),
    ("0:16", "0:20", "v56 user manual"),
    ("0:35", "0:38", "v56 user manual"),
    ("1:00", "1:04", "v56 user manual"),
    ("1:28", "1:31", "v56 user manual"),
    ("2:14", "2:18", "v56 user manual"),
    ("2:28", "2:32", "v56 user manual"),
    ("3:19", "3:23", "v56 user manual"),
    ("4:40", "4:43", "v56 user manual"),
    ("5:02", "5:06", "v56 user manual"),
    ("5:35", "5:39", "v56 user manual"),
    ("5:53", "5:56", "v56 user manual"),
    ("6:58", "7:02", "v56 user manual"),
    ("7:58", "8:01", "v56 user manual"),
    ("9:36", "9:39", "v56 user manual"),
    ("9:51", "9:55", "v56 user manual"),
    ("10:43", "10:45.5", "v56 user manual"),
    ("11:36", "11:39", "v56 user manual"),
    ("12:08", "12:12", "v56 user manual"),
    ("12:21", "12:24", "v56 user manual"),
    ("13:54", "13:57", "v56 user manual"),
    ("14:14", "14:18", "v56 user manual"),
    ("14:31", "14:34", "v56 user manual"),
]

END_BEFORE_CAMERA = {1}
YOUTUBE_URL = "https://www.youtube.com/watch?v=ZuYGaOl1vgg"


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


def fmt_time(sec: float) -> str:
    if sec >= 3600:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        if abs(s - round(s)) < 0.05:
            return f"{h}:{m:02d}:{int(round(s)):02d}"
        return f"{h}:{m:02d}:{s:.1f}"
    m = int(sec // 60)
    s = sec - m * 60
    if abs(s - round(s)) < 0.05:
        return f"{m}:{int(round(s)):02d}"
    return f"{m}:{s:.1f}"


def scene_cuts(path: Path, t0: float, t1: float, thresh: float = 0.25) -> list[float]:
    cmd = [
        "ffmpeg", "-hide_banner", "-ss", str(max(0, t0 - 5)), "-to", str(t1 + 3),
        "-i", str(path), "-vf", f"select='gt(scene,{thresh})',showinfo", "-f", "null", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    base = max(0, t0 - 5)
    cuts = []
    for line in (out.stderr + out.stdout).splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            cuts.append(base + float(m.group(1)))
    return sorted(set(cuts))


def apply_camera_qc(clips: list[tuple[str, str, str]], source: Path) -> list[tuple[str, str, str]]:
    out = list(clips)
    for idx in END_BEFORE_CAMERA:
        start_s, end_s, _note = out[idx - 1]
        t0, t1 = parse_time(start_s), parse_time(end_s)
        cuts = [c for c in scene_cuts(source, t0, t1) if t0 < c <= t1 + 0.5]
        if cuts:
            new_end = cuts[-1] - 0.2
            if new_end > t0 + 0.5:
                out[idx - 1] = (
                    start_s,
                    fmt_time(new_end),
                    "v56 QC - end before camera cut",
                )
    return out


def get_fps(video: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video),
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
        "-ss", f"{start_sec:.6f}", "-i", str(source), "-t", f"{duration:.6f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True)


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/fall_dataset_clips/056_V56")
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "source.mp4"

    clips = list(MANUAL_CLIPS)
    if source.exists():
        clips = apply_camera_qc(clips, source)

    fps = get_fps(source) if source.exists() else 30.0
    manual, manifest = [], []

    for i, (start_s, end_s, note) in enumerate(clips, 1):
        name = f"fall_{i:03d}.mp4"
        start_f = int(round(parse_time(start_s) * fps))
        end_f = int(round(parse_time(end_s) * fps))
        if end_f <= start_f:
            end_f = start_f + int(round(2.5 * fps))
        manual.append({"clip": name, "start": start_s, "end": end_s, "start_frame": start_f, "end_frame": end_f, "note": note})
        manifest.append({"clip": name, "mode": "manual_qc_v56", "start_frame": start_f, "end_frame": end_f, "duration_sec": round((end_f - start_f) / fps, 2), "note": note, "source_clip_v56": "user_manual"})

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "video_info.json").write_text(json.dumps({"video_id": "V56", "folder": "056_V56", "youtube_url": YOUTUBE_URL, "note": f"{len(manual)} user manual clips"}, indent=2) + "\n")
    print(f"Wrote {len(manual)} entries -> {folder / 'manual_timings.json'}")

    if not source.exists():
        print(f"Missing {source} - add source video then re-run to cut clips.")
        return

    for clip_file in folder.glob("fall_*.mp4"):
        clip_file.unlink()

    print(f"Cutting {len(manual)} V56 clips @ {fps:.3f} fps")
    ok = 0
    for entry in manual:
        out_path = folder / entry["clip"]
        try:
            cut_clip(source, out_path, entry["start_frame"], entry["end_frame"], fps)
            ok += 1
            dur = (entry["end_frame"] - entry["start_frame"]) / fps
            print(f"  OK {entry['clip']}  {entry['start']}-{entry['end']}  {dur:.2f}s")
        except subprocess.CalledProcessError as exc:
            print(f"  FAIL {entry['clip']}: {exc}")
    print(f"Done: {ok}/{len(manual)}")


if __name__ == "__main__":
    main()
