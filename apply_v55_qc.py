#!/usr/bin/env python3
"""Apply V55 manual QC: cut fall clips from user-provided timestamps."""
import json
import re
import subprocess
import sys
from pathlib import Path

MANUAL_CLIPS = [
    ("0:01", "0:05", "v55 user manual"),
    ("0:21", "0:25", "v55 user manual"),
    ("0:26", "0:28", "v55 user manual"),
    ("1:10", "1:14", "v55 user manual"),
    ("1:18", "1:22", "v55 user manual"),
    ("1:24", "1:27", "v55 user manual"),
    ("2:11", "2:15", "v55 user manual"),
    ("2:44", "2:48", "v55 user manual"),
    ("2:54", "2:57", "v55 user manual"),
    ("2:58", "3:01", "v55 user manual"),
    ("3:17", "3:19.3", "v55 user manual"),
    ("3:25", "3:28", "v55 user manual"),
    ("3:31", "3:34", "v55 user manual"),
    ("3:58", "4:01", "v55 user manual"),
    ("4:45", "4:49", "v55 user manual"),
    ("4:59", "5:03", "v55 user manual"),
    ("5:11", "5:13", "v55 user manual"),
    ("5:24", "5:27", "v55 user manual"),
    ("5:54", "5:56", "v55 user manual"),
    ("6:08", "6:11", "v55 user manual - cut before camera cut"),
    ("6:42", "6:45", "v55 user manual"),
    ("7:08", "7:11", "v55 user manual"),
    ("7:18", "7:21", "v55 user manual"),
    ("7:42", "7:45", "v55 user manual"),
    ("7:49", "7:52", "v55 user manual"),
    ("8:38", "8:42", "v55 user manual"),
    ("9:35", "9:39", "v55 user manual"),
    ("10:10", "10:13", "v55 user manual"),
    ("12:15", "12:18", "v55 user manual"),
    ("12:26", "12:28", "v55 user manual"),
    ("12:48", "12:51", "v55 user manual"),
    ("12:56", "12:59", "v55 user manual"),
    ("13:05", "13:09", "v55 user manual"),
    ("13:25", "13:28", "v55 user manual"),
    ("13:52", "13:56", "v55 user manual"),
    ("14:09", "14:13", "v55 user manual"),
    ("14:39", "14:43", "v55 user manual"),
    ("14:56", "15:00", "v55 user manual"),
    ("15:27", "15:30", "v55 user manual"),
    ("15:35", "15:39", "v55 user manual"),
    ("16:26", "16:27.5", "v55 user manual"),
    ("16:46", "16:48", "v55 user manual"),
    ("17:31", "17:33", "v55 user manual"),
    ("18:35", "18:39", "v55 user manual"),
    ("18:45", "18:48", "v55 user manual - start after camera cut"),
    ("19:03", "19:05.5", "v55 user manual"),
    ("19:22", "19:25", "v55 user manual"),
]

END_BEFORE_CAMERA = {20}
START_AFTER_CAMERA = {45}
YOUTUBE_URL = "https://www.youtube.com/watch?v=8Q6HFIfVF-o"


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
        start_s, end_s, _ = out[idx - 1]
        t0, t1 = parse_time(start_s), parse_time(end_s)
        cuts = [c for c in scene_cuts(source, t0, t1) if t0 < c <= t1 + 0.5]
        if cuts:
            new_end = cuts[-1] - 0.2
            if new_end > t0 + 0.5:
                out[idx - 1] = (start_s, fmt_time(new_end), "v55 QC - end before camera cut")
    for idx in START_AFTER_CAMERA:
        start_s, end_s, _ = out[idx - 1]
        t0, t1 = parse_time(start_s), parse_time(end_s)
        cuts = [c for c in scene_cuts(source, t0, t1) if t0 - 0.5 <= c < t1]
        if cuts:
            new_start = cuts[0] + 0.3
            if new_start < t1 - 0.5:
                out[idx - 1] = (fmt_time(new_start), end_s, "v55 QC - start after camera cut")
    return out


def get_fps(video: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
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
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/fall_dataset_clips/055_V55")
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
        manifest.append({"clip": name, "mode": "manual_qc_v55", "start_frame": start_f, "end_frame": end_f, "duration_sec": round((end_f - start_f) / fps, 2), "note": note, "source_clip_v55": "user_manual"})
    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "video_info.json").write_text(json.dumps({"video_id": "V55", "folder": "055_V55", "youtube_url": YOUTUBE_URL, "note": f"{len(manual)} user manual clips"}, indent=2) + "\n")
    print(f"Wrote {len(manual)} entries -> {folder / 'manual_timings.json'}")
    if not source.exists():
        print(f"Missing {source} - add source video then re-run to cut clips.")
        return
    for f in folder.glob("fall_*.mp4"):
        f.unlink()
    print(f"Cutting {len(manual)} V55 clips @ {fps:.3f} fps")
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
