#!/usr/bin/env python3
"""Apply V52 manual QC: cut fall clips from user-provided timestamps."""
import json
import re
import subprocess
import sys
from pathlib import Path

MANUAL_CLIPS = [
    ("0:44", "0:48", "v52 user manual"),
    ("0:55", "0:57.8", "v52 user manual"),
    ("1:15", "1:18", "v52 user manual"),
    ("1:29", "1:32", "v52 user manual"),
    ("1:51", "1:54", "v52 user manual"),
    ("2:25", "2:28", "v52 user manual"),
    ("2:33", "2:37", "v52 user manual"),
    ("2:50", "2:54", "v52 user manual"),
    ("3:06", "3:10", "v52 user manual"),
    ("4:09", "4:13", "v52 user manual"),
    ("4:20", "4:24", "v52 user manual"),
    ("4:25", "4:29", "v52 user manual"),
    ("4:31", "4:34", "v52 user manual"),
    ("4:42", "4:45", "v52 user manual"),
    ("4:58", "5:02", "v52 user manual"),
    ("5:05", "5:07.8", "v52 user manual — cut before camera cut"),
    ("5:23", "5:26", "v52 user manual"),
    ("5:28", "5:32", "v52 user manual — cut before camera cut"),
    ("5:34", "5:37", "v52 user manual"),
    ("5:43", "5:47", "v52 user manual"),
    ("5:49", "5:53", "v52 user manual"),
    ("5:54", "5:56.5", "v52 user manual"),
    ("5:59", "6:02", "v52 user manual"),
    ("6:11", "6:15", "v52 user manual"),
    ("6:30", "6:33", "v52 user manual — cut before camera cut"),
    ("6:40", "6:44", "v52 user manual"),
    ("6:51", "6:53", "v52 user manual"),
    ("6:55", "6:58", "v52 user manual"),
    ("6:59", "7:03", "v52 user manual"),
    ("7:05", "7:07", "v52 user manual"),
    ("7:09", "7:12", "v52 user manual"),
    ("7:14", "7:17", "v52 user manual"),
    ("7:31", "7:35", "v52 user manual"),
    ("7:38", "7:42", "v52 user manual"),
    ("7:56", "8:00", "v52 user manual"),
    ("8:07", "8:10", "v52 user manual"),
    ("8:12", "8:16", "v52 user manual"),
    ("8:22", "8:26", "v52 user manual"),
    ("8:28", "8:32", "v52 user manual"),
    ("8:40", "8:43", "v52 user manual"),
    ("8:47", "8:51", "v52 user manual"),
    ("9:07", "9:11", "v52 user manual"),
    ("9:15", "9:17", "v52 user manual"),
    ("9:19", "9:22", "v52 user manual"),
    ("9:28", "9:32", "v52 user manual"),
    ("9:34", "9:37", "v52 user manual"),
    ("9:44", "9:48", "v52 user manual"),
    ("9:58", "10:02", "v52 user manual"),
    ("10:10", "10:14", "v52 user manual"),
]

END_BEFORE_CAMERA = {16, 18, 25}

YOUTUBE_URL = "https://www.youtube.com/watch?v=I8d6YId5Y3g"


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
                    "v52 QC — end before camera cut",
                )
    return out


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
        "data/fall_dataset_clips/052_V52"
    )
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
            "mode": "manual_qc_v52",
            "start_frame": start_f,
            "end_frame": end_f,
            "duration_sec": round((end_f - start_f) / fps, 2),
            "note": note,
            "source_clip_v52": "user_manual",
        })

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "video_info.json").write_text(json.dumps({
        "video_id": "V52",
        "folder": "052_V52",
        "youtube_url": YOUTUBE_URL,
        "note": f"{len(manual)} user manual clips",
    }, indent=2) + "\n")
    print(f"Wrote {len(manual)} entries -> {folder / 'manual_timings.json'}")

    if not source.exists():
        print(f"Missing {source} — add source video then re-run to cut clips.")
        return

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    print(f"Cutting {len(manual)} V52 clips @ {fps:.3f} fps")
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
