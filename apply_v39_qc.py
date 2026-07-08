#!/usr/bin/env python3
"""Apply V39 manual QC: cut fall clips from user-provided timestamps."""
import json
import re
import subprocess
import sys
from pathlib import Path

MANUAL_CLIPS = [
    ("0:10", "0:14", "v39 user manual"),
    ("0:20", "0:24", "v39 user manual"),
    ("1:42", "1:46", "v39 user manual"),
    ("2:15", "2:18", "v39 user manual"),
    ("2:25", "2:28.5", "v39 user manual"),
    ("5:20", "5:24", "v39 user manual"),
    ("5:30", "5:34", "v39 user manual"),
    ("5:55", "5:59", "v39 user manual"),
    ("6:09", "6:13", "v39 user manual"),
    ("6:26", "6:30", "v39 user manual"),
    ("6:52", "6:56", "v39 user manual"),
    ("7:19", "7:23", "v39 user manual"),
    ("7:24", "7:27", "v39 user manual"),
    ("7:31", "7:34", "v39 user manual — start when camera changes (QC may adjust)"),
    ("7:37", "7:40", "v39 user manual"),
    ("7:45", "7:48", "v39 user manual"),
    ("9:31", "9:34", "v39 user manual"),
    ("11:22", "11:25", "v39 user manual"),
    ("11:31", "11:34", "v39 user manual"),
    ("11:45", "11:49", "v39 user manual"),
    ("12:01", "12:04", "v39 QC — start +1 sec"),
    ("14:04", "14:07", "v39 user manual"),
    ("14:08", "14:12", "v39 user manual"),
    ("14:28", "14:30", "v39 user manual"),
    ("14:48", "14:51", "v39 user manual"),
    ("15:25", "15:29", "v39 user manual"),
    ("15:36", "15:40", "v39 user manual"),
    ("15:49", "15:53", "v39 user manual"),
    ("16:22", "16:26", "v39 user manual"),
    ("16:33", "16:37", "v39 user manual"),
    ("16:50", "16:53", "v39 user manual"),
    ("17:04", "17:08", "v39 user manual"),
    ("17:24", "17:27", "v39 user manual"),
    ("17:36", "17:40", "v39 user manual"),
    ("18:20", "18:24", "v39 user manual"),
]

YOUTUBE_URL = "https://www.youtube.com/watch?v=SU1Yi3ijYtY"


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


def scene_cut_after(path: Path, t0: float, t1: float, thresh: float = 0.25) -> float | None:
    cmd = [
        "ffmpeg", "-hide_banner", "-ss", str(max(0, t0 - 5)), "-to", str(t1 + 2),
        "-i", str(path), "-vf", f"select='gt(scene,{thresh})',showinfo", "-f", "null", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    base = max(0, t0 - 5)
    cuts = []
    for line in (out.stderr + out.stdout).splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            cuts.append(base + float(m.group(1)))
    cuts = [c for c in sorted(set(cuts)) if t0 - 2 <= c <= t1]
    return cuts[0] if cuts else None


def fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    if abs(s - round(s)) < 0.05:
        return f"{m}:{int(round(s)):02d}"
    return f"{m}:{s:.1f}"


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


def apply_camera_qc(clips: list[tuple[str, str, str]], source: Path) -> list[tuple[str, str, str]]:
    """Adjust clip 14 (7:31-7:34) start after detected camera cut."""
    out = list(clips)
    idx = 13  # fall_014
    start_s, end_s, note = out[idx]
    t0, t1 = parse_time(start_s), parse_time(end_s)
    cut = scene_cut_after(source, t0, t1)
    if cut is not None:
        new_start = cut + 0.3
        if new_start < t1 - 0.5:
            out[idx] = (
                fmt_time(new_start),
                end_s,
                "v39 QC — start after camera cut to falling person",
            )
    return out


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/fall_dataset_clips/039_V39"
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
            "mode": "manual_qc_v39",
            "start_frame": start_f,
            "end_frame": end_f,
            "duration_sec": round((end_f - start_f) / fps, 2),
            "note": note,
            "source_clip_v39": "user_manual",
        })

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "video_info.json").write_text(json.dumps({
        "video_id": "V39",
        "folder": "039_V39",
        "youtube_url": YOUTUBE_URL,
        "note": f"{len(manual)} user manual clips",
    }, indent=2) + "\n")
    print(f"Wrote {len(manual)} entries -> {folder / 'manual_timings.json'}")

    if not source.exists():
        print(f"Missing {source} — add source video then re-run to cut clips.")
        return

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    print(f"Cutting {len(manual)} V39 clips @ {fps:.3f} fps")
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
