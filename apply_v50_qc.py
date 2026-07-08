#!/usr/bin/env python3
"""Apply V50 manual QC: cut fall clips from user-provided timestamps."""
import json
import re
import subprocess
import sys
from pathlib import Path

MANUAL_CLIPS = [
    ("0:21", "0:24", "v50 user manual"),
    ("0:51", "0:55", "v50 user manual — start after camera changes"),
    ("1:22", "1:26", "v50 user manual"),
    ("1:28", "1:32", "v50 user manual"),
    ("2:00", "2:03", "v50 user manual"),
    ("2:04", "2:08", "v50 user manual"),
    ("2:43", "2:46", "v50 user manual"),
    ("2:50", "2:54", "v50 user manual"),
    ("2:55", "2:57", "v50 user manual"),
    ("3:54", "3:57", "v50 user manual"),
    ("4:07", "4:11", "v50 user manual"),
    ("6:34", "6:37", "v50 user manual"),
    ("9:19", "9:23", "v50 user manual"),
    ("9:27", "9:31", "v50 user manual"),
    ("10:40", "10:43", "v50 user manual"),
    ("11:52", "11:56", "v50 user manual"),
    ("13:30", "13:32", "v50 user manual"),
    ("13:33", "13:36", "v50 user manual"),
    ("13:37", "13:40", "v50 user manual"),
    ("16:42", "16:46", "v50 user manual"),
    ("17:40", "17:42", "v50 user manual"),
    ("18:30", "18:33", "v50 user manual"),
    ("19:12", "19:16", "v50 user manual"),
    ("20:04", "20:07", "v50 user manual"),
    ("21:57", "22:01", "v50 user manual"),
    ("22:33", "22:36", "v50 user manual"),
    ("22:38", "22:41", "v50 user manual"),
    ("22:52", "22:55", "v50 user manual — cut before camera changes"),
    ("22:56", "22:59", "v50 user manual"),
    ("23:08", "23:10", "v50 user manual"),
    ("23:36", "23:39", "v50 user manual"),
    ("23:49", "23:52.3", "v50 user manual"),
    ("24:11", "24:15", "v50 user manual"),
    ("24:45", "24:48", "v50 user manual"),
    ("25:23", "25:26", "v50 user manual"),
    ("26:23", "26:26", "v50 user manual"),
    ("26:47", "26:50", "v50 user manual"),
    ("28:15", "28:18", "v50 user manual"),
    ("29:18", "29:21", "v50 user manual"),
    ("29:28", "29:31", "v50 user manual"),
    ("30:37", "30:41", "v50 user manual"),
    ("31:51", "31:55", "v50 user manual"),
    ("32:22", "32:26", "v50 user manual"),
    ("33:04", "33:07", "v50 user manual"),
    ("35:02", "35:05", "v50 user manual"),
]

START_AFTER_CAMERA = {2}
END_BEFORE_CAMERA = {28}

YOUTUBE_URL = "https://www.youtube.com/watch?v=ClxxqUGsj54"


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


def scene_cuts_sensitive(path: Path, t0: float, t1: float) -> list[float]:
    cuts = scene_cuts(path, t0, t1, thresh=0.25)
    if not cuts:
        cuts = scene_cuts(path, t0, t1, thresh=0.2)
    return cuts


def apply_camera_qc(clips: list[tuple[str, str, str]], source: Path) -> list[tuple[str, str, str]]:
    out = list(clips)
    for idx in START_AFTER_CAMERA:
        start_s, end_s, _note = out[idx - 1]
        t0, t1 = parse_time(start_s), parse_time(end_s)
        cuts = scene_cuts_sensitive(source, t0, t1 + 1)
        in_window = [c for c in cuts if t0 <= c <= t1 + 0.5]
        before_start = [c for c in cuts if t0 - 2 <= c < t0]
        pick = in_window[0] if in_window else (before_start[-1] if before_start else None)
        if pick is not None:
            new_start = pick + 0.3
            if new_start < t1 - 0.5:
                out[idx - 1] = (
                    fmt_time(new_start),
                    end_s,
                    "v50 QC — start after camera cut to falling person",
                )
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
                    "v50 QC — end before camera cut",
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
        "data/fall_dataset_clips/050_V50"
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
            "mode": "manual_qc_v50",
            "start_frame": start_f,
            "end_frame": end_f,
            "duration_sec": round((end_f - start_f) / fps, 2),
            "note": note,
            "source_clip_v50": "user_manual",
        })

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "video_info.json").write_text(json.dumps({
        "video_id": "V50",
        "folder": "050_V50",
        "youtube_url": YOUTUBE_URL,
        "note": f"{len(manual)} user manual clips",
    }, indent=2) + "\n")
    print(f"Wrote {len(manual)} entries -> {folder / 'manual_timings.json'}")

    if not source.exists():
        print(f"Missing {source} — add source video then re-run to cut clips.")
        return

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    print(f"Cutting {len(manual)} V50 clips @ {fps:.3f} fps")
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
