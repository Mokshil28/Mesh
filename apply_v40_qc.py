#!/usr/bin/env python3
"""Apply V40 manual QC: cut fall clips from user-provided timestamps."""
import json
import re
import subprocess
import sys
from pathlib import Path

MANUAL_CLIPS = [
    ("0:50", "0:54", "v40 user manual"),
    ("2:13", "2:16", "v40 user manual"),
    ("5:45", "5:48", "v40 user manual — extend end +1s or before camera cut"),
    ("6:25", "6:29", "v40 user manual"),
    ("7:00", "7:03", "v40 user manual — end before camera cut"),
    ("12:29", "12:31.5", "v40 user manual"),
    ("12:46", "12:49", "v40 user manual"),
    ("15:17", "15:21", "v40 user manual"),
    ("15:33", "15:36", "v40 user manual"),
    ("15:41", "15:45", "v40 user manual"),
    ("17:17", "17:19", "v40 user manual"),
    ("18:17", "18:21", "v40 user manual"),
    ("24:28", "24:32", "v40 user manual"),
    ("24:50", "24:54", "v40 user manual"),
    ("24:55", "24:58", "v40 user manual"),
    ("30:14", "30:18", "v40 user manual"),
    ("30:55", "30:59", "v40 user manual"),
    ("31:36", "31:39", "v40 user manual"),
    ("32:46", "32:49", "v40 user manual"),
    ("35:29", "35:32", "v40 user manual"),
    ("37:47", "37:50", "v40 user manual"),
    ("37:51", "37:53", "v40 user manual"),
    ("37:53", "37:56", "v40 user manual"),
    ("38:05", "38:09", "v40 user manual"),
    ("41:30", "41:34", "v40 user manual"),
    ("41:50", "41:54", "v40 user manual"),
    ("42:03", "42:06.5", "v40 user manual"),
    ("45:00", "45:03", "v40 user manual"),
    ("46:49", "46:53", "v40 user manual"),
    ("47:02", "47:04", "v40 user manual"),
    ("49:51", "49:54", "v40 user manual"),
    ("50:14", "50:17.7", "v40 user manual"),
    ("50:35", "50:38", "v40 user manual"),
    ("50:53", "50:56", "v40 user manual"),
    ("51:43", "51:46", "v40 user manual"),
    ("52:25", "52:28", "v40 user manual"),
    ("52:41", "52:45", "v40 user manual"),
    ("52:58", "53:02", "v40 user manual"),
    ("53:16", "53:20", "v40 user manual"),
    ("53:31", "53:35", "v40 user manual"),
    ("53:47", "53:51", "v40 user manual"),
    ("54:00", "54:02.3", "v40 user manual"),
    ("54:43", "54:56", "v40 user manual — end before camera cut"),
    ("56:23", "56:26", "v40 user manual"),
    ("56:30", "56:33", "v40 user manual"),
    ("57:01", "57:04", "v40 user manual"),
    ("57:08", "57:11", "v40 user manual"),
    ("57:26", "57:29", "v40 user manual"),
    ("57:36", "57:38.2", "v40 user manual"),
    ("57:43", "57:46", "v40 user manual"),
    ("58:12", "58:16", "v40 user manual"),
    ("58:34", "58:37", "v40 user manual"),
    ("58:40", "58:43", "v40 user manual"),
    ("1:00:34", "1:00:38", "v40 user manual"),
    ("1:00:50", "1:00:54", "v40 user manual"),
    ("1:06:13", "1:06:17", "v40 user manual"),
    ("1:06:22", "1:06:26", "v40 user manual"),
    ("1:06:52", "1:06:54", "v40 user manual"),
]

# 1-based indices with end-before-camera QC
END_BEFORE_CAMERA = {5, 43}
START_AFTER_CAMERA = {4, 24, 27, 34, 45}
EXTEND_END_OR_BEFORE = {3, 14, 42}  # try +1s at end, else end before next camera cut
EXTEND_END_SEC = 1.0

YOUTUBE_URL = "https://www.youtube.com/watch?v=TC6B78DoCcc"


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
                    "v40 QC — start after camera cut to falling person",
                )
    for idx in EXTEND_END_OR_BEFORE:
        start_s, end_s, _note = out[idx - 1]
        t0, t1 = parse_time(start_s), parse_time(end_s)
        target_end = t1 + EXTEND_END_SEC
        cuts = scene_cuts_sensitive(source, t0, target_end + 0.5)
        after_end = [c for c in cuts if t1 < c <= target_end + 0.05]
        in_clip = [c for c in cuts if t0 < c <= t1 + 0.05]
        if after_end:
            new_end = after_end[0] - 0.2
            note = "v40 QC — extend end blocked by camera cut"
        elif in_clip:
            new_end = in_clip[-1] - 0.2
            note = "v40 QC — end before camera cut"
        else:
            new_end = target_end
            note = f"v40 QC — extend end +{EXTEND_END_SEC}s"
        if new_end > t0 + 0.5:
            out[idx - 1] = (start_s, fmt_time(new_end), note)
    for idx in END_BEFORE_CAMERA:
        start_s, end_s, note = out[idx - 1]
        t0, t1 = parse_time(start_s), parse_time(end_s)
        cuts = [c for c in scene_cuts(source, t0, t1) if t0 < c <= t1 + 0.5]
        if cuts:
            new_end = cuts[-1] - 0.2
            if new_end > t0 + 0.5:
                out[idx - 1] = (
                    start_s,
                    fmt_time(new_end),
                    "v40 QC — end before camera cut",
                )
    return out


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/fall_dataset_clips/040_V40"
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
            "mode": "manual_qc_v40",
            "start_frame": start_f,
            "end_frame": end_f,
            "duration_sec": round((end_f - start_f) / fps, 2),
            "note": note,
            "source_clip_v40": "user_manual",
        })

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "video_info.json").write_text(json.dumps({
        "video_id": "V40",
        "folder": "040_V40",
        "youtube_url": YOUTUBE_URL,
        "note": f"{len(manual)} user manual clips",
    }, indent=2) + "\n")
    print(f"Wrote {len(manual)} entries -> {folder / 'manual_timings.json'}")

    if not source.exists():
        print(f"Missing {source} — add source video then re-run to cut clips.")
        return

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    print(f"Cutting {len(manual)} V40 clips @ {fps:.3f} fps")
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
