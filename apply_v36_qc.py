#!/usr/bin/env python3
"""Apply V36 manual QC: cut clips from user-provided mm:ss windows."""
import json
import re
import subprocess
import sys
from pathlib import Path

# (start, end, note) — user V36 manual list
MANUAL_CLIPS = [
    ("0:02", "0:06", "qc_v36 manual"),
    ("0:08", "0:11", "qc_v36 manual"),
    ("0:57", "1:00", "qc_v36 pre-fall motion"),
    ("1:16", "1:19", "qc_v36 manual"),
    ("1:20", "1:24", "qc_v36 manual"),
    ("2:19", "2:22", "qc_v36 manual"),
    ("3:11", "3:15", "qc_v36 scene change to person about to fall"),
    ("4:03", "4:06", "qc_v36 manual"),
    ("4:36", "4:39", "qc_v36 person about to fall"),
    ("4:56", "5:00", "qc_v36 manual"),
    ("5:12", "5:15", "qc_v36 person about to fall"),
    ("5:35", "5:39", "qc_v36 manual"),
    ("5:55", "5:59", "qc_v36 manual"),
    ("6:40", "6:44", "qc_v36 manual"),
    ("7:01", "7:05", "qc_v36 manual"),
    ("7:07", "7:09", "qc_v36 manual"),
    ("7:18", "7:22", "qc_v36 manual"),
    ("7:50", "7:53", "qc_v36 person about to fall"),
    ("8:02", "8:05", "qc_v36 manual"),
    ("9:38", "9:41", "qc_v36 manual"),
    ("10:17", "10:20", "qc_v36 manual"),
    ("10:45", "10:49", "qc_v36 manual"),
    ("12:16", "12:19", "qc_v36 manual"),
    ("13:04", "13:06", "qc_v36 manual"),
    ("13:08", "13:11", "qc_v36 cut before clip change"),
    ("13:27", "13:31", "qc_v36 manual"),
    ("14:14", "14:17", "qc_v36 manual"),
    ("15:14", "15:17", "qc_v36 cut before scene change"),
    ("15:35", "15:39", "qc_v36 manual"),
    ("15:40", "15:44", "qc_v36 manual"),
    ("16:52", "16:56", "qc_v36 manual"),
    ("17:26", "17:30", "qc_v36 after scene cut, end before next scene"),
    ("17:58", "18:01", "qc_v36 manual"),
    ("18:03", "18:05", "qc_v36 person on ground before scene exit"),
    ("18:27", "18:30", "qc_v36 start after scene change"),
    ("18:46", "18:49", "qc_v36 manual"),
    ("19:09", "19:12", "qc_v36 manual"),
    ("19:25", "19:28", "qc_v36 after scene change, person about to fall"),
    ("19:33", "19:35", "qc_v36 manual"),
    ("20:32", "20:35", "qc_v36 manual"),
    ("20:55", "20:57", "qc_v36 manual"),
    ("21:29", "21:32", "qc_v36 manual"),
    ("22:16", "22:19", "qc_v36 cut before next clip"),
    ("22:42", "22:46", "qc_v36 manual"),
    ("23:15", "23:18", "qc_v36 manual"),
    ("25:14", "25:17", "qc_v36 manual"),
    ("25:51", "25:55", "qc_v36 manual"),
    ("26:09", "26:11", "qc_v36 manual"),
    ("26:12", "26:15", "qc_v36 manual"),
    ("26:58", "27:03", "qc_v36 manual"),
    ("27:14", "27:17", "qc_v36 manual"),
    ("27:28", "27:32", "qc_v36 manual"),
    ("28:10", "28:14", "qc_v36 manual"),
    ("28:36", "28:38", "qc_v36 manual"),
    ("29:52", "29:55", "qc_v36 manual"),
    ("30:08", "30:11", "qc_v36 manual"),
    ("31:17", "31:20", "qc_v36 manual"),
    ("31:21", "31:24", "qc_v36 manual"),
    ("31:54", "31:57", "qc_v36 manual"),
    ("32:28", "32:32", "qc_v36 manual"),
    ("33:14", "33:16", "qc_v36 manual"),
    ("33:53", "33:56", "qc_v36 manual"),
    ("35:28", "35:31", "qc_v36 manual"),
    ("35:51", "35:55", "qc_v36 manual"),
]


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


def load_scenes(folder: Path) -> list[tuple[int, int]]:
    p = folder / "scene_cut_manifest.json"
    if not p.exists():
        return []
    entries = json.loads(p.read_text())
    return [(int(e["segment_start_frame"]), int(e["segment_end_frame"])) for e in entries]


def scene_at(scenes, frame: int) -> tuple[int, int, int] | None:
    for i, (s, e) in enumerate(scenes):
        if s <= frame < e:
            return i, s, e
    return None


def scene_before(scenes, frame: int) -> tuple[int, int, int] | None:
    best = None
    for i, (s, e) in enumerate(scenes):
        if s <= frame:
            best = (i, s, e)
    return best


def next_scene(scenes, frame: int) -> tuple[int, int, int] | None:
    for i, (s, e) in enumerate(scenes):
        if s > frame:
            return i, s, e
    return None


def adjust_clip(start_f: int, end_f: int, note: str, scenes, fps: float) -> tuple[int, int]:
    note_l = note.lower()
    margin = 5

    if "scene change" in note_l or "after scene" in note_l or "after scene cut" in note_l:
        sc = scene_at(scenes, start_f) or scene_before(scenes, start_f)
        if sc:
            _, s, e = sc
            start_f = max(start_f, s + int(round(1.0 * fps)))
            end_f = min(end_f, e - margin)

    if "person about to fall" in note_l and "after scene" not in note_l:
        sc = scene_at(scenes, start_f) or scene_before(scenes, start_f)
        if sc:
            _, s, e = sc
            if start_f < s + margin:
                start_f = s + int(round(1.0 * fps))
            end_f = min(end_f, e - margin)

    if "pre-fall" in note_l:
        sc = scene_at(scenes, start_f) or scene_before(scenes, start_f)
        if sc:
            _, s, e = sc
            if start_f < s:
                start_f = s + int(round(0.8 * fps))
            end_f = min(end_f, e - margin)

    if "cut before" in note_l or "before scene" in note_l or "before clip" in note_l or "before next" in note_l:
        sc = scene_at(scenes, end_f) or scene_before(scenes, end_f)
        if sc:
            _, s, e = sc
            end_f = min(end_f, e - margin)

    if "after scene cut" in note_l and "end before" in note_l:
        sc = scene_at(scenes, start_f) or scene_before(scenes, start_f)
        if sc:
            _, s, e = sc
            start_f = max(start_f, s + int(round(1.0 * fps)))
            end_f = min(end_f, e - margin)

    if "on ground" in note_l:
        sc = scene_at(scenes, end_f) or scene_before(scenes, end_f)
        if sc:
            _, s, e = sc
            end_f = min(end_f, e - margin)

    if end_f <= start_f:
        end_f = start_f + int(round(2.5 * fps))

    # never shrink below ~2s if user asked for longer window
    user_dur = end_f - start_f
    if user_dur < int(round(2.0 * fps)):
        pass  # already extended above
    return start_f, end_f


# Manual frame overrides where auto scene logic over-clamped short scenes
FRAME_OVERRIDES: dict[str, tuple[int, int]] = {
    "fall_003.mp4": (1734, 1797),   # 0:58-1:00 pre-fall start +1s
    "fall_007.mp4": (5755, 5835),   # scene_030 +1s -> 3:12-3:15
    "fall_011.mp4": (9329, 9418),   # 5:12-5:15 cut first frame
    "fall_028.mp4": (27348, 27408), # 15:15-15:17 old end through fall, before scene_142
    "fall_032.mp4": (31320, 31390), # 17:27-17:30 after scene_160 cut
    "fall_035.mp4": (33148, 33189), # 18:28-18:30 after scene_168 cut
    "fall_038.mp4": (34873, 34922), # 19:27-19:28 after scene_177 cut
}


def mmss(frame: int, fps: float) -> str:
    s = frame / fps
    m = int(s // 60)
    sec = int(round(s - m * 60))
    return f"{m}:{sec:02d}"


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/fall_dataset_clips/035_V36"
    )
    folder = folder.resolve()
    source = folder / "source.mp4"
    if not source.exists():
        print(f"Missing source: {source}")
        sys.exit(1)

    fps = get_fps(source)
    scenes = load_scenes(folder)
    manual, manifest = [], []

    for i, (start_s, end_s, note) in enumerate(MANUAL_CLIPS, 1):
        name = f"fall_{i:03d}.mp4"
        start_f = int(round(parse_time(start_s) * fps))
        end_f = int(round(parse_time(end_s) * fps))
        if scenes:
            start_f, end_f = adjust_clip(start_f, end_f, note, scenes, fps)
        if name in FRAME_OVERRIDES:
            start_f, end_f = FRAME_OVERRIDES[name]
        manual.append({
            "clip": name,
            "start": mmss(start_f, fps),
            "end": mmss(end_f, fps),
            "start_frame": start_f,
            "end_frame": end_f,
            "note": note,
        })
        manifest.append({
            "clip": name,
            "mode": "manual_qc_v36",
            "start_frame": start_f,
            "end_frame": end_f,
            "duration_sec": round((end_f - start_f) / fps, 2),
            "note": note,
            "source_clip_v36": "user_manual",
        })

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Cutting {len(manual)} V36 manual clips @ {fps:.3f} fps")
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
