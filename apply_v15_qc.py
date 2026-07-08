#!/usr/bin/env python3
"""Apply V15 manual QC: deletes, custom mm:ss windows, new source clips."""
import json
import re
import subprocess
import sys
from pathlib import Path

DELETE = {
    2, 3, 4, 5, 7, 8, 9, 12, 13, 14,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    32, 34, 35, 37, 38,
}

# Existing clip number -> (start, end) in source video
CUSTOM_TIMES = {
    15: ("3:00", "3:07"),
    28: ("4:15", "4:18"),
    29: ("5:00", "5:03"),
    33: ("5:27", "5:30"),
    36: ("5:55", "5:59"),
    40: ("6:21", "6:24"),
}

# New clips cut directly from source (not from prior fall_*.mp4)
NEW_CLIPS = [
    ("7:31", "7:35", "qc_v15 manual 7:31-7:35"),
    ("9:11", "9:16", "qc_v15 manual 9:11-9:16"),
    ("11:24", "11:28", "qc_v15 manual 11:24-11:28"),
    ("14:00", "14:05", "qc_v15 manual 14:00-14:05"),
    ("15:59", "16:04", "qc_v15 manual 15:59-16:04"),
    ("16:35", "16:38", "qc_v15 manual 16:35-16:38"),
    ("17:18", "17:22", "qc_v15 manual 17:18-17:22"),
    ("17:25", "17:29", "qc_v15 manual 17:25-17:29"),
    ("18:08", "18:12", "qc_v15 manual 18:08-18:12"),
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


def clip_num(name: str) -> int:
    return int(name.replace("fall_", "").replace(".mp4", ""))


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
        "data/fall_dataset_clips/014_V15"
    )
    folder = folder.resolve()
    source = folder / "source.mp4"
    fps = get_fps(source)
    manifest = json.loads((folder / "manifest.json").read_text())
    by_num = {clip_num(e["clip"]): e for e in manifest}

    rejected = []
    kept: list[dict] = []

    for num in sorted(by_num):
        if num in DELETE:
            rejected.append({
                "clip": by_num[num]["clip"],
                "source_clip": f"fall_{num:03d}.mp4",
                "reason": "user_qc_delete_v15",
            })
            continue

        keep_from_1_30 = 1 <= num <= 30 and num not in DELETE
        keep_custom = num in CUSTOM_TIMES
        if not keep_from_1_30 and not keep_custom:
            rejected.append({
                "clip": by_num[num]["clip"],
                "source_clip": f"fall_{num:03d}.mp4",
                "reason": "user_qc_drop_v15_not_in_keep_list",
            })
            continue

        e = by_num[num]
        if num in CUSTOM_TIMES:
            start_s, end_s = CUSTOM_TIMES[num]
            note = f"qc_v15 custom {start_s}-{end_s}"
            entry = {
                **e,
                "start_frame": int(round(parse_time(start_s) * fps)),
                "end_frame": int(round(parse_time(end_s) * fps)),
                "note": note,
                "mode": "fall_trim_manual_qc",
                "_old_num": num,
                "_start": start_s,
                "_end": end_s,
            }
        else:
            entry = {
                **e,
                "note": e.get("note", ""),
                "mode": "fall_trim_auto_v6",
                "_old_num": num,
            }
        kept.append(entry)

    for idx, (start_s, end_s, note) in enumerate(NEW_CLIPS):
        kept.append({
            "clip": "_new",
            "mode": "manual_qc_v15",
            "start_frame": int(round(parse_time(start_s) * fps)),
            "end_frame": int(round(parse_time(end_s) * fps)),
            "note": note,
            "_old_num": 20000 + idx,
            "_start": start_s,
            "_end": end_s,
            "_is_new": True,
        })

    kept.sort(key=lambda x: x["_old_num"])

    manual = []
    new_manifest = []
    for i, e in enumerate(kept, 1):
        name = f"fall_{i:03d}.mp4"
        old = e.pop("_old_num")
        is_new = e.pop("_is_new", False)
        start_s = e.pop("_start", None)
        end_s = e.pop("_end", None)
        start_f = e["start_frame"]
        end_f = e["end_frame"]

        manual.append({
            "clip": name,
            "start_frame": start_f,
            "end_frame": end_f,
            **({"start": start_s, "end": end_s} if start_s else {}),
            "note": e.get("note", ""),
        })

        entry = {k: v for k, v in e.items() if k != "clip"}
        entry["clip"] = name
        entry["duration_sec"] = round((end_f - start_f) / fps, 2)
        if is_new:
            entry["source_clip_v15"] = "new_manual"
        else:
            entry["source_clip_v15"] = f"fall_{old:03d}.mp4"
        new_manifest.append(entry)

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(new_manifest, indent=2) + "\n")

    rej_path = folder / "rejected_clips.json"
    prev = json.loads(rej_path.read_text()) if rej_path.exists() else []
    rej_path.write_text(json.dumps(prev + rejected, indent=2) + "\n")

    print(f"Deleted/dropped {len(rejected)} | Kept {len(manual)} clips")

    ok = 0
    for entry in manual:
        out = folder / entry["clip"]
        try:
            cut_clip(source, out, entry["start_frame"], entry["end_frame"], fps)
            ok += 1
            dur = (entry["end_frame"] - entry["start_frame"]) / fps
            print(f"  OK {entry['clip']}  {dur:.2f}s  {entry.get('note', '')}")
        except subprocess.CalledProcessError as exc:
            print(f"  FAIL {entry['clip']}: {exc}")

    print(f"Done: {ok}/{len(manual)}")


if __name__ == "__main__":
    main()
