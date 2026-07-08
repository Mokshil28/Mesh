#!/usr/bin/env python3
"""Apply V05 manual QC: deletes, caps, scene offsets, motion-based trim."""
import json
import subprocess
import sys
from pathlib import Path

FPS = 24000 / 1001  # 23.976

DELETE = {
    1, 2, 5, 12, 13, 14, 38, 42, 43, 45, 48, 49, 50, 52, 53, 67, 68,
}

CAP_FROM_START = {28: 2.0}
START_PLUS_1 = {54: 1.0}  # skip scene change at start (bowler in scene)
EXTEND_END_1 = {66: 1.0}

# fall contact ~24040, +2s on ground, include pre-fall
CLIP_70 = {"start_pre_sec": 1.2, "post_contact_sec": 2.0}


def sec_to_frames(sec: float) -> int:
    return int(round(sec * FPS))


def cut_clip(source: Path, out: Path, start_f: int, end_f: int) -> None:
    start_sec = start_f / FPS
    duration = max((end_f - start_f) / FPS, 0.1)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_sec:.6f}", "-i", str(source),
        "-t", f"{duration:.6f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def clip_num(name: str) -> int:
    return int(name.replace("fall_", "").replace(".mp4", ""))


def frames_to_sec(frame: int) -> float:
    return round(frame / FPS, 3)


def apply_edit(num: int, e: dict, source: Path) -> tuple[int, int, str]:
    start = e["start_frame"]
    end = e["end_frame"]

    if num == 70:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from detect_fall_clips_frames import motion_profile, find_onset_contact_end

        region_s = max(0, start - sec_to_frames(2))
        region_e = end + sec_to_frames(3)
        fi, sc = motion_profile(source, region_s, region_e)
        onset, contact, motion_end, _, _ = find_onset_contact_end(
            fi, sc, region_s, region_e
        )
        pre = sec_to_frames(CLIP_70["start_pre_sec"])
        post = sec_to_frames(CLIP_70["post_contact_sec"])
        start = max(region_s, onset - pre)
        end = min(region_e, contact + post)
        return start, end, "qc_v05 fall +2s on ground, no scene change"

    if num in CAP_FROM_START:
        dur = CAP_FROM_START[num]
        end = start + sec_to_frames(dur)
        return start, end, f"qc_v05 cap {dur}s from clip start"

    if num in START_PLUS_1:
        off = START_PLUS_1[num]
        start = start + sec_to_frames(off)
        return start, end, f"qc_v05 start +{off}s skip scene cut"

    if num in EXTEND_END_1:
        end = end + sec_to_frames(EXTEND_END_1[num])
        return start, end, "qc_v05 extend +1s at end"

    return start, end, e.get("note", "")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/fall_dataset_clips/005_V05"
    )
    folder = folder.resolve()
    source = folder / "source.mp4"
    if not source.exists():
        print(f"Missing {source}")
        sys.exit(1)

    manifest = json.loads((folder / "manifest.json").read_text())
    by_num = {clip_num(e["clip"]): e for e in manifest}

    rejected = []
    kept = []

    for num in sorted(by_num):
        e = by_num[num]
        if num in DELETE:
            rejected.append({
                "clip": e["clip"],
                "source_clip": f"fall_{num:03d}.mp4",
                "reason": "user_qc_delete_v05",
            })
            continue

        start, end, note = apply_edit(num, e, source)
        kept.append({
            **e,
            "start_frame": start,
            "end_frame": end,
            "start_sec": frames_to_sec(start),
            "end_sec": frames_to_sec(end),
            "duration_sec": round((end - start) / FPS, 3),
            "note": note,
            "mode": "fall_trim_manual_qc",
            "_old_num": num,
        })

    manual = []
    new_manifest = []
    for i, e in enumerate(kept, 1):
        name = f"fall_{i:03d}.mp4"
        old = e.pop("_old_num")
        manual.append({
            "clip": name,
            "start_frame": e["start_frame"],
            "end_frame": e["end_frame"],
            "note": e["note"],
        })
        entry = {k: v for k, v in e.items() if k != "clip"}
        entry["clip"] = name
        entry["source_clip_v05"] = f"fall_{old:03d}.mp4"
        new_manifest.append(entry)

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(new_manifest, indent=2) + "\n")

    rej_path = folder / "rejected_clips.json"
    prev = json.loads(rej_path.read_text()) if rej_path.exists() else []
    rej_path.write_text(json.dumps(prev + rejected, indent=2) + "\n")

    print(f"Deleted {len(DELETE)} | Kept {len(manual)}")

    ok = 0
    for entry in manual:
        out = folder / entry["clip"]
        try:
            cut_clip(source, out, entry["start_frame"], entry["end_frame"])
            ok += 1
            dur = (entry["end_frame"] - entry["start_frame"]) / FPS
            print(f"  OK {entry['clip']}  {dur:.2f}s  {entry['note']}")
        except subprocess.CalledProcessError as exc:
            print(f"  FAIL {entry['clip']}: {exc}")

    print(f"Done: {ok}/{len(manual)}")


if __name__ == "__main__":
    main()
