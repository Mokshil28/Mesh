#!/usr/bin/env python3
"""Apply V14 manual QC: deletes, merges, scene windows, extends."""
import json
import subprocess
import sys
from pathlib import Path

FPS = 25.0
MARGIN = 5

DELETE = {
    5, 9, 17, 18, 19, 20, 21, 31, 32, 33, 35, 36, 37, 38, 39,
    41, 42, 43, 44, 45, 48, 51, 52, 53, 54, 58, 65,
    75, 76, 77, 78, 79, 82, 83, 84, 85, 92, 95, 96, 98,
    99, 100, 101, 102, 107, 111, 114, 115, 116, 117, 118,
    122, 126, 128, 129, 131, 134,
    137, 138, 139, 140, 141, 142,
    148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
    164, 168, 171, 172, 173, 174, 175, 177, 184, 188,
}

# Clip cut / missing fall motion — use full scene window
FULL_SCENE = {
    2, 3, 6, 8, 10, 23, 24, 25, 27, 29,
    60, 61, 62, 64, 66, 67, 68, 70, 71, 73, 74,
    80, 81, 86, 87, 88, 89, 90, 91, 94,
    103, 106, 109, 110, 112, 113, 119, 121, 125, 127,
    133, 135, 136, 146,
    165, 166, 167, 169, 179, 180,
}

START_MINUS_1 = {30, 34, 130}
EXTEND_END_1 = {40, 59, 69, 72, 147, 163, 181}
SCENE_PLUS_1_END_PLUS_1 = {176, 178}
SCENE_START_OFFSET = {182: 1.0}
CAP_FROM_START = {183: 4.0}

MERGE_GROUPS = [
    ([49, 50], "qc_v14 merge 49+50 fall motion"),
    ([55, 56], "qc_v14 merge 55+56 fall motion"),
    ([143, 144, 145], "qc_v14 merge/fix 143-145 fall motion"),
]


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


def scene_bounds(entries: list[dict]) -> tuple[int, int]:
    ss = min(e["scene_start_frame"] for e in entries)
    se = max(e["scene_end_frame"] for e in entries)
    return ss + MARGIN, se - MARGIN


def apply_edit(num: int, e: dict) -> tuple[int, int, str]:
    ss = e["scene_start_frame"]
    se = e["scene_end_frame"]
    start = e["start_frame"]
    end = e["end_frame"]

    if num in FULL_SCENE:
        return ss + MARGIN, se - MARGIN, "qc_v14 full scene fall motion"

    if num in SCENE_START_OFFSET:
        off = SCENE_START_OFFSET[num]
        start = ss + sec_to_frames(off)
        end = min(se - MARGIN, end + sec_to_frames(1.0))
        return start, end, f"qc_v14 scene +{off}s skip cut, +1s end"

    if num in SCENE_PLUS_1_END_PLUS_1:
        start = ss + sec_to_frames(1.0)
        end = min(se - MARGIN, end + sec_to_frames(1.0))
        return start, end, "qc_v14 scene +1s start, +1s end"

    for dur, nums in ((CAP_FROM_START[k], {k}) for k in CAP_FROM_START):
        if num in nums:
            end = min(start + sec_to_frames(dur), se - MARGIN)
            return start, end, f"qc_v14 cap {dur}s from clip start"

    if num in START_MINUS_1:
        start = max(ss + MARGIN, start - sec_to_frames(1.0))
        return start, end, "qc_v14 start -1s"

    if num in EXTEND_END_1:
        end = min(se - MARGIN, end + sec_to_frames(1.0))
        return start, end, "qc_v14 extend +1s at end"

    return start, end, e.get("note", "")


def build_merge_entry(nums: list[int], by_num: dict, note: str) -> dict:
    entries = [by_num[n] for n in nums]
    start, end = scene_bounds(entries)
    base = {**entries[0]}
    base["scene_start_frame"] = min(e["scene_start_frame"] for e in entries)
    base["scene_end_frame"] = max(e["scene_end_frame"] for e in entries)
    base["source_scene"] = "+".join(e["source_scene"] for e in entries)
    base["start_frame"] = start
    base["end_frame"] = end
    base["duration_sec"] = round((end - start) / FPS, 2)
    base["note"] = note
    base["mode"] = "fall_trim_manual_qc"
    base["_merged_from"] = nums
    return base


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/fall_dataset_clips/013_V14"
    )
    folder = folder.resolve()
    source = folder / "source.mp4"
    manifest = json.loads((folder / "manifest.json").read_text())
    by_num = {clip_num(e["clip"]): e for e in manifest}

    merged_nums: set[int] = set()
    for nums, _ in MERGE_GROUPS:
        merged_nums.update(nums)

    rejected = []
    kept: list[dict] = []

    for nums, note in MERGE_GROUPS:
        if any(n in DELETE for n in nums):
            for n in nums:
                if n in by_num:
                    rejected.append({
                        "clip": by_num[n]["clip"],
                        "source_clip": f"fall_{n:03d}.mp4",
                        "reason": "user_qc_delete_v14_partial_merge",
                    })
            continue
        entry = build_merge_entry(nums, by_num, note)
        entry["_old_num"] = nums[0]
        kept.append(entry)

    for num in sorted(by_num):
        if num in merged_nums or num in DELETE:
            if num in DELETE:
                rejected.append({
                    "clip": by_num[num]["clip"],
                    "source_clip": f"fall_{num:03d}.mp4",
                    "reason": "user_qc_delete_v14",
                })
            continue

        e = by_num[num]
        start, end, note = apply_edit(num, e)
        kept.append({
            **e,
            "start_frame": start,
            "end_frame": end,
            "duration_sec": round((end - start) / FPS, 2),
            "note": note,
            "mode": "fall_trim_manual_qc",
            "_old_num": num,
        })

    # preserve source order by old clip number
    kept.sort(key=lambda x: x["_old_num"])

    manual = []
    new_manifest = []
    for i, e in enumerate(kept, 1):
        name = f"fall_{i:03d}.mp4"
        old = e.pop("_old_num")
        merged = e.pop("_merged_from", None)
        manual.append({
            "clip": name,
            "start_frame": e["start_frame"],
            "end_frame": e["end_frame"],
            "note": e["note"],
        })
        entry = {k: v for k, v in e.items() if k != "clip"}
        entry["clip"] = name
        if merged:
            entry["source_clip_v14"] = [f"fall_{n:03d}.mp4" for n in merged]
        else:
            entry["source_clip_v14"] = f"fall_{old:03d}.mp4"
        new_manifest.append(entry)

    for f in folder.glob("fall_*.mp4"):
        f.unlink()

    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(new_manifest, indent=2) + "\n")

    rej_path = folder / "rejected_clips.json"
    prev = json.loads(rej_path.read_text()) if rej_path.exists() else []
    rej_path.write_text(json.dumps(prev + rejected, indent=2) + "\n")

    print(f"Deleted {len(DELETE)} | Merged groups {len(MERGE_GROUPS)} | Kept {len(manual)}")

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
