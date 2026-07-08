#!/usr/bin/env python3
"""Detect fall clips via scene cuts + motion, write frame timings, cut with ffmpeg."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Fall-arc trimming: one scene = one fall, end right after rest (2-4s typical)
SCENE_EDGE_MARGIN = 5
PRE_ONSET_SEC = 0.8       # brief pre-fall motion
POST_REST_SEC = 0.8       # brief rest on ground after fall
MIN_CLIP_SEC = 2.0
MAX_CLIP_SEC = 4.0        # typical clip length cap
FULL_SCENE_MAX_SEC = 3.0  # only keep whole shot if very short
MAX_FALL_ARC_SEC = 4.0
REST_SEARCH_SEC = 2.5     # look for settle soon after impact
REST_LOW_FRAC = 0.30
REST_HOLD_SEC = 0.35
SECOND_PEAK_FRAC = 0.42   # cut before a second fall / camera-event spike
MIN_SCENE_SEC = 1.5
SCENE_THRESHOLD = 0.35
MAX_ONSET_SCENE_FRAC = 0.85   # AFV falls often land 4-6s into a short scene
MAX_ONSET_SEC = 12.0          # AFV scenes can be 10-14s; fall may land 8-10s in
MIN_RISE_RATIO = 1.10         # softer — slow/staged falls have gentler rise
MIN_PEAK_BASELINE_RATIO = 1.25
MAX_FALL_SPAN_SEC = 3.0       # allow slower topples
MIN_FALL_SCORE = 120.0        # accept strong motion peaks even if rise is soft
REJECT_SCENE_IDS: set[int] = set()


def get_fps(video: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,nb_frames,duration",
            "-of", "json", str(video),
        ],
        text=True,
    )
    info = json.loads(out)["streams"][0]
    rate = info["r_frame_rate"]
    if "/" in rate:
        num, den = rate.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(rate)
    return fps


def get_frame_count(video: Path, fps: float) -> int:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", str(video),
        ],
        text=True,
    ).strip()
    if out.isdigit():
        return int(out)
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n > 0:
        return n
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)], text=True).strip())
    return int(dur * fps)


def detect_scene_cuts(video: Path, fps: float, total_frames: int) -> list[int]:
    """Return frame indices where a new scene starts (after frame 0)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-i", str(video),
        "-filter:v", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    cuts = [0]
    for line in proc.stderr.splitlines():
        if "pts_time:" in line:
            for part in line.split():
                if part.startswith("pts_time:"):
                    t = float(part.split(":")[1])
                    frame = int(round(t * fps))
                    if frame > cuts[-1] + sec_frames(MIN_SCENE_SEC, fps):
                        cuts.append(frame)
    if cuts[-1] < total_frames - sec_frames(MIN_SCENE_SEC, fps):
        cuts.append(total_frames)
    return cuts


def motion_profile(video: Path, start_f: int, end_f: int, sample_step: int = 2) -> tuple[list[int], list[float]]:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames_idx = []
    scores = []
    prev_gray = None
    f = start_f
    while f < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (f - start_f) % sample_step == 0:
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                h = diff.shape[0]
                weights = np.ones(h, dtype=np.float32)
                weights[h // 3:] *= 1.8
                weighted = diff * weights[:, None]
                score = float(np.mean(weighted))
                frames_idx.append(f)
                scores.append(score)
            prev_gray = gray
        else:
            cap.grab()
        f += 1
    cap.release()
    return frames_idx, scores


def find_onset_contact_end(
    frames_idx: list[int],
    scores: list[float],
    scene_start: int,
    scene_end: int,
    fps: float = 30.0,
) -> tuple[int, int, int, float, bool]:
    """Find fall motion; prefer earliest strong peak in scene. Returns late_misdetect flag."""
    if len(scores) < 3:
        mid = frames_idx[len(frames_idx) // 2] if frames_idx else scene_start
        return mid, mid, mid, 0.0, False

    arr = np.array(scores, dtype=np.float32)
    smooth = smooth_scores(scores.tolist() if hasattr(scores, 'tolist') else list(scores))
    global_max = float(np.max(smooth))
    if global_max <= 0:
        mid = frames_idx[len(frames_idx) // 2]
        return mid, mid, mid, 0.0, False

    scene_len = max(scene_end - scene_start, 1)
    onset_cutoff = min(
        scene_start + int(scene_len * MAX_ONSET_SCENE_FRAC),
        scene_start + sec_frames(MAX_ONSET_SEC, fps),
    )

    # Strongest local maximum in the searchable window (skip opening noise)
    baseline = float(np.median(smooth[: max(3, len(smooth) // 4)]))
    peak_thresh = max(baseline * MIN_PEAK_BASELINE_RATIO, global_max * 0.25, 1.0)
    skip_until = scene_start + sec_frames(1.0, fps)
    peak_i = None
    best_val = 0.0
    for i in range(1, len(smooth) - 1):
        frame = frames_idx[i]
        if frame > onset_cutoff:
            break
        if frame < skip_until:
            continue
        if smooth[i] < peak_thresh:
            continue
        if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            if float(smooth[i]) > best_val:
                best_val = float(smooth[i])
                peak_i = i

    if peak_i is None:
        candidates = [i for i, f in enumerate(frames_idx) if f <= onset_cutoff]
        if not candidates:
            mid = frames_idx[len(frames_idx) // 2]
            return mid, mid, mid, 0.0, True
        local = smooth[candidates]
        peak_local_i = int(np.argmax(local))
        peak_i = candidates[peak_local_i]
        if float(local[peak_local_i]) < peak_thresh:
            return frames_idx[candidates[0]], frames_idx[candidates[0]], frames_idx[candidates[0]], 0.0, True

    contact = frames_idx[peak_i]
    peak_val = float(smooth[peak_i])
    onset_rel = (contact - scene_start) / scene_len
    late_misdetect = (
        onset_rel > MAX_ONSET_SCENE_FRAC
        or (contact - scene_start) > sec_frames(MAX_ONSET_SEC, fps)
    )

    onset_thresh = peak_val * 0.22
    onset_i = peak_i
    for i in range(peak_i, -1, -1):
        if smooth[i] < onset_thresh:
            onset_i = min(i + 1, peak_i)
            break
        onset_i = i
    onset = frames_idx[onset_i]

    end_thresh = peak_val * 0.20
    motion_end_i = peak_i
    for i in range(peak_i, len(smooth)):
        if smooth[i] >= end_thresh:
            motion_end_i = i
    motion_end = frames_idx[motion_end_i]

    return onset, contact, motion_end, peak_val * 10, late_misdetect


def sec_frames(sec: float, fps: float) -> int:
    return int(round(sec * fps))


def smooth_scores(scores: list[float]) -> np.ndarray:
    arr = np.array(scores, dtype=np.float32)
    if len(arr) < 3:
        return arr
    kernel = np.array([0.2, 0.6, 0.2])
    return np.convolve(arr, kernel, mode="same")


def nearest_index(frames_idx: list[int], frame: int) -> int:
    return min(range(len(frames_idx)), key=lambda i: abs(frames_idx[i] - frame))


def validate_fall_pattern(
    frames_idx: list[int],
    scores: list[float],
    onset: int,
    contact: int,
    scene_start: int,
    scene_end: int,
    fps: float,
    fall_score: float = 0.0,
) -> str | None:
    """Reject unless motion follows fall arc: pre-rise -> impact -> settle."""
    scene_len = scene_end - scene_start
    if scene_len < sec_frames(MIN_SCENE_SEC, fps):
        return "scene_too_short"

    contact_rel = (contact - scene_start) / max(scene_len, 1)
    if contact_rel > MAX_ONSET_SCENE_FRAC and (contact - scene_start) > sec_frames(MAX_ONSET_SEC, fps):
        return "fall_too_late_in_scene"

    if len(scores) < 4:
        return "insufficient_motion_samples"

    smooth = smooth_scores(scores)
    onset_i = nearest_index(frames_idx, onset)
    contact_i = nearest_index(frames_idx, contact)
    peak_val = float(smooth[contact_i])
    if peak_val <= 0:
        return "no_motion"

    before = [float(smooth[i]) for i, f in enumerate(frames_idx) if f < onset]
    baseline = float(np.median(before)) if before else peak_val * 0.2
    weak_peak = peak_val < max(baseline * MIN_PEAK_BASELINE_RATIO, 1.0)
    if weak_peak and fall_score < MIN_FALL_SCORE:
        return "weak_rise_from_baseline"

    onset_val = float(smooth[onset_i])
    if peak_val < onset_val * MIN_RISE_RATIO and fall_score < MIN_FALL_SCORE:
        return "no_impact_rise"

    fall_span = contact - onset
    if fall_span > sec_frames(MAX_FALL_SPAN_SEC, fps):
        return "fall_span_too_long"

    search_until = min(scene_end, contact + sec_frames(REST_SEARCH_SEC, fps))
    after = [i for i, f in enumerate(frames_idx) if contact <= f <= search_until]
    if len(after) < 2:
        if fall_score >= MIN_FALL_SCORE:
            return None
        return "no_post_impact_data"

    rest_thresh = peak_val * REST_LOW_FRAC
    hold_f = sec_frames(REST_HOLD_SEC, fps)
    low_run = 0
    prev_frame = contact
    settled = False
    for i in after:
        frame = frames_idx[i]
        step = max(frame - prev_frame, 1)
        prev_frame = frame
        if float(smooth[i]) <= rest_thresh:
            low_run += step
            if low_run >= hold_f:
                settled = True
                break
        else:
            low_run = 0
    if not settled and fall_score < MIN_FALL_SCORE:
        return "no_settle_after_impact"

    return None


def should_reject(
    scene_id: int,
    frames_idx: list[int],
    scores: list[float],
    onset: int,
    contact: int,
    scene_start: int,
    scene_end: int,
    fps: float,
    fall_score: float = 0.0,
) -> str | None:
    if scene_id in REJECT_SCENE_IDS:
        return "user_qc_reject"
    return validate_fall_pattern(
        frames_idx, scores, onset, contact, scene_start, scene_end, fps, fall_score
    )


def find_first_fall_end(
    frames_idx: list[int],
    scores: list[float],
    contact: int,
    scene_end: int,
    fps: float,
) -> int:
    """End after the first fall settles; cut before second fall or camera spike."""
    search_until = min(scene_end - SCENE_EDGE_MARGIN, contact + sec_frames(REST_SEARCH_SEC, fps))
    after = [i for i, f in enumerate(frames_idx) if contact <= f <= search_until]
    if not after:
        return min(scene_end - SCENE_EDGE_MARGIN, contact + sec_frames(1.2, fps))

    window_scores = [scores[i] for i in after]
    peak_val = float(max(window_scores))
    if peak_val <= 0:
        return min(scene_end - SCENE_EDGE_MARGIN, contact + sec_frames(1.2, fps))

    rest_thresh = peak_val * REST_LOW_FRAC
    second_thresh = peak_val * SECOND_PEAK_FRAC
    hold_f = sec_frames(REST_HOLD_SEC, fps)

    settle_idx = None
    low_run = 0
    prev_frame = contact
    for pos, i in enumerate(after):
        frame = frames_idx[i]
        step = max(frame - prev_frame, 1)
        prev_frame = frame
        if scores[i] <= rest_thresh:
            low_run += step
            if low_run >= hold_f:
                settle_idx = pos
                break
        else:
            low_run = 0

    if settle_idx is None:
        return min(scene_end - SCENE_EDGE_MARGIN, contact + sec_frames(1.2, fps))

    rest_frame = frames_idx[after[settle_idx]]
    for i in after[settle_idx + 1:]:
        if scores[i] >= second_thresh:
            cut = max(contact + sec_frames(0.5, fps), frames_idx[i] - sec_frames(0.25, fps))
            return min(cut, rest_frame + sec_frames(0.5, fps))
    return rest_frame


def find_rest_frame(
    frames_idx: list[int],
    scores: list[float],
    contact: int,
    motion_end: int,
    scene_end: int,
    fps: float,
) -> int:
    """Frame where the first fall motion has settled."""
    return find_first_fall_end(frames_idx, scores, contact, scene_end, fps)


def trim_clip_bounds(
    onset: int,
    contact: int,
    motion_end: int,
    scene_start: int,
    scene_end: int,
    fps: float,
    late_misdetect: bool = False,
    frames_idx: list[int] | None = None,
    scores: list[float] | None = None,
) -> tuple[int, int]:
    """One fall per clip: brief pre-fall, full fall, brief rest; 2-4s typical."""
    margin = SCENE_EDGE_MARGIN
    scene_len = scene_end - scene_start
    min_f = sec_frames(MIN_CLIP_SEC, fps)
    max_f = sec_frames(MAX_CLIP_SEC, fps)
    full_max_f = sec_frames(FULL_SCENE_MAX_SEC, fps)
    pre_f = sec_frames(PRE_ONSET_SEC, fps)
    post_rest_f = sec_frames(POST_REST_SEC, fps)

    if scene_len <= full_max_f:
        return scene_start + margin, scene_end - margin

    if frames_idx and scores:
        rest = find_first_fall_end(frames_idx, scores, contact, scene_end, fps)
    else:
        rest = min(scene_end - margin, contact + sec_frames(1.2, fps))

    # Start at scene cut when fall is within the scene (skip previous-clip lead-in)
    scene_pre_f = sec_frames(0.35, fps)
    start = max(scene_start + margin, scene_start + scene_pre_f)
    end = min(scene_end - margin, rest + post_rest_f)
    end = max(end, contact + sec_frames(0.5, fps))
    end = min(end, scene_end - margin, start + max_f)

    if end - start < min_f:
        need = min_f - (end - start)
        start = max(scene_start + margin, start - need // 2)
        end = min(scene_end - margin, end + (need - need // 2), start + max_f)

    return start, end


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
    if len(sys.argv) < 2:
        print("Usage: detect_fall_clips_frames.py <clip_folder> [source.mp4]")
        sys.exit(1)

    folder = Path(sys.argv[1]).resolve()
    folder.mkdir(parents=True, exist_ok=True)

    old_manifest = folder / "manifest.json"
    if old_manifest.exists():
        backup = folder / "scene_cut_manifest.json"
        if not backup.exists():
            old_manifest.rename(backup)
            print(f"Backed up old manifest -> {backup.name}")

    source = Path(sys.argv[2]) if len(sys.argv) > 2 else folder / "source.mp4"
    if not source.exists():
        print(f"Missing source: {source}")
        sys.exit(1)

    fps = get_fps(source)
    total = get_frame_count(source, fps)
    duration_min = total / fps / 60
    print(f"Source: {source.name} | {fps:.3f} fps | {total} frames | {duration_min:.1f} min")

    print("Detecting scene cuts...")
    cuts = detect_scene_cuts(source, fps, total)
    scenes = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    min_scene_f = sec_frames(MIN_SCENE_SEC, fps)
    scenes = [(s, e) for s, e in scenes if e - s >= min_scene_f]
    print(f"Found {len(scenes)} scenes")

    manual = []
    manifest = []
    rejected = []
    clip_num = 0

    for si, (scene_start, scene_end) in enumerate(scenes, 1):
        frames_idx, scores = motion_profile(source, scene_start, scene_end)
        if not frames_idx:
            rejected.append({"scene": si, "reason": "no_motion_data"})
            continue

        onset, contact, motion_end, fall_score, late_misdetect = find_onset_contact_end(
            frames_idx, scores, scene_start, scene_end, fps
        )
        reject_reason = should_reject(
            si, frames_idx, scores, onset, contact, scene_start, scene_end, fps, fall_score
        )
        if reject_reason:
            rejected.append({
                "scene": si,
                "reason": reject_reason,
                "onset_frame": onset,
                "contact_frame": contact,
                "fall_score": round(fall_score, 1),
            })
            continue

        if frames_idx and scores:
            rest_frame = find_rest_frame(
                frames_idx, scores, contact, motion_end, scene_end, fps
            )
        else:
            rest_frame = motion_end

        start_f, end_f = trim_clip_bounds(
            onset, contact, motion_end, scene_start, scene_end, fps,
            late_misdetect, frames_idx, scores,
        )

        clip_num += 1
        name = f"fall_{clip_num:03d}.mp4"
        manual.append({
            "clip": name,
            "start_frame": start_f,
            "end_frame": end_f,
            "note": f"scene_{si:03d} pre_fall_to_rest",
        })
        manifest.append({
            "clip": name,
            "mode": "fall_trim_auto_v9",
            "source_scene": f"scene_{si:03d}",
            "scene_start_frame": scene_start,
            "scene_end_frame": scene_end,
            "onset_frame": onset,
            "contact_frame": contact,
            "motion_end_frame": motion_end,
            "rest_frame": rest_frame,
            "late_misdetect": late_misdetect,
            "start_frame": start_f,
            "end_frame": end_f,
            "onset_sec": round(onset / fps, 2),
            "contact_sec": round(contact / fps, 2),
            "duration_sec": round((end_f - start_f) / fps, 2),
            "fall_score": round(fall_score, 1),
        })

    manual_path = folder / "manual_timings.json"
    manifest_path = folder / "manifest.json"
    rejected_path = folder / "rejected_clips.json"
    manual_path.write_text(json.dumps(manual, indent=2) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    rejected_path.write_text(json.dumps(rejected, indent=2) + "\n")
    print(f"Wrote {len(manual)} clips, rejected {len(rejected)} scenes")

    # Remove stale clips from previous run
    for old in folder.glob("fall_*.mp4"):
        old.unlink()

    ok = 0
    for entry in manual:
        out = folder / entry["clip"]
        try:
            cut_clip(source, out, entry["start_frame"], entry["end_frame"], fps)
            ok += 1
            dur = (entry["end_frame"] - entry["start_frame"]) / fps
            print(f"  OK {entry['clip']}  {dur:.2f}s  frames {entry['start_frame']}-{entry['end_frame']}")
        except subprocess.CalledProcessError as exc:
            print(f"  FAIL {entry['clip']}: {exc}")

    print(f"Done: {ok}/{len(manual)} clips in {folder}")


if __name__ == "__main__":
    main()
