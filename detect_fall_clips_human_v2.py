#!/usr/bin/env python3
"""Fall-clip extractor v2 — person-track based, tuned to a strict fall framework.

Each kept clip follows: person in motion (about to fall) -> fall motion ->
impact -> complete fall on the ground (settled / at rest). Capped at 4-5 s.

Improvements over detect_fall_clips_human.py:
  1. Real rest detection  — finds where center velocity actually settles after
     impact (held below a threshold), instead of blindly using the track end.
  2. Aspect-ratio fall cue — bbox tall->wide flip (w/h grows) is used as a
     primary fall signal, catching falls toward/away from camera and rejecting
     crouch/walk false positives.
  3. Track-break repair   — association falls back to center-distance and bridges
     short gaps, so fast fall motion doesn't snap the track mid-fall.
  4. Finer timing         — FRAME_STRIDE 5->4 (drop to 3 on GPU for tighter cuts).

Requires conda env 4D-humans (detectron2). Reuses the detector + base helpers.

Usage:  python detect_fall_clips_human_v2.py <clip_folder> [source.mp4]
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from detect_fall_clips_frames import (
    SCENE_EDGE_MARGIN,
    detect_scene_cuts,
    get_fps,
    get_frame_count,
    sec_frames,
    cut_clip,
)
from detect_fall_clips_human import build_detector, detect_people, box_center

# --- clip framework (your spec) --------------------------------------------
PRE_ONSET_SEC = 0.7      # lead-in: person in motion, about to fall
POST_REST_SEC = 0.6      # brief hold once the fall is complete
MIN_CLIP_SEC = 2.0
MAX_CLIP_SEC = 5.0       # hard cap: clip must not exceed ~4-5 s
MIN_SCENE_SEC = 1.5

# --- detection / tracking ---------------------------------------------------
FRAME_STRIDE = 4         # every Nth frame; set 3 on GPU for finer timing
MIN_TRACK_LEN = 4
IOU_MATCH = 0.2          # primary association
DIST_MATCH_FRAC = 0.75   # fallback: center dist < frac * mean box diagonal
MAX_GAP_SAMPLES = 3      # bridge this many missed detections before ending a track

# --- fall acceptance --------------------------------------------------------
MIN_DROP_FRAC = 0.12     # vertical center drop as fraction of frame height
MIN_DROP_PX = 30.0       # absolute floor for the drop
MIN_ASPECT_GAIN = 1.4    # w/h must grow >= this OR ...
LYING_ASPECT = 0.9       # ... final bbox is roughly horizontal (w/h >= this)

# --- rest / settle detection ------------------------------------------------
REST_SEARCH_SEC = 2.5    # look this far past impact for a settle
REST_HOLD_SEC = 0.35     # low motion must hold this long to count as "at rest"
REST_SPEED_FRAC = 0.08   # settled when per-sample center move < frac of body height


# ---------------------------------------------------------------------------
# tracking with gap-bridging + distance fallback
# ---------------------------------------------------------------------------
def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _diag(box):
    return float(np.hypot(box[2] - box[0], box[3] - box[1]))


def _match_score(prev_box, box):
    """1.0-ish for a good match, 0 for no match. IoU first, distance fallback."""
    iou = _iou(prev_box, box)
    if iou >= IOU_MATCH:
        return 1.0 + iou
    d = np.linalg.norm(box_center(prev_box) - box_center(box))
    gate = DIST_MATCH_FRAC * 0.5 * (_diag(prev_box) + _diag(box))
    if d <= gate and gate > 0:
        return 1.0 - d / gate      # 0..1, lower than any IoU match
    return -1.0


def build_tracks(frame_boxes):
    """frame_boxes: list of (frame_idx, boxes). Returns list of tracks.

    A track is a list of (frame_idx, box). Tracks survive up to MAX_GAP_SAMPLES
    missed detections so a fast fall (low IoU between frames) is not split.
    """
    active = []   # dicts: {pts: [(f,box)], last: box, miss: int}
    done = []
    for frame_idx, boxes in frame_boxes:
        used = set()
        # greedily match existing tracks to the best available box
        order = sorted(active, key=lambda t: -len(t["pts"]))
        for tr in order:
            best_j, best_s = -1, 0.0
            for j, box in enumerate(boxes):
                if j in used:
                    continue
                s = _match_score(tr["last"], box)
                if s > best_s:
                    best_s, best_j = s, j
            if best_j >= 0:
                tr["pts"].append((frame_idx, boxes[best_j]))
                tr["last"] = boxes[best_j]
                tr["miss"] = 0
                used.add(best_j)
            else:
                tr["miss"] += 1
        # retire tracks that have gone missing too long
        still = []
        for tr in active:
            if tr["miss"] > MAX_GAP_SAMPLES:
                done.append(tr)
            else:
                still.append(tr)
        active = still
        # unmatched boxes seed new tracks
        for j, box in enumerate(boxes):
            if j not in used:
                active.append({"pts": [(frame_idx, box)], "last": box, "miss": 0})
    done.extend(active)
    return [tr["pts"] for tr in done]


def box_metrics(box):
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5, w, h


def pick_falling_track(tracks):
    """Prefer the track with the most downward motion AND tall->wide change."""
    best, best_score = None, -1e9
    for tr in tracks:
        if len(tr) < MIN_TRACK_LEN:
            continue
        m = np.array([box_metrics(b) for _, b in tr], dtype=np.float32)
        cy, w, h = m[:, 1], m[:, 2], m[:, 3]
        dy = np.diff(cy)
        downward = float(np.sum(np.clip(dy, 0, None)))
        drop = float(cy[-1] - cy[0])
        aspect0 = float(w[0] / h[0])
        aspect1 = float(w[-1] / h[-1])
        aspect_gain = aspect1 - aspect0
        score = downward + 0.75 * drop + 60.0 * max(aspect_gain, 0.0) + 0.05 * len(tr)
        if score > best_score:
            best_score, best = score, tr
    return best


# ---------------------------------------------------------------------------
# fall arc: onset -> impact -> settled (complete fall)
# ---------------------------------------------------------------------------
def analyze_track(tr, fps, frame_h):
    """Return (reason|None, meta) describing the fall arc along a track."""
    frames = np.array([f for f, _ in tr], dtype=np.int32)
    m = np.array([box_metrics(b) for _, b in tr], dtype=np.float32)
    cx, cy, w, h = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
    n = len(frames)
    if n < MIN_TRACK_LEN:
        return "track_too_short", {}

    dy = np.diff(cy)                       # +ve = moving down
    max_dy = float(np.max(dy)) if len(dy) else 0.0
    med_h = float(np.median(h))
    samp_dt = FRAME_STRIDE / fps           # seconds between track samples

    # impact = frame of peak downward speed
    impact_c = int(np.argmax(dy)) + 1 if len(dy) else n - 1
    impact_c = min(impact_c, n - 1)

    # onset = start of the contiguous downward run before impact
    onset_thr = max(0.2 * max_dy, 0.02 * med_h)
    onset_c = impact_c
    c = impact_c
    while c >= 1 and dy[c - 1] >= onset_thr:
        onset_c = c - 1
        c -= 1

    # rest = first place center velocity settles low and holds, after impact
    hold_samples = max(1, int(round(REST_HOLD_SEC / samp_dt)))
    search_end_frame = frames[impact_c] + sec_frames(REST_SEARCH_SEC, fps)
    rest_c = None
    run_start, run = None, 0
    for c in range(impact_c + 1, n):
        if frames[c] > search_end_frame:
            break
        move = float(np.hypot(cx[c] - cx[c - 1], cy[c] - cy[c - 1]))
        if move <= REST_SPEED_FRAC * h[c]:
            if run == 0:
                run_start = c
            run += 1
            if run >= hold_samples:
                rest_c = run_start
                break
        else:
            run = 0
    if rest_c is None:
        # never settled (e.g. person rising again) -> cut at end of search window
        within = [c for c in range(impact_c + 1, n) if frames[c] <= search_end_frame]
        rest_c = within[-1] if within else min(impact_c + 1, n - 1)

    # --- is this actually a fall? -----------------------------------------
    total_drop = float(cy[rest_c] - cy[onset_c])
    drop_frac = total_drop / max(frame_h, 1.0)
    aspect0 = float(w[onset_c] / h[onset_c])
    aspect1 = float(w[rest_c] / h[rest_c])
    aspect_gain = aspect1 / max(aspect0, 1e-3)

    dropped = total_drop >= MIN_DROP_PX and drop_frac >= MIN_DROP_FRAC
    lay_down = aspect_gain >= MIN_ASPECT_GAIN or aspect1 >= LYING_ASPECT
    if not dropped and not lay_down:
        return "not_a_fall", {
            "total_drop_px": round(total_drop, 1),
            "drop_frac": round(drop_frac, 3),
            "aspect_gain": round(aspect_gain, 2),
        }
    if not dropped and aspect_gain < MIN_ASPECT_GAIN:
        return "insufficient_drop", {
            "total_drop_px": round(total_drop, 1),
            "drop_frac": round(drop_frac, 3),
        }

    meta = {
        "onset_frame": int(frames[onset_c]),
        "contact_frame": int(frames[impact_c]),
        "rest_frame": int(frames[rest_c]),
        "total_drop_px": round(total_drop, 1),
        "drop_frac": round(drop_frac, 3),
        "aspect_start": round(aspect0, 2),
        "aspect_end": round(aspect1, 2),
        "aspect_gain": round(aspect_gain, 2),
        "settled": rest_c is not None,
        "track_len": n,
    }
    return None, meta


def clip_bounds(meta, scene_start, scene_end, fps):
    """about-to-fall -> ... -> complete fall, clamped to [MIN, MAX] sec."""
    margin = SCENE_EDGE_MARGIN
    pre_f = sec_frames(PRE_ONSET_SEC, fps)
    post_f = sec_frames(POST_REST_SEC, fps)
    min_f = sec_frames(MIN_CLIP_SEC, fps)
    max_f = sec_frames(MAX_CLIP_SEC, fps)

    start = max(scene_start + margin, meta["onset_frame"] - pre_f)
    end = min(scene_end - margin, meta["rest_frame"] + post_f)
    end = max(end, meta["contact_frame"] + sec_frames(0.5, fps))
    end = min(end, start + max_f, scene_end - margin)   # enforce the 5 s cap

    if end - start < min_f:
        need = min_f - (end - start)
        start = max(scene_start + margin, start - need // 2)
        end = min(scene_end - margin, end + (need - need // 2), start + max_f)
    return int(start), int(end)


def analyze_scene(source, detector, scene_start, scene_end, fps, frame_h):
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, scene_start)
    frame_boxes = []
    frame_idx = scene_start
    while frame_idx < scene_end:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - scene_start) % FRAME_STRIDE == 0:
            boxes, _ = detect_people(detector, frame, score_thresh=0.5)
            frame_boxes.append((frame_idx, boxes))
        else:
            cap.grab()
        frame_idx += 1
    cap.release()

    track = pick_falling_track(build_tracks(frame_boxes))
    if track is None:
        return "no_falling_person", {}
    reason, meta = analyze_track(track, fps, frame_h)
    if reason:
        return reason, meta
    start_f, end_f = clip_bounds(meta, scene_start, scene_end, fps)
    meta["start_frame"] = start_f
    meta["end_frame"] = end_f
    meta["duration_sec"] = round((end_f - start_f) / fps, 2)
    return None, meta


def load_scenes(source, fps, total):
    manifest = source.parent / "scene_cut_manifest.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text())
        scenes = [
            (int(e["segment_start_frame"]), int(e["segment_end_frame"]))
            for e in entries if "segment_start_frame" in e
        ]
        min_scene_f = sec_frames(MIN_SCENE_SEC, fps)
        scenes = [(s, e) for s, e in scenes if e - s >= min_scene_f]
        if scenes:
            print(f"Loaded {len(scenes)} scenes from {manifest.name}", flush=True)
            return scenes
    cuts = detect_scene_cuts(source, fps, total)
    scenes = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    min_scene_f = sec_frames(MIN_SCENE_SEC, fps)
    return [(s, e) for s, e in scenes if e - s >= min_scene_f]


def frame_height(source):
    cap = cv2.VideoCapture(str(source))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return h if h > 0 else 1080


def main():
    if len(sys.argv) < 2:
        print("Usage: detect_fall_clips_human_v2.py <clip_folder> [source.mp4]")
        sys.exit(1)

    folder = Path(sys.argv[1]).resolve()
    source = Path(sys.argv[2]) if len(sys.argv) > 2 else folder / "source.mp4"
    if not source.exists():
        print(f"Missing source: {source}")
        sys.exit(1)

    print("Loading person detector (Detectron2 regnety)...", flush=True)
    detector = build_detector()

    fps = get_fps(source)
    total = get_frame_count(source, fps)
    frame_h = frame_height(source)
    print(f"Source: {source.name} | {fps:.3f} fps | {total} frames | H={frame_h}", flush=True)

    scenes = load_scenes(source, fps, total)
    print(f"Scenes to analyze: {len(scenes)}", flush=True)

    manual, manifest, rejected = [], [], []
    clip_num = 0
    for si, (scene_start, scene_end) in enumerate(scenes, 1):
        reason, meta = analyze_scene(source, detector, scene_start, scene_end, fps, frame_h)
        if reason:
            rejected.append({"scene": si, "reason": reason, **meta})
            print(f"  skip scene_{si:03d} ({reason})", flush=True)
            continue
        clip_num += 1
        name = f"fall_{clip_num:03d}.mp4"
        manual.append({
            "clip": name,
            "start_frame": meta["start_frame"],
            "end_frame": meta["end_frame"],
            "note": f"scene_{si:03d} v2 about_to_fall_to_complete_fall",
        })
        manifest.append({
            "clip": name,
            "mode": "fall_trim_human_v2",
            "source_scene": f"scene_{si:03d}",
            "scene_start_frame": scene_start,
            "scene_end_frame": scene_end,
            **meta,
        })
        print(f"  keep scene_{si:03d} -> {name}  {meta['duration_sec']:.2f}s", flush=True)

    for old in folder.glob("fall_*.mp4"):
        old.unlink()
    (folder / "manual_timings.json").write_text(json.dumps(manual, indent=2) + "\n")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "rejected_clips.json").write_text(json.dumps(rejected, indent=2) + "\n")

    ok = 0
    for entry in manual:
        out = folder / entry["clip"]
        try:
            cut_clip(source, out, entry["start_frame"], entry["end_frame"], fps)
            ok += 1
        except subprocess.CalledProcessError as exc:
            print(f"  FAIL {entry['clip']}: {exc}")
    print(f"Done: {ok}/{len(manual)} v2 clips, rejected {len(rejected)}")


if __name__ == "__main__":
    main()
