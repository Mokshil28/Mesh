#!/usr/bin/env python3
"""Detect fall clips using person detection + falling-body tracking (Detectron2).

Unlike pixel-motion scoring, this checks:
  - a person is present
  - their bbox drops (falling down)
  - motion settles after impact (rest on ground)

Requires conda env 4D-humans (detectron2). Much closer to what you want visually.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from detect_fall_clips_frames import (
    SCENE_EDGE_MARGIN,
    detect_scene_cuts,
    get_fps,
    get_frame_count,
    sec_frames,
    cut_clip,
)

PRE_ONSET_SEC = 0.8
POST_REST_SEC = 0.8
MIN_CLIP_SEC = 2.0
MAX_CLIP_SEC = 4.0
MIN_SCENE_SEC = 1.5
FRAME_STRIDE = 8
DETECT_MAX_SIDE = 640
MIN_TRACK_LEN = 4
MIN_DROP_PX = 35.0
MIN_DOWNWARD_SUM = 50.0


def build_detector():
    import torch.nn as nn
    from detectron2 import model_zoo
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy

    detectron2_cfg = model_zoo.get_config(
        'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True
    )
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.55
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.5
    predictor = DefaultPredictor_Lazy(detectron2_cfg)

    # RegNet weights use SyncBatchNorm; convert for CPU/MPS inference on Mac
    def _convert_syncbn(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.SyncBatchNorm):
                bn = nn.BatchNorm2d(
                    child.num_features, child.eps, child.momentum,
                    child.affine, child.track_running_stats,
                )
                if child.affine:
                    bn.weight.data = child.weight.data.detach().clone()
                    bn.bias.data = child.bias.data.detach().clone()
                bn.running_mean = child.running_mean
                bn.running_var = child.running_var
                bn.num_batches_tracked = child.num_batches_tracked
                setattr(module, name, bn)
            else:
                _convert_syncbn(child)
        return module

    if not torch.cuda.is_available():
        predictor.model = _convert_syncbn(predictor.model)
        predictor.model.eval().to(predictor.device)
    return predictor


def detect_people(detector, img_cv2, score_thresh=0.55):
    h, w = img_cv2.shape[:2]
    scale = DETECT_MAX_SIDE / max(h, w)
    if scale < 1.0:
        small = cv2.resize(img_cv2, (int(w * scale), int(h * scale)))
        det_out = detector(small)
        inv = 1.0 / scale
    else:
        det_out = detector(img_cv2)
        inv = 1.0
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > score_thresh)
    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy() * inv
    scores = det_instances.scores[valid_idx].cpu().numpy()
    return boxes, scores


def box_center(box):
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def build_tracks(frame_boxes):
    tracks = []
    for frame_idx, boxes in frame_boxes:
        if len(boxes) == 0:
            continue
        used = set()
        for track in tracks:
            prev_box = track[-1][1]
            ious = [box_iou(prev_box, b) for b in boxes]
            if not ious:
                continue
            best = int(np.argmax(ious))
            if ious[best] >= 0.25 and best not in used:
                track.append((frame_idx, boxes[best]))
                used.add(best)
        for b_idx, box in enumerate(boxes):
            if b_idx not in used:
                tracks.append([(frame_idx, box)])
    return tracks


def pick_falling_track(tracks):
    best_track = None
    best_score = -1e9
    for track in tracks:
        if len(track) < 3:
            continue
        centers = np.array([box_center(box) for _, box in track], dtype=np.float32)
        dy = centers[1:, 1] - centers[:-1, 1]
        downward = float(np.sum(np.clip(dy, 0, None)))
        drop = float(centers[-1, 1] - centers[0, 1])
        height_change = float(centers[0, 1] - centers[-1, 1])
        score = downward + 0.75 * drop + 0.02 * len(track) + 0.1 * height_change
        if score > best_score:
            best_score = score
            best_track = track
    return best_track


def box_metrics(box: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return cx, cy, w, h


def analyze_scene_fall(
    source: Path,
    detector,
    scene_start: int,
    scene_end: int,
    fps: float,
) -> tuple[str | None, dict]:
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
    if track is None or len(track) < MIN_TRACK_LEN:
        return "no_falling_person", {}

    centers = np.array([box_metrics(box)[0:2] for _, box in track], dtype=np.float32)
    heights = np.array([box_metrics(box)[3] for _, box in track], dtype=np.float32)
    frames = np.array([f for f, _ in track], dtype=np.int32)

    dy = centers[1:, 1] - centers[:-1, 1]
    downward = float(np.sum(np.clip(dy, 0, None)))
    total_drop = float(centers[-1, 1] - centers[0, 1])
    height_ratio = float(heights[-1] / max(heights[0], 1.0))

    if downward < MIN_DOWNWARD_SUM or total_drop < MIN_DROP_PX:
        return "person_not_falling", {
            "downward": round(downward, 1),
            "total_drop": round(total_drop, 1),
        }

    # impact ~ max downward speed; rest ~ after center stops rising
    impact_i = int(np.argmax(dy)) + 1 if len(dy) else len(track) - 1
    impact_frame = int(frames[min(impact_i, len(frames) - 1)])
    onset_frame = int(frames[max(0, impact_i - 2)])
    rest_frame = int(frames[-1])

    # reject if person stays tall (not on ground)
    if height_ratio > 0.82 and total_drop < MIN_DROP_PX * 1.5:
        return "no_lie_down", {"height_ratio": round(height_ratio, 2)}

    margin = SCENE_EDGE_MARGIN
    pre_f = sec_frames(PRE_ONSET_SEC, fps)
    post_f = sec_frames(POST_REST_SEC, fps)
    start_f = max(scene_start + margin, onset_frame - pre_f)
    end_f = min(scene_end - margin, rest_frame + post_f, start_f + sec_frames(MAX_CLIP_SEC, fps))
    end_f = max(end_f, impact_frame + sec_frames(0.5, fps))

    if end_f - start_f < sec_frames(MIN_CLIP_SEC, fps):
        need = sec_frames(MIN_CLIP_SEC, fps) - (end_f - start_f)
        start_f = max(scene_start + margin, start_f - need // 2)
        end_f = min(scene_end - margin, end_f + need // 2)

    meta = {
        "onset_frame": onset_frame,
        "contact_frame": impact_frame,
        "rest_frame": rest_frame,
        "start_frame": start_f,
        "end_frame": end_f,
        "downward_px": round(downward, 1),
        "total_drop_px": round(total_drop, 1),
        "height_ratio": round(height_ratio, 2),
        "track_len": len(track),
    }
    return None, meta


def load_scenes(source: Path, fps: float, total: int) -> list[tuple[int, int]]:
    """Use cached scene_cut_manifest.json when present to skip slow ffmpeg pass."""
    manifest = source.parent / "scene_cut_manifest.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text())
        scenes = [
            (int(e["segment_start_frame"]), int(e["segment_end_frame"]))
            for e in entries
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


def main():
    if len(sys.argv) < 2:
        print("Usage: detect_fall_clips_human.py <clip_folder> [source.mp4]")
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
    print(f"Source: {source.name} | {fps:.3f} fps | {total} frames", flush=True)

    scenes = load_scenes(source, fps, total)
    print(f"Scenes to analyze: {len(scenes)}", flush=True)

    manual, manifest, rejected = [], [], []
    clip_num = 0

    for si, (scene_start, scene_end) in enumerate(scenes, 1):
        reason, meta = analyze_scene_fall(source, detector, scene_start, scene_end, fps)
        if reason:
            rejected.append({
                "scene": si,
                "reason": reason,
                **meta,
            })
            print(f"  skip scene_{si:03d} ({reason})", flush=True)
            continue

        clip_num += 1
        name = f"fall_{clip_num:03d}.mp4"
        start_f, end_f = meta["start_frame"], meta["end_frame"]
        manual.append({
            "clip": name,
            "start_frame": start_f,
            "end_frame": end_f,
            "note": f"scene_{si:03d} human_fall_track",
        })
        manifest.append({
            "clip": name,
            "mode": "fall_trim_human_detectron",
            "source_scene": f"scene_{si:03d}",
            "scene_start_frame": scene_start,
            "scene_end_frame": scene_end,
            **meta,
            "duration_sec": round((end_f - start_f) / fps, 2),
        })
        print(f"  keep scene_{si:03d} -> {name}  {(end_f-start_f)/fps:.2f}s", flush=True)

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

    print(f"Done: {ok}/{len(manual)} human-validated clips, rejected {len(rejected)}")


if __name__ == "__main__":
    main()
