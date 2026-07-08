#!/usr/bin/env python3
"""One-command export: falling-person mesh + source video + interactive viewer.

Tight per-frame tracking (every frame, minimal smoothing) so the mesh follows
the falling body as closely as HMR2/SMPL allows. Pause any frame for full 360°
mesh inspection.

Re-run any time:
    ./run_fall_002.sh

Build viewer only (skip slow HMR inference if mesh already exported):
    ./run_fall_002.sh --build-only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2
from hmr2.utils import recursive_to
from hmr2.utils.renderer import cam_crop_to_full

from export_mesh_sequence import build_html, find_grounded_frame, quantize
from video_demo import (
    build_detector,
    detect_people,
    match_subject_box,
    scan_falling_track,
    tight_track_box,
)


def smpl_display_transform(verts: np.ndarray) -> np.ndarray:
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    hom = np.c_[verts, np.ones(len(verts))]
    return (rot @ hom.T).T[:, :3]


def infer_overlay_verts(img_cv2, box, model, model_cfg, device):
    """Per-frame camera-space mesh aligned to the video (no temporal smoothing)."""
    dataset = ViTDetDataset(model_cfg, img_cv2, box[None])
    batch = recursive_to(
        next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))),
        device,
    )
    with torch.no_grad():
        out = model(batch)

    pred_cam = out['pred_cam']
    box_center = batch['box_center'].float()
    box_size = batch['box_size'].float()
    img_size = batch['img_size'].float()
    focal = float(model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max())
    cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal)[0].detach().cpu().numpy()
    verts = out['pred_vertices'][0].detach().cpu().numpy()
    overlay = smpl_display_transform(verts + cam_t).astype(np.float32)
    return overlay, focal


def write_mesh_bin(path: Path, frames: np.ndarray, fps: float) -> None:
    t, v, _ = frames.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('<IIf', t, v, fps))
        frames.astype('<f4').tofile(f)


def load_mesh_bin(path: Path) -> tuple[np.ndarray, float]:
    with open(path, 'rb') as f:
        t, v, fps = struct.unpack('<IIf', f.read(12))
        data = np.fromfile(f, dtype='<f4').reshape(t, v, 3)
    return data, fps


def export_mesh(
    video_path: Path,
    out_dir: Path,
    detector,
    model,
    model_cfg,
    device,
    frame_stride: int,
    smooth_alpha: float,
    det_score: float,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    track = scan_falling_track(cap, detector, frame_stride=1)
    if track is None:
        raise RuntimeError(f'No falling person found in {video_path}')
    print(f'Locked falling subject ({len(track)} detections)', flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, out_dir / 'source.mp4')

    prev_box = track[0][1].astype(np.float32)
    smooth_state = prev_box.copy()
    frame_idx = 0
    processed = 0
    frames = []
    focal_length = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        boxes, scores = detect_people(detector, frame, score_thresh=det_score)
        matched = match_subject_box(prev_box, boxes, scores, tight=True)
        smooth_state = tight_track_box(smooth_state, matched, alpha=smooth_alpha)
        prev_box = smooth_state

        overlay, focal_length = infer_overlay_verts(frame, smooth_state, model, model_cfg, device)
        frames.append(overlay)
        print(f'{video_path.stem}: frame {processed} (src {frame_idx})', flush=True)
        processed += 1
        frame_idx += 1

    cap.release()
    if not frames:
        raise RuntimeError('No frames processed')

    seq = np.stack(frames, axis=0)
    out_fps = fps / frame_stride
    write_mesh_bin(out_dir / 'mesh_overlay.bin', seq, out_fps)

    faces = model.smpl.faces.astype(np.uint32)
    (out_dir / 'faces.bin').write_bytes(faces.astype('<u4').tobytes())

    meta = {
        'video': str(video_path),
        'fps': out_fps,
        'width': width,
        'height': height,
        'focal_length': focal_length,
        'num_frames': int(seq.shape[0]),
        'num_vertices': int(seq.shape[1]),
        'has_video_bg': True,
        'tracking': 'tight per-frame HMR2, frame_stride=1, minimal box smoothing',
        'note': 'Play = mesh on video. Pause = 360 mesh inspect. SMPL hands are approximate.',
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def build_viewer(out_dir: Path, out_html: Path, title: str, video_src_rel: str) -> None:
    mesh_path = out_dir / 'mesh_overlay.bin'
    if not mesh_path.exists():
        raise FileNotFoundError(f'Missing {mesh_path} — run without --build-only first')

    seq, fps = load_mesh_bin(mesh_path)
    faces = np.frombuffer((out_dir / 'faces.bin').read_bytes(), dtype='<u4').reshape(-1, 3)
    meta_src = json.loads((out_dir / 'meta.json').read_text())

    grounded = find_grounded_frame(seq)
    seq_q, lo, span = quantize(seq)

    meta = {
        'title': title,
        'source': Path(meta_src.get('video', 'clip.mp4')).name,
        'fps': fps,
        'grounded_frame': grounded,
        'note': 'Play mesh on video · Pause for 360° mesh inspect',
    }
    camera = {
        'width': int(meta_src['width']),
        'height': int(meta_src['height']),
        'focal': float(meta_src['focal_length']),
    }

    html = build_html(
        seq_q, lo, span, faces, fps, meta, grounded,
        video_src=video_src_rel,
        camera=camera,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)
    print(f'Saved viewer: {out_html}  ({out_html.stat().st_size / 1e6:.1f} MB)', flush=True)
    print(f'Clip: {seq.shape[0]} frames @ {fps:.2f} fps ({seq.shape[0] / fps:.2f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser(description='Export interactive fall mesh viewer with video background')
    ap.add_argument('--video', required=True)
    ap.add_argument('--out_dir', default='fall_out/fall_002_mesh')
    ap.add_argument('--out_html', default='fall_out/V001_fall_002_viewer.html')
    ap.add_argument('--title', default='')
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    ap.add_argument('--detector', default='regnety', choices=['vitdet', 'regnety'])
    ap.add_argument('--frame_stride', type=int, default=1)
    ap.add_argument('--smooth_alpha', type=float, default=0.88,
                    help='Box follow strength (higher = tighter tracking, default 0.88)')
    ap.add_argument('--det_score', type=float, default=0.45,
                    help='Person detection threshold (lower = keep lock in hard frames)')
    ap.add_argument('--build-only', action='store_true', help='Rebuild HTML from existing mesh_overlay.bin')
    args = ap.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_html = Path(args.out_html)
    title = args.title or f'{video_path.parent.name} {video_path.stem} — Falling Person Mesh'

    if not args.build_only:
        if 'PYOPENGL_PLATFORM' in os.environ and os.environ['PYOPENGL_PLATFORM'] == 'egl':
            del os.environ['PYOPENGL_PLATFORM']
        download_models(CACHE_DIR_4DHUMANS)
        model, model_cfg = load_hmr2(args.checkpoint)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()
        detector = build_detector(args.detector)
        export_mesh(
            video_path, out_dir, detector, model, model_cfg, device,
            args.frame_stride, args.smooth_alpha, args.det_score,
        )

    video_src_rel = os.path.relpath(out_dir / 'source.mp4', out_html.parent)
    build_viewer(out_dir, out_html, title, video_src_rel)


if __name__ == '__main__':
    main()
