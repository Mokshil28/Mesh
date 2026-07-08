#!/usr/bin/env python3
"""Track the falling person and export a mesh-only viewer — no video, no person visible.

Only the SMPL mesh is shown, animated at the original clip speed.

Usage:
    python export_fall_mesh_only.py \\
        --video data/fall_dataset_clips/001_V01/fall_006.mp4 \\
        --out_html fall_out/V001_fall_006_viewer.html

Re-run:
    ./run_fall_006.sh
    ./run_fall_006.sh --build-only
"""
from __future__ import annotations

import argparse
import json
import os
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

from export_mesh_sequence import align_for_ground_view, build_html, find_grounded_frame, quantize, to_y_up
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


def infer_mesh_frame(img_cv2, box, model, model_cfg, device):
    """Pelvis-centered mesh for mesh-only playback (person replaced by mesh)."""
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
    focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal)[0].detach().cpu().numpy()

    verts = out['pred_vertices'][0].detach().cpu().numpy()
    joints = out['pred_keypoints_3d'][0].detach().cpu().numpy()
    world = verts + cam_t
    pelvis = joints[0] + cam_t
    centered = smpl_display_transform(world - pelvis)
    return centered.astype(np.float32), float(focal)


def write_mesh_bin(path: Path, frames: np.ndarray, fps: float) -> None:
    t, v, _ = frames.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('<IIf', t, v, fps))
        frames.astype('<f4').tofile(f)


def load_mesh_bin(path: Path) -> tuple[np.ndarray, float]:
    with open(path, 'rb') as f:
        t, v, fps = struct.unpack('<IIf', f.read(12))
        return np.fromfile(f, dtype='<f4').reshape(t, v, 3), fps


def export_sequence(
    video_path: Path,
    out_dir: Path,
    detector,
    model,
    model_cfg,
    device,
    frame_stride: int,
    smooth_alpha: float,
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
    prev_box = track[0][1].astype(np.float32)
    smooth_state = prev_box.copy()
    frames = []
    focal = None
    frame_idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        boxes, scores = detect_people(detector, frame, score_thresh=0.45)
        matched = match_subject_box(prev_box, boxes, scores, tight=True)
        smooth_state = tight_track_box(smooth_state, matched, alpha=smooth_alpha)
        prev_box = smooth_state

        mesh, focal = infer_mesh_frame(frame, smooth_state, model, model_cfg, device)
        frames.append(mesh)
        print(f'{video_path.stem}: frame {processed} (src {frame_idx})', flush=True)
        processed += 1
        frame_idx += 1

    cap.release()
    if not frames:
        raise RuntimeError('No frames processed')

    seq = np.stack(frames, axis=0)
    out_fps = fps / frame_stride
    write_mesh_bin(out_dir / 'mesh.bin', seq, out_fps)

    faces = model.smpl.faces.astype(np.uint32)
    (out_dir / 'faces.bin').write_bytes(faces.astype('<u4').tobytes())

    meta = {
        'video': str(video_path),
        'fps': out_fps,
        'width': width,
        'height': height,
        'focal_length': focal,
        'num_frames': int(seq.shape[0]),
        'num_vertices': int(seq.shape[1]),
        'mode': 'mesh_only',
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def build_viewer(out_dir: Path, out_html: Path, title: str) -> None:
    seq, fps = load_mesh_bin(out_dir / 'mesh.bin')
    faces = np.frombuffer((out_dir / 'faces.bin').read_bytes(), dtype='<u4').reshape(-1, 3)

    seq = to_y_up(seq)
    grounded = find_grounded_frame(seq)
    seq = align_for_ground_view(seq, grounded)
    seq_q, lo, span = quantize(seq)

    meta = {
        'title': title,
        'source': json.loads((out_dir / 'meta.json').read_text()).get('video', ''),
        'fps': fps,
        'grounded_frame': grounded,
        'note': 'Mesh only — falling person replaced by SMPL mesh',
    }
    html = build_html(seq_q, lo, span, faces, fps, meta, grounded, video_src=None, camera=None)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)
    print(f'Saved mesh-only viewer: {out_html}  ({out_html.stat().st_size / 1e6:.1f} MB)', flush=True)
    print(f'{seq.shape[0]} frames @ {fps:.2f} fps ({seq.shape[0] / fps:.2f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser(description='Mesh-only falling person viewer')
    ap.add_argument('--video', required=True)
    ap.add_argument('--out_dir', default='')
    ap.add_argument('--out_html', default='')
    ap.add_argument('--title', default='')
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    ap.add_argument('--detector', default='regnety', choices=['vitdet', 'regnety'])
    ap.add_argument('--frame_stride', type=int, default=1)
    ap.add_argument('--smooth_alpha', type=float, default=0.88)
    ap.add_argument('--build-only', action='store_true')
    args = ap.parse_args()

    video_path = Path(args.video)
    stem = video_path.stem
    out_dir = Path(args.out_dir or f'fall_out/{stem}_mesh')
    out_html = Path(args.out_html or f'fall_out/V001_{stem}_viewer.html')
    title = args.title or f'V001 {stem} — Mesh Only Fall'

    if not args.build_only:
        if 'PYOPENGL_PLATFORM' in os.environ and os.environ['PYOPENGL_PLATFORM'] == 'egl':
            del os.environ['PYOPENGL_PLATFORM']
        download_models(CACHE_DIR_4DHUMANS)
        model, model_cfg = load_hmr2(args.checkpoint)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()
        detector = build_detector(args.detector)
        export_sequence(
            video_path, out_dir, detector, model, model_cfg, device,
            args.frame_stride, args.smooth_alpha,
        )

    build_viewer(out_dir, out_html, title)

    out_mp4 = out_html.parent / out_html.name.replace('_viewer.html', '_mesh.mp4')
    from render_mesh_video import load_mesh_bin  # noqa: F401 — ensure import works
    import subprocess
    subprocess.run([
        os.environ.get('PYTHON', '/Users/mshah76/miniforge3/envs/4D-humans/bin/python'),
        str(Path(__file__).resolve().parent / 'render_mesh_video.py'),
        '--mesh_dir', str(out_dir),
        '--out_video', str(out_mp4),
    ], check=True)
    print(f'Saved mesh video: {out_mp4}', flush=True)


if __name__ == '__main__':
    main()
