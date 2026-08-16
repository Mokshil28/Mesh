#!/usr/bin/env python3
"""Bake an accurate overlay.mp4 from an existing mesh_overlay.bin + source.mp4.

Uses the same pyrender IntrinsicsCamera as HMR2 — no re-detection needed.
Mesh verts in mesh_overlay.bin are already in the OpenGL display frame
(smpl_display_transform applied during export).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

# Prefer EGL on Linux; leave default on macOS.
if sys.platform != 'darwin' and 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pyrender

from export_interactive_fall import load_mesh_bin
from video_demo import open_video_writer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def render_frame(verts: np.ndarray, faces: np.ndarray, focal: float, width: int, height: int) -> np.ndarray:
    """RGBA float render of a pre-display-transformed mesh (camera at identity)."""
    colors = np.tile(np.array([*LIGHT_BLUE, 1.0], dtype=np.float32), (verts.shape[0], 1))
    mesh = trimesh.Trimesh(vertices=verts.copy(), faces=faces.copy(), vertex_colors=colors, process=False)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.35, 0.35, 0.35))
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    camera = pyrender.IntrinsicsCamera(
        fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0, zfar=1e12,
    )
    scene.add(camera, pose=np.eye(4))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return color.astype(np.float32) / 255.0


def bake_overlay(clip_dir: Path, video_path: Path | None = None) -> Path:
    mesh_path = clip_dir / 'mesh_overlay.bin'
    faces_path = clip_dir / 'faces.bin'
    meta_path = clip_dir / 'meta.json'
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    seq, fps = load_mesh_bin(mesh_path)
    faces = np.frombuffer(faces_path.read_bytes(), dtype='<u4').reshape(-1, 3)
    meta = json.loads(meta_path.read_text())
    focal = float(meta['focal_length'])
    width = int(meta['width'])
    height = int(meta['height'])

    source = video_path or (clip_dir / 'source.mp4')
    if not source.exists():
        raise FileNotFoundError(source)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {source}')

    out_path = clip_dir / 'overlay.mp4'
    writer = open_video_writer(str(out_path), fps, (width, height))
    n = seq.shape[0]
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            # If video shorter than mesh, reuse last readable behaviour: black pad avoid
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        rgba = render_frame(seq[i], faces, focal, width, height)
        rgb = rgba[:, :, :3]
        a = rgba[:, :, 3:]
        img = frame.astype(np.float32)[:, :, ::-1] / 255.0  # BGR->RGB
        comp = img * (1.0 - a) + rgb * a
        out_bgr = (255.0 * comp[:, :, ::-1]).astype(np.uint8)
        writer.write(out_bgr)
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f'  baked {i + 1}/{n}', flush=True)
    cap.release()
    writer.release()
    # OpenCV often writes MPEG-4 Part 2 (mp4v), which browsers reject. Force H.264.
    ensure_h264(out_path)
    print(f'Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)', flush=True)
    return out_path


def ensure_h264(path: Path) -> None:
    """Re-encode to H.264 yuv420p so Safari/Chrome can play the file."""
    import os
    import shutil
    import subprocess

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        for cand in (
            os.path.expanduser('~/miniconda3/envs/4d-humans-gpu/bin/ffmpeg'),
            os.path.expanduser('~/miniconda3/bin/ffmpeg'),
            os.path.expanduser('~/miniforge3/envs/4D-humans/bin/ffmpeg'),
            '/usr/bin/ffmpeg',
        ):
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                ffmpeg = cand
                break
    if ffmpeg is None:
        print(f'WARNING: ffmpeg not found; left {path} as-is (may not play in browsers)', flush=True)
        return

    tmp = path.with_suffix('.h264tmp.mp4')
    cmd = [
        ffmpeg, '-y', '-i', str(path),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '18',
        '-movflags', '+faststart', '-an', str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tmp.replace(path)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f'WARNING: could not H.264-encode {path}: {e}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip_dir', required=True, help='Directory with mesh_overlay.bin / faces.bin / meta.json')
    ap.add_argument('--video', default='', help='Optional source video path override')
    args = ap.parse_args()
    clip_dir = Path(args.clip_dir)
    video = Path(args.video) if args.video else None
    bake_overlay(clip_dir, video)


if __name__ == '__main__':
    main()
