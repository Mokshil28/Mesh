#!/usr/bin/env python3
"""Render a 360° orbit video of the fallen-person mesh (matches the HTML viewer)."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh

from build_grounded_viewer import load_mesh_bin
from export_mesh_sequence import align_for_ground_view, find_grounded_frame
from video_demo import open_video_writer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
BG_COLOR = (0.043, 0.055, 0.078, 1.0)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    up = np.array([0.0, 1.0, 0.0]) if up is None else up.astype(np.float64)
    eye = eye.astype(np.float64)
    target = target.astype(np.float64)
    forward = eye - target
    forward /= np.linalg.norm(forward) + 1e-9
    right = np.cross(up, forward)
    right /= np.linalg.norm(right) + 1e-9
    cam_up = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = cam_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def mesh_bounds(verts: np.ndarray) -> tuple[np.ndarray, float, float]:
    center = verts.mean(axis=0)
    radii = np.linalg.norm(verts - center[None, :], axis=1)
    max_r = float(radii.max())
    min_y = float(verts[:, 1].min())
    return center, max_r, min_y


def make_ground_plane(min_y: float, half: float) -> trimesh.Trimesh:
    plane = trimesh.creation.box(extents=[half * 2, 0.02, half * 2])
    plane.apply_translation([0.0, min_y - 0.01, 0.0])
    return plane


def render_orbit_frame(
    verts: np.ndarray,
    faces: np.ndarray,
    yaw: float,
    pitch: float,
    dist: float,
    target: np.ndarray,
    width: int,
    height: int,
    grid_path: trimesh.Trimesh | None,
) -> np.ndarray:
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(*LIGHT_BLUE, 1.0),
    )
    mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, faces), material=material)

    scene = pyrender.Scene(bg_color=BG_COLOR, ambient_light=(0.35, 0.38, 0.45))
    scene.add(mesh, 'body')
    if grid_path is not None:
        grid_mat = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            baseColorFactor=(0.10, 0.12, 0.16, 1.0),
        )
        scene.add(pyrender.Mesh.from_trimesh(grid_path, material=grid_mat), 'grid')

    eye = target + dist * np.array([
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
        np.cos(pitch) * np.cos(yaw),
    ])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.5, aspectRatio=width / height)
    cam_pose = look_at(eye, target)
    scene.add(camera, pose=cam_pose)

    for phi in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        lp = np.array([np.cos(phi), 0.8, np.sin(phi)]) * 4.0 + target
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=look_at(lp, target))

    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    rgb = (color[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def export_orbit_video(
    mesh_dir: Path,
    out_video: Path,
    frame_idx: int | None = None,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    duration: float = 8.0,
    pitch: float = 0.22,
    start_yaw: float = 0.4,
    show_grid: bool = True,
) -> int:
    seq, _ = load_mesh_bin(mesh_dir / 'mesh.bin')
    faces = np.frombuffer((mesh_dir / 'faces.bin').read_bytes(), dtype='<u4').reshape(-1, 3)

    grounded = find_grounded_frame(seq)
    seq = align_for_ground_view(seq, grounded)
    fidx = grounded if frame_idx is None else int(frame_idx)
    verts = seq[fidx]

    center, max_r, min_y = mesh_bounds(verts)
    target = center.copy()
    target[1] += max_r * 0.15
    dist = max_r * 2.8

    half = max(1.2, max_r * 2.2)
    grid = make_ground_plane(min_y, half) if show_grid else None

    n_frames = max(int(round(fps * duration)), 1)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(str(out_video), fps, (width, height))

    print(f'Rendering {n_frames} frames @ {fps:.0f} fps (pose frame {fidx})...', flush=True)
    for i in range(n_frames):
        t = i / n_frames
        yaw = start_yaw + t * 2.0 * np.pi
        img = render_orbit_frame(verts, faces, yaw, pitch, dist, target, width, height, grid)
        writer.write(img)
        if i % 15 == 0 or i == n_frames - 1:
            print(f'  frame {i + 1}/{n_frames}', flush=True)

    writer.release()
    return fidx


def main():
    ap = argparse.ArgumentParser(description='Export 360° orbit video of fallen mesh pose')
    ap.add_argument('--mesh_dir', default='fall_out/fall_002_mesh')
    ap.add_argument('--out_video', default='fall_out/V001_fall_002_orbit.mp4')
    ap.add_argument('--frame', type=int, default=None, help='Mesh frame (default: auto grounded)')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--duration', type=float, default=8.0, help='Seconds for one full rotation')
    ap.add_argument('--no_grid', action='store_true')
    args = ap.parse_args()

    fidx = export_orbit_video(
        Path(args.mesh_dir),
        Path(args.out_video),
        frame_idx=args.frame,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        show_grid=not args.no_grid,
    )
    out = Path(args.out_video)
    print(f'Saved orbit video: {out}  ({out.stat().st_size / 1e6:.1f} MB)', flush=True)
    print(f'Grounded pose frame: {fidx}', flush=True)


if __name__ == '__main__':
    main()
