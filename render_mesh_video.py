#!/usr/bin/env python3
"""Render mesh.bin sequence to an MP4 video (mesh only, dark background)."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh

from export_mesh_sequence import align_for_ground_view, find_grounded_frame, to_y_up
from export_orbit_video import BG_COLOR, LIGHT_BLUE, look_at
from video_demo import open_video_writer


def load_mesh_bin(path: Path) -> tuple[np.ndarray, float]:
    with open(path, 'rb') as f:
        t, v, fps = struct.unpack('<IIf', f.read(12))
        data = np.fromfile(f, dtype='<f4').reshape(t, v, 3)
    return data, fps


def render_frame(verts, faces, center, max_r, width, height, yaw=0.4, pitch=0.22):
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(*LIGHT_BLUE, 1.0),
    )
    mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, faces), material=material)
    scene = pyrender.Scene(bg_color=BG_COLOR, ambient_light=(0.35, 0.38, 0.45))
    scene.add(mesh, 'body')

    target = center + np.array([0.0, max_r * 0.12, 0.0])
    dist = max(max_r * 2.6, 0.9)
    eye = target + dist * np.array([
        np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw),
    ])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.5, aspectRatio=width / height)
    scene.add(camera, pose=look_at(eye, target))
    for phi in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        lp = np.array([np.cos(phi), 0.8, np.sin(phi)]) * 4.0 + target
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=look_at(lp, target))

    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    return cv2.cvtColor((color[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh_dir', required=True)
    ap.add_argument('--out_video', required=True)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    args = ap.parse_args()

    mesh_dir = Path(args.mesh_dir)
    seq, fps = load_mesh_bin(mesh_dir / 'mesh.bin')
    faces = np.frombuffer((mesh_dir / 'faces.bin').read_bytes(), dtype='<u4').reshape(-1, 3)

    seq = to_y_up(seq)
    grounded = find_grounded_frame(seq)
    seq = align_for_ground_view(seq, grounded)

    # global center for lock-in-place (mesh stays centered while falling)
    global_c = seq.reshape(-1, 3).mean(axis=0)

    out = Path(args.out_video)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(str(out), fps, (args.width, args.height))

    print(f'Rendering {seq.shape[0]} frames @ {fps:.1f} fps ...', flush=True)
    for i, verts in enumerate(seq):
        c = verts.mean(axis=0)
        centered = verts - c + global_c
        max_r = float(np.linalg.norm(centered - global_c, axis=1).max())
        img = render_frame(centered, faces, global_c, max_r, args.width, args.height)
        writer.write(img)
        if i % 10 == 0:
            print(f'  {i + 1}/{seq.shape[0]}', flush=True)

    writer.release()
    print(f'Saved: {out}  ({out.stat().st_size / 1e6:.1f} MB)', flush=True)


if __name__ == '__main__':
    main()
