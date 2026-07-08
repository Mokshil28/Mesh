#!/usr/bin/env python3
"""Track falling person, export per-frame SMPL mesh, build interactive 360 viewer."""
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
from hmr2.utils.renderer import Renderer, cam_crop_to_full

from video_demo import (
    build_detector,
    detect_people,
    match_subject_box,
    open_video_writer,
    render_mesh_on_frame,
    scan_falling_track,
    smooth_box,
)

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def smpl_display_transform(verts: np.ndarray) -> np.ndarray:
    """Match demo.py / renderer orientation for consistent 360 viewing."""
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    hom = np.c_[verts, np.ones(len(verts))]
    return (rot @ hom.T).T[:, :3]


def infer_frame(img_cv2, box, model, model_cfg, device):
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
    return centered.astype(np.float32), verts, cam_t


def write_mesh_bin(path: Path, frames: np.ndarray, fps: float) -> None:
    """frames: (T, V, 3) float32, pelvis-centered display coords."""
    t, v, _ = frames.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('<IIf', t, v, fps))
        frames.astype('<f4').tofile(f)


def write_viewer_html(path: Path, title: str) -> None:
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title} — 360° mesh</title>
  <style>
    body {{ margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif; }}
    #hud {{ position: fixed; left: 12px; top: 12px; z-index: 2; background: rgba(0,0,0,.55); padding: 10px 14px; border-radius: 8px; }}
    #bar {{ position: fixed; left: 0; right: 0; bottom: 0; z-index: 2; background: rgba(0,0,0,.7); padding: 12px 16px; display: flex; gap: 12px; align-items: center; }}
    input[type=range] {{ flex: 1; }}
    canvas {{ display: block; }}
    a {{ color: #8cf; }}
  </style>
</head>
<body>
<div id="hud">
  <div><b>{title}</b></div>
  <div>Drag to orbit · scroll to zoom · slider = time</div>
  <div id="status">Loading…</div>
</div>
<div id="bar">
  <button id="play">Play</button>
  <input id="frame" type="range" min="0" max="0" value="0" step="1"/>
  <span id="label">0 / 0</span>
</div>
<script type="importmap">
{{ "imports": {{ "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                 "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/" }} }}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const status = document.getElementById('status');
const slider = document.getElementById('frame');
const label = document.getElementById('label');
const playBtn = document.getElementById('play');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.01, 100);
camera.position.set(0, 0.2, 2.8);
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.5, 0);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 0.9);
key.position.set(2, 4, 3);
scene.add(key);
const fill = new THREE.DirectionalLight(0xaaccff, 0.35);
fill.position.set(-3, 1, -2);
scene.add(fill);
scene.add(new THREE.GridHelper(4, 20, 0x333333, 0x222222));

let frames = null;
let faces = null;
let fps = 30;
let mesh = null;
let geom = null;
let playing = false;
let lastT = 0;

async function load() {{
  const [binRes, faceRes, metaRes] = await Promise.all([
    fetch('mesh.bin'), fetch('faces.bin'), fetch('meta.json')
  ]);
  const meta = await metaRes.json();
  fps = meta.fps;
  const buf = await binRes.arrayBuffer();
  const view = new DataView(buf);
  const T = view.getUint32(0, true);
  const V = view.getUint32(4, true);
  const fileFps = view.getFloat32(8, true);
  fps = fileFps || meta.fps;
  const flat = new Float32Array(buf, 12);
  frames = [];
  for (let t = 0; t < T; t++) {{
    const start = t * V * 3;
    frames.push(flat.subarray(start, start + V * 3));
  }}
  faces = new Uint32Array(await (await faceRes).arrayBuffer());

  geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(V * 3), 3));
  geom.setIndex(new THREE.BufferAttribute(faces, 1));
  geom.computeVertexNormals();
  mesh = new THREE.Mesh(geom, new THREE.MeshPhongMaterial({{
    color: 0xa6bdd8, flatShading: false, shininess: 20
  }}));
  scene.add(mesh);

  slider.max = T - 1;
  label.textContent = `0 / ${{T - 1}}`;
  status.textContent = `${{T}} frames · ${{fps.toFixed(1)}} fps · ${{V}} verts`;
  setFrame(0);
}}

function setFrame(i) {{
  if (!frames) return;
  i = Math.max(0, Math.min(frames.length - 1, i));
  slider.value = i;
  label.textContent = `${{i}} / ${{frames.length - 1}}`;
  geom.attributes.position.array.set(frames[i]);
  geom.attributes.position.needsUpdate = true;
  geom.computeVertexNormals();
}}

slider.addEventListener('input', () => setFrame(+slider.value));
playBtn.addEventListener('click', () => {{ playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; lastT = performance.now(); }});

addEventListener('resize', () => {{
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});

function animate(t) {{
  requestAnimationFrame(animate);
  if (playing && frames) {{
    const dt = (t - lastT) / 1000;
    lastT = t;
    let next = +slider.value + dt * fps;
    if (next >= frames.length) next = 0;
    setFrame(Math.floor(next));
  }}
  controls.update();
  renderer.render(scene, camera);
}}

load().then(() => requestAnimationFrame(animate)).catch(err => {{
  status.textContent = 'Load failed: ' + err;
  console.error(err);
}});
</script>
</body>
</html>"""
    path.write_text(html)


def export_clip(
    video_path: str,
    out_folder: str,
    detector,
    model,
    model_cfg,
    renderer,
    device,
    frame_stride: int,
    smooth_alpha: float,
    save_overlay: bool,
) -> Path:
    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    track = scan_falling_track(cap, detector, frame_stride)
    if track is None:
        raise RuntimeError(f'No falling person found in {video_path}')

    writer = None
    if save_overlay:
        writer = open_video_writer(str(out / 'overlay.mp4'), max(fps / frame_stride, 1.0), (width, height))

    frame_verts = []
    frame_indices = []
    prev_box = track[0][1].astype(np.float32)
    smooth_state = prev_box.copy()
    frame_idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        boxes, scores = detect_people(detector, frame)
        matched = match_subject_box(prev_box, boxes, scores)
        smooth_state = smooth_box(smooth_state, matched, alpha=smooth_alpha)
        prev_box = smooth_state

        centered, _, _ = infer_frame(frame, smooth_state, model, model_cfg, device)
        frame_verts.append(centered)
        frame_indices.append(frame_idx)

        if writer is not None:
            overlay = render_mesh_on_frame(frame, smooth_state, model, model_cfg, renderer, device)
            writer.write(overlay)

        print(f'frame {processed} (src {frame_idx})', flush=True)
        processed += 1
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if not frame_verts:
        raise RuntimeError('No frames processed')

    frames_arr = np.stack(frame_verts, axis=0)
    faces = model.smpl.faces.astype(np.uint32)

    write_mesh_bin(out / 'mesh.bin', frames_arr, fps / frame_stride)
    (out / 'faces.bin').write_bytes(faces.astype('<u4').tobytes())
    meta = {
        'video': str(video_path),
        'fps': fps / frame_stride,
        'num_frames': int(frames_arr.shape[0]),
        'num_vertices': int(frames_arr.shape[1]),
        'frame_indices': frame_indices,
        'note': 'Vertices pelvis-centered, display-oriented for 360 viewer',
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')

    title = Path(video_path).stem
    write_viewer_html(out / 'viewer.html', title)
    print(f'Saved viewer: {out / "viewer.html"}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description='Export 360 mesh viewer for a fall clip')
    parser.add_argument('--video', required=True)
    parser.add_argument('--out_folder', default='fall_out/mesh_view')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--detector', default='regnety', choices=['vitdet', 'regnety'])
    parser.add_argument('--frame_stride', type=int, default=1, help='1 = every frame (best accuracy)')
    parser.add_argument('--smooth_alpha', type=float, default=0.55)
    parser.add_argument('--no_overlay', action='store_true')
    args = parser.parse_args()

    if 'PYOPENGL_PLATFORM' in os.environ and os.environ['PYOPENGL_PLATFORM'] == 'egl':
        del os.environ['PYOPENGL_PLATFORM']

    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    detector = build_detector(args.detector)
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    export_clip(
        args.video,
        args.out_folder,
        detector,
        model,
        model_cfg,
        renderer,
        device,
        args.frame_stride,
        args.smooth_alpha,
        save_overlay=not args.no_overlay,
    )


if __name__ == '__main__':
    main()
