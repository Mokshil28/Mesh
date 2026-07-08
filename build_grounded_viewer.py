#!/usr/bin/env python3
"""Build a self-contained 360° viewer from an existing mesh.bin export."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from export_mesh_sequence import (
    align_for_ground_view,
    build_html,
    find_grounded_frame,
    quantize,
)


def load_mesh_bin(path: Path) -> tuple[np.ndarray, float]:
    with open(path, 'rb') as f:
        t, v, fps = struct.unpack('<IIf', f.read(12))
        data = np.fromfile(f, dtype='<f4').reshape(t, v, 3)
    return data, fps


def main():
    ap = argparse.ArgumentParser(description='Build grounded 360 viewer from mesh.bin')
    ap.add_argument('--mesh_dir', default='fall_out/fall_002_mesh')
    ap.add_argument('--out_html', default='fall_out/V001_fall_002_viewer.html')
    ap.add_argument('--title', default='V001 fall_002 — Falling Person 3D Mesh')
    args = ap.parse_args()

    mesh_dir = Path(args.mesh_dir)
    seq, fps = load_mesh_bin(mesh_dir / 'mesh.bin')
    faces = np.frombuffer((mesh_dir / 'faces.bin').read_bytes(), dtype='<u4').reshape(-1, 3)
    meta_src = json.loads((mesh_dir / 'meta.json').read_text()) if (mesh_dir / 'meta.json').exists() else {}

    grounded = find_grounded_frame(seq)
    seq = align_for_ground_view(seq, grounded)
    seq_q, lo, span = quantize(seq)

    meta = {
        'title': args.title,
        'source': Path(meta_src.get('video', 'fall_002.mp4')).name,
        'fps': fps,
        'grounded_frame': grounded,
        'note': 'Falling person only; orbit defaults to grounded pose',
    }
    html = build_html(seq_q, lo, span, faces, fps, meta, grounded)

    out = Path(args.out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f'Grounded pose frame: {grounded} / {seq.shape[0] - 1}')
    print(f'Saved: {out}  ({out.stat().st_size / 1e6:.1f} MB)')
    print('Open in a browser — plays at clip speed; pause any frame and drag to orbit.')


if __name__ == '__main__':
    main()
