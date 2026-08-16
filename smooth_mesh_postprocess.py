#!/usr/bin/env python3
"""Temporal smoothing on an existing mesh_overlay.bin (no re-inference)."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from export_interactive_fall import build_viewer, load_mesh_bin, write_mesh_bin
from video_demo import stabilize_mesh_root


def smooth_sequence(seq: np.ndarray, alpha: float) -> np.ndarray:
    """EMA across frames. alpha=1 keeps raw; lower = smoother."""
    if alpha >= 1.0 or seq.shape[0] < 2:
        return seq
    out = seq.copy()
    prev = seq[0].copy()
    out[0] = prev
    for i in range(1, seq.shape[0]):
        cur = alpha * seq[i] + (1.0 - alpha) * prev
        out[i] = cur
        prev = cur
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description='Smooth mesh_overlay.bin temporally')
    ap.add_argument('--clip_dir', required=True, help='e.g. fall_out/V01/fall_002')
    ap.add_argument('--mesh_smooth', type=float, default=0.28,
                    help='Blend toward current frame (0.28 = 72%% prior frame)')
    ap.add_argument('--root_smooth', type=float, default=0.30,
                    help='Smooth global mesh translation (lower = steadier)')
    ap.add_argument('--out_html', default='', help='Viewer HTML path (optional)')
    ap.add_argument('--title', default='')
    ap.add_argument('--mesh-only-360', action='store_true')
    ap.add_argument('--side-by-side', action='store_true',
                    help='Original video left; mesh + background + 360° right')
    ap.add_argument('--no-backup', action='store_true')
    args = ap.parse_args()

    clip_dir = Path(args.clip_dir)
    mesh_path = clip_dir / 'mesh_overlay.bin'
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    seq, fps = load_mesh_bin(mesh_path)
    print(f'Loaded {mesh_path.name}: {seq.shape[0]} frames, smooth={args.mesh_smooth:.2f}', flush=True)

    if not args.no_backup:
        backup = clip_dir / 'mesh_overlay_raw.bin'
        if not backup.exists():
            shutil.copy2(mesh_path, backup)
            print(f'Backup: {backup}', flush=True)

    smoothed = stabilize_mesh_root(seq, root_alpha=args.root_smooth)
    smoothed = smooth_sequence(smoothed, args.mesh_smooth)
    write_mesh_bin(mesh_path, smoothed, fps)
    print(f'Wrote smoothed mesh: {mesh_path}', flush=True)

    out_html = Path(args.out_html) if args.out_html else clip_dir.parent / f'{clip_dir.name}_viewer.html'
    title = args.title or f'{clip_dir.name} — smoothed mesh'
    source = clip_dir / 'source.mp4'
    import os
    video_rel = os.path.relpath(source, out_html.parent) if source.exists() else None
    build_viewer(
        clip_dir, out_html, title,
        video_src_rel=video_rel,
        overlay_src_rel=None,
        mesh_only_360=args.mesh_only_360 and not args.side_by_side,
        side_by_side=args.side_by_side,
    )


if __name__ == '__main__':
    main()
