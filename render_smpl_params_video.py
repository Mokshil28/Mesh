#!/usr/bin/env python3
"""Render an animated SMPL reconstruction from smpl_params.npz to MP4."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import smplx
import torch
from render_mesh_video import render_frame

def main():
    p=argparse.ArgumentParser(); p.add_argument('params', type=Path); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--width', type=int, default=640); p.add_argument('--height', type=int, default=480)
    p.add_argument('--smpl-model', type=Path, default=Path.home()/'.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl')
    a=p.parse_args()
    with np.load(a.params) as d:
        betas=d['betas'].astype(np.float32); orient=d['global_orient'].astype(np.float32); pose=d['body_pose'].astype(np.float32); transl=d['transl'].astype(np.float32); fps=float(d['fps'])
    model=smplx.SMPL(str(a.smpl_model), batch_size=len(orient), create_transl=False)
    with torch.no_grad():
        verts=model(betas=torch.from_numpy(betas).expand(len(orient),-1), global_orient=torch.from_numpy(orient), body_pose=torch.from_numpy(pose)).vertices.numpy()+transl[:,None,:]
    center=verts.reshape(-1,3).mean(0); radius=float(np.linalg.norm(verts-center,axis=2).max())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    writer=cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (a.width,a.height))
    if not writer.isOpened(): raise RuntimeError(f'Cannot open video writer: {a.out}')
    try:
        for i, frame in enumerate(verts):
            writer.write(render_frame(frame, model.faces, center, radius, a.width, a.height))
            if i % 30 == 0: print(f'{a.params.parent.name}: {i+1}/{len(verts)}', flush=True)
    finally: writer.release()
    print(f'Wrote {a.out}', flush=True)
if __name__=='__main__': main()
