#!/usr/bin/env python3
"""Generate synthetic radar arrays directly from a fitted SMPL parameter file.

The output matches ``mesh_to_radar_signal.py`` and is suitable as an mmAP
preparation input (range/Doppler and micro-Doppler arrays).  It is a physics-
inspired simulation, not measured radar data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import smplx
import torch

from mesh_to_radar_signal import (
    mesh_micro_doppler, range_doppler, save_micro_doppler, save_preview,
    save_range_time, simulate,
)


def smpl_vertices(params: Path, model_path: Path) -> tuple[list[np.ndarray], float, np.ndarray]:
    """Rebuild the per-frame body vertices saved by sam_mesh_to_smpl.py."""
    with np.load(params) as saved:
        required = ("betas", "global_orient", "body_pose", "transl")
        absent = [key for key in required if key not in saved]
        if absent:
            raise ValueError(f"{params} is missing: {', '.join(absent)}")
        orient = saved["global_orient"].astype(np.float32)
        pose = saved["body_pose"].astype(np.float32)
        transl = saved["transl"].astype(np.float32)
        betas = saved["betas"].astype(np.float32)
        fps = float(saved["fps"]) if "fps" in saved else 30.0
        names = saved["frame_files"] if "frame_files" in saved else np.array(
            [f"frame_{index:06d}.ply" for index in range(len(orient))]
        )
    frames = len(orient)
    if frames < 2 or pose.shape[0] != frames or transl.shape[0] != frames:
        raise ValueError("SMPL pose/orientation/translation arrays must have the same frame count (>=2)")
    model = smplx.SMPL(str(model_path), batch_size=frames, create_transl=False)
    with torch.no_grad():
        out = model(
            betas=torch.from_numpy(betas).expand(frames, -1),
            global_orient=torch.from_numpy(orient),
            body_pose=torch.from_numpy(pose),
        )
    vertices = out.vertices.numpy() + transl[:, None, :]
    return [frame.astype(np.float64) for frame in vertices], fps, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smpl_params", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smpl-model", type=Path,
                        default=Path.home() / ".cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl")
    parser.add_argument("--target-range", type=float, default=4.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--radar-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--radar-sign", choices=(-1.0, 1.0), type=float, default=-1.0)
    parser.add_argument("--points", type=int, default=2000)
    parser.add_argument("--range-bins", type=int, default=128)
    parser.add_argument("--doppler-window", type=int, default=32)
    parser.add_argument("--velocity-bins", type=int, default=128)
    parser.add_argument("--velocity-limit", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not args.smpl_model.is_file():
        parser.error(f"SMPL model not found: {args.smpl_model}")

    frames, fps, names = smpl_vertices(args.smpl_params, args.smpl_model)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(frames[0]))
    if len(indices) > args.points:
        indices = rng.choice(indices, size=args.points, replace=False)
    frames = [frame[indices] for frame in frames]
    profiles, ranges, meta = simulate(frames, fps, 77.0, 1000.0, args.range_bins,
                                      args.target_range, args.mesh_scale,
                                      args.radar_axis, args.radar_sign)
    rd, velocity = range_doppler(profiles, fps, 77.0, args.doppler_window)
    micro, micro_velocity, body_range, micro_time = mesh_micro_doppler(
        frames, fps, args.target_range, args.mesh_scale, args.radar_axis,
        args.radar_sign, args.velocity_bins, args.velocity_limit)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "synthetic_radar.npz", range_profiles=profiles,
                        range_doppler=rd, range_m=ranges, velocity_mps=velocity,
                        micro_doppler_power=micro, micro_doppler_velocity_mps=micro_velocity,
                        micro_doppler_time_s=micro_time, body_centre_range_m=body_range,
                        frame_files=names, source="fitted SMPL parameters", **meta)
    save_preview(args.out / "range_doppler.png", rd, ranges, velocity)
    save_range_time(args.out / "range_time.png", profiles, ranges, fps)
    save_micro_doppler(args.out / "micro_doppler.png", micro, micro_velocity, micro_time)
    print(f"Wrote {args.out / 'synthetic_radar.npz'}")
    print(f"Frames: {len(frames)} | range resolution: {meta['range_resolution_m']:.3f} m")


if __name__ == "__main__":
    main()
