#!/usr/bin/env python3
"""Fit SMPL parameters to a SAM-Body4D mesh sequence and export a 360° viewer.

This uses only SAM mesh output; it does not call HMR.  The fitted parameters
are an approximation of the SAM surface and should be quality-checked before
training.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
import trimesh
import smplx

from mesh_viewer_html import align_for_ground_view, build_html, find_grounded_frame, quantize


def read_vertices(path: Path) -> np.ndarray:
    """Read SAM's binary-little-endian PLY vertices (x/y/z are first fields)."""
    with path.open("rb") as f:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            header.extend(line)
        lines = header.decode("ascii").splitlines()
        n = next(int(line.split()[2]) for line in lines if line.startswith("element vertex "))
        props, in_vertex = [], False
        for line in lines:
            x = line.split()
            if x[:2] == ["element", "vertex"]:
                in_vertex = True
            elif x[:1] == ["element"]:
                in_vertex = False
            elif in_vertex and x[:1] == ["property"] and len(x) == 3:
                props.append((x[1], x[2]))
        kinds = {"char": "b", "uchar": "B", "short": "h", "ushort": "H", "int": "i", "uint": "I", "float": "f", "double": "d"}
        fmt = "<" + "".join(kinds[k] for k, _ in props)
        size = struct.calcsize(fmt)
        raw = f.read(n * size)
    if [name for _, name in props[:3]] != ["x", "y", "z"] or any(kind != "float" for kind, _ in props[:3]):
        raise ValueError(f"Unsupported PLY vertex layout: {path}")
    # SAM PLYs use three float32 coordinates followed by four uchar colours.
    row = np.frombuffer(raw, dtype=np.uint8).reshape(n, size)
    return np.frombuffer(row[:, :12].copy().tobytes(), dtype="<f4").reshape(n, 3).astype(np.float32)


def load_sequence(mesh_dir: Path, sample_points: int, seed: int) -> tuple[list[Path], np.ndarray]:
    # External drives copied through Finder can contain macOS resource-fork
    # sidecars named ._00000001.ply.  They are not PLY meshes.
    files = sorted(p for p in mesh_dir.glob("*.ply") if not p.name.startswith("._"))
    if len(files) < 2:
        raise FileNotFoundError(f"Need at least two .ply frames in {mesh_dir}")
    raw = [read_vertices(p) for p in files]
    common = min(len(v) for v in raw)
    rng = np.random.default_rng(seed)
    indices = np.arange(common) if common <= sample_points else rng.choice(common, sample_points, replace=False)
    return files, np.stack([v[indices] for v in raw])


def fit_smpl(target: np.ndarray, model_path: Path, iterations: int, device: torch.device) -> dict[str, np.ndarray]:
    """Jointly fit all frames with surface Chamfer and temporal smoothing."""
    target_t = torch.tensor(target, device=device)
    frames = len(target)
    model = smplx.SMPL(str(model_path), batch_size=frames, create_transl=False).to(device)
    # A shared body shape; pose and translation vary with the fall motion.
    betas = torch.zeros((1, 10), device=device, requires_grad=True)
    orient = torch.zeros((frames, 3), device=device, requires_grad=True)
    pose = torch.zeros((frames, 69), device=device, requires_grad=True)
    transl = torch.tensor(target.mean(axis=1), device=device, requires_grad=True)
    # Template is centred near the origin; map it roughly to SAM's first body.
    with torch.no_grad():
        transl.sub_(transl[0] - target_t[0].mean(0))
    optimizer = torch.optim.Adam([betas, orient, pose, transl], lr=0.035)
    sample = min(512, model.v_template.shape[0])
    gen = torch.Generator(device=device).manual_seed(17)
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        output = model(betas=betas.expand(frames, -1), global_orient=orient, body_pose=pose)
        verts = output.vertices + transl[:, None, :]
        smpl_idx = torch.randperm(verts.shape[1], generator=gen, device=device)[:sample]
        # Symmetric nearest-surface distance. Targets were already sampled
        # from the SAM surface, avoiding a large all-vertex distance matrix.
        distance = torch.cdist(verts[:, smpl_idx], target_t)
        chamfer = distance.min(2).values.mean() + distance.min(1).values.mean()
        smooth = ((pose[1:] - pose[:-1]) ** 2).mean() + ((transl[1:] - transl[:-1]) ** 2).mean() * 0.5
        shape_prior = (betas ** 2).mean()
        loss = chamfer + 0.08 * smooth + 0.005 * shape_prior
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == iterations:
            print(f"fit {step + 1}/{iterations}: chamfer={chamfer.item():.4f} m loss={loss.item():.4f}", flush=True)
    with torch.no_grad():
        output = model(betas=betas.expand(frames, -1), global_orient=orient, body_pose=pose)
        verts = (output.vertices + transl[:, None, :]).cpu().numpy().astype(np.float32)
    return {"betas": betas.detach().cpu().numpy().astype(np.float32),
            "global_orient": orient.detach().cpu().numpy().astype(np.float32),
            "body_pose": pose.detach().cpu().numpy().astype(np.float32),
            "transl": transl.detach().cpu().numpy().astype(np.float32),
            "fitted_vertices": verts}


def stabilize_rigid_jitter(sequence: np.ndarray, window: int) -> np.ndarray:
    """Remove isolated whole-body translation jitter without freezing a fall.

    A short median trajectory follows the sustained downward fall while rejecting
    one/two-frame SAM tracking jumps. The same correction is applied to every
    vertex, so body pose and mesh shape are unchanged.
    """
    if window < 3 or len(sequence) < 3:
        return sequence
    if window % 2 == 0:
        window += 1
    centres = sequence.mean(axis=1)
    radius = window // 2
    padded = np.pad(centres, ((radius, radius), (0, 0)), mode="edge")
    stable = np.stack([np.median(padded[i:i + window], axis=0) for i in range(len(centres))])
    return sequence - (centres - stable)[:, None, :]


def center_fall_at_start(sequence: np.ndarray) -> np.ndarray:
    """Make an inspection view with the upright body centred on its floor.

    Initial feet define y=0.  Horizontal centre movement is removed so an
    orbit viewer follows the person rather than looking like a moving camera;
    vertical motion is intentionally preserved to show the fall.
    """
    out = sequence.copy()
    start = out[0].mean(axis=0)
    ground_y = float(out[0, :, 1].min())
    centres = out.mean(axis=1)
    out[:, :, 0] -= centres[:, None, 0]
    out[:, :, 2] -= centres[:, None, 2]
    out[:, :, 1] -= ground_y
    return out


def write_360_viewer(mesh_files: list[Path], full_mesh_dir: Path, out_html: Path, fps: float,
                     stabilize_window: int = 0, center_at_start: bool = False) -> None:
    full = np.stack([read_vertices(p) for p in mesh_files])
    if stabilize_window:
        full = stabilize_rigid_jitter(full, stabilize_window)
    first = trimesh.load(str(mesh_files[0]), process=False)
    faces = np.asarray(first.faces, dtype=np.uint32)
    grounded = find_grounded_frame(full)
    view = center_fall_at_start(full) if center_at_start else align_for_ground_view(full, grounded)
    q, lo, span = quantize(view)
    html = build_html(q, lo, span, faces, fps, {"title": f"SAM 360° mesh · {full_mesh_dir.parent.name}",
        "fps": fps, "grounded_frame": grounded,
        "note": "SAM mesh only. Pause and drag to inspect the fall from any angle."
                + (f" Rigid jitter stabilized with a {stabilize_window}-frame median." if stabilize_window else "")
                + (" Starts centered on the initial ground plane; horizontal translation is viewer-only." if center_at_start else "")}, grounded,
        mesh_only_360=True, pause_inspect_360=True)
    out_html.write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_dir", type=Path, help="SAM mesh_4d_individual/<person-id> directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smpl-model", type=Path, default=Path.home() / ".cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--points", type=int, default=384, help="SAM surface points sampled per frame")
    ap.add_argument("--iterations", type=int, default=80)
    ap.add_argument("--stabilize-window", type=int, default=3,
                    help="Odd median window for 360-view rigid-jitter stabilization; 0 disables it")
    ap.add_argument("--center-at-start", action="store_true",
                    help="Viewer only: centre the person at the initial floor and retain vertical fall")
    ap.add_argument("--no-viewer", action="store_true",
                    help="Write only smpl_params.npz; useful for large parameter-only batches")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if not args.smpl_model.is_file():
        ap.error(f"SMPL model not found: {args.smpl_model}")
    files, target = load_sequence(args.mesh_dir, args.points, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(files)} SAM mesh frames; fitting SMPL on CPU.")
    result = fit_smpl(target, args.smpl_model, args.iterations, torch.device("cpu"))
    np.savez_compressed(args.out / "smpl_params.npz", **{k: v for k, v in result.items() if k != "fitted_vertices"},
                        fps=np.float32(args.fps), frame_files=np.array([p.name for p in files]),
                        source="SAM-Body4D mesh fit")
    if not args.no_viewer:
        write_360_viewer(files, args.mesh_dir, args.out / "sam_mesh_360.html", args.fps,
                         args.stabilize_window, args.center_at_start)
    print(f"Wrote {args.out / 'smpl_params.npz'}")
    if not args.no_viewer:
        print(f"Wrote {args.out / 'sam_mesh_360.html'}")


if __name__ == "__main__":
    main()
