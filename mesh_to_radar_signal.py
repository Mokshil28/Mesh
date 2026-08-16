#!/usr/bin/env python3
"""Create a simple synthetic FMCW-radar dataset from a 4D mesh sequence.

This is intended for simulation/training.  It turns each mesh frame into a
complex range profile and range--Doppler maps; it is not a replacement for
calibrated measurements from a physical radar.

Example:
  python3 mesh_to_radar_signal.py \
    fall_out/sam_gpu_batch_raw/039_V39/fall_022/mesh_4d_individual/1 \
    --out fall_out/radar_sim/039_V39_fall_022
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

C_M_PER_S = 299_792_458.0


def read_vertices(path: Path) -> np.ndarray:
    """Read x/y/z vertices from the binary-little-endian PLY files SAM emits."""
    with path.open("rb") as f:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            header.extend(line)
        lines = header.decode("ascii").splitlines()
        if "format binary_little_endian 1.0" not in lines:
            raise ValueError(f"Only binary_little_endian PLY is supported: {path}")
        vertex_count = 0
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in lines:
            fields = line.split()
            if fields[:2] == ["element", "vertex"]:
                vertex_count, in_vertex = int(fields[2]), True
            elif fields and fields[0] == "element":
                in_vertex = False
            elif in_vertex and fields[:1] == ["property"] and len(fields) == 3:
                properties.append((fields[1], fields[2]))
        formats = {"char": "b", "uchar": "B", "short": "h", "ushort": "H",
                   "int": "i", "uint": "I", "float": "f", "double": "d"}
        fmt = "<" + "".join(formats[kind] for kind, _ in properties)
        row_size = struct.calcsize(fmt)
        raw = f.read(vertex_count * row_size)
    if len(raw) != vertex_count * row_size:
        raise ValueError(f"Truncated vertex data: {path}")
    values = np.frombuffer(raw, dtype=np.dtype([(name, "<f4" if kind == "float" else
        "<f8" if kind == "double" else "u1" if kind == "uchar" else "i1" if kind == "char" else
        "<i4") for kind, name in properties]), count=vertex_count)
    # SAM meshes use float x/y/z.  A clear error is safer than silently using
    # a wrong dtype for unusual PLY property layouts.
    if any(kind != "float" for kind, _ in properties[:3]) or [n for _, n in properties[:3]] != ["x", "y", "z"]:
        raise ValueError(f"Unexpected vertex layout in {path}; expected float x/y/z first")
    return np.column_stack((values["x"], values["y"], values["z"])).astype(np.float64)


def frame_vertices(mesh_dir: Path, points: int, seed: int) -> tuple[list[Path], list[np.ndarray]]:
    files = sorted(mesh_dir.glob("*.ply"))
    if not files:
        raise FileNotFoundError(f"No .ply frames in {mesh_dir}")
    rng = np.random.default_rng(seed)
    raw_frames = [read_vertices(file) for file in files]
    # SAM normally preserves vertex ordering across a sequence. Sampling the
    # same indices in every frame lets us derive each surface point's radial
    # velocity, rather than comparing unrelated random vertices.
    common_count = min(len(vertices) for vertices in raw_frames)
    indices = np.arange(common_count)
    if common_count > points:
        indices = rng.choice(common_count, size=points, replace=False)
    frames = [vertices[indices] for vertices in raw_frames]
    return files, frames


def simulate(frames: list[np.ndarray], fps: float, carrier_ghz: float, bandwidth_mhz: float,
             range_bins: int, target_range: float, mesh_scale: float,
             radar_axis: str, radar_sign: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return complex range profiles and a calibrated metadata dictionary."""
    wavelength = C_M_PER_S / (carrier_ghz * 1e9)
    range_resolution = C_M_PER_S / (2 * bandwidth_mhz * 1e6)
    ranges = (np.arange(range_bins) + 0.5) * range_resolution
    profiles = np.zeros((len(frames), range_bins), dtype=np.complex64)

    initial_center = frames[0].mean(axis=0)
    axis = {"x": 0, "y": 1, "z": 2}[radar_axis]
    transverse = [i for i in range(3) if i != axis]
    centroid_ranges = []
    for index, vertices in enumerate(frames):
        # Rotate/relabel the chosen source axis into the radar boresight (+X).
        # The first-frame body centre is target_range metres away, while all
        # later whole-body and limb motion is retained.
        delta = (vertices - initial_center) * mesh_scale
        xyz = np.column_stack((target_range + radar_sign * delta[:, axis],
                               delta[:, transverse[0]], delta[:, transverse[1]]))
        centroid_ranges.append(float(np.linalg.norm(xyz.mean(axis=0))))
        distance = np.linalg.norm(xyz, axis=1)
        valid = (distance >= 0) & (distance < range_bins * range_resolution)
        distance = distance[valid]
        # Each sampled surface point is a weak isotropic scatterer.  The phase
        # is the two-way propagation phase; 1/r^2 is an amplitude-only path loss.
        phase = np.exp(-1j * 4 * np.pi * distance / wavelength) / np.maximum(distance, 0.25) ** 2
        bins = np.floor(distance / range_resolution).astype(int)
        np.add.at(profiles[index], bins, phase)

    metadata = {
        "frame_rate_hz": fps,
        "carrier_frequency_ghz": carrier_ghz,
        "bandwidth_mhz": bandwidth_mhz,
        "wavelength_m": wavelength,
        "range_resolution_m": range_resolution,
        "range_bin_centres_m": ranges,
        "target_range_m": target_range,
        "mesh_scale_to_m": mesh_scale,
        "source_mesh_radar_axis": radar_axis,
        "source_mesh_radar_sign": radar_sign,
        "centroid_range_m": np.array(centroid_ranges),
        "time_s": np.arange(len(frames)) / fps,
    }
    return profiles, ranges, metadata


def range_doppler(profiles: np.ndarray, fps: float, carrier_ghz: float, window: int) -> tuple[np.ndarray, np.ndarray]:
    window = min(window, len(profiles))
    if window < 2:
        raise ValueError("Need at least two mesh frames for Doppler")
    slow = profiles[-window:] * np.hanning(window)[:, None]
    rd = np.fft.fftshift(np.fft.fft(slow, axis=0), axes=0)
    doppler_hz = np.fft.fftshift(np.fft.fftfreq(window, d=1 / fps))
    wavelength = C_M_PER_S / (carrier_ghz * 1e9)
    return rd.astype(np.complex64), doppler_hz * wavelength / 2


def save_preview(path: Path, rd: np.ndarray, ranges: np.ndarray, velocity: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    image = 20 * np.log10(np.abs(rd).T + 1e-8)
    image -= image.max()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(image, origin="lower", aspect="auto", cmap="magma",
              extent=[velocity[0], velocity[-1], ranges[0], ranges[-1]], vmin=-45, vmax=0)
    ax.set(xlabel="Radial velocity (m/s)", ylabel="Range (m)", title="Synthetic range–Doppler map (final window)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def save_range_time(path: Path, profiles: np.ndarray, ranges: np.ndarray, fps: float) -> None:
    """Save the complete clip's echo energy, not just one Doppler window."""
    import matplotlib.pyplot as plt
    image = 20 * np.log10(np.abs(profiles).T + 1e-8)
    image -= image.max()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(image, origin="lower", aspect="auto", cmap="magma",
              extent=[0, (len(profiles) - 1) / fps, ranges[0], ranges[-1]], vmin=-45, vmax=0)
    ax.set(xlabel="Time (s)", ylabel="Range (m)", title="Synthetic range–time map (whole clip)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def mesh_micro_doppler(frames: list[np.ndarray], fps: float, target_range: float, mesh_scale: float,
                       radar_axis: str, radar_sign: float, velocity_bins: int,
                       velocity_limit: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make a velocity-time map from tracked mesh vertices.

    This is a mesh-derived micro-Doppler proxy: each vertex's finite-difference
    radial velocity is accumulated at every video frame.  It preserves fast
    limb/torso motion visible in the mesh, but it is not raw chirp-rate IQ.
    """
    axis = {"x": 0, "y": 1, "z": 2}[radar_axis]
    transverse = [i for i in range(3) if i != axis]
    initial_center = frames[0].mean(axis=0)
    positions = []
    centres = []
    for vertices in frames:
        delta = (vertices - initial_center) * mesh_scale
        xyz = np.column_stack((target_range + radar_sign * delta[:, axis],
                               delta[:, transverse[0]], delta[:, transverse[1]]))
        positions.append(xyz)
        centres.append(xyz.mean(axis=0))
    distances = np.linalg.norm(np.stack(positions), axis=2)
    radial_velocity = np.diff(distances, axis=0) * fps
    edges = np.linspace(-velocity_limit, velocity_limit, velocity_bins + 1)
    centres_v = (edges[:-1] + edges[1:]) / 2
    power = np.zeros((len(radial_velocity), velocity_bins), dtype=np.float32)
    for i, (velocity, distance) in enumerate(zip(radial_velocity, distances[1:])):
        # 1/r^4 is power loss for a monostatic radar; it prevents distant
        # vertices from dominating the feature merely by count.
        power[i], _ = np.histogram(velocity, bins=edges, weights=1 / np.maximum(distance, .25) ** 4)
    body_range = np.linalg.norm(np.stack(centres), axis=1)
    return power, centres_v, body_range, np.arange(1, len(frames)) / fps


def save_micro_doppler(path: Path, power: np.ndarray, velocity: np.ndarray, time_s: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    image = 10 * np.log10(power.T + 1e-12)
    image -= image.max()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(image, origin="lower", aspect="auto", cmap="magma",
              extent=[time_s[0], time_s[-1], velocity[0], velocity[-1]], vmin=-45, vmax=0)
    ax.set(xlabel="Time (s)", ylabel="Radial velocity (m/s)",
           title="Mesh-derived micro-Doppler proxy (whole clip)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh_dir", type=Path, help="Directory containing sequential .ply mesh frames")
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    p.add_argument("--fps", type=float, default=30, help="Mesh/video frame rate")
    p.add_argument("--carrier-ghz", type=float, default=77, help="Radar carrier frequency")
    p.add_argument("--bandwidth-mhz", type=float, default=1000, help="FMCW sweep bandwidth")
    p.add_argument("--range-bins", type=int, default=128)
    p.add_argument("--target-range", type=float, default=4.0, help="First-frame body-centre range in metres")
    p.add_argument("--mesh-scale", type=float, default=1.0, help="Multiply mesh units by this to obtain metres")
    p.add_argument("--radar-axis", choices=("x", "y", "z"), default="z",
                   help="Mesh axis pointing toward/away from the radar (inspect/calibrate this)")
    p.add_argument("--radar-sign", type=float, choices=(-1.0, 1.0), default=-1.0,
                   help="Whether positive or negative selected-axis motion increases radar range")
    p.add_argument("--points", type=int, default=2000, help="Surface vertices sampled per frame")
    p.add_argument("--doppler-window", type=int, default=32)
    p.add_argument("--velocity-bins", type=int, default=128)
    p.add_argument("--velocity-limit", type=float, default=8.0,
                   help="Maximum absolute radial velocity in the micro-Doppler proxy (m/s)")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    if args.fps <= 0 or args.carrier_ghz <= 0 or args.bandwidth_mhz <= 0 or args.points <= 0:
        p.error("fps, carrier frequency, bandwidth, and points must be positive")

    files, frames = frame_vertices(args.mesh_dir, args.points, args.seed)
    profiles, ranges, meta = simulate(frames, args.fps, args.carrier_ghz, args.bandwidth_mhz,
                                      args.range_bins, args.target_range, args.mesh_scale,
                                      args.radar_axis, args.radar_sign)
    rd, velocity = range_doppler(profiles, args.fps, args.carrier_ghz, args.doppler_window)
    micro_power, micro_velocity, body_range, micro_time = mesh_micro_doppler(
        frames, args.fps, args.target_range, args.mesh_scale, args.radar_axis,
        args.radar_sign, args.velocity_bins, args.velocity_limit)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "synthetic_radar.npz", range_profiles=profiles,
                        range_doppler=rd, range_m=ranges, velocity_mps=velocity,
                        micro_doppler_power=micro_power, micro_doppler_velocity_mps=micro_velocity,
                        micro_doppler_time_s=micro_time, body_centre_range_m=body_range,
                        frame_files=np.array([f.name for f in files]), **meta)
    save_preview(args.out / "range_doppler.png", rd, ranges, velocity)
    save_range_time(args.out / "range_time.png", profiles, ranges, args.fps)
    save_micro_doppler(args.out / "micro_doppler.png", micro_power, micro_velocity, micro_time)
    print(f"Wrote {args.out / 'synthetic_radar.npz'}")
    print(f"Wrote {args.out / 'range_doppler.png'}")
    print(f"Wrote {args.out / 'range_time.png'}")
    print(f"Wrote {args.out / 'micro_doppler.png'}")
    print(f"Frames: {len(files)} | range resolution: {meta['range_resolution_m']:.3f} m")


if __name__ == "__main__":
    main()
