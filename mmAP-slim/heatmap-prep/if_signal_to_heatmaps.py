#!/usr/bin/env python3
"""Convert simulator IF signals into mmAP-compatible radar heatmaps.

The fall simulator writes complex IF data as ``if_signal.npy`` with shape
``(frames, tx, rx, chirps, adc_samples)``.  This implements the same signal
processing convention as ``bin2npy/main_time_dop_read_all.py`` but reads that
array directly and avoids its unavailable ``torchaudio`` dependency.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft


def db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(values, 1e-12))


def sliding_sums(values: np.ndarray, window: int, hop: int) -> np.ndarray:
    """Sum a 1D time series over overlapping windows without huge tensors."""
    total = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    starts = np.arange(0, len(values) - window + 1, hop)
    return total[starts + window] - total[starts]


def heatmaps(signal: np.ndarray, *, range_bins: int = 150, window: int = 256,
             hop: int = 16, angle_bins: int = 128) -> dict[str, np.ndarray]:
    if signal.ndim != 5:
        raise ValueError(f"Expected (frames, tx, rx, chirps, samples); got {signal.shape}")
    frames, tx, rx, chirps, samples = signal.shape
    if tx < 3 or rx < 4:
        raise ValueError(f"Signal shape is too small for mmAP settings: {signal.shape}")

    # Keep the processing deterministic for repeatable train/test data.
    virtual = np.asarray(signal, dtype=np.complex64).transpose(1, 2, 0, 3, 4)
    virtual = virtual.reshape(tx * rx, frames, chirps, samples)
    ranged = np.fft.fft(virtual * np.hamming(samples), axis=-1)[..., :range_bins]
    ranged = ranged - ranged.mean(axis=-2, keepdims=True)  # clutter removal across chirps
    sequence = ranged.reshape(tx * rx, frames * chirps, range_bins)
    if sequence.shape[1] < window:
        raise ValueError("Not enough IF samples for one heatmap window")

    # Time-range: energy per range bin in each sliding time window.
    range_energy = np.abs(sequence[0])  # (time, range)
    range_total = np.vstack((np.zeros((1, range_bins)), np.cumsum(range_energy, axis=0)))
    starts = np.arange(0, sequence.shape[1] - window + 1, hop)
    time_range = db((range_total[starts + window] - range_total[starts]).T)

    # Time-angle: use the same eight virtual-array positions as mmAP's
    # simulator path: TX0 + TX2, four receivers each.
    array_idx = [0, 1, 2, 3, 8, 9, 10, 11]
    if sequence.shape[0] <= max(array_idx):
        array_idx = list(range(min(sequence.shape[0], 8)))
    angle_source = sequence[array_idx]
    angled = np.fft.fftshift(np.fft.fft(angle_source, n=angle_bins, axis=0), axes=0)
    angle_energy = np.abs(angled).sum(axis=2)  # (angle, time)
    time_angle = db(np.stack([sliding_sums(row, window, hop) for row in angle_energy]))

    # Time-doppler: STFT of each range-bin time series, then sum energy over
    # range. torch.stft replaces the original torchaudio Spectrogram call.
    doppler_input = sequence[0].T  # (range, time)
    _, _, doppler = stft(
        doppler_input, fs=1.0, window=np.hamming(window), nperseg=window,
        noverlap=window - hop, nfft=window, return_onesided=False,
        boundary=None, padded=False, axis=-1,
    )
    time_doppler = db(np.abs(np.fft.fftshift(doppler, axes=1)).sum(axis=0))

    return {
        "angle": time_angle.astype(np.float32),
        "doppler": time_doppler.astype(np.float32),
        "range": time_range.astype(np.float32),
    }


def save_preview(data: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.imshow(data, origin="lower", aspect="auto", cmap="magma")
    plt.title(title)
    plt.xlabel("time window")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="Simulator if_signal.npy")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--range-bins", type=int, default=150)
    ap.add_argument("--no-preview", action="store_true",
                    help="Write only .npy arrays (recommended for large batches)")
    args = ap.parse_args()

    signal = np.load(args.input, allow_pickle=False)
    outputs = heatmaps(signal, range_bins=args.range_bins)
    args.out.mkdir(parents=True, exist_ok=True)
    for modality, values in outputs.items():
        np.save(args.out / f"{modality}.npy", values)
        if not args.no_preview:
            save_preview(values, f"{modality.title()} heatmap", args.out / f"{modality}.png")
        print(f"{modality}: {values.shape} -> {args.out / (modality + '.npy')}")


if __name__ == "__main__":
    main()
