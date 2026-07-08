"""Plot pelvis height and trunk angle from video_demo.py signals.json."""
import argparse
import json
import os

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signals', type=str, default='video_out/signals.json')
    parser.add_argument('--out_folder', type=str, default='video_out')
    args = parser.parse_args()

    with open(args.signals, encoding='utf-8') as f:
        signals = json.load(f)
    if not signals:
        print('No signals found.')
        return

    times = [row['time_sec'] for row in signals]
    pelvis_h = [row['pelvis_height'] for row in signals]
    trunk = [row['trunk_angle_deg'] for row in signals]

    os.makedirs(args.out_folder, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(times, pelvis_h, color='#2563eb')
    axes[0].set_ylabel('Pelvis height (rel.)')
    axes[0].set_title('Pelvis height over time')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, trunk, color='#dc2626')
    axes[1].set_ylabel('Trunk angle (deg)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Trunk angle vs vertical')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(args.out_folder, 'signals_plot.png')
    fig.savefig(plot_path, dpi=150)
    print(f'Saved plot: {plot_path}')


if __name__ == '__main__':
    main()
