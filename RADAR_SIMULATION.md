# Mesh-to-radar simulation

`mesh_to_radar_signal.py` converts the sequential SAM-Body4D `.ply` meshes
into a **synthetic** 77 GHz FMCW-radar representation. It does not turn a
mesh into a signal that a physical radar can transmit or receive. Instead, it
models the echoes that an idealised radar at a chosen position would receive
from the moving mesh. This is useful for training and testing a fall-detection
model before collecting real radar data.

The conversion is:

```text
4D mesh vertices -> surface scatterers -> complex range profiles -> range-Doppler map
```

For every mesh frame, sampled vertices become weak reflectors. Their distance
to the radar determines the two-way phase and range bin. Changes over time
create Doppler, so the final range-Doppler map captures radial human motion.

## Run one clip

Activate the environment that contains NumPy (the project `4D-humans`
environment includes it), then run:

```bash
cd /Users/mshah76/Downloads/Mesh-main
conda activate 4D-humans
python mesh_to_radar_signal.py \
  fall_out/sam_gpu_batch_raw/039_V39/fall_022/mesh_4d_individual/1 \
  --out fall_out/radar_sim/039_V39_fall_022
```

The output folder contains:

- `synthetic_radar.npz`: model-ready arrays: `range_profiles` (complex,
  frames × range bins), `range_doppler` (complex, velocity × range), axes,
  frame names, and all simulation settings.
- `range_doppler.png`: an inspection image of the final Doppler window.
- `range_time.png`: full-clip echo energy by range and time.
- `micro_doppler.png`: full-clip mesh-derived radial-velocity map. This is a
  kinematic proxy from 30 FPS mesh motion, not chirp-rate raw radar IQ.

The defaults represent a 77 GHz radar with 1 GHz bandwidth: about 0.15 m
range resolution, a 4 m initial target range, 30 FPS mesh timing, and a
32-frame Doppler window. `range_time.png` shows the whole clip's motion;
`range_doppler.png` is only the final Doppler window.

## Important calibration choices

- `--target-range 4`: moves the first-frame body centre to 4 m in front of
  the simulated radar. Change this to match your intended installation.
- `--radar-axis z --radar-sign -1`: states which mesh coordinate is the
  radar's toward/away direction. This must match the camera/radar geometry.
  The default fits the downloaded SAM clip convention, where depth changes
  mostly appear along negative Z; it is not a general calibration.
- For a ceiling radar, use `--radar-axis y --radar-sign -1` when mesh Y is
  upward. A downward fall then increases the simulated sensor-to-body range.
- `--mesh-scale 1`: SAM mesh coordinates are assumed to be metres. Adjust it
  only after checking a known body height; a roughly 1.7 m adult should span
  about 1.7 units after scaling.
- `--fps 30`: must match the source clip frame rate for Doppler velocity to
  have correct units.
- This first version models the body only. It excludes room walls, furniture,
  radar antenna patterns, multipath, noise, and material-specific reflectivity.
  Those need to be added or calibrated against recordings from the exact radar
  device before treating the result as real-world radar data.
