# Running Mesh-main on this Mac

## What is already installed

- **Miniforge** at `~/miniforge3`
- **Conda env** `4D-humans` (Python 3.10, PyTorch 1.11, detectron2 0.6, hmr2, etc.)
- **HMR2 model weights** (~5 GB) at `~/.cache/4DHumans/`

## What you still need (manual)

### 1. SMPL body model (required)

The demo cannot run without this file.

1. Register at https://smplify.is.tue.mpg.de/
2. Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
3. Copy it to:

```
/Users/mshah76/Downloads/Mesh-main/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

4. Convert it once:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate 4D-humans
cd /Users/mshah76/Downloads/Mesh-main
python -c "from hmr2.models import convert_pkl; convert_pkl('data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', '$HOME/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl')"
```

### 2. Fall video clips (required to re-run processing)

Put your input `.mp4` files here:

```
/Users/mshah76/Downloads/Mesh-main/data/fall_dataset/clips/
```

(Previous **output** videos are in `fall_out/FALL_*/overlay.mp4`, but the original input clips were not included in the zip.)

## Run

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate 4D-humans
cd /Users/mshah76/Downloads/Mesh-main
bash run_all_falls.sh
```

Or a single video:

```bash
python video_demo.py --video path/to/clip.mp4 --out_folder fall_out --detector regnety
```

## Re-install everything from scratch

```bash
bash setup_mac.sh
```

## Zip results for your professor

```bash
bash share_with_prof.sh
```
