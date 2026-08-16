# Binary fall vs non-fall preparation

The source heatmaps are complete-clip arrays.  `prepare_binary_fall_dataset.py`
validates them, assigns each subject/session folder to exactly one split, and
turns each clip into fixed 128-time-bin windows that the existing mmAP loader
can read.

## Current fall-only validation

```bash
python heatmap-prep/prepare_binary_fall_dataset.py \
  --fall-root '/Volumes/data/fall down/radar_data/mmAP_heatmap_gpu_clips' \
  --out '/Volumes/data/fall down/radar_data/fall_nonfall_binary'
```

This writes manifests only. It deliberately does not create a one-class
training dataset.

## After non-fall heatmaps arrive

**Recommendation:** keep non-fall ≈ fall count (~1:1) for the first training run.
A 1:10 non-fall majority makes the model lazy (always predict non-fall) and slows
iteration. Use all ~16k later with class weights / focal loss if you need more
motion diversity.

```bash
# Balanced (~1574 fall + ~1574 stratified non-fall), group-safe splits + windows
python heatmap-prep/prepare_binary_fall_dataset.py \
  --fall-root '/Volumes/data/fall down/radar_data/mmAP_heatmap_gpu_clips' \
  --nonfall-root '/Volumes/data/fall down/radar_data/mmAP_heatmap_nonfall_sim' \
  --out '/Volumes/data/fall down/radar_data/fall_nonfall_binary_balanced' \
  --balance-to-fall \
  --write-windows

python heatmap-prep/verify_binary_dataset.py \
  --out '/Volumes/data/fall down/radar_data/fall_nonfall_binary_balanced'
```

## Train binary classifier (SMPL heatmaps only)

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mmap
cd mmAP-slim
bash train_fall_nonfall_binary.sh
```

Uses `run_finetuning_heatmap_wholemodel.py` on the balanced SMPL dataset
(`fall_nonfall_binary_balanced`). Checkpoints and logs land in
`output/finetune/fall_nonfall_binary_smpl/`. Keep the test split sealed until
final evaluation.

The output is compatible with mmAP's `MultiTaskImageFolder` loader:

```text
dataset/train/{angle,doppler,range}/{fall,non_fall}/*.npy
dataset/val/{angle,doppler,range}/{fall,non_fall}/*.npy
dataset/test/{angle,doppler,range}/{fall,non_fall}/*.npy
```

Only training and validation are used while selecting a model. The test split
must remain untouched until final evaluation. The `Vxx` session/subject folder
is the split group, preventing windows from the same source session from
appearing in multiple splits.
