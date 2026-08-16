# Radar-Based Fall Detection from Simulated Human Motion

This repo documents an end-to-end ML pipeline that turns short video clips of falls into simulated radar data, then trains a classifier to distinguish falls from non-falls. Large datasets, licensed body model files (SMPL), and trained checkpoints are kept offline due to size and licensing. The code here is fully inspectable; full retraining requires that private data.

---

## Process overview

| Step | What happens | Core tools |
|---|---|---|
| 1. Collect video clips | 2,000+ annotated fall/non-fall clips, 2–5 seconds each | Manual curation + clip organization |
| 2. Organize the dataset | Label and structure clips for downstream processing | Folder/manifest conventions in this repo |
| 3. Reconstruct 3D human mesh | Recover full-body 3D motion from video; represent with SMPL | [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-vith), SMPL |
| 4. Mesh → radar data | Simulate radar returns: IF signal → range, Doppler, angle heatmaps | Custom Python/NumPy radar simulator |
| 5. Dataset engineering | Fix FPS mismatch, balance classes, safe splits | Custom Python + Bash ETL |
| 6. Train the classifier | Fine-tune a multimodal ViT on the heatmaps | PyTorch, MultiMAE/mmAP, timm, einops |
| 7. Scaled training | Run on GPU clusters with fault recovery | CUDA, NVIDIA RTX A5000/A6000, tmux, conda, rsync |
| 8. Evaluate | Window- and clip-level Acc, F1, confusion matrices | scikit-learn, matplotlib |
| 9. Current work | Collecting real-world radar to test sim→real transfer | In progress |

---

## What I built, in detail

### 1. Collect video clips

I collected and curated a library of **2,000+ short annotated video clips**, typically **2–5 seconds** each, covering fall activities.

### 2. Organize the dataset

Clips were labeled and organized into a structured dataset so they could be processed consistently through the rest of the pipeline (fall vs non-fall folders, versioned clip IDs, QC of usable sequences).

### 3. Reconstruct 3D human mesh

Each video clip was reconstructed into a **3D human mesh** — a time sequence of 3D body geometry — so the radar simulator has a moving body to “illuminate,” not RGB pixels.

**Models / tools at this stage:**
- **[SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-vith)** (`facebook/sam-3d-body-vith` on Hugging Face) — Meta’s promptable full-body **3D human mesh recovery (HMR)** model for robust in-the-wild pose/mesh estimation ([model card](https://huggingface.co/facebook/sam-3d-body-vith), code: [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body))
- **SMPL** (Skinned Multi-Person Linear model) — compact parametric body shape/pose representation used as the body prior for downstream mesh export and radar simulation (separately licensed; **not** redistributed in this repo)

Supporting scripts in this repo clean and export motion for simulation:
- `export_mesh_sequence.py`
- `stabilize_smpl_params.py`
- `smooth_mesh_postprocess.py`
- `sam_mesh_to_smpl.py` / `batch_sam_mesh_to_smpl.sh` (mesh → SMPL bridging where needed)
- `render_smpl_params_video.py`, `render_mesh_video.py` (QC visualization)

### 4. Mesh → radar data

Using the reconstructed motion, a **custom radar simulation stack** computes what a virtual FMCW-style radar would detect from that body motion:

| Script | Role |
|---|---|
| `mesh_to_radar_signal.py` | Mesh vertices → synthetic radar (scatterers → range / Doppler) |
| `smpl_to_radar_signal.py` | SMPL-parameter path into the same sim |
| `RADAR_SIMULATION.md` | Physics/software notes for this stage |

This produces:

- **IF (intermediate frequency) signal** — raw simulated radar return for the motion  
- **Range heatmap** — distance of body / body parts over time  
- **Doppler–time heatmap** — radial speed over time  
- **Angle heatmap** — angular direction relative to the sensor over time  

Heatmaps are materialized into structured **NumPy `.npy`** tensors with `mmAP-slim/heatmap-prep/`:

| Script | Role |
|---|---|
| `if_signal_to_heatmaps.py` | IF / spectrogram → mmAP-ready heatmaps |
| `heatmap_sim_prep.py` | Simulation heatmap preparation helpers |
| `heatmap_rcd_prep.py` | Range / channel prep utilities |
| `materialize_nonfall_spectrograms.py` | Batch materialize spectrograms → `.npy` (also reused for 10 FPS fall) |

### 5. Dataset engineering — the key software engineering challenge

**The problem:** fall and non-fall sequences were originally generated at inconsistent frame rates (~30 FPS-style vs. 10 FPS). Left unfixed, this creates a “domain fingerprint” — the model could learn fall vs non-fall from timing artifacts rather than motion, producing misleading accuracy that would fail on real radar.

**The fix:** custom Python + Bash ETL rebuilt a **synchronized 10 FPS dataset** across both classes:

| Tool | Role |
|---|---|
| `mmAP-slim/heatmap-prep/prepare_binary_fall_dataset.py` | Balanced fall/non-fall windows, group-safe splits |
| `mmAP-slim/heatmap-prep/verify_binary_dataset.py` | Sanity-check dataset layout / counts |
| `mmAP-slim/heatmap-prep/calibrate_nonfall_to_fall.py` | Optional amplitude / norm alignment |
| `hpc/setup_10fps_mmap_pipeline.sh` | One-shot 10 FPS materialize + dataset build |

Included safeguards:
- **Class-balanced sampling** — non-fall down-sampled to match fall (~**1,372** each)  
- **Class-weighted cross-entropy** (`--class_weight` in training)  
- **Fixed 128-bin** temporal windows for consistent ViT input shape  
- **Group-safe splits** + **sealed test** held out until final reporting  

**Final matched training dataset: 2,744 clips (1,372 fall + 1,372 non-fall).**

### 6. Train the classifier

Fine-tuned **mmAP**, a multimodal Vision Transformer on **MultiMAE / MultiViT** (`multivit_base`), jointly encoding **range + Doppler + angle** heatmaps.

| Component | Role |
|---|---|
| **PyTorch** | Training framework |
| **timm** / **einops** | ViT building blocks / tensor ops |
| `run_finetuning_heatmap_wholemodel.py` | Main fine-tune entry point |
| `cfgs/finetune/fall_nonfall_binary_10fps.yaml` | Experiment config |
| `--class_balance` / `--class_weight` | Sampler + loss weighting |

Two training configurations were compared (sealed-test, **clip-level** mean P(fall)):

| Setting | Accuracy | Balanced Accuracy | F1 (macro) |
|---|---|---|---|
| Batch **64**, **64** epochs | ~98.6% | ~98.7% | ~0.98 |
| Batch **32**, **30** epochs | ~97.8% | ~98.2% | ~0.98 |

Best checkpoint selected by **validation balanced accuracy**, not raw accuracy alone.

### 7. Training at scale

Training ran on **CUDA**-enabled **NVIDIA RTX A5000 / A6000** GPUs, with:
- **tmux + conda** for long, resumable jobs (`hpc/setup_mmap_cci.sh`, train launchers in `hpc/`)
- **rsync / scp** for code and dataset sync
- **Checkpoint resume** after OOM / NaN so progress is not lost  
  - `hpc/run_mmap_10fps_train_cci.sh` (batch-32 path)  
  - `hpc/run_mmap_10fps_bs64_csgpu3.sh` (batch-64 path)

### 8. Evaluation harness

`mmAP-slim/eval_binary_fall_metrics.py` (+ **scikit-learn**, **matplotlib**) reports **accuracy, balanced accuracy, precision, recall, F1, and confusion matrices** at:

- **Window-level** — each short heatmap window  
- **Clip-level** — aggregate per video (majority vote or mean fall probability) ← closest to a deployed detector  

Launcher: `hpc/run_eval_10fps_metrics.sh`

**Headline sealed-test result: ~98.6% clip-level accuracy, ~98.7% balanced accuracy, F1 ≈ 0.98** (batch-64 model).

### 9. Current work: real-world data validation

The numbers above are on **simulated** radar. I am collecting **real physical radar** of fall and non-fall activity and routing it through the same heatmap pipeline so the sim-trained model can be tested on real hardware. Open question: does sim performance transfer?

---

## Offline artifacts (data drive)

Backed up outside git (external drive):

```text
…/fall down/backup_mmap_10fps/
  checkpoints/
    bs32_checkpoint-best.pth
    bs64_checkpoint-best.pth
  results_bs32/   # eval_val, eval_test, log.txt, train.log
  results_bs64/
```

Prepared dataset:

```text
…/fall down/radar_data/fall_nonfall_binary_10fps/dataset/{train,val,test}
```

---

## Repository map

| Path | Role |
|---|---|
| `mmAP-slim/` | Classifier code, MultiMAE modules, training & eval |
| `mmAP-slim/heatmap-prep/` | Signal → heatmap → dataset builders |
| `mmAP-slim/cfgs/` | YAML experiment configs |
| `mmAP-slim/eval_binary_fall_metrics.py` | Full metric suite |
| `hpc/` | Environment setup, dataset build, train/eval launchers |
| `mesh_to_radar_signal.py`, `smpl_to_radar_signal.py` | Mesh/SMPL → simulated radar |
| `export_mesh_*.py`, `render_*.py` | Mesh visualization / export |
| `RADAR_SIMULATION.md` | Radar simulation details |

Heatmap `.npy` datasets, trained `.pth` checkpoints, SMPL `.pkl` files, and private logs stay offline.

---

## Design choices

1. **Simulation-first** — fast iteration without staging thousands of real falls; sim→real is the active validation phase  
2. **Matched 10 FPS** — removes a spurious domain fingerprint between fall and non-fall  
3. **Balanced classes + class-weighted loss** — two safeguards against majority-class bias  
4. **Sealed test set** — val for model selection; test only for final numbers  
5. **Clip-level metrics** — closer to how a real-time detector would decide  
6. **Balanced accuracy / F1** — more honest than accuracy alone when a split is uneven  

---

## Setup (software only — full reproduction needs private data)

You can install dependencies and inspect all code paths. Full retrain needs the private 10 FPS heatmaps, SMPL license files, and checkpoints.

```bash
git clone <THIS_REPO_URL> Mesh-main
cd Mesh-main

conda create -n mmap-gpu python=3.10 pip -y
conda activate mmap-gpu

# Linux + NVIDIA GPU (CUDA 12.1 wheels):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Mac (CPU / Apple Silicon):
# pip install torch torchvision

pip install -r mmAP-slim/requirements.txt
pip install 'timm==0.4.12' 'einops==0.3.2' 'numpy==1.26.4' \
  scikit-learn matplotlib tqdm PyYAML opencv-python-headless

python - <<'PY'
import torch, timm
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("timm", timm.__version__)
PY
```

**SAM 3D Body** (mesh recovery, separate from mmAP training) is obtained from Hugging Face after accepting the model terms:  
[facebook/sam-3d-body-vith](https://huggingface.co/facebook/sam-3d-body-vith) — follow Meta’s [INSTALL / demo instructions](https://github.com/facebookresearch/sam-3d-body) for that stack.

### With the private dataset + checkpoint

```bash
conda activate mmap-gpu
cd mmAP-slim

python eval_binary_fall_metrics.py \
  --checkpoint /path/to/checkpoint-best.pth \
  --data_root /path/to/fall_nonfall_binary_10fps/dataset \
  --split test \
  --output_dir /tmp/mmap_eval_test
```

```bash
python run_finetuning_heatmap_wholemodel.py \
  --config cfgs/finetune/fall_nonfall_binary_10fps.yaml \
  --data_path /path/to/dataset/train \
  --eval_data_path /path/to/dataset/val \
  --nb_classes 2 \
  --batch_size 64 \
  --epochs 64 \
  --class_balance \
  --class_weight \
  --output_dir output/finetune/my_run
```

---

## Related docs

- `RADAR_SIMULATION.md` — mesh → radar simulation details  
- `mmAP-slim/BINARY_FALL_CLASSIFICATION.md` — dataset preparation notes  
- `mmAP-slim/requirements.txt` — Python dependencies  

---

## License

See `LICENSE.md`. Upstream MultiMAE / research components may carry additional terms. **SMPL** and **SAM 3D Body** ([SAM License](https://huggingface.co/facebook/sam-3d-body-vith)) have separate licenses and are not redistributed in this repository.
