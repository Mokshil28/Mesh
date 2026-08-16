# mmAP Radar Heatmap Processing

This repository contains scripts for processing radar data and training mmAP (Multi-Modal Autoencoder for Radar Processing) models on radar heatmap data.

## Python Version

- **Required Python Version**: Python 3.8+
- **Tested Python Version**: Python 3.8 (as specified in requirements.txt)

## Installation Requirements

### Environment Setup

We recommend creating a new [conda](https://docs.conda.io/en/latest/) virtual environment:

```bash
conda create -n mmap python=3.8 -y
conda activate mmap
```

### PyTorch Installation

Install [PyTorch](https://pytorch.org/) 1.10.0+ and [torchvision](https://pytorch.org/vision/stable/index.html) 0.11.1+:

```bash
conda install pytorch=1.10.0 torchvision=0.11.1 -c pytorch -y
```

### Required Packages

Install all other required packages from requirements.txt:

```bash
pip install -r requirements.txt
```

### Optional Performance Optimization

For improved data loading performance, consider replacing Pillow with [Pillow-SIMD](https://github.com/uploadcare/pillow-simd):

```bash
sh tools/install_pillow_simd.sh
```

## Workflow Overview

The mmAP training pipeline consists of three main stages:

### 1. Data Preparation
- **Input**: Raw radar binary files (.bin format)
- **Process**: Convert radar signals to multi-modal heatmaps
- **Output**: Training-ready heatmap datasets (angle, doppler, range)
- **Script**: `heatmap-prep/bin2npy/main.py`

### 2. Pre-training on Generated Dataset
- **Input**: Multi-modal heatmap data or ImageNet with pseudo-labels
- **Process**: Self-supervised pre-training using masked autoencoding
- **Output**: Pre-trained mmAP model weights
- **Scripts**: `run_pretraining_heatmap.py`

### 3. Fine-tuning on Recorded Dataset
- **Input**: Pre-trained model + labeled radar classification data
- **Process**: Supervised fine-tuning for specific radar tasks
- **Output**: Task-specific trained model
- **Scripts**: 
  - `run_finetuning_heatmap_clsheadonly.py` (classification head only)
  - `run_finetuning_heatmap_wholemodel.py` (full model)
  - `run_finetuning_heatmap-single_cls.py` (single domain)


## Dataset Structure

For simplicity and uniformity, all datasets should be structured in the following way:

```
/path/to/data/
├── train/
│   ├── angle/
│   │   └── activity1/
│   │       ├── sample1.npy
│   │       └── sample2.npy
│   ├── doppler/
│   │   └── activity1/
│   │       ├── sample1.npy
│   │       └── sample2.npy
│   └── range/
│       └── activity1/
│           ├── sample1.npy
│           └── sample2.npy
└── val/
    ├── angle/
    │   └── activity2/
    │       ├── sample3.npy
    │       └── sample4.npy
    ├── doppler/
    │   └── activity2/
    │       ├── sample3.npy
    │       └── sample4.npy
    └── range/
        └── activity2/
            ├── sample3.npy
            └── sample4.npy
```

**Key Requirements**:
- The folder structure and filenames should match across modalities
- For radar heatmaps, we use `angle`, `doppler`, and `range` as modality names
- Each modality contains activity-based subfolders for classification tasks
- Data files are stored as `.npy` format for efficient loading

## Script Functionality

### 1. `heatmap-prep/bin2npy/main.py`

**Purpose**: Radar data preprocessing and heatmap generation

**Functionality**:
- Reads binary radar data files (.bin format)
- Processes raw radar frames with Range FFT
- Generates three types of heatmaps:
  - **Time-Angle (TA)**: Angular information over time
  - **Time-Doppler (TD)**: Velocity/motion information over time  
  - **Time-Range (TR)**: Distance information over time
- Splits data into training (80%) and evaluation (20%) sets
- Saves processed heatmaps as numpy arrays for training

**Key Features**:
- Configurable frame parameters (ADC samples, antennas, chirps)
- Automatic train/eval split
- Multiple radar modalities extraction

```bash
cd heatmap-prep/bin2npy/
python main.py
```

### 2. `run_pretraining_heatmap.py`

**Purpose**: Pre-training mmAP specifically for radar heatmap data

**Functionality**:
- Adapted version of mmAP pre-training for radar domains
- Supports angle, doppler, and range heatmap modalities
- Custom input/output adapters for heatmap data
- Specialized loss functions for radar data (MaskedL1Loss)

**Radar Domains**:
- **Angle**: 128×128 patches with (128,1) patch size
- **Doppler**: 256×128 patches with (256,1) patch size  
- **Range**: 100×128 patches with (100,1) patch size

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_pretraining_heatmap.py \
    --in_domains angle-doppler-range \
    --out_domains angle-doppler-range \
    --batch_size 32 \
    --epochs 100 \
    --output_dir ./output/pretrain \
    --config cfgs/pretrain/heatmap_pretrain.yaml
```

### 3. `run_finetuning_heatmap_clsheadonly.py`

**Purpose**: Fine-tuning with frozen backbone (classification head only)

**Functionality**:
- Loads pre-trained mmAP model
- Freezes encoder and input adapters
- Only trains the linear classification head
- Efficient fine-tuning approach for limited data
- Supports confusion matrix generation

**Training Strategy**:
- Frozen: Encoder + Input Adapters
- Trainable: Classification Head only
- Faster training with lower computational requirements

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 run_finetuning_heatmap_clsheadonly.py \
    --config cfgs/finetune/cls/ft_heatmap_clshead_100e_mmap-b.yaml \
    --finetune /path/to/mmap_weights \
    --data_path /path/to/heatmap/train \
    --eval_data_path /path/to/heatmap/val
```

### 4. `run_finetuning_heatmap_wholemodel.py`

**Purpose**: Full model fine-tuning for classification

**Functionality**:
- Fine-tunes entire pre-trained mmAP model
- Trains all parameters (encoder + adapters + classification head)
- More comprehensive adaptation to target domain
- Higher computational requirements but potentially better performance

**Training Strategy**:
- Trainable: All model parameters
- Layer-wise learning rate decay support
- Full model adaptation

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 run_finetuning_heatmap_wholemodel.py \
    --config cfgs/finetune/cls/ft_heatmap_full_100e_mmap-b.yaml \
    --finetune /path/to/mmap_weights \
    --data_path /path/to/heatmap/train \
    --eval_data_path /path/to/heatmap/val
```

### 5. `run_finetuning_heatmap-single_cls.py`

**Purpose**: Single-domain classification fine-tuning

**Functionality**:
- Fine-tuning for single heatmap modality classification
- Simplified input handling for single domain
- Supports mixup and augmentation strategies
- Direct classification without multi-modal complexity

**Use Case**:
- When only one type of heatmap data is available
- Baseline comparison with multi-modal approaches
- Simpler deployment scenarios

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 run_finetuning_heatmap-single_cls.py \
    --config cfgs/finetune/cls/ft_single_heatmap_100e_mmap-b.yaml \
    --finetune /path/to/mmap_weights \
    --data_path /path/to/single_heatmap/train \
    --eval_data_path /path/to/single_heatmap/val
```

## Configuration Files

The training scripts support both YAML config files and command-line arguments:
- **Pre-training configs**: [`cfgs/pretrain/`](cfgs/pretrain/)
- **Fine-tuning configs**: [`cfgs/finetune/`](cfgs/finetune/)

Config file arguments override default arguments, and command-line arguments override both default and config arguments.

### **Important**: 
> When changing settings, modify the `output_dir` and `wandb_run_name` (if logging is activated) to reflect the changes.

## Experiment Logging

To activate logging to [Weights & Biases](https://docs.wandb.ai/), either edit the config files or use the `--log_wandb` flag along with other logging arguments:

```bash
--log_wandb --wandb_project your_project --wandb_entity your_entity
```

## Model Formats

All fine-tuning scripts support models in the mmAP/MultiViT format. Pre-trained models using the timm/ViT format can be converted using the `vit2mmap_converter.py` script.

## Reference
For more pre-training and fine-tuning steps, check out [MultiMAE](https://github.com/EPFL-VILAB/MultiMAE) for orginal implmentation.