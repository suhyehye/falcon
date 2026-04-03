<div align="center">

# FALCON

### Fine-grained Alignment for Counting Objects in Industry

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A two-stage referring-expression counting pipeline that combines **SAM3** for text-prompted object detection with **PE-Core + CoOp** for fine-grained attribute classification.

[Getting Started](#getting-started) &bull; [Usage](#usage) &bull; [Results](#results) &bull; [Model Architecture](#model-architecture)

</div>

---

## Overview

FALCON tackles the challenge of **counting objects by referring expressions** in industrial inspection scenarios. Given an image and a natural language expression (e.g., *"crushed blue cuboid potentiometer"*), the model detects and counts all matching instances.

<details>
<summary><b>Pipeline</b></summary>

```
                        Input Image
                            |
                            v
                +-----------------------+
                |        SAM3           |  text prompt: base class name
                |  (Object Detection)   |  e.g., "blue cuboid potentiometer"
                +-----------------------+
                            |
                            v
                   NMS Filtering + Crop
                            |
                            v
                +-----------------------+
                |   PE-Core + CoOp      |  learnable context vectors
                |  (Attribute Classify)  |  e.g., "normal" vs "crushed"
                +-----------------------+
                            |
                            v
              Per-expression Count --> MAE / RMSE
```

</details>

## Getting Started

### Prerequisites

- CUDA-compatible GPU
- Conda (Miniconda or Anaconda)

### Installation

```bash
# Clone the repository
git clone https://github.com/suhyehye/FALCON.git
cd FALCON

# Create conda environment
conda create -n falcon python=3.12 -y
conda activate falcon

# Install dependencies
pip install -r requirements.txt

# Install SAM3 module
cd sam3 && pip install -e . && cd ..
```

### Data Preparation

```
../dataset/ioc/
├── anno/
│   ├── annotations.json      # Per-image point annotations with ref expressions
│   └── split_config.json     # Train / test split
└── <class_name>/
    └── *.jpg
```

Place the SAM3 checkpoint at `checkpoints/sam3.pt`.

### Dependencies

| Package | Version |
|---------|---------|
| Python | 3.12 |
| PyTorch | 2.9.1+cu128 |
| open_clip_torch | 3.2.0 |
| timm | 1.0.24 |

## Usage

### 1. Train CoOp

Fine-tune learnable context vectors on the training split:

```bash
python train_coop.py
```

Outputs:
- Context vectors: `checkpoints/coop_language_guided/coop_language_guided_model.pth`
- Class list: `checkpoints/coop_language_guided/classes.json`

### 2. Evaluate

```bash
# CoOp (fine-tuned)
python test_coop.py

# Zero-shot baseline (no CoOp)
python test_baseline.py
```

Visualization results are saved to `coop_result/<class_name>/<image_name>/<ref_exp>.jpg`.

### 3. Ablation Study

Compare performance across different numbers of learnable context vectors:

```bash
python ablation_vectors.py
```

## Results

### Context Vector Ablation (`n_ctx`)

| n_ctx | MAE | RMSE |
|:-----:|:----:|:-----:|
| 2 | 2.00 | 4.58 |
| **4** | **1.80** | **4.49** |
| 8 | 6.29 | 9.25 |

## Model Architecture

| Component | Details |
|-----------|---------|
| **Vision Encoder** | PE-Core ViT-bigG/14-448 (`hf-hub:timm/PE-Core-bigG-14-448`) |
| **Object Detector** | SAM3 (text-prompted grounding + segmentation) |
| **Prompt Tuning** | CoOp — 4 learnable context vectors, dim=1280 |
| **Optimizer** | SGD (lr=0.002, momentum=0.9, weight_decay=5e-4) |
| **Scheduler** | Cosine Annealing |
| **Epochs** | 10 |

## Project Structure

```
FALCON/
├── train_coop.py          # CoOp training (SAM3 crop + PE-Core classification)
├── test_coop.py           # Evaluation with CoOp (MAE/RMSE + visualization)
├── test_baseline.py       # Zero-shot PE-Core baseline evaluation
├── ablation_vectors.py    # Context vector count ablation (n_ctx = 2, 4, 8)
├── requirements.txt
├── sam3/                  # SAM3 module (text-prompted detection)
└── checkpoints/
    └── sam3.pt            # SAM3 checkpoint
```

## Acknowledgements

- [SAM3](https://github.com/facebookresearch/sam3) — Segment Anything Model 3 by Meta AI
- [PE-Core](https://github.com/baaivision/PE-Core) — Parameter-Efficient Core for Vision-Language Models
- [CoOp](https://github.com/KaiyangZhou/CoOp) — Learning to Prompt for Vision-Language Models
