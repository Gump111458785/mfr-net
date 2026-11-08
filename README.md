# MFR-Net: Multi-Scale Feature Rectification and Uncertainty Calibration for Visual Classification

Official implementation of the paper **"Multi-Scale Feature Rectification and Uncertainty Calibration for Robust Visual Classification"**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/CUDA-GPU_Required-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

**MFR-Net** introduces a lightweight architectural and loss formulation framework that tackles representation degradation and probability miscalibration in visual classification:
1. **Multi-Scale Feature Rectification (MFR) Module**: Dynamic spatial-frequency feature recalibration that suppresses spurious high-frequency noise while preserving fine-grained boundary semantics.
2. **Uncertainty-Aware Calibration (UAC) Loss**: Jointly penalizes overconfident boundary predictions via margin-aware regularized cross-entropy and soft calibration constraints.
3. **Comprehensive Multi-Dataset Benchmarking**: Rigorously evaluated across CIFAR-10, CIFAR-100, and SVHN over five quantitative metrics (Top-1, Top-5, Macro-F1, ECE, and Latency).

---

## ⚙️ System & Hardware Requirements

- **Operating System**: Linux / macOS with CUDA support
- **Hardware**: NVIDIA GPU (RTX 3080/3090, A100, V100, or equivalent with >= 8GB VRAM). **CUDA is strictly required for experimental reproduction.**
- **Software**: Python >= 3.8, CUDA >= 11.6, cuDNN >= 8.0

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository and navigate into it
git clone https://github.com/reproduction-benchmarks/mfr-net.git
cd mfr-net

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Dataset Download

Running experiments requires downloading the benchmark datasets (`CIFAR-10`, `CIFAR-100`, `SVHN`):

```bash
# Download all datasets automatically to ./data/cache
bash scripts/download_data.sh

# Or download a specific dataset using Python:
python data/download_data.py --dataset cifar100 --data_dir ./data/cache
```

---

## 🔬 Training & Evaluation

### Training on GPU

Train MFR-Net on CIFAR-100:
```bash
python scripts/train.py \
    --model mfr_net \
    --dataset cifar100 \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --output_dir ./checkpoints
```

Train Baselines (ResNet-18, ResNet-50, MobileNetV2):
```bash
python scripts/train.py --model resnet18 --dataset cifar100
python scripts/train.py --model resnet50 --dataset cifar100
python scripts/train.py --model mobilenet_v2 --dataset cifar100
```

### Evaluation & Metric Computation

Evaluate checkpoint on test sets (computes Top-1, Top-5, Macro-F1, and 15-bin ECE):
```bash
python scripts/evaluate.py \
    --model mfr_net \
    --dataset cifar100 \
    --checkpoint ./checkpoints/mfr_net_cifar100_seed42_best.pth
```

### Hardware Efficiency Benchmarking

Measures model parameters (M), FLOPs (G), GPU inference latency (ms), and throughput (samples/s):
```bash
python scripts/benchmark_efficiency.py
```

---

## 📊 Experimental Results (Reported in Paper)

### Table 1: Multi-Dataset Performance & Calibration Comparison

| Dataset | Architecture | Top-1 Acc (%) | Top-5 Acc (%) | Macro-F1 (%) | ECE (%) ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CIFAR-10** | ResNet-18 | 94.82 | 99.68 | 94.75 | 3.42 |
| | ResNet-50 | 95.36 | 99.74 | 95.29 | 4.15 |
| | MobileNetV2 | 93.91 | 99.52 | 93.84 | 4.86 |
| | **MFR-Net (Ours)** | **96.48** | **99.85** | **96.42** | **1.85** |
| **CIFAR-100** | ResNet-18 | 76.54 | 93.81 | 76.32 | 8.92 |
| | ResNet-50 | 79.24 | 94.65 | 79.08 | 9.64 |
| | MobileNetV2 | 74.88 | 92.73 | 74.65 | 11.20 |
| | **MFR-Net (Ours)** | **81.42** | **95.90** | **81.28** | **4.98** |
| **SVHN** | ResNet-18 | 96.15 | 99.82 | 96.08 | 2.76 |
| | ResNet-50 | 96.42 | 99.87 | 96.35 | 3.12 |
| | MobileNetV2 | 95.60 | 99.75 | 95.51 | 3.84 |
| | **MFR-Net (Ours)** | **97.35** | **99.92** | **97.30** | **1.42** |

### Table 2: Model Complexity & Hardware Latency (NVIDIA GPU)

| Architecture | Params (M) | FLOPs (G) | GPU Latency (ms) | Throughput (img/s) |
| :--- | :---: | :---: | :---: | :---: |
| MobileNetV2 | 2.30 | 0.32 | 3.12 | 1420 |
| ResNet-18 | 11.22 | 1.82 | 4.25 | 1180 |
| ResNet-50 | 23.71 | 4.12 | 8.64 | 620 |
| **MFR-Net (Ours)** | **14.38** | **2.26** | **5.10** | **985** |

### Table 3: Ablation Study on CIFAR-100

| Configuration | MFR Module | UAC Loss | Temp Scaling | Top-1 Acc (%) | Macro-F1 (%) | ECE (%) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| (a) Baseline (ResNet-18) | ✗ | ✗ | ✗ | 76.54 | 76.32 | 8.92 |
| (b) + MFR Module | ✓ | ✗ | ✗ | 79.88 | 79.71 | 7.45 |
| (c) + UAC Loss | ✗ | ✓ | ✗ | 77.82 | 77.60 | 5.86 |
| (d) MFR + UAC | ✓ | ✓ | ✗ | 81.15 | 81.02 | 5.12 |
| (e) Full MFR-Net | ✓ | ✓ | ✓ | **81.42** | **81.28** | **4.98** |

---

## 🔁 Automated Reproduction Verification

To verify that your trained models replicate the results within target tolerance:
```bash
python scripts/verify_reproduction.py --reference ./results/paper_reference_results.json --tolerance 0.75
```

Or execute the complete reproduction pipeline in one command:
```bash
bash scripts/reproduce_all.sh
```

---

## 📁 Repository Structure

```
mfr-net/
├── README.md                          # Project documentation and reproduction guide
├── requirements.txt                   # Environment dependencies
├── setup.py                           # Package installation script
├── paper/
│   ├── paper.md                       # Full academic conference paper (English)
│   ├── paper_zh.md                    # Academic conference paper (Chinese)
│   └── paper_benchmark.json           # Machine-readable benchmarks for automated evaluators
├── data/
│   ├── download_data.py               # Dataset acquisition utility (CIFAR-10, CIFAR-100, SVHN)
│   └── dataset.py                     # Multi-dataset loader pipelines
├── models/
│   ├── baselines.py                   # Baseline architectures (ResNet-18/50, MobileNetV2)
│   └── mfr_net.py                     # Proposed MFR-Net architecture & MFR block
├── utils/
│   ├── metrics.py                     # Multi-metric computation (Top-1, Top-5, Macro-F1, ECE)
│   ├── loss.py                        # Uncertainty-Aware Calibration Loss (UAC)
│   └── logger.py                      # Experiment logging and table formatting
├── scripts/
│   ├── download_data.sh               # One-click dataset downloader
│   ├── train.py                       # Model training script (GPU required)
│   ├── evaluate.py                    # Evaluation & inference testing script
│   ├── benchmark_efficiency.py        # GPU latency, throughput & FLOPs benchmark
│   ├── reproduce_all.sh               # Master one-click full reproduction bash pipeline
│   └── verify_reproduction.py         # Automated comparison against paper results
└── results/
    └── paper_reference_results.json   # Ground truth experimental metrics from paper
```
