# Transfer Learning on Oxford-IIIT Pet (Limited Data)

Comparing three training paradigms for image classification under a limited-data regime, using ResNet-18 on a 10-class subset of Oxford-IIIT Pet (~2,000 images). Includes an EfficientNet-B0 efficiency benchmark.

## Overview

Evaluates `from scratch` (B1), `feature extraction` (B2), and `progressive fine-tuning` (B3), plus an augmentation control test (B4) and a modern-architecture comparison with EfficientNet-B0 (B5).

## Experimental Protocol

- Architecture: ResNet-18 (ImageNet `IMAGENET1K_V1`)
- Input: 224x224, standard ImageNet normalization
- Batch size 32, AdamW optimizer (weight decay 1e-2), CrossEntropyLoss
- Early stopping (patience 4) monitoring validation loss
- Fixed random seed 42 (PyTorch, NumPy, Python)
- Stratified split: 70% train / 15% val / 15% test

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebook.ipynb              # main notebook (executed end-to-end)
├── report.pdf                  # analysis report
└── results/
    ├── metrics_log.csv         # raw metrics log
    ├── metrics_log.json
    ├── loss_curves.png         # train vs val loss (B1/B2/B3)
    └── failure_analysis.png    # 8 misclassified samples
```

## How to Run

1. Open `notebook.ipynb` in Google Colab (GPU T4 runtime).
2. `Runtime > Run all`. The Oxford-IIIT Pet dataset is downloaded automatically.
3. Outputs: `metrics_log.csv/.json`, `loss_curves.png`, `failure_analysis.png`.

Local: `pip install -r requirements.txt` (requires an NVIDIA/CUDA GPU; CPU is substantially slower).

## Results

| Condition | Strategy | Test Acc | Macro F1 |
|-----------|----------|----------|----------|
| B1 | from scratch | 0.4636 | 0.4231 |
| B2 | feature extraction | 0.9073 | 0.9094 |
| B3 | progressive fine-tuning | 0.8874 | 0.8880 |
| B5 | EfficientNet-B0 (feature) | 0.9272 | 0.9281 |

Transfer learning decisively outperforms training from scratch. Under limited data, feature extraction is the most efficient and robust choice, while EfficientNet-B0 achieves the highest Macro F1 with the fewest trainable parameters.
