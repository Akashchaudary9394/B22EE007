# Robust Environmental Sound Classification using Transfer Learning with Responsible AI Analysis

**IIT Jodhpur | Speech Understanding Project**  
**Team:** Prem Kumar (B22AI031) · Akash Chaudhary (B22EE007) · V.K Santosh (B22AI049)

---

## Overview

This project implements a complete pipeline for **Environmental Sound Classification (ESC)** using:

- **Dataset**: UrbanSound8K (8,732 clips, 10 urban sound categories, 8 folds)
- **Features**: 128-bin Log Mel Spectrogram (224×224) + 40-coefficient MFCC
- **Models**: YAMNet (MobileNetV2 backbone) + ResNet-50 with transfer learning
- **Training**: 8-fold cross-validation, Adam optimizer, early stopping, class weighting
- **Responsible AI**: Fairness analysis, robustness under noise, Grad-CAM explainability

---

## Project Structure

```
esc_project/
├── data/
│   ├── UrbanSound8K/           # Raw dataset (auto-created)
│   └── features/               # Cached .npy features per fold
├── src/
│   ├── data/
│   │   ├── download_dataset.py  # Dataset acquisition + synthetic fallback
│   │   ├── preprocess.py        # Mel spectrogram + MFCC extraction
│   │   └── augmentation.py      # SpecAugment, noise, stretch, pitch shift
│   ├── models/
│   │   ├── yamnet_transfer.py   # YAMNet (MobileNetV2 backbone, pretrained)
│   │   ├── resnet50_transfer.py # ResNet-50 (ImageNet pretrained, fine-tuned)
│   │   └── ensemble.py          # Soft voting ensemble
│   ├── responsible_ai/
│   │   ├── fairness.py          # Demographic Parity Index, bias detection
│   │   ├── robustness.py        # SNR degradation curves
│   │   └── explainability.py    # Grad-CAM visualizations
│   ├── visualization/
│   │   └── dashboard.py         # All publication figures
│   ├── train.py                 # 8-fold CV training
│   └── evaluate.py              # Metrics, confusion matrix
├── outputs/
│   ├── checkpoints/             # Model checkpoints (.pt)
│   ├── figures/                 # All plots (PNG)
│   └── results/                 # Metrics CSV/JSON
├── run_all.py                   # Single entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Recommended)

```bash
# Full pipeline with default settings (synthetic dataset, ~2-4 hours on CPU)
python run_all.py

# Quick smoke test (1 fold, 5 epochs, small synthetic dataset, ~15 minutes)
python run_all.py --quick

# Train YAMNet only on folds 1 and 2 for 20 epochs
python run_all.py --model yamnet --folds 1 2 --epochs 20
```

### 3. Individual Phases

```bash
# Phase 1+2: Dataset setup + feature extraction only
python src/train.py --preprocess-only

# Phase 3: Train a specific model
python src/train.py --model yamnet --epochs 30
python src/train.py --model resnet50 --epochs 30

# Phase 4: Evaluate trained models
python src/evaluate.py --model both

# Phase 5a: Fairness analysis
python src/responsible_ai/fairness.py --model both

# Phase 5b: Robustness testing
python src/responsible_ai/robustness.py --model both

# Phase 5c: Grad-CAM explainability
python src/responsible_ai/explainability.py --model both --samples 3

# Phase 6: Generate all figures
python src/visualization/dashboard.py
```

### 4. Using Real UrbanSound8K Dataset

Place the UrbanSound8K dataset in:
```
data/UrbanSound8K/
├── audio/
│   ├── fold1/  ... ├── fold8/
└── metadata/
    └── UrbanSound8K.csv
```

Or if you have Kaggle credentials:
```bash
kaggle datasets download -d andrewmvd/urban-sound-classification --unzip -p data/
```

---

## Sound Classes (UrbanSound8K)

| ID | Class | Description |
|---|---|---|
| 0 | air_conditioner | HVAC systems |
| 1 | car_horn | Vehicle horns |
| 2 | children_playing | Outdoor play |
| 3 | dog_bark | Dog sounds |
| 4 | drilling | Construction |
| 5 | engine_idling | Running engines |
| 6 | gun_shot | Gunfire |
| 7 | jackhammer | Pneumatic tools |
| 8 | siren | Emergency vehicles |
| 9 | street_music | Busking/outdoor music |

---

## Architecture Details

### Feature Extraction Pipeline
```
Raw Audio (WAV) → Load @ 22.05 kHz → Bandpass Filter (200–8000 Hz)
→ STFT → Mel Filterbank (128 bins) → Log Power → Resize 224×224
→ 3-channel image (for CNN)  +  40-coeff MFCC (mean+std → 80-dim vector)
```

### YAMNet Model (Transfer Learning)
- **Backbone**: MobileNetV2 pretrained on ImageNet
- **Frozen**: All feature layers except last 3 blocks
- **Head**: Linear(1280, 512) → BN → Dropout → Linear(512, 10)
- **MFCC stream**: Linear(80, 64) → fusion → 10-class output

### ResNet-50 Model (Transfer Learning)
- **Backbone**: ResNet-50 pretrained on ImageNet (V2 weights)
- **Frozen**: Layers 1–3 (only Layer-4 + FC fine-tuned)
- **Head**: GAP → Linear(2048, 256) → BN → Dropout → Linear(256, 10)

### Training
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Loss**: Cross-Entropy with inverse-frequency class weights
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Early stopping**: patience=5
- **Data augmentation**: SpecAugment, horizontal flip (training only)

---

## Responsible AI

### Fairness
- Per-class precision, recall, F1-score breakdown
- **Demographic Parity Index (DPI)**: std(TPR per class) / mean(TPR)
- Underperforming class detection (F1 < mean − 1σ)

### Robustness
- Controlled noise injection: Gaussian, traffic, crowd
- SNR levels: 20, 10, 5, 0, −5 dB
- Accuracy vs. SNR degradation curves per noise type and per class

### Explainability (Grad-CAM)
- Gradient-weighted Class Activation Mapping on mel spectrograms
- Highlights frequency-time regions driving each classification
- Gallery: 10 classes × 3 samples each

---

## Expected Deliverables

| Deliverable | Status | Location |
|---|---|---|
| Feature extraction code | ✅ | `src/data/preprocess.py` |
| Data augmentation | ✅ | `src/data/augmentation.py` |
| YAMNet model | ✅ | `src/models/yamnet_transfer.py` |
| ResNet-50 model | ✅ | `src/models/resnet50_transfer.py` |
| Ensemble model | ✅ | `src/models/ensemble.py` |
| 8-fold CV Training | ✅ | `src/train.py` |
| Evaluation metrics | ✅ | `src/evaluate.py` |
| Fairness analysis | ✅ | `src/responsible_ai/fairness.py` |
| Robustness testing | ✅ | `src/responsible_ai/robustness.py` |
| Grad-CAM explainability | ✅ | `src/responsible_ai/explainability.py` |
| Visualization dashboard | ✅ | `src/visualization/dashboard.py` |
| Single entry point | ✅ | `run_all.py` |

---

## References

[1] Salamon, J., Jacoby, C., & Bello, J.P. (2014). A Dataset and Taxonomy for Urban Sound Research. ACM Multimedia 2014. [UrbanSound8K]

[2] Palanisamy, K., Singhania, D., & Yao, A. (2020). Rethinking CNN Models for Audio Classification. arXiv:2007.11154.

[3] Gemmeke, J.F. et al. (2017). AudioSet: An Ontology and Human-Labeled Dataset for Audio Events. ICASSP 2017. [YAMNet]

[4] Gong, Y., Liu, Y.A., & Glass, J. (2021). AST: Audio Spectrogram Transformer. Interspeech 2021.

[5] Selvaraju, R.R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV 2017.

[6] Park, D.S. et al. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. Interspeech 2019.

[7] Hendrycks, D. & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR 2019.

---

## License

This project is for academic purposes at IIT Jodhpur, Speech Understanding course.
