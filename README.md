# 🚀 Off‑Road Semantic Segmentation with DINOv2

> Lightweight, fast‑training semantic segmentation pipeline using **DINOv2 features + ConvNeXt‑style head**, optimized for hackathon‑scale datasets and limited GPU time.

---

## 🧠 Overview

This project focuses on **pixel‑level off‑road scene understanding** using a frozen **DINOv2 Vision Transformer backbone** and a lightweight **segmentation head**.

The goal was to:

- Improve **baseline IoU (~0.29)**
- Train **multiple diverse models** quickly
- Apply **TTA + weighted ensembling**
- Achieve **~0.48 validation IoU** within strict time limits

---

## 📊 Final Results

| Model | Description | Val IoU | Dice / F1 |
|------|-------------|---------|-----------|
| Baseline | Frozen DINOv2 + simple head | ~0.29 | ~0.44 |
| A–C | Different seeds + augmentations | ~0.46–0.47 | ~0.62 |
| D | Larger backbone / variation | ~0.34 | ~0.50 |
| E | Fine‑tuning variant | ~0.29 | ~0.44 |
| **Ensemble (best)** | Weighted + TTA | **~0.48** | **~0.65** |

> **Key insight:** Ensembling diverse lightweight models provided the biggest gain under time constraints.

---

## 🗂️ Project Structure

```
project/
│
├── models/                # Trained model weights (not included in repo)
├── scripts/               # Training, evaluation, and inference scripts
│   ├── train_*.py
│   ├── evaluate_*.py
│   ├── ensemble_*.py
│   └── visualize.py
│
├── outputs/
│   ├── submission_masks/
│   ├── submission_visuals/
│   └── plots/
│
├── docs/
│   └── report.pdf
│
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

The dataset is **not included** in this repository.

Expected structure:

```
DATASET_ROOT/
├── train/
│   ├── Color_Images/
│   └── Segmentation/
├── val/
│   ├── Color_Images/
│   └── Segmentation/
└── test/
    └── Color_Images/
```

➡️ **Dataset download links:**

- Training set: **[ADD LINK HERE]**
- Validation set: **[ADD LINK HERE]**
- Test set: **[ADD LINK HERE]**

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/offroad-segmentation.git

cd offroad-segmentation

pip install -r requirements.txt
```

---

## 🏋️ Training

Example:

```bash
python scripts/train_model_A.py --seed 42
```

Train multiple diverse models for ensembling:

```bash
python scripts/train_model_B.py --seed 43
python scripts/train_model_C.py --seed 44
python scripts/train_model_D.py --seed 45
python scripts/train_model_E.py --seed 46
```

---

## 🔍 Evaluation

Compute validation IoU:

```bash
python scripts/evaluate_model.py
```

---

## 🧪 Test‑Time Augmentation + Ensemble

```bash
python scripts/ensemble_inference.py
```

Outputs:

- `submission_masks/` → grayscale label masks
- `submission_visuals/` → colored visualization

---

## 📈 Visualizations Included

- Training loss vs epoch
- Validation IoU vs epoch
- Dice/F1 curves
- Pixel accuracy curves
- Sample predictions

These are stored in:

```
outputs/plots/
```

---
## 🖼️ Segmentation Results

<p align="center">
  Input Image:<img src="outputs/visuals/input_1.png" width="400" height="250"/>

  Predicted Output: <img src="outputs/visuals/input_2.png" width="400" height="250"/>
</p>

---

## 🏎️ Key Techniques Used

- **Frozen DINOv2 backbone** for fast convergence
- **ConvNeXt‑style segmentation head**
- **Strong augmentations** (flip, rotate, color jitter, cutout)
- **Mixed precision training** for speed
- **Cosine LR scheduling**
- **TTA + weighted ensemble** for final boost

---

## 🧾 Report

Full methodology and analysis available in:

```
docs/report.pdf
```

---

## 🤝 Acknowledgements

- Meta AI — **DINOv2**
- Albumentations — augmentation library
- PyTorch — training framework

---

## ⭐ If this helped

Drop a ⭐ on the repo — it helps a lot!

