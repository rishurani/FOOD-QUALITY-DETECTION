# FreshVision AI - Dataset Guide

This directory holds the labeled image datasets used for training, validating, and testing the deep transfer learning model.

## 1. Directory Structure

The training pipeline (`train_model.py`) expects the dataset to be organized in the following directory hierarchy:

```
dataset/
├── train/
│   ├── fresh/      # Training images of fresh produce
│   └── spoiled/    # Training images of spoiled/decayed produce
├── validation/
│   ├── fresh/      # Validation images for early stopping & hyperparameter tuning
│   └── spoiled/    # Validation images of spoiled produce
└── test/
    ├── fresh/      # Strictly held-out test images for final unbiased metrics
    └── spoiled/    # Held-out test images of spoiled produce
```

### Supported Image Formats
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- WebP (`.webp`)
- BMP (`.bmp`)

All images are dynamically converted to RGB, rescaled to **224 &times; 224 pixels**, and normalized into the range `[-1.0, 1.0]` during inference and training to match MobileNetV2's exact input distribution.

---

## 2. Quick Start: Built-in Sample Dataset Generator

For instant local testing, offline development, or demonstration without downloading multi-gigabyte files, run:

```bash
python generate_sample_dataset.py
```

This generates 184 structured, balanced image samples across `train`, `validation`, and `test` splits with synthetic visual features (vibrant produce skin vs. enzymatic browning, irregular mold blemishes, and textural necrosis).

---

## 3. Recommended Production Benchmark Datasets

For training production-grade models, we recommend the following benchmark computer vision datasets:

1. **Fruits Fresh and Rotten for Classification (Kaggle)**:
   - Link: [Kaggle - Fruits Fresh and Rotten](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
   - Contains ~13,000 images of fresh and rotten apples, bananas, and oranges.
2. **Fruit and Vegetable Freshness Dataset (Kaggle)**:
   - Link: [Kaggle - Fruit and Veg Freshness](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables)
   - Multi-produce dataset covering tomatoes, bell peppers, carrots, cucumbers, and apples.
3. **Mendeley Data - Fruit Quality Dataset**:
   - Contains high-resolution laboratory and orchard images under standardized illuminations.

### How to use an external dataset:
1. Download and extract the archive.
2. Split images into `train/` (70%), `validation/` (15%), and `test/` (15%).
3. Place fresh images into `fresh/` subfolders and spoiled images into `spoiled/` subfolders.
4. Run:
   ```bash
   python train_model.py --epochs 20 --batch_size 32
   ```
