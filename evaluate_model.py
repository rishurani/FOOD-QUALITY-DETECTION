#!/usr/bin/env python3
"""
FreshVision AI - Model Evaluation Script
Evaluates an existing trained model checkpoint against held-out test data
and generates quantitative classification metrics and confusion matrix plots.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Food Quality Model")
    parser.add_argument('--model_path', type=str, default='models/food_quality_model.keras', help='Path to trained model')
    parser.add_argument('--test_dir', type=str, default='dataset/test', help='Directory containing held-out test data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save updated evaluation metrics')
    return parser.parse_args()

def plot_confusion_matrix(cm, output_dir: Path):
    """Visualizes and saves the held-out test confusion matrix."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Greens, alpha=0.8)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=str(cm[i, j]), va='center', ha='center', size='xx-large', weight='bold')

    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.ylabel('True Ground Truth', fontsize=11, fontweight='bold')
    plt.title('Confusion Matrix (Held-out Test Split)', fontsize=12, fontweight='bold', pad=20)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Fresh', 'Spoiled'])
    ax.set_yticklabels(['Fresh', 'Spoiled'])
    plt.colorbar(cax)
    plt.tight_layout()

    cm_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")

def main():
    args = parse_args()
    model_path = Path(args.model_path)
    test_path = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        fallback = Path('models/food_quality_model.h5')
        if fallback.exists():
            model_path = fallback
        else:
            logger.error(f"Model file '{model_path}' not found. Please run 'python train_model.py' first.")
            sys.exit(1)

    if not test_path.exists():
        logger.error(f"Test directory '{test_path}' does not exist.")
        sys.exit(1)

    fresh_dir = test_path / 'fresh'
    spoiled_dir = test_path / 'spoiled'
    if not fresh_dir.exists() or not spoiled_dir.exists():
        logger.error(f"Test directory must contain 'fresh' and 'spoiled' subfolders.")
        sys.exit(1)

    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    logger.info(f"Loading model checkpoint from {model_path}...")
    model = tf.keras.models.load_model(str(model_path))

    logger.info(f"Loading test images from {test_path}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        labels='inferred',
        label_mode='binary',
        class_names=['fresh', 'spoiled'],
        image_size=(224, 224),
        batch_size=args.batch_size,
        shuffle=False
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred = (np.array(y_pred_probs) >= 0.5).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Read existing metrics if available to keep training curves intact
    metrics_file = output_dir / 'model_metrics.json'
    existing = {}
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    existing.update({
        'model_name': 'MobileNetV2-FoodQuality',
        'model_version': 'v1.0.0-MobileNetV2',
        'test_accuracy': round(float(test_acc), 4),
        'test_loss': round(float(test_loss), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'confusion_matrix': {
            'true_fresh': int(tn),
            'false_spoiled': int(fp),
            'false_fresh': int(fn),
            'true_spoiled': int(tp)
        }
    })

    if 'splits' not in existing:
        existing['splits'] = {}
    existing['splits']['test_samples'] = len(y_true)

    with open(metrics_file, 'w') as f:
        json.dump(existing, f, indent=2)

    plot_confusion_matrix(cm, output_dir)

    print("\n" + "=" * 65)
    print(" HELD-OUT TEST EVALUATION REPORT ")
    print("=" * 65)
    print(f" Test Accuracy:    {test_acc * 100:.2f}%")
    print(f" Precision:        {prec * 100:.2f}%")
    print(f" Recall:           {rec * 100:.2f}%")
    print(f" F1-Score:         {f1 * 100:.2f}%")
    print("\nClassification Matrix:")
    print(f"  True Fresh (TN): {tn:4d}   |  False Spoiled (FP): {fp:4d}")
    print(f"  False Fresh (FN): {fn:4d}   |  True Spoiled (TP):  {tp:4d}")
    print("=" * 65 + "\n")
    print(classification_report(y_true, y_pred, target_names=['Fresh', 'Spoiled'], digits=4))

if __name__ == '__main__':
    main()
