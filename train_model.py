#!/usr/bin/env python3
"""
FreshVision AI - MobileNetV2 Training Pipeline
Trains a deep convolutional neural network via transfer learning for food quality classification.
Saves model weights, class indices, evaluation metrics, and performance plots.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Set seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Food Quality Classification Model via MobileNetV2")
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Root dataset directory')
    parser.add_argument('--epochs', type=int, default=15, help='Number of initial training epochs')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5, help='Fine-tuning learning rate')
    parser.add_argument('--fine_tune', action='store_true', default=True, help='Enable fine-tuning of upper layers')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save trained model & metrics')
    return parser.parse_args()

def validate_dataset_structure(dataset_path: Path):
    """
    Validates that dataset/ contains train/, validation/, and test/ splits,
    each with fresh/ and spoiled/ subdirectories with at least 1 image.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory '{dataset_path}' does not exist. "
            "Please create it or run 'python generate_sample_dataset.py' to generate a demo dataset."
        )

    required_splits = ['train', 'validation', 'test']
    required_classes = ['fresh', 'spoiled']
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    split_counts = {}

    for split in required_splits:
        split_dir = dataset_path / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing split directory '{split_dir}'. Dataset must contain 'train', 'validation', and 'test' folders."
            )

        split_counts[split] = {}
        for cls_name in required_classes:
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(
                    f"Missing class directory '{cls_dir}'. Each split must contain 'fresh' and 'spoiled' folders."
                )

            images = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            if len(images) == 0:
                raise ValueError(
                    f"Directory '{cls_dir}' contains 0 valid images. At least 1 image per class is required."
                )
            split_counts[split][cls_name] = len(images)

    logger.info("Dataset structure verified successfully:")
    for split, counts in split_counts.items():
        logger.info(f"  {split.upper()}: {counts['fresh']} fresh, {counts['spoiled']} spoiled (Total: {sum(counts.values())})")

    return split_counts

def build_model(input_shape=(224, 224, 3)):
    """
    Constructs the MobileNetV2 transfer learning architecture:
    1. Preprocessing layer: MobileNetV2 normalization [-1, 1]
    2. MobileNetV2 feature extractor (ImageNet weights, pre-frozen)
    3. GlobalAveragePooling2D
    4. Dropout(0.3)
    5. Dense(1, activation='sigmoid') for Fresh (0) vs Spoiled (1)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.random.set_seed(SEED)

    # Base feature extractor
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze pre-trained weights
    base_model.trainable = False

    # Data augmentation for training only
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ], name="data_augmentation")

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = data_augmentation(inputs)
    # MobileNetV2 expects values in [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="dropout_regularization")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="food_quality_output")(x)

    model = models.Model(inputs, outputs, name="FreshVision_MobileNetV2")
    return model, base_model

def plot_and_save_curves(history, output_dir: Path):
    """Saves loss and accuracy curves as a high-resolution PNG."""
    import matplotlib.pyplot as plt

    epochs = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], 'b-o', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'g-s', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], 'r-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'm-s', label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved learning curves plot to {plot_path}")

def plot_confusion_matrix(cm, output_dir: Path):
    """Visualizes and saves the held-out test confusion matrix."""
    import matplotlib.pyplot as plt
    import numpy as np

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
    dataset_path = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate dataset
    try:
        split_counts = validate_dataset_structure(dataset_path)
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        sys.exit(1)

    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    img_size = (224, 224)
    batch_size = args.batch_size

    # 2. Load dataset splits via tf.keras.utils.image_dataset_from_directory
    logger.info("Loading training dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path / 'train',
        labels='inferred',
        label_mode='binary',
        class_names=['fresh', 'spoiled'],
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=SEED
    )

    logger.info("Loading validation dataset...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path / 'validation',
        labels='inferred',
        label_mode='binary',
        class_names=['fresh', 'spoiled'],
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    logger.info("Loading held-out test dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path / 'test',
        labels='inferred',
        label_mode='binary',
        class_names=['fresh', 'spoiled'],
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Performance optimization: prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # 3. Save class indices
    class_indices = {'fresh': 0, 'spoiled': 1}
    with open(output_dir / 'class_indices.json', 'w') as f:
        json.dump(class_indices, f, indent=2)

    # 4. Build Model
    model, base_model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 5. Callbacks
    checkpoint_path = output_dir / 'food_quality_model.keras'
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # 6. Initial Training (Frozen base)
    logger.info(f"Starting initial training for {args.epochs} epochs...")
    history_initial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # 7. Optional Fine-Tuning
    full_history = {
        'accuracy': [float(x) for x in history_initial.history['accuracy']],
        'val_accuracy': [float(x) for x in history_initial.history['val_accuracy']],
        'loss': [float(x) for x in history_initial.history['loss']],
        'val_loss': [float(x) for x in history_initial.history['val_loss']]
    }

    if args.fine_tune and args.fine_tune_epochs > 0:
        logger.info("Unfreezing upper layers of MobileNetV2 for fine-tuning...")
        base_model.trainable = True
        # Fine-tune only from layer 100 onwards
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=len(history_initial.epoch),
            callbacks=callbacks
        )

        full_history['accuracy'].extend([float(x) for x in history_fine.history['accuracy']])
        full_history['val_accuracy'].extend([float(x) for x in history_fine.history['val_accuracy']])
        full_history['loss'].extend([float(x) for x in history_fine.history['loss']])
        full_history['val_loss'].extend([float(x) for x in history_fine.history['val_loss']])

    # 8. Save Final Model (both .keras and fallback .h5)
    model.save(str(checkpoint_path))
    fallback_path = output_dir / 'food_quality_model.h5'
    try:
        model.save(str(fallback_path))
    except Exception as e:
        logger.warning(f"Could not save .h5 fallback format: {e}")

    logger.info(f"Model saved successfully to {checkpoint_path}")

    # 9. Evaluate on Held-Out Test Set
    logger.info("Evaluating model on held-out test dataset...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    # Collect ground truths and raw predictions
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred = (np.array(y_pred_probs) >= 0.5).astype(int)

    # Calculate scikit-learn metrics
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # 10. Record Metrics to JSON
    total_train = sum(split_counts['train'].values())
    total_val = sum(split_counts['validation'].values())
    total_test = sum(split_counts['test'].values())

    metrics_record = {
        'model_name': 'MobileNetV2-FoodQuality',
        'model_version': 'v1.0.0-MobileNetV2',
        'train_accuracy': round(full_history['accuracy'][-1], 4),
        'train_loss': round(full_history['loss'][-1], 4),
        'val_accuracy': round(full_history['val_accuracy'][-1], 4),
        'val_loss': round(full_history['val_loss'][-1], 4),
        'test_accuracy': round(float(test_acc), 4),
        'test_loss': round(float(test_loss), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'splits': {
            'train_samples': total_train,
            'val_samples': total_val,
            'test_samples': total_test
        },
        'confusion_matrix': {
            'true_fresh': int(tn),      # Label 0 correctly identified
            'false_spoiled': int(fp),   # Label 0 misclassified as 1
            'false_fresh': int(fn),     # Label 1 misclassified as 0
            'true_spoiled': int(tp)     # Label 1 correctly identified
        },
        'history': {
            'epochs': list(range(1, len(full_history['accuracy']) + 1)),
            'accuracy': [round(x, 4) for x in full_history['accuracy']],
            'val_accuracy': [round(x, 4) for x in full_history['val_accuracy']],
            'loss': [round(x, 4) for x in full_history['loss']],
            'val_loss': [round(x, 4) for x in full_history['val_loss']]
        }
    }

    metrics_file = output_dir / 'model_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_record, f, indent=2)

    logger.info(f"Evaluation metrics saved to {metrics_file}")
    logger.info(f"Final Test Accuracy: {test_acc * 100:.2f}% | Precision: {prec * 100:.2f}% | Recall: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%")

    # 11. Generate plots
    full_history['epochs'] = list(range(1, len(full_history['accuracy']) + 1))
    plot_and_save_curves(full_history, output_dir)
    plot_confusion_matrix(cm, output_dir)

    print("\n" + "=" * 65)
    print(" Training and Evaluation Completed Successfully! ")
    print(f" Model Saved: {checkpoint_path}")
    print(f" Metrics Saved: {metrics_file}")
    print(f" Held-Out Test Accuracy: {test_acc * 100:.2f}%")
    print("=" * 65 + "\n")

if __name__ == '__main__':
    main()
