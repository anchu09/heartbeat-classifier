"""Training pipeline for the heartbeat classifier.

hbc-train --data-root . --output-dir results/run_01
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from heartbeat_classifier import config
from heartbeat_classifier.data.generator import ECGDataGenerator, ecg_generator_to_tf_dataset
from heartbeat_classifier.data.loader import build_train_val_test_split, collect_segment_paths
from heartbeat_classifier.models.cnn import build_model

logger = logging.getLogger(__name__)


def run_training(data_root: Path, output_dir: Path) -> None:
    """Load segments, fit the model with early stopping, and save all artifacts."""
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ecg_dir = data_root / config.DATA_DIRS["sliced_ecg"]
    annotation_dir = data_root / config.DATA_DIRS["sliced_annotations"]

    for required in (ecg_dir, annotation_dir):
        if not required.is_dir():
            raise FileNotFoundError(
                f"Required data directory not found: {required}. "
                "Run the preprocessing pipeline first."
            )

    # ── Train / val / test split ──────────────────────────────────────────────
    record_ids = sorted({p.parent.name for p in ecg_dir.glob("*/*.csv")})
    if not record_ids:
        raise RuntimeError(f"No ECG segment files found under {ecg_dir}.")

    train_ids, val_ids, test_ids = build_train_val_test_split(
        record_ids, file_annotations=config.FILE_ANNOTATIONS
    )
    logger.info("Train records (%d): %s", len(train_ids), train_ids)
    logger.info("Val   records (%d): %s", len(val_ids), val_ids)
    logger.info("Test  records (%d): %s", len(test_ids), test_ids)

    train_x, train_y = collect_segment_paths(train_ids, ecg_dir, annotation_dir)
    val_x, val_y = collect_segment_paths(val_ids, ecg_dir, annotation_dir)
    test_x, test_y = collect_segment_paths(test_ids, ecg_dir, annotation_dir)

    if not train_x:
        raise RuntimeError("No training segments found.")
    if not val_x:
        raise RuntimeError("No validation segments found.")
    if not test_x:
        raise RuntimeError("No test segments found.")

    # ── Data generators → tf.data pipelines ──────────────────────────────────
    train_gen = ECGDataGenerator(train_x, train_y, batch_size=config.BATCH_SIZE, augment=True)
    val_gen = ECGDataGenerator(val_x, val_y, batch_size=config.BATCH_SIZE, augment=False)
    test_gen = ECGDataGenerator(test_x, test_y, batch_size=config.BATCH_SIZE, augment=False)

    ds_train = (
        ecg_generator_to_tf_dataset(train_gen)
        .shuffle(len(train_x))
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_val = (
        ecg_generator_to_tf_dataset(val_gen).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )
    ds_test = (
        ecg_generator_to_tf_dataset(test_gen).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )

    # ── Compute class weights (MIT-BIH is heavily imbalanced toward N) ────────
    all_labels = []
    for ann_path in train_y:
        ann_df = pd.read_csv(ann_path, sep=" ", header=None)
        all_labels.extend(ann_df.iloc[:, 2].tolist())
    unique_classes = sorted(set(all_labels))
    weights = compute_class_weight("balanced", classes=np.array(unique_classes), y=np.array(all_labels))
    class_weight = {i: w for i, w in enumerate(weights)}
    logger.info("Class weights: %s", {c: f"{w:.2f}" for c, w in zip(unique_classes, weights)})

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(input_shape=(config.WINDOW_SIZE, 2))
    logger.info("Model parameters: %d", model.count_params())

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_model_path = output_dir / "model_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=config.LR_MIN,
            verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info(
        "Starting training: up to %d epochs, early stopping patience %d.",
        config.EPOCHS,
        config.EARLY_STOPPING_PATIENCE,
    )
    history = model.fit(
        ds_train,
        epochs=config.EPOCHS,
        validation_data=ds_val,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # ── Save artifacts ────────────────────────────────────────────────────────
    model.save(output_dir / "model.keras")
    logger.info("Final model saved to %s", output_dir / "model.keras")
    logger.info("Best model saved to %s", best_model_path)

    label_encoder = LabelEncoder().fit(config.ARRHYTHMIA_CLASSES)
    with (output_dir / "label_encoder.pkl").open("wb") as fh:
        pickle.dump(label_encoder, fh)
    logger.info("Label encoder saved to %s", output_dir / "label_encoder.pkl")

    _plot_training_history(history.history, output_dir)

    with (output_dir / "training_history.json").open("w") as fh:
        json.dump(history.history, fh, indent=2)
    logger.info("Training history saved to %s", output_dir / "training_history.json")

    # ── Final evaluation on held-out test set ─────────────────────────────────
    logger.info("Evaluating on held-out test set …")
    test_metrics = model.evaluate(ds_test, verbose=0)
    for name, value in zip(model.metrics_names, test_metrics):
        logger.info("test_%s: %.4f", name, value)

    logger.info("Training complete.")


def _plot_training_history(history: dict, output_dir: Path) -> None:
    """Save training/validation loss and accuracy curves to *output_dir*."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Categorical cross-entropy")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "accuracy" in history:
        axes[1].plot(history["accuracy"], label="train")
        axes[1].plot(history["val_accuracy"], label="val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry-point for the training pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train the ECG arrhythmia classification model.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Repository root directory (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/training"),
        help="Directory for model checkpoints and training plots.",
    )
    args = parser.parse_args()
    run_training(data_root=args.data_root, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
