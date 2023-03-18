"""Keras Sequence data generator for ECG segment batches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from heartbeat_classifier import config
from heartbeat_classifier.augmentation.transforms import apply_random_transform


_CLASS_TO_IDX: dict[str, int] = {cls: i for i, cls in enumerate(config.ARRHYTHMIA_CLASSES)}


def _load_annotation(path: Path) -> np.ndarray:
    """Load an annotation slice and return a (WINDOW_SIZE, NUM_CLASSES) one-hot int32 array."""
    df = pd.read_csv(path, sep=" ", header=None)
    indices = df.iloc[:, 2].map(_CLASS_TO_IDX).to_numpy(dtype=np.int32)
    return np.eye(config.NUM_CLASSES, dtype=np.int32)[indices]


class ECGDataGenerator(Sequence):
    """Load ECG segments and annotation masks from disk on demand.

    X: (WINDOW_SIZE, 2) float32 — two ECG channels, comma-separated on disk.
    y: (WINDOW_SIZE, NUM_CLASSES) int32 one-hot mask derived from the space-separated
       annotation slices (col 2 holds the AAMI label string).
    """

    def __init__(
        self,
        x_paths: list[Path],
        y_paths: list[Path],
        batch_size: int,
        augment: bool = False,
    ) -> None:
        if len(x_paths) != len(y_paths):
            raise ValueError(
                f"x_paths and y_paths must have equal length, "
                f"got {len(x_paths)} vs {len(y_paths)}."
            )
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self) -> int:
        return int(np.ceil(len(self.x_paths) / self.batch_size))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        batch_x_paths = self.x_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y_paths = self.y_paths[idx * self.batch_size : (idx + 1) * self.batch_size]

        # ECG slices: comma-separated (signal_slicer uses pandas default sep=",")
        batch_x = np.asarray(
            [pd.read_csv(p, header=None).values for p in batch_x_paths], dtype=np.float32
        )
        batch_y = np.asarray([_load_annotation(p) for p in batch_y_paths])

        if self.augment:
            for i in range(batch_x.shape[0]):
                tid = np.random.randint(0, config.NUM_AUGMENT_TRANSFORMS)
                batch_x[i : i + 1], batch_y[i : i + 1] = apply_random_transform(
                    tid, batch_x[i : i + 1], batch_y[i : i + 1]
                )

        return batch_x, batch_y

    def __repr__(self) -> str:
        return (
            f"ECGDataGenerator(n_samples={len(self.x_paths)}, "
            f"batch_size={self.batch_size}, augment={self.augment})"
        )


def ecg_generator_to_tf_dataset(generator: ECGDataGenerator) -> tf.data.Dataset:
    """Convert an ECGDataGenerator to a tf.data.Dataset of individual samples."""
    def _gen():
        for i in range(len(generator)):
            x, y = generator[i]
            yield from zip(x, y, strict=True)

    return tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(config.WINDOW_SIZE, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(config.WINDOW_SIZE, config.NUM_CLASSES), dtype=tf.int32),
        ),
    )
