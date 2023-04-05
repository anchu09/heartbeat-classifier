"""Prediction and WFDB annotation evaluation pipeline.

hbc-evaluate --model results/training/model.keras --data-root . --output-dir results/eval_01

Runs inference on the held-out test set, applies moving-average smoothing,
decodes predictions to AAMI beat labels, and writes WFDB annotation files
for beat-by-beat comparison via wrann/bxb.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import IO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from heartbeat_classifier import config
from heartbeat_classifier.data.generator import ECGDataGenerator, ecg_generator_to_tf_dataset
from heartbeat_classifier.data.loader import (
    build_train_val_test_split,
    collect_segment_paths,
)
from heartbeat_classifier.utils.time_utils import (
    moving_average_kernel,
    pad_timestamp,
    seconds_to_timestamp,
)

logger = logging.getLogger(__name__)


def run_evaluation(
    model_path: Path,
    data_root: Path,
    output_dir: Path,
) -> None:
    """Run inference on the test set and write WFDB beat-by-beat results."""
    model_path = Path(model_path)
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ecg_dir = data_root / config.DATA_DIRS["sliced_ecg"]
    annotation_dir = data_root / config.DATA_DIRS["sliced_annotations"]

    for required in (ecg_dir, annotation_dir):
        if not required.is_dir():
            raise FileNotFoundError(
                f"Required data directory not found: {required}. "
                "Run the preprocessing pipeline first."
            )

    logger.info("Loading model from %s", model_path)
    model: tf.keras.Model = tf.keras.models.load_model(model_path)

    # ── Reproduce test split ──────────────────────────────────────────────────
    record_ids = sorted({p.parent.name for p in ecg_dir.glob("*/*.csv")})
    _, _, test_ids = build_train_val_test_split(record_ids, file_annotations=config.FILE_ANNOTATIONS)

    test_x, test_y = collect_segment_paths(test_ids, ecg_dir, annotation_dir)
    if not test_x:
        raise RuntimeError(f"No test segments found under {ecg_dir}.")

    test_gen = ECGDataGenerator(test_x, test_y, batch_size=config.BATCH_SIZE, augment=False)
    ds_test = ecg_generator_to_tf_dataset(test_gen).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    logger.info("Running inference on %d test segments.", len(test_x))
    raw_predictions: np.ndarray = model.predict(ds_test)

    # ── Smooth predictions ────────────────────────────────────────────────────
    smoothed = apply_moving_average(raw_predictions, config.MOVING_AVG_ORDER)

    # ── Decode to beat labels ─────────────────────────────────────────────────
    encoder_path = model_path.parent / "label_encoder.pkl"
    if encoder_path.is_file():
        with encoder_path.open("rb") as fh:
            label_encoder = pickle.load(fh)
        logger.info("Loaded label encoder from %s", encoder_path)
    else:
        logger.warning(
            "label_encoder.pkl not found next to model; "
            "fitting from config.ARRHYTHMIA_CLASSES — ensure class list matches training."
        )
        label_encoder = LabelEncoder()
        label_encoder.fit(config.ARRHYTHMIA_CLASSES)
    decoded = decode_predictions(smoothed, label_encoder)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bxb_dir = output_dir / "bxb" / run_timestamp
    bxb_dir.mkdir(parents=True, exist_ok=True)

    test_record_ids = sorted({path.parent.name for path in test_y})

    write_annotation_files(
        decoded=decoded,
        test_paths=test_y,
        record_ids=test_record_ids,
        bxb_dir=bxb_dir,
        original_database_dir=data_root / config.DATA_DIRS["original_database"],
    )

    _run_wfdb_evaluation(
        bxb_dir=bxb_dir,
        record_ids=test_record_ids,
    )

    plot_predictions(
        smoothed=smoothed,
        test_x=test_x,
        test_y=test_y,
        output_dir=output_dir / "plots",
    )

    logger.info("Evaluation complete. Results written to %s", output_dir)


def apply_moving_average(
    predictions: np.ndarray,
    half_order: int,
) -> np.ndarray:
    """Apply a uniform moving-average filter to each class channel of predictions."""
    kernel = moving_average_kernel(half_order)
    smoothed = predictions.copy()
    for i in range(smoothed.shape[0]):
        for k in range(smoothed.shape[2]):
            smoothed[i, :, k] = np.convolve(smoothed[i, :, k], kernel, mode="same")
    return smoothed


def decode_predictions(
    predictions: np.ndarray,
    label_encoder: LabelEncoder,
) -> np.ndarray:
    """Argmax predictions and decode to AAMI beat labels. Returns shape (WINDOW_SIZE, N)."""
    max_indices = np.argmax(predictions, axis=-1)
    n_segments, window_size = max_indices.shape
    flat_labels = label_encoder.inverse_transform(max_indices.ravel())
    return flat_labels.reshape(n_segments, window_size).T


def write_annotation_files(
    decoded: np.ndarray,
    test_paths: list[Path],
    record_ids: list[str],
    bxb_dir: Path,
    original_database_dir: Path,
) -> None:
    """Write one WFDB-format annotation CSV per test record into bxb_dir."""
    segment_counts: dict[str, int] = {}
    for path in test_paths:
        rid = path.parent.name
        segment_counts[rid] = segment_counts.get(rid, 0) + 1

    col_start = 0
    for record_id in record_ids:
        col_end = col_start + segment_counts[record_id]
        record_labels = decoded[:, col_start:col_end]
        full_signal = record_labels.flatten(order="F")

        out_path = bxb_dir / f"{record_id}.csv"
        with out_path.open("w") as fh:
            prev_label = config.SILENCE_LABEL
            prev_start = 0

            for sample_idx, label in enumerate(full_signal):
                if label == config.SILENCE_LABEL:
                    continue
                if label == prev_label:
                    continue

                # Emit the midpoint of the previous run
                if prev_label != config.SILENCE_LABEL:
                    midpoint = (prev_start + sample_idx) // 2
                    _write_annotation_row(fh, midpoint, prev_label)

                prev_label = label
                prev_start = sample_idx

            # Flush last run
            if prev_label != config.SILENCE_LABEL:
                midpoint = (prev_start + len(full_signal)) // 2
                _write_annotation_row(fh, midpoint, prev_label)

        col_start = col_end

        # Copy reference files needed by WFDB tools
        for ext in (".atr", ".hea", ".qrs"):
            src = original_database_dir / f"{record_id}{ext}"
            if src.is_file():
                shutil.copy(src, bxb_dir / f"{record_id}{ext}")


def _write_annotation_row(fh: IO[str], sample_idx: int, label: str) -> None:
    """Write one WFDB annotation row. sub/chan/num are 0 (unused in beat files)."""
    time_seconds = sample_idx / config.SAMPLE_RATE
    timestamp = seconds_to_timestamp(time_seconds)
    padding = pad_timestamp(timestamp)
    line = f"{padding}{timestamp}{sample_idx:9d}     {label}    0    0    0\n"
    fh.write(line)


def _run_wfdb_evaluation(
    bxb_dir: Path,
    record_ids: list[str],
) -> None:
    """Run wrann/rdann/bxb for each record. Requires wfdb-tools on PATH.

    Per-record failures are logged as warnings so the remaining records
    are still processed.
    """
    results_file = bxb_dir / "bxb_results.txt"

    for record_id in sorted(record_ids):
        record_base = str(bxb_dir / record_id)
        ann_csv = bxb_dir / f"{record_id}.csv"

        commands = [
            ["wrann", "-r", record_base, "-a", "myqrs"],
            ["rdann", "-r", record_base, "-a", "myqrs"],
            ["bxb", "-r", record_base, "-a", "atr", "myqrs"],
            ["bxb", "-r", record_base, "-a", "atr", "qrs"],
        ]

        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    input=ann_csv.read_text() if cmd[0] == "wrann" else None,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if result.stdout:
                    with results_file.open("a") as fh:
                        fh.write(f"\n{'─' * 80}\n{record_id}: {' '.join(cmd)}\n")
                        fh.write(result.stdout)
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "WFDB command failed for record %s: %s\n%s",
                    record_id,
                    " ".join(cmd),
                    exc.stderr,
                )
            except FileNotFoundError:
                logger.warning(
                    "WFDB tool not found: %s. "
                    "Install wfdb-tools to enable beat-by-beat evaluation.",
                    cmd[0],
                )
                return


def plot_predictions(
    smoothed: np.ndarray,
    test_x: list[Path],
    test_y: list[Path],
    output_dir: Path,
) -> None:
    """Save one ECG + prediction vs ground-truth overlay plot per test segment.

    Each plot shows the raw ECG signal (channel 0) together with per-class
    predicted (PRED) and original (ORIG) one-hot annotation traces, matching
    the ARRHYTHMIA_CLASSES label mapping.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    _class_to_idx = {cls: i for i, cls in enumerate(config.ARRHYTHMIA_CLASSES)}
    pred_onehot = np.eye(config.NUM_CLASSES, dtype=np.float32)[np.argmax(smoothed, axis=-1)]
    decoder_label = "  ".join(f"{cls}={i}" for i, cls in enumerate(config.ARRHYTHMIA_CLASSES))

    for i, (ecg_path, ann_path) in enumerate(zip(test_x, test_y)):
        ecg = pd.read_csv(ecg_path, header=None).values.astype(np.float32)
        ann_df = pd.read_csv(ann_path, sep=" ", header=None)
        true_indices = ann_df.iloc[:, 2].map(_class_to_idx).to_numpy(dtype=np.int32)
        true_onehot = np.eye(config.NUM_CLASSES, dtype=np.float32)[true_indices]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(ecg[:, 0] + 1, label=decoder_label, color="steelblue", linewidth=0.8)

        for k in range(config.NUM_CLASSES - 1):  # F N Q S V (0-4)
            ax.plot(pred_onehot[i, :, k], label=f"PRED: {k}")
            ax.plot(true_onehot[:, k] - 0.05 - k / 100, label=f"ORIG: {k}")

        z_idx = config.NUM_CLASSES - 1
        ax.plot(pred_onehot[i, :, z_idx] - 0.4, label=f"PRED: {z_idx}", color="brown")
        ax.plot(true_onehot[:, z_idx] - 0.45, label=f"ORIG: {z_idx}", color="black")

        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel("Samples")
        ax.set_ylabel("ECG and annotations")
        ax.set_title(f"{i}: {ecg_path.stem}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.subplots_adjust(right=0.78)
        fig.savefig(output_dir / f"{i}_{ecg_path.stem}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    logger.info("Saved %d prediction plots to %s", len(test_x), output_dir)


def main() -> None:
    """CLI entry-point for the evaluation pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate a trained heartbeat-classifier model.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the saved .keras model file.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Repository root directory (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for annotation files and evaluation results.",
    )
    args = parser.parse_args()
    run_evaluation(
        model_path=args.model,
        data_root=args.data_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
