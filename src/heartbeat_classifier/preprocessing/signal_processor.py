"""ECG signal preprocessing: zero-padding, windowing, and normalization.

Pipeline:
1. fill_zero_padding       – label every sample (silence between beats)
2. apply_annotation_window – spread each beat label over ±bandwidth samples
3. normalize_signals       – per-channel min-max scaling to SCALER_RANGE
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from heartbeat_classifier import config


def fill_zero_padding(
    annotations: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Produce dense per-sample annotation files covering [0, SIGNAL_LENGTH).

    Beat positions get their original label; all other samples get SILENCE_LABEL.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = config.SIGNAL_LENGTH
    indices = np.arange(n)
    timestamps = np.round(indices / config.SAMPLE_RATE, 4)

    for record_id, df in annotations.items():
        col2 = np.full(n, config.SILENCE_LABEL, dtype=object)
        col3 = np.zeros(n, dtype=int)
        col4 = np.zeros(n, dtype=int)
        col5 = np.zeros(n, dtype=int)

        for _, row in df.iterrows():
            beat_idx = int(row.iloc[1])
            if 0 <= beat_idx < n:
                col2[beat_idx] = str(row.iloc[2])

        out_df = pd.DataFrame(
            {0: timestamps, 1: indices, 2: col2, 3: col3, 4: col4, 5: col5}
        )
        out_df.to_csv(output_dir / f"{record_id}.csv", index=False, header=False, sep=" ")


def apply_annotation_window(
    padded_dir: Path,
    output_dir: Path,
    bandwidth: float = config.ANNOTATION_BANDWIDTH,
) -> None:
    """Extend each beat label ±bandwidth seconds around its sample position."""
    padded_dir = Path(padded_dir)
    output_dir = Path(output_dir)

    if not padded_dir.is_dir():
        raise FileNotFoundError(f"Padded annotation directory not found: {padded_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    half_window = round(bandwidth * config.SAMPLE_RATE)

    for csv_path in sorted(padded_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, sep=r"\s+", header=None, on_bad_lines="warn", engine="python")
        beat_positions = df.index[df.iloc[:, 2] != config.SILENCE_LABEL].tolist()

        for pos in beat_positions:
            beat_row = df.iloc[pos, 2:]
            start = max(0, pos - half_window)
            end = min(len(df), pos + half_window)
            df.iloc[start:end, 2:] = beat_row.values

        df.to_csv(output_dir / csv_path.name, index=False, header=False, sep=" ")


def normalize_signals(
    ecg_dir: Path,
    output_dir: Path,
) -> None:
    """Min-max normalize each ECG channel to SCALER_RANGE.

    Normalization is local to each recording — absolute amplitude differences
    across recordings are not preserved.
    """
    ecg_dir = Path(ecg_dir)
    output_dir = Path(output_dir)

    if not ecg_dir.is_dir():
        raise FileNotFoundError(f"ECG directory not found: {ecg_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = MinMaxScaler(feature_range=config.SCALER_RANGE)

    for csv_path in sorted(ecg_dir.glob("*.csv")):
        df = pd.read_csv(
            csv_path,
            index_col=None,
            on_bad_lines="warn",
            delimiter="\t",
            header=None,
        )
        for col in [1, 2]:
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

        df.iloc[:, 1:3].to_csv(
            output_dir / csv_path.name,
            index=False,
            header=False,
            sep=" ",
        )


def main() -> None:
    """CLI entry-point: run the full ECG preprocessing pipeline."""
    import argparse
    import logging

    from heartbeat_classifier.data.loader import load_annotation_files
    from heartbeat_classifier.preprocessing.annotation_processor import standardize_annotations
    from heartbeat_classifier.preprocessing.signal_slicer import (
        slice_annotations,
        slice_ecg_signals,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run the ECG preprocessing pipeline.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Repository root directory (default: current directory).",
    )
    root = parser.parse_args().data_root

    logger.info("Step 1/5 – Normalizing ECG signals …")
    normalize_signals(
        ecg_dir=root / config.DATA_DIRS["original_ecg"],
        output_dir=root / config.DATA_DIRS["normalized_ecg"],
    )

    logger.info("Step 2/5 – Filling silence labels between beats …")
    annotations = load_annotation_files(root / config.DATA_DIRS["original_annotations"])
    fill_zero_padding(
        annotations=annotations,
        output_dir=root / config.DATA_DIRS["annotations_padding"],
    )

    logger.info("Step 3/5 – Applying annotation window …")
    apply_annotation_window(
        padded_dir=root / config.DATA_DIRS["annotations_padding"],
        output_dir=root / config.DATA_DIRS["annotations_window"],
    )

    logger.info("Step 4/5 – Standardizing beat codes to ANSI/AAMI classes …")
    standardize_annotations(
        input_dir=root / config.DATA_DIRS["annotations_window"],
        output_dir=root / config.DATA_DIRS["standardized_annotations"],
    )

    logger.info("Step 5/5 – Slicing signals and annotations into windows …")
    slice_ecg_signals(
        normalized_ecg_dir=root / config.DATA_DIRS["normalized_ecg"],
        output_dir=root / config.DATA_DIRS["sliced_ecg"],
    )
    slice_annotations(
        annotation_dir=root / config.DATA_DIRS["standardized_annotations"],
        output_dir=root / config.DATA_DIRS["sliced_annotations"],
    )

    logger.info("Preprocessing complete.")
