"""Slice full-length ECG recordings and annotations into fixed-size windows."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from heartbeat_classifier import config


def slice_ecg_signals(
    normalized_ecg_dir: Path,
    output_dir: Path,
    window_size: int = config.WINDOW_SIZE,
) -> None:
    """Partition normalized ECG files into window_size-sample chunks per record."""
    normalized_ecg_dir = Path(normalized_ecg_dir)
    output_dir = Path(output_dir)

    if not normalized_ecg_dir.is_dir():
        raise FileNotFoundError(f"Normalized ECG directory not found: {normalized_ecg_dir}")

    for csv_path in sorted(normalized_ecg_dir.glob("*.csv")):
        record_id = csv_path.stem
        record_out = output_dir / record_id
        record_out.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(
            csv_path,
            index_col=None,
            on_bad_lines="warn",
            delimiter=" ",
            header=None,
        )

        for chunk_idx, (_, chunk) in enumerate(df.groupby(df.index // window_size), start=1):
            out_path = record_out / f"{record_id}_part{chunk_idx:03d}.csv"
            chunk.to_csv(
                out_path,
                index=False,
                header=False,
                quoting=csv.QUOTE_NONE,
                escapechar="\\",
            )


def slice_annotations(
    annotation_dir: Path,
    output_dir: Path,
    window_size: int = config.WINDOW_SIZE,
) -> None:
    """Partition standardized annotation files into window_size-sample chunks per record."""
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)

    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    for csv_path in sorted(annotation_dir.glob("*.csv")):
        record_id = csv_path.stem
        record_out = output_dir / record_id
        record_out.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path, sep=r"\s+", header=None, on_bad_lines="warn", engine="python")

        for chunk_idx, (_, chunk) in enumerate(df.groupby(df.index // window_size), start=1):
            out_path = record_out / f"{record_id}_part{chunk_idx:03d}.csv"
            chunk.to_csv(out_path, index=False, header=False, sep=" ")
