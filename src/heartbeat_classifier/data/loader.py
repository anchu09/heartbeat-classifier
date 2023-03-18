"""Functions for loading ECG records and building train/test splits."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from heartbeat_classifier import config

logger = logging.getLogger(__name__)


def load_ecg_signals(directory: Path) -> dict[str, pd.DataFrame]:
    """Load normalized ECG CSVs from *directory*. Returns {record_id: DataFrame}.

    Files are tab-separated with two signal channels (no header).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"ECG directory not found: {directory}")

    records: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(directory.glob("*.csv")):
        record_id = csv_path.stem
        records[record_id] = pd.read_csv(
            csv_path,
            index_col=None,
            on_bad_lines="warn",
            delimiter="\t",
            header=None,
        )
    return records


def load_annotation_files(directory: Path) -> dict[str, pd.DataFrame]:
    """Load annotation CSVs from *directory*. Returns {record_id: DataFrame}.

    Files are whitespace-separated; column 2 holds the beat label.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {directory}")

    annotations: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(directory.glob("*.csv")):
        record_id = csv_path.stem
        annotations[record_id] = pd.read_csv(
            csv_path,
            sep=r"\s+",
            header=None,
            on_bad_lines="skip",
            engine="python",
        )
    return annotations


def build_train_val_test_split(
    record_ids: list[str],
    file_annotations: dict[str, str] | None = None,
    test_size: float = config.TRAIN_TEST_SPLIT,
    val_size: float = config.TRAIN_VAL_SPLIT,
    random_state: int = config.RANDOM_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Split record_ids into train, validation, and test sets.

    The test set uses the same first split as a standalone train/test split would.
    The val set is carved from the training portion. Falls back to random
    splitting if any class has too few samples to stratify the val split.
    Returns sorted (train_ids, val_ids, test_ids).
    """
    if file_annotations is not None:
        known = [rid for rid in record_ids if rid in file_annotations]
        if not known:
            raise ValueError("None of the provided record_ids appear in file_annotations.")
        strata = [file_annotations[rid] for rid in known]
        train_val_ids, test_ids = train_test_split(
            known, test_size=test_size, stratify=strata, random_state=random_state
        )
    else:
        if not record_ids:
            raise ValueError("record_ids must not be empty.")
        train_val_ids, test_ids = train_test_split(
            record_ids, test_size=test_size, random_state=random_state
        )

    val_fraction = val_size / (1.0 - test_size)
    try:
        tv_strata = (
            [file_annotations[rid] for rid in train_val_ids]
            if file_annotations is not None
            else None
        )
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_fraction,
            stratify=tv_strata,
            random_state=random_state,
        )
    except ValueError:
        logger.warning(
            "Stratified val split failed (too few samples in some class); "
            "falling back to random split."
        )
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_fraction, random_state=random_state
        )

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def collect_segment_paths(
    record_ids: list[str],
    ecg_dir: Path,
    annotation_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Return sorted (ecg_paths, annotation_paths) for the given record IDs.

    Raises ValueError if ECG and annotation segment counts differ for any record.
    """
    ecg_paths: list[Path] = []
    annotation_paths: list[Path] = []

    for record_id in record_ids:
        record_ecg = sorted(Path(ecg_dir).glob(f"{record_id}/*.csv"))
        record_ann = sorted(Path(annotation_dir).glob(f"{record_id}/*.csv"))
        if len(record_ecg) != len(record_ann):
            raise ValueError(
                f"Record {record_id}: {len(record_ecg)} ECG segments but "
                f"{len(record_ann)} annotation segments. Re-run preprocessing."
            )
        ecg_paths.extend(record_ecg)
        annotation_paths.extend(record_ann)

    return ecg_paths, annotation_paths
