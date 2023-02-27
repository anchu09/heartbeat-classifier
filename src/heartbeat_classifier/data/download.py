"""Download the MIT-BIH Arrhythmia Database and extract signals and annotations.

hbc-download --data-root .

Requires the 'data' optional dependency: uv sync --extra data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

logger = logging.getLogger(__name__)

RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def download(db_dir: Path) -> None:
    db_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MIT-BIH Arrhythmia Database to %s …", db_dir)
    wfdb.dl_database("mitdb", dl_dir=str(db_dir))
    logger.info("Download complete.")


def extract(db_dir: Path, ecg_dir: Path, ann_dir: Path) -> None:
    ecg_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    for record_id in RECORDS:
        record_path = str(db_dir / record_id)

        record = wfdb.rdrecord(record_path)
        # forward-fill then zero to handle the rare missing samples in MIT-BIH
        signals = pd.DataFrame(record.p_signal).ffill().fillna(0.0).to_numpy()

        ecg_df = pd.DataFrame(signals)
        ecg_df.insert(0, "idx", range(len(ecg_df)))
        ecg_df.to_csv(ecg_dir / f"{record_id}.csv", sep="\t", header=False, index=False)

        ann = wfdb.rdann(record_path, "atr")
        timestamps = np.array(ann.sample) / record.fs
        ann_df = pd.DataFrame({
            0: timestamps,
            1: ann.sample,
            2: ann.symbol,
            3: 0,
            4: 0,
            5: 0,
        })
        ann_df.to_csv(ann_dir / f"{record_id}.csv", sep=" ", header=False, index=False)

        logger.info("Extracted record %s (%d samples, %d annotations)",
                    record_id, len(signals), len(ann.sample))

    logger.info("All %d records extracted.", len(RECORDS))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download and extract the MIT-BIH database.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Repository root directory (default: current directory).",
    )
    root = parser.parse_args().data_root

    db_dir  = root / "original_database"
    ecg_dir = root / "semi_preprocessed_signals" / "original_ecg"
    ann_dir = root / "semi_preprocessed_signals" / "original_annotations"

    download(db_dir)
    extract(db_dir, ecg_dir, ann_dir)


if __name__ == "__main__":
    main()
