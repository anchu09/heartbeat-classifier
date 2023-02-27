"""Map raw MIT-BIH beat codes to the ANSI/AAMI standard five-class scheme.

ANSI/AAMI EC57:1998 defines five morphological beat classes:
  N  – Normal and bundle-branch-block beats
  S  – Supraventricular ectopic beats
  V  – Ventricular ectopic beats
  Q  – Unknown / paced beats
  F  – Fusion beats

Any code not covered by the standard is mapped to ``SILENCE_LABEL`` (``"Z"``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from heartbeat_classifier import config


def standardize_annotations(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Map raw MIT-BIH beat codes to ANSI/AAMI class labels using BEAT_CLASS_MAP.

    Unknown codes fall back to SILENCE_LABEL.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Annotation input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, sep=r"\s+", header=None, on_bad_lines="warn", engine="python")

        df.iloc[:, 2] = df.iloc[:, 2].map(config.BEAT_CLASS_MAP).fillna(config.SILENCE_LABEL)

        df.to_csv(output_dir / csv_path.name, index=False, header=False, sep=" ")
