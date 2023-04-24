"""Unit tests for preprocessing and utility functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heartbeat_classifier import config
from heartbeat_classifier.preprocessing.signal_processor import (
    fill_zero_padding,
    normalize_signals,
)
from heartbeat_classifier.utils.time_utils import (
    moving_average_kernel,
    seconds_to_timestamp,
    timestamp_to_seconds,
)


def test_fill_zero_padding_writes_silence_between_beats(tmp_path, monkeypatch):
    """Samples between annotated beats should receive SILENCE_LABEL."""
    monkeypatch.setattr(config, "SIGNAL_LENGTH", 100)
    df = pd.DataFrame(
        {0: [0.0, 0.138], 1: [10, 50], 2: ["N", "V"], 3: [0, 0], 4: [0, 0], 5: [0, 0]}
    )
    fill_zero_padding({"100": df}, tmp_path)

    out = (tmp_path / "100.csv").read_text().splitlines()
    for line in out[:10]:
        assert line.split()[2] == config.SILENCE_LABEL, f"Expected silence, got: {line}"
    beat_lines = [row for row in out if row.split()[1] == "10"]
    assert beat_lines and beat_lines[0].split()[2] == "N"


def test_normalize_signals_output_in_scaler_range(tmp_path):
    """All output values must lie within SCALER_RANGE after normalization."""
    ecg_dir = tmp_path / "ecg"
    out_dir = tmp_path / "ecg_norm"
    ecg_dir.mkdir()

    # col 0 = sample index, col 1 = channel 1, col 2 = channel 2 (tab-separated)
    rng = np.random.default_rng(0)
    ecg = rng.standard_normal((200, 2)) * 500
    indices = np.arange(200).reshape(-1, 1)
    data = np.hstack([indices, ecg])
    pd.DataFrame(data).to_csv(ecg_dir / "record100.csv", index=False, header=False, sep="\t")

    normalize_signals(ecg_dir, out_dir)

    result = pd.read_csv(out_dir / "record100.csv", header=None, sep=" ").values.astype(float)
    lo, hi = config.SCALER_RANGE
    assert result.min() >= lo - 1e-9
    assert result.max() <= hi + 1e-9


def test_timestamp_roundtrip():
    """seconds_to_timestamp and timestamp_to_seconds must be inverses."""
    for seconds in (0.0, 1.5, 60.0, 123.456, 3599.999):
        ts = seconds_to_timestamp(seconds)
        recovered = timestamp_to_seconds(ts)
        assert abs(recovered - seconds) < 0.001, f"Roundtrip failed for {seconds}s"


def test_timestamp_to_seconds_bad_format_raises():
    with pytest.raises(ValueError, match="Cannot parse"):
        timestamp_to_seconds("not_a_timestamp")


@pytest.mark.parametrize("half_order", [1, 5, 25, 51])
def test_moving_average_kernel(half_order):
    kernel = moving_average_kernel(half_order)
    assert len(kernel) == 2 * half_order + 1
    assert abs(kernel.sum() - 1.0) < 1e-10
    assert np.allclose(kernel, kernel[0])  # uniform window


def test_annotation_half_window_matches_formula():
    expected = round(config.ANNOTATION_BANDWIDTH * config.SAMPLE_RATE)
    assert config.ANNOTATION_HALF_WINDOW == expected


def test_beat_class_map_only_maps_to_valid_classes():
    valid = set(config.ARRHYTHMIA_CLASSES) - {config.SILENCE_LABEL}
    for code, cls in config.BEAT_CLASS_MAP.items():
        assert cls in valid, f"Beat code {code!r} maps to invalid class {cls!r}"


def test_all_transform_ids_are_dispatchable():
    """Every ID in [0, NUM_AUGMENT_TRANSFORMS) must dispatch without error."""
    from heartbeat_classifier.augmentation.transforms import apply_random_transform

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, config.WINDOW_SIZE, 2)).astype(np.float32)
    y = np.zeros((1, config.WINDOW_SIZE, config.NUM_CLASSES), dtype=np.int32)
    for tid in range(config.NUM_AUGMENT_TRANSFORMS):
        x_out, y_out = apply_random_transform(tid, x.copy(), y.copy())
        assert x_out.shape == x.shape
        assert y_out.shape == y.shape
