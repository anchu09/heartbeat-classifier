"""Time and string formatting utilities for WFDB annotation output."""

from __future__ import annotations

import numpy as np


def seconds_to_timestamp(total_seconds: float) -> str:
    """Convert seconds to 'M:SS.mmm' format (e.g. 65.32 → '1:05.320')."""
    minutes, remaining = divmod(total_seconds, 60)
    return f"{int(minutes)}:{remaining:06.3f}"


def pad_timestamp(timestamp: str, total_width: int = 12) -> str:
    """Return spaces so that padding + timestamp fills total_width (WFDB column format)."""
    return " " * (total_width - len(timestamp))


def timestamp_to_seconds(time_str: str) -> float:
    """Parse 'M:SS.mmm' into total seconds. Raises ValueError on bad format."""
    try:
        minutes_part, seconds_part = time_str.split(":")
        seconds, milliseconds = seconds_part.split(".")
        return int(minutes_part) * 60 + int(seconds) + int(milliseconds) / 1000
    except ValueError as exc:
        raise ValueError(f"Cannot parse timestamp {time_str!r}. Expected 'MM:SS.mmm'.") from exc


def moving_average_kernel(half_order: int) -> np.ndarray:
    """Return a normalized uniform FIR kernel of length 2*half_order+1 that sums to 1."""
    length = 2 * half_order + 1
    return np.repeat(1.0 / length, length)
