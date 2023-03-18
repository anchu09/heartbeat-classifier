"""Random signal augmentation transforms for ECG training data.

Each transform accepts and returns ``(batch_x, batch_y)`` where:
- ``batch_x``: ``(1, WINDOW_SIZE, 2)`` float32 array — one sample, two channels.
- ``batch_y``: ``(1, WINDOW_SIZE, NUM_CLASSES)`` int32 one-hot annotation mask.

Transforms add realistic noise and distortions found in ambulatory ECG recordings:
low/high/band-pass filtering, baseline wander, 50 Hz power-line interference,
and white Gaussian noise. SNR values are in decibels (properly converted to
linear ratio before computing noise power).
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sig

from heartbeat_classifier import config

_FS = config.SAMPLE_RATE


def apply_random_transform(
    transform_id: int,
    batch_x: np.ndarray,
    batch_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply transform transform_id to batch_x; 0 = identity. Raises ValueError if out of range."""
    if not (0 <= transform_id < config.NUM_AUGMENT_TRANSFORMS):
        raise ValueError(
            f"transform_id must be in [0, {config.NUM_AUGMENT_TRANSFORMS}), got {transform_id}."
        )
    if transform_id == 0:
        return batch_x, batch_y
    if transform_id == 1:
        return _low_pass(batch_x), batch_y
    if transform_id == 2:
        return _high_pass(batch_x), batch_y
    if transform_id == 3:
        return _band_pass(batch_x), batch_y
    if transform_id == 4:
        return _add_baseline_wander(batch_x), batch_y
    if transform_id == 5:
        return _add_50hz_interference(batch_x), batch_y
    # transform_id == 6
    return _add_white_noise(batch_x), batch_y


# ── Private helpers ───────────────────────────────────────────────────────────


def _apply_filter(batch_x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply a zero-phase IIR filter to every channel of batch_x."""
    result = batch_x.copy()
    for channel in range(result.shape[2]):
        result[0, :, channel] = sig.filtfilt(b, a, result[0, :, channel])
    return result


def _db_to_noise(signal_power: float, snr_db: float) -> float:
    """Convert an SNR value in dB to the corresponding noise power."""
    snr_linear = 10.0 ** (snr_db / 10.0)
    return signal_power / snr_linear


# ── Transform implementations ─────────────────────────────────────────────────


def _low_pass(batch_x: np.ndarray) -> np.ndarray:
    """Low-pass filter with a random cutoff in the configured range."""
    lo, hi = config.LP_CUTOFF_RANGE
    cutoff = np.random.uniform(lo, hi)
    b, a = sig.butter(6, cutoff / (_FS / 2.0), btype="lowpass")
    return _apply_filter(batch_x, b, a)


def _high_pass(batch_x: np.ndarray) -> np.ndarray:
    """High-pass filter with a random cutoff in the configured range."""
    lo, hi = config.HP_CUTOFF_RANGE
    cutoff = np.random.uniform(lo, hi)
    b, a = sig.butter(6, cutoff / (_FS / 2.0), btype="highpass")
    return _apply_filter(batch_x, b, a)


def _band_pass(batch_x: np.ndarray) -> np.ndarray:
    """Band-pass filter combining high-pass and low-pass cutoff ranges."""
    low = np.random.uniform(*config.HP_CUTOFF_RANGE)
    high = np.random.uniform(*config.LP_CUTOFF_RANGE)
    b, a = sig.butter(3, [low / (_FS / 2.0), high / (_FS / 2.0)], btype="bandpass")
    return _apply_filter(batch_x, b, a)


def _add_white_noise(batch_x: np.ndarray) -> np.ndarray:
    """Add white Gaussian noise at a random SNR in [NOISE_SNR_MIN_DB, NOISE_SNR_MAX_DB] dB."""
    snr_db = np.random.uniform(config.NOISE_SNR_MIN_DB, config.NOISE_SNR_MAX_DB)
    result = batch_x.copy()
    for channel in range(result.shape[2]):
        signal_power = float(np.mean(result[0, :, channel] ** 2))
        noise_power = _db_to_noise(signal_power, snr_db)
        noise = np.random.normal(0.0, np.sqrt(noise_power), result.shape[1])
        result[0, :, channel] += noise
    return result


def _add_baseline_wander(batch_x: np.ndarray) -> np.ndarray:
    """Add sinusoidal baseline wander with a random frequency in the configured range."""
    freq = np.random.uniform(*config.BASELINE_WANDER_FREQ)
    time = np.arange(batch_x.shape[1]) / _FS
    result = batch_x.copy()
    for channel in range(result.shape[2]):
        amplitude = np.random.uniform(0.0, float(np.mean(np.abs(result[0, :, channel]))))
        result[0, :, channel] += amplitude * np.sin(2 * np.pi * freq * time)
    return result


def _add_50hz_interference(batch_x: np.ndarray) -> np.ndarray:
    """Add 50 Hz power-line interference with random harmonics at a random SNR in dB."""
    snr_db = np.random.uniform(config.NOISE_SNR_MIN_DB, config.NOISE_SNR_MAX_DB)
    t = np.arange(batch_x.shape[1]) / _FS

    f0 = config.POWERLINE_FREQ
    num_harmonics = np.random.randint(1, config.POWERLINE_MAX_HARMONICS)
    harmonic_indices = np.sort(
        np.random.choice(range(2, config.POWERLINE_MAX_HARMONICS + 1), num_harmonics, replace=False)
    )
    noise_template = np.sin(2 * np.pi * f0 * t)
    for h in harmonic_indices:
        noise_template += np.sin(2 * np.pi * f0 * h * t)

    result = batch_x.copy()
    for channel in range(result.shape[2]):
        signal_power = float(np.mean(result[0, :, channel] ** 2))
        noise_power = _db_to_noise(signal_power, snr_db)
        noise_amplitude = np.sqrt(noise_power / np.mean(noise_template**2))
        result[0, :, channel] += noise_amplitude * noise_template
    return result
