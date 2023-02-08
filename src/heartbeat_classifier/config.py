"""Central configuration — all constants live here."""

from __future__ import annotations

# ── Signal & model constants ──────────────────────────────────────────────────
SAMPLE_RATE: int = 360  # MIT-BIH sampling frequency (Hz)
SIGNAL_LENGTH: int = 650_000  # Samples per full recording
WINDOW_SIZE: int = 5_000
ANNOTATION_BANDWIDTH: float = 0.15  # Seconds around each beat label
ANNOTATION_HALF_WINDOW: int = round(ANNOTATION_BANDWIDTH * SAMPLE_RATE)

# ── Training hyper-parameters ─────────────────────────────────────────────────
RANDOM_SEED: int = 42
BATCH_SIZE: int = 32
EPOCHS: int = 100
EARLY_STOPPING_PATIENCE: int = 10
TRAIN_TEST_SPLIT: float = 0.30
TRAIN_VAL_SPLIT: float = 0.15
DROPOUT_RATE: float = 0.3
L2_REGULARIZATION: float = 1e-4
MOVING_AVG_ORDER: int = 51       # Moving-average filter half-order

# ── Optimizer & learning-rate schedule ───────────────────────────────────────
LEARNING_RATE: float = 1e-3
LR_REDUCE_FACTOR: float = 0.5
LR_REDUCE_PATIENCE: int = 5
LR_MIN: float = 1e-6

# ── Architecture ──────────────────────────────────────────────────────────────
ENC_FILTERS: tuple[int, ...] = (64, 128, 512)
ENC_CONV_BLOCKS: tuple[int, ...] = (4, 4, 4)
DEC_FILTERS: int = 256
DENSE_UNITS: tuple[int, int] = (256, 128)
DECODER_KERNEL_SIZE: int = 8

# ── Normalization ─────────────────────────────────────────────────────────────
SCALER_RANGE: tuple[float, float] = (-1.0, 1.0)

# ── Augmentation ─────────────────────────────────────────────────────────────
NUM_AUGMENT_TRANSFORMS: int = 7  # Number of transforms (ids 0..6; 0 = identity)
NOISE_SNR_MIN_DB: float = 10.0  # dB
NOISE_SNR_MAX_DB: float = 30.0  # dB
LP_CUTOFF_RANGE: tuple[float, float] = (30.0, 100.0)   # Low-pass cutoff (Hz)
HP_CUTOFF_RANGE: tuple[float, float] = (0.5, 1.0)      # High-pass cutoff (Hz)
BASELINE_WANDER_FREQ: tuple[float, float] = (0.05, 0.5) # Wander frequency (Hz)
POWERLINE_FREQ: float = 50.0                             # Power-line frequency (Hz)
POWERLINE_MAX_HARMONICS: int = 5                         # Max harmonic order

# ── Arrhythmia classes ────────────────────────────────────────────────────────
ARRHYTHMIA_CLASSES: list[str] = ["F", "N", "Q", "S", "V", "Z"]
NUM_CLASSES: int = len(ARRHYTHMIA_CLASSES)
SILENCE_LABEL: str = "Z"  # Sentinel for non-beat samples

# ── ANSI/AAMI beat-code → standard class mapping ─────────────────────────────
# Based on American National Standard ANSI/AAMI EC57:1998
BEAT_CLASS_MAP: dict[str, str] = {
    # Normal beats
    "N": "N",
    "L": "N",
    "R": "N",
    "B": "N",
    "e": "N",
    "n": "N",
    # Supraventricular ectopic beats
    "a": "S",
    "J": "S",
    "A": "S",
    "S": "S",
    "j": "S",
    # Ventricular ectopic beats
    "V": "V",
    "E": "V",
    # Unknown / paced beats
    "/": "Q",
    "f": "Q",
    "Q": "Q",
    # Fusion beats
    "F": "F",
}

# ── MIT-BIH record → dominant-class label (used for stratified split) ────────
FILE_ANNOTATIONS: dict[str, str] = {
    "100": "S",
    "101": "S",
    "102": "Q",
    "103": "S",
    "104": "Q",
    "105": "Q",
    "106": "V",
    "107": "Q",
    "108": "F",
    "109": "F",
    "111": "V",
    "112": "S",
    "113": "S",
    "114": "F",
    "115": "N",
    "116": "S",
    "117": "S",
    "118": "S",
    "119": "V",
    "121": "S",
    "122": "N",
    "123": "V",
    "124": "F",
    "200": "F",
    "201": "F",
    "202": "F",
    "203": "F",
    "205": "F",
    "207": "S",
    "208": "F",
    "209": "S",
    "210": "F",
    "212": "N",
    "213": "F",
    "214": "F",
    "215": "F",
    "217": "Q",
    "219": "F",
    "220": "S",
    "221": "V",
    "222": "S",
    "223": "F",
    "228": "S",
    "230": "V",
    "231": "S",
    "232": "S",
    "233": "F",
    "234": "S",
}

# ── Data layout (relative to repo root) ──────────────────────────────────────
DATA_DIRS = {
    "original_database": "original_database",
    "original_ecg": "semi_preprocessed_signals/original_ecg",
    "original_annotations": "semi_preprocessed_signals/original_annotations",
    "annotations_padding": "semi_preprocessed_signals/annotations_padding",
    "annotations_window": "semi_preprocessed_signals/annotations_padding_and_window",
    "standardized_annotations": "semi_preprocessed_signals/standardized_annotations",
    "normalized_ecg": "semi_preprocessed_signals/normalized_ecg",
    "sliced_ecg": "sliced_signals/ecgs",
    "sliced_annotations": "sliced_signals/annotations",
}
