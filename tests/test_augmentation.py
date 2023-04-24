"""Unit tests for ECG augmentation transforms."""

import numpy as np
import pytest

from heartbeat_classifier import config
from heartbeat_classifier.augmentation.transforms import apply_random_transform


def _make_batch(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, config.WINDOW_SIZE, 2)).astype(np.float32)
    y = np.zeros((1, config.WINDOW_SIZE, config.NUM_CLASSES), dtype=np.int32)
    y[:, :, 1] = 1  # all samples labeled as class 1
    return x, y


def test_identity_transform_unchanged():
    x, y = _make_batch()
    x_out, y_out = apply_random_transform(0, x, y)
    np.testing.assert_array_equal(x_out, x)
    np.testing.assert_array_equal(y_out, y)


@pytest.mark.parametrize("tid", range(config.NUM_AUGMENT_TRANSFORMS))
def test_all_transforms_preserve_shape(tid):
    """Every transform must return arrays with the same shape as the input."""
    x, y = _make_batch(seed=tid)
    x_out, y_out = apply_random_transform(tid, x, y)
    assert x_out.shape == x.shape, f"transform {tid}: x shape changed"
    assert y_out.shape == y.shape, f"transform {tid}: y shape changed"


@pytest.mark.parametrize("tid", range(1, config.NUM_AUGMENT_TRANSFORMS))
def test_non_identity_transforms_modify_signal(tid):
    """Transforms 1-6 should change at least some signal values."""
    x, y = _make_batch(seed=tid)
    x_out, _ = apply_random_transform(tid, x, y)
    assert not np.allclose(x_out, x), f"transform {tid} did not modify the signal"


def test_invalid_transform_id_raises():
    x, y = _make_batch()
    with pytest.raises(ValueError, match="transform_id"):
        apply_random_transform(config.NUM_AUGMENT_TRANSFORMS, x, y)
    with pytest.raises(ValueError):
        apply_random_transform(-1, x, y)
