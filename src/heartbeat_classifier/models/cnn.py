"""Dilated 1-D encoder-decoder CNN for per-sample ECG arrhythmia classification.

Each conv block: Conv1D → LayerNorm → BatchNorm → Dropout.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Conv1DTranspose,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MaxPooling1D,
)

from heartbeat_classifier import config


def _conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    strides: int,
    dilation_rate: int,
) -> tf.Tensor:
    """Single Conv1D → LayerNorm → BatchNorm → Dropout block."""
    x = Conv1D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding="same",
        activation="elu",
        kernel_regularizer=regularizers.l2(config.L2_REGULARIZATION),
    )(x)
    x = LayerNormalization(axis=-1)(x)
    x = BatchNormalization()(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    return x


def build_model(
    input_shape: tuple[int, int] = (config.WINDOW_SIZE, 2),
    kernel_size: int = 7,
    strides: int = 1,
    dilation_rate: int = 2,
) -> tf.keras.Model:
    """Build and compile the arrhythmia classification model."""
    inputs = Input(shape=input_shape, name="input")

    # ── Encoder ───────────────────────────────────────────────────────────────
    x = inputs
    for n_blocks, n_filters in zip(config.ENC_CONV_BLOCKS, config.ENC_FILTERS):
        for _ in range(n_blocks):
            x = _conv_block(x, n_filters, kernel_size, strides, dilation_rate)
        x = MaxPooling1D(pool_size=2)(x)

    # ── Decoder ───────────────────────────────────────────────────────────────
    out = x
    for _ in range(len(config.ENC_FILTERS)):
        out = Conv1DTranspose(
            config.DEC_FILTERS,
            kernel_size=config.DECODER_KERNEL_SIZE,
            strides=2,
            padding="same",
            activation="elu",
            kernel_regularizer=regularizers.l2(config.L2_REGULARIZATION),
        )(out)

    out = Dense(config.DENSE_UNITS[0], activation="elu")(out)
    out = LayerNormalization(axis=-1)(out)
    out = BatchNormalization()(out)
    out = Dropout(config.DROPOUT_RATE)(out)

    out = Dense(config.DENSE_UNITS[1], activation="elu")(out)
    out = Dense(config.NUM_CLASSES, activation="softmax")(out)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
