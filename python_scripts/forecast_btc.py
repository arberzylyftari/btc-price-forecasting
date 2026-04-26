#!/usr/bin/env python3
"""Training and validating an LSTM model for BTC price forecasting."""

import numpy as np
import tensorflow as tf


WINDOW = 1440
BATCH  = 512
TARGET = 3


def make_dataset(data, shuffle=False):
    """Build a tf.data.Dataset of sliding windows."""
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(WINDOW + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(WINDOW + 1))
    ds = ds.map(lambda w: (w[:WINDOW], w[WINDOW, TARGET]))
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(input_shape):
    """Build LSTM model."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True,
                             input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def main():
    print("Loading data...")
    train_scaled = np.load('train_scaled.npy')
    val_scaled   = np.load('val_scaled.npy')
    test_scaled  = np.load('test_scaled.npy')

    print(f"Train: {train_scaled.shape}")
    print(f"Val  : {val_scaled.shape}")
    print(f"Test : {test_scaled.shape}")

    print("Building datasets...")
    train_ds = make_dataset(train_scaled, shuffle=True)
    val_ds   = make_dataset(val_scaled)
    test_ds  = make_dataset(test_scaled)

    print("Building model...")
    model = build_model((WINDOW, train_scaled.shape[1]))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            save_best_only=True
        )
    ]

    print("Training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks
    )

    print("Evaluating...")
    val_mse  = model.evaluate(val_ds)
    test_mse = model.evaluate(test_ds)
    print(f"\nValidation MSE : {val_mse:.6f}")
    print(f"Test MSE       : {test_mse:.6f}")

    model.save('final_model.keras')
    print("Model saved.")


if __name__ == '__main__':
    main()