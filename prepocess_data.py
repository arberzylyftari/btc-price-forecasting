#!/usr/bin/env python3
"""Preprocesses raw Coinbase BTC/USD 1-min data for RNN forecasting."""

import numpy as np
import pandas as pd


def load_and_clean(path):
    """Load CSV, sort by time, forward-fill missing rows."""
    df = pd.read_csv(path)
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s')

    hour = df['datetime'].dt.hour
    dow  = df['datetime'].dt.dayofweek

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_sin']  = np.sin(2 * np.pi * dow / 7)
    df['day_cos']  = np.cos(2 * np.pi * dow / 7)

    features = ['Open', 'High', 'Low', 'Close',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    df = df[features].ffill().dropna().reset_index(drop=True)
    return df


def normalize(train, val, test):
    """MinMax scale fitted on train only."""
    data_min = train.min(axis=0)
    data_max = train.max(axis=0)

    def scale(arr):
        return np.clip((arr - data_min) / (data_max - data_min), 0.0, 1.0)

    return scale(train), scale(val), scale(test), data_min, data_max


def main():
    path = ('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

    print("Loading and cleaning data...")
    df = load_and_clean(path)
    print(f"Clean rows: {len(df):,}")

    n  = len(df)
    t1 = int(n * 0.80)
    t2 = int(n * 0.90)

    train = df.iloc[:t1].values.astype(np.float32)
    val   = df.iloc[t1:t2].values.astype(np.float32)
    test  = df.iloc[t2:].values.astype(np.float32)

    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    print("Normalizing...")
    train_s, val_s, test_s, d_min, d_max = normalize(train, val, test)

    print("Saving...")
    np.save('train_scaled.npy', train_s)
    np.save('val_scaled.npy',   val_s)
    np.save('test_scaled.npy',  test_s)
    np.savez('scaler.npz', data_min=d_min, data_max=d_max)

    print("Done. Files saved: train_scaled.npy, val_scaled.npy, "
          "test_scaled.npy, scaler.npz")


if __name__ == '__main__':
    main()