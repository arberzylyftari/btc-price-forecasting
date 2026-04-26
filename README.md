# BTC Price Forecasting with LSTM

A deep learning project that uses a stacked LSTM neural network to forecast
Bitcoin closing prices one minute into the future, trained on four years of
Coinbase BTC/USD historical trade data.

**Live demo:** https://arberzylyftari.github.io/btc-price-forecasting

## Results

| Metric | Value |
|--------|-------|
| Test MSE | 6.0e-06 |
| Test MAE | $38.44 |
| Mean error | $31.16 |
| Std error | $36.45 |
| Actual mean price | $6,933.35 |
| Predicted mean price | $6,964.52 |

The model achieves a mean absolute error of **$38.44 on a $6,933 asset - 0.55% error**.

## Live Demo

The project website includes a **Historical Replay** feature that runs the
actual trained LSTM model in the browser using TensorFlow.js. It loads the
real held-out 2018 test windows, runs inference client-side, and reveals
both the prediction and the actual next-minute price.

## Project Structure

btc-price-forecasting/
├── Notebooks/
│   ├── 01_data_exploration.ipynb    - Dataset analysis and preprocessing decisions
│   ├── 02_preprocessing.ipynb       - Data cleaning, feature engineering, scaling
│   └── 03_model_training.ipynb      - Model architecture, training and evaluation
├── tfjs_model/
│   ├── model.json                   - TensorFlow.js model config
│   └── group1-shard1of1.bin         - Model weights
├── test_scaled.npy                  - Scaled test split (used by replay demo)
├── actuals.npy                      - Held-out actual prices in USD
├── predictions.npy                  - Model predictions in USD
├── train_scaled.npy                 - Scaled training split
├── val_scaled.npy                   - Scaled validation split
├── scaler.pkl                       - Fitted MinMaxScaler
├── index.html                       - Portfolio website with replay demo
└── README.md

## Dataset

- **Source:** Coinbase BTC/USD historical trade data
- **Period:** December 2014 - January 2019
- **Granularity:** One row per 60-second trading window
- **Total rows:** 2,099,760
- **Columns:** Timestamp, Open, High, Low, Close, Volume_(BTC), Volume_(Currency), Weighted_Price

## Methodology

### Data Exploration (Notebook 1)

A systematic audit of the raw dataset prior to any transformation:

- **109,069 rows (5.19%)** have all values as NaN - inactive trading windows, not corrupted data
- **58,354 timestamps** are absent from the CSV entirely - surfaced by reindexing to a complete 60-second DatetimeIndex
- The longest consecutive gap spans **38.4 hours**, concentrated in the 2014–2016 sparse period
- BTC price autocorrelation at lag 1 day is **0.9971** - justifying 24 hours of lookback

### Preprocessing (Notebook 2)

| Step | Description |
|------|-------------|
| Reindex | Rebuild complete 60-second DatetimeIndex - surfaces 58,354 hidden missing rows |
| Forward-fill | Carry last known price forward into all inactive windows |
| Trim | Remove data before 2017-01-01 - eliminates 38h gap artifacts |
| Features | Retain Close + log₁p(Volume) + cyclic time encodings |
| Split | Chronological split - no shuffling |
| Scale | MinMaxScaler fitted on training data only |

### Feature Engineering

| Feature | Decision | Rationale |
|---------|----------|-----------|
| Close | Retained | Primary prediction target |
| Open, High, Low | Dropped | Correlation ≥ 0.999 with Close - no independent signal |
| Weighted_Price | Dropped | Mathematical duplicate of Close |
| Volume_(BTC) | Retained (log₁p) | Only independent feature (corr = 0.15) |
| Volume_(Currency) | Dropped | Collinear with Volume_(BTC) |
| Timestamp | Encoded | Sin/cos cyclic features for hour and day-of-week |

**Final feature set:** `Close`, `volume_log`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`

### Model Architecture (Notebook 3)

Input → (1440, 6)
LSTM(128, return_sequences=True)
Dropout(0.2)
LSTM(64, return_sequences=True)
Dropout(0.2)
LSTM(32, return_sequences=False)
Dropout(0.1)
Dense(1) → predicted Close price


| Parameter | Value |
|-----------|-------|
| Loss | Mean Squared Error |
| Optimizer | AdamW (lr=0.0005, weight_decay=1e-4) |
| Learning rate schedule | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | patience=7, restore best weights |
| Batch size | 32 |
| Window size | 1,440 steps (24 hours) |
| Horizon | 1 step (1 minute ahead) |

### Data Pipeline

`tf.data.Dataset` streams sliding windows directly from scaled arrays with
`shift=5` - one window every 5 minutes. This avoids materialising millions
of (1440, 6) arrays in RAM while still providing dense, overlapping training sequences.

## Training Splits

| Split | Period | Rows |
|-------|--------|------|
| Train | 2017-01-01 → 2018-05-01 | 698,400 |
| Validation | 2018-05-01 → 2018-06-01 | 44,640 |
| Test | 2018-06-01 → 2018-08-01 | 87,840 |

All three splits fall within the 2017–2018 price regime to ensure consistent
price distribution between training and evaluation.

## Training Progression

| Run | MAE (USD) | Change |
|-----|-----------|--------|
| Run 1 | $608.41 | Baseline - test set covered the 2018 crash (distribution shift) |
| Run 2 | $91.79 | Fixed splits - stable price regime, 2-layer LSTM |
| Run 3 | $32.68 | 3-layer LSTM + AdamW + ReduceLROnPlateau |
| Run 4 (final) | $38.44 | Retrain after session loss - consistent result |

## Requirements

tensorflow>=2.12
numpy
pandas
scikit-learn
matplotlib

## Key Findings

- Forward-filling inactive windows preserves the regular time grid that LSTMs require
- Trimming to 2017-01-01 eliminates long flat-line artifacts from 38h gaps
- Fitting the MinMaxScaler on training data only prevents leakage of future price ranges
- Cyclic sin/cos encoding ensures the model understands temporal periodicity correctly
- All OHLC columns are correlated ≥ 0.999 - retaining only Close avoids redundancy
- Distribution shift between training and test regimes causes larger errors than model capacity

## Author
Arber Zylyftari - Holberton School