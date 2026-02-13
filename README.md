# Adaptive Benign Overfitting (ABO)

C++ implementation of **Adaptive Benign Overfitting (ABO)** for online time-series forecasting using **over-parameterised Recursive Least Squares (RLS)** with **Random Fourier Features (RFFs)** and **numerically stable QR / Givens updates**.
Paper available at: [Arxiv Preprint 2601.22200](https://arxiv.org/abs/2601.22200)
This repository accompanies ongoing research on **double descent**, **numerical stability**, and **online learning under non-stationarity**, with applications to **financial** and **energy** time-series.

---

## Features

- Adaptive Benign Overfitting (ABO) with Random Fourier Features
- Exponentially weighted RLS with QR / Givens rotations
- Stable operation in full-rank and rank-deficient regimes
- Double-descent and interpolation-limit experiments
- Baselines:
  - QRD-RLS
  - KRLS-RBF

- High-frequency EUR/USD and electricity load forecasting

---

## Build

Compile all experiments with:

```bash
make -j
```

Compiled binaries are placed in:

```text
bin/
```

---

## Large files (Git LFS)

This repository uses **Git Large File Storage (LFS)** for datasets.

Before cloning, install Git LFS:

```bash
git lfs install
```

After cloning the repository, fetch the data files:

```bash
git lfs pull
```

---

## Data

The EUR/USD experiments use a normalized price series stored at:

```text
data/EURUSD/raw_norm_EURUSD.csv
```

This file is tracked using **Git LFS**.

---

## Quick Start

### EUR/USD best-model evaluation (recommended)

The binary `gridsearch_eurusd_test_best` runs a **cross-validated evaluation**
of the best-performing hyperparameters on the EUR/USD dataset for:

- **ABO (proposed)**
- **QRD-RLS**
- **KRLS-RBF**

Each model is evaluated over multiple rolling validation folds, and results
are written to a CSV file.

#### Example (paper configuration)

```bash
./bin/gridsearch_eurusd_test_best \
  --run abo,qrd,krls \
  --first_date 5376 \
  --start_k 0 \
  --end_k 5 \
  --val_length 1344 \
  --warmup 50 \
  --abo_lags 19 \
  --abo_window 20 \
  --abo_sigma 6.50586 \
  --abo_log2D 11 \
  --qrd_lags 48 \
  --qrd_window 128 \
  --krls_lags 25 \
  --krls_window 261 \
  --krls_sigma 4.2 \
  --out_csv results/gridsearch/EURUSD/best_test.csv
```

This command:

- Runs **5 validation folds**
- Uses **fixed best hyperparameters**
- Reports per-fold and averaged performance
- Saves results to a single CSV file

---

## Command-line Arguments

### Common options

| Flag           | Description                                           |
| -------------- | ----------------------------------------------------- |
| `--run`        | Models to run: `abo`, `qrd`, `krls` (comma-separated) |
| `--first_date` | Initial index in the EUR/USD series                   |
| `--start_k`    | First validation fold (inclusive)                     |
| `--end_k`      | Last validation fold (exclusive)                      |
| `--val_length` | Validation length per fold                            |
| `--warmup`     | Warm-up iterations excluded from timing               |
| `--out_csv`    | Output CSV file path                                  |

---

### ABO (Adaptive Benign Overfitting)

| Flag           | Description                              |
| -------------- | ---------------------------------------- |
| `--abo_lags`   | Number of input lags                     |
| `--abo_window` | RLS window size                          |
| `--abo_sigma`  | RFF kernel bandwidth                     |
| `--abo_D`      | RFF feature dimension                    |
| `--abo_log2D`  | Use `D = 2^{log2D}` instead of `--abo_D` |

---

### QRD-RLS

| Flag           | Description          |
| -------------- | -------------------- |
| `--qrd_lags`   | Number of input lags |
| `--qrd_window` | RLS window size      |

---

### KRLS-RBF

| Flag            | Description                     |
| --------------- | ------------------------------- |
| `--krls_lags`   | Number of input lags            |
| `--krls_window` | Kernel dictionary / window size |
| `--krls_sigma`  | RBF kernel bandwidth            |

---

## Other binaries

Additional experiments included in the repository:

```bash
./bin/dd_test
./bin/EURUSD_test
./bin/elect_test
```

These are used for ablation studies, double-descent analysis, and dataset-specific experiments.

---

## Results

All experiment outputs are stored as CSV files under:

```text
results/
```

Each CSV contains:

- Per-fold MSE and residual variance
- Timing statistics (µs and seconds)
- Model configuration metadata

These files are used directly to generate the tables and figures in the paper.

---

## Authors

Luis Ontaneda
Dr. Nick Firoozye
