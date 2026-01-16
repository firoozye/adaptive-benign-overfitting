# Adaptive Benign Overfitting (ABO)

C++ implementation of **Adaptive Benign Overfitting (ABO)** for online time-series forecasting using **over-parameterised Recursive Least Squares (RLS)** with **Random Fourier Features (RFFs)** and **QR / Givens updates**.

This repository accompanies ongoing research on **double descent**, **numerical stability**, and **online learning under non-stationarity**.

---

## Features

- Online RLS with Random Fourier Features
- Numerically stable QR-based updates (Givens rotations)
- Double descent and stability experiments
- Baselines: QRD-RLS and KRLS-RBF
- Applications to financial and energy time-series

---

## Build

```bash
make -j
```

Binaries are produced in:

```text
bin/
```

---

## Large files (Git LFS)

This repository uses **Git Large File Storage (LFS)** for datasets.

Before cloning, make sure Git LFS is installed:

```bash
git lfs install
```

After cloning, fetch the data files with:

```bash
git lfs pull
```

---

## Quick Start

Run the best-performing EUR/USD configuration:

```bash
./bin/gridsearch_eurusd_test_best
```

Other examples:

```bash
./bin/dd_test
./bin/EURUSD_test
./bin/elect_test
```

---

## Results

All experiment outputs (CSV + plots) are stored in:

```text
results/
```

---

## Citation

```bibtex
@article{abo2025,
  title={Adaptive Benign Overfitting for Online Learning in Non-Stationary Time-Series},
  author={Ontaneda, Luis and Firoozye, Nick},
  journal={IEEE Transactions on Signal Processing},
  year={2025}
}
```

---

## Authors

Luis Ontaneda  
Dr Nick Firoozye
