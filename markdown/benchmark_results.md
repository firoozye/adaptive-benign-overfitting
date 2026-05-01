# ABO Benchmark Results

Machine: 4 × 3900 MHz, L1=32KiB, L2=1MiB, L3=8.4MiB
Date: 2026-03-23
Benchmark: `BM_ABO_RFF` — 1000 streaming steps of downdate + pred + update per iteration,
sliding window N=20, synthetic AR series (paper Eq. 25), 7 lags, λ=1.

## New refactored code (`-O3 -march=native`)

| D | Time (ms) | CPU (ms) | Iterations |
|---|---|---|---|
| 2 | 4.94 | 4.95 | 139 |
| 16 | 6.53 | 6.55 | 106 |
| 128 | 16.46 | 16.49 | 42 |
| 1,024 | 160.1 | 157.6 | 4 |
| 16,384 | 4,913 | 4,807 | 1 |

## New refactored code (no optimization, `DEBUG=1`)

| D | Time (ms) | CPU (ms) | Iterations |
|---|---|---|---|
| 2 | 8.00 | 8.00 | 88 |
| 16 | 10.30 | 10.28 | 68 |
| 128 | 22.25 | 22.24 | 32 |
| 1,024 | 467.7 | 356.2 | 2 |
| 16,384 | 6,344 | 5,826 | 1 |

## Speedup from `-O3 -march=native`

| D | Speedup |
|---|---|
| 2 | 1.6× |
| 16 | 1.6× |
| 128 | 1.3× |
| 1,024 | 2.9× |
| 16,384 | 1.3× |

The 2.9× at D=1024 reflects Givens rotation loops vectorizing well with AVX2/native SIMD.

## Notes on old-vs-new comparison

The old benchmark (`e577e46`, pre-refactor) cannot be run directly:
- References `timing_test_non_linear_ts.cc` (file is `.cpp`) — makefile bug in old code
- Reads LFS CSV files (`data/non_linear_ts_lags.csv`) which require `git lfs pull`

Key differences between old and new implementations:
- **Old**: `downdate()` took no args, read oldest row from stored `X_` member; compiled without `-O3`
- **New**: `X_` removed; caller passes `z_old` explicitly; pre-allocated scratch buffers eliminate all hot-path `new[]`/`delete[]`; compiled with `-O3 -march=native`

To run the old benchmark: `git lfs pull` then `git checkout e577e46 && make benchmarks && ./timing_test`

## Build instructions

```bash
# Dependencies
sudo apt-get install -y libopenblas-dev liblapacke-dev

# Submodules (Eigen)
git submodule update --init libs/eigen

# Google Benchmark (build once)
cd /tmp && git clone https://github.com/google/benchmark.git gbenchmark
cd gbenchmark && cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON \
    -DBENCHMARK_ENABLE_GTEST_TESTS=OFF -B build && cmake --build build -j$(nproc)
mkdir -p /path/to/repo/libs/benchmark/build/src
cp /tmp/gbenchmark/build/src/libbenchmark.a /path/to/repo/libs/benchmark/build/src/
cp -r /tmp/gbenchmark/include/benchmark /path/to/repo/libs/benchmark/include/

# Build and run
make benchmarks
./timing_test
```
