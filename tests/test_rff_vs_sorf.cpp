// test_rff_vs_sorf.cpp
//
// Compares GaussianRFF vs SORF on two axes:
//
//   1. MSE VARIANCE — mean and std-dev of test-MSE over N_SEEDS independent
//      random draws (seeds 0..N_SEEDS-1) at each D.
//      Both methods approximate the same RBF kernel, so means should match;
//      SORF's orthogonal draws should reduce variance at intermediate D.
//
//   2. TIMING — microseconds per streaming step (transform + ABO update/downdate)
//      averaged over N_SEEDS runs. Theoretical crossover: SORF O(D log d_pad)
//      vs RFF O(d*D), so speedup ~ d/(3*log2(d_pad)) at large D.
//
// Data: synthetic nonlinear TS  x_t = 2*x_{t-1}/(1+0.8*x_{t-1}^2) + U(-1,1)
//       L=25 lags (higher than paper's L=7 to make SORF speedup visible),
//       window N=30, ff=1, n_its=2000.

#include "abo/dd_test.h"
#include "abo/sorf.h"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using Clock    = std::chrono::high_resolution_clock;
using Us       = std::chrono::microseconds;

// ---------------------------------------------------------------------------
// Generate synthetic data (series seed fixed; RFF seeds vary separately)
// ---------------------------------------------------------------------------
static void make_data(int LAG, int N, int N_ITS,
                      MatrixXd &X_init, MatrixXd &X_stream,
                      std::vector<double> &y_init, std::vector<double> &y_stream)
{
    const int BURNIN  = 500;
    const int T_TOTAL = BURNIN + LAG + N + N_ITS;

    std::srand(99999);  // fixed series seed
    auto unif = []() {
        return 2.0 * (static_cast<double>(std::rand()) / RAND_MAX) - 1.0;
    };

    std::vector<double> s(T_TOTAL);
    s[0] = unif();
    for (int t = 1; t < T_TOTAL; ++t)
    {
        double x = s[t - 1];
        s[t] = 2.0 * x / (1.0 + 0.8 * x * x) + unif();
    }
    std::vector<double> sc(s.begin() + BURNIN, s.end());

    const int NROWS = (int)sc.size() - LAG;
    MatrixXd X_full(NROWS, LAG);
    std::vector<double> y_full(NROWS);
    for (int i = 0; i < NROWS; ++i)
    {
        for (int j = 0; j < LAG; ++j) X_full(i, j) = sc[i + j];
        y_full[i] = sc[i + LAG];
    }

    // Standardise on init batch
    for (int j = 0; j < LAG; ++j)
    {
        double mu = 0.0;
        for (int i = 0; i < N; ++i) mu += X_full(i, j);
        mu /= N;
        double var = 0.0;
        for (int i = 0; i < N; ++i) { double d = X_full(i,j)-mu; var += d*d; }
        double sd = std::sqrt(var / N);
        if (sd < 1e-12) sd = 1.0;
        for (int i = 0; i < NROWS; ++i) X_full(i, j) = (X_full(i, j) - mu) / sd;
    }

    X_init   = X_full.topRows(N);
    X_stream = X_full.middleRows(N, N_ITS);
    y_init  .assign(y_full.begin(),     y_full.begin() + N);
    y_stream.assign(y_full.begin() + N, y_full.begin() + N + N_ITS);
}

// ---------------------------------------------------------------------------
// Run one experiment: {method, rff_seed, D} → {test_mse, us_per_step}
// method: 0 = GaussianRFF, 1 = SORF
// ---------------------------------------------------------------------------
static std::pair<double,double> run_one(
    int method, int rff_seed, int D,
    const MatrixXd &X_init, const MatrixXd &X_stream,
    const std::vector<double> &y_init, const std::vector<double> &y_stream)
{
    const int N     = (int)X_init.rows();
    const int LAG   = (int)X_init.cols();
    const int N_ITS = (int)X_stream.rows();

    GaussianRFF *rff  = nullptr;
    SORF        *sorf = nullptr;
    if (method == 0) rff  = new GaussianRFF(LAG, D, 1.0, rff_seed);
    else             sorf = new SORF       (LAG, D, 1.0, rff_seed);

    auto tfm_row = [&](const MatrixXd &src, int row) -> std::vector<double> {
        MatrixXd z = rff ? rff->transform(src.row(row)) : sorf->transform(src.row(row));
        return std::vector<double>(z.data(), z.data() + D);
    };
    auto tfm_mat = [&](const MatrixXd &src) -> std::vector<double> {
        MatrixXd z = rff ? rff->transform(src) : sorf->transform(src);
        return std::vector<double>(z.data(), z.data() + D);
    };

    // Build init batch (column-major, stride N)
    std::vector<double> X_c(static_cast<size_t>(N) * D);
    {
        MatrixXd Z = rff ? rff->transform_matrix(X_init) : sorf->transform_matrix(X_init);
        for (int j = 0; j < D; ++j)
            for (int i = 0; i < N; ++i)
                X_c[i + j * N] = Z(i, j);
    }

    std::vector<double> y_in = y_init;
    ABO abo(X_c.data(), y_in.data(), N, 1.0, D, N);

    std::vector<std::vector<double>> X_raw_ring(N, std::vector<double>(LAG));
    std::vector<double> y_ring(N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < LAG; ++j) X_raw_ring[i][j] = X_init(i, j);
        y_ring[i] = y_init[i];
    }
    int ring_idx = 0;

    double   test_mse = 0.0;
    long long total_us = 0;

    for (int i = 0; i < N_ITS; ++i)
    {
        auto t0 = Clock::now();

        std::vector<double> z_new = tfm_row(X_stream, i);
        double y_true = y_stream[i];

        if (abo.n_obs_ == N)
        {
            MatrixXd raw(1, LAG);
            for (int j = 0; j < LAG; ++j) raw(0, j) = X_raw_ring[ring_idx][j];
            std::vector<double> z_old = tfm_mat(raw);
            abo.downdate(z_old.data(), y_ring[ring_idx]);
        }

        for (int j = 0; j < LAG; ++j) X_raw_ring[ring_idx][j] = X_stream(i, j);
        y_ring[ring_idx] = y_true;
        ring_idx = (ring_idx + 1) % N;

        double pred = abo.pred(z_new.data());
        test_mse   += (pred - y_true) * (pred - y_true);
        abo.update(z_new.data(), y_true);

        total_us += std::chrono::duration_cast<Us>(Clock::now() - t0).count();
    }

    delete rff;
    delete sorf;

    return {test_mse / N_ITS, static_cast<double>(total_us) / N_ITS};
}

// ---------------------------------------------------------------------------
int main()
{
    const int LAG     = 25;
    const int N       = 30;
    const int N_ITS   = 2000;
    const int N_SEEDS = 5;

    // Theoretical SORF speedup at large D: d / (3 * log2(d_pad))
    int d_pad = 1; while (d_pad < LAG) d_pad <<= 1;
    double theory_speedup = static_cast<double>(LAG) / (3.0 * std::log2(d_pad));

    std::vector<int> D_vals = {4, 16, 64, 256, 1024, 4096, 16384};

    MatrixXd X_init, X_stream;
    std::vector<double> y_init, y_stream;
    make_data(LAG, N, N_ITS, X_init, X_stream, y_init, y_stream);

    auto mean = [](const std::vector<double> &v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto sd = [&](const std::vector<double> &v) {
        double m = mean(v), s = 0.0;
        for (double x : v) s += (x - m) * (x - m);
        return std::sqrt(s / (v.size() - 1));
    };

    const int W = 11;
    std::cout << "\nRFF vs SORF — MSE variance and timing (L=" << LAG
              << ", N=" << N << ", n_its=" << N_ITS
              << ", seeds=0.." << N_SEEDS-1 << ")\n";
    std::cout << std::string(105, '-') << "\n";
    std::cout << std::left
              << std::setw(7)  << "D"
              << std::setw(W)  << "RFF mean"
              << std::setw(W)  << "RFF std"
              << std::setw(W)  << "RFF us/step"
              << std::setw(W)  << "SORF mean"
              << std::setw(W)  << "SORF std"
              << std::setw(W)  << "SORF us/step"
              << std::setw(8)  << "speedup"
              << "note\n";
    std::cout << std::string(105, '-') << "\n";

    for (int D : D_vals)
    {
        std::vector<double> rm, rs, sm, ss;

        std::cout << "  D=" << D << " running..." << std::flush;
        for (int seed = 0; seed < N_SEEDS; ++seed)
        {
            auto [r_mse, r_us] = run_one(0, seed, D, X_init, X_stream, y_init, y_stream);
            auto [s_mse, s_us] = run_one(1, seed, D, X_init, X_stream, y_init, y_stream);
            rm.push_back(r_mse); rs.push_back(r_us);
            sm.push_back(s_mse); ss.push_back(s_us);
        }
        std::cout << "\r";  // overwrite progress line

        double speedup = mean(rs) / mean(ss);
        std::string note = (D == N) ? "<-- I*" : "";
        if (speedup > 1.3) note += "  SORF faster";

        std::cout << std::left  << std::setw(7) << D
                  << std::fixed << std::setprecision(5)
                  << std::setw(W) << mean(rm)
                  << std::setw(W) << sd(rm)
                  << std::setprecision(2)
                  << std::setw(W) << mean(rs)
                  << std::setprecision(5)
                  << std::setw(W) << mean(sm)
                  << std::setw(W) << sd(sm)
                  << std::setprecision(2)
                  << std::setw(W) << mean(ss)
                  << std::setprecision(2)
                  << std::setw(8) << speedup
                  << note << "\n";
    }

    std::cout << std::string(105, '-') << "\n";
    std::cout << "\nTheoretical max SORF speedup (L=" << LAG
              << ", d_pad=" << d_pad << "): "
              << std::fixed << std::setprecision(1) << theory_speedup << "x\n"
              << "  (speedup = d / (3 * log2(d_pad)) — measured at large D)\n"
              << "  us/step includes transform + ABO update + downdate\n";

    return 0;
}
