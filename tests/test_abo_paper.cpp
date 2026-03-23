// test_abo_paper.cpp
// Reproduces the double-descent experiment from the ABO paper (Table 2 / Figure 2).
//
// Paper setup (Section 3.4):
//   Series:  x_t = 2*x_{t-1} / (1 + 0.8*x_{t-1}^2) + eps_t,  eps_t ~ U(-1,1)
//   Lags:    L = 7
//   Window:  N = 20
//   ff:      lambda = 1 (Table 2)
//   n_its:   10000 streaming updates
//   D range: 2^1 .. 2^14 (powers of 2) + D=20 (interpolation threshold I*)
//
// For each D we report:
//   - Test  MSE: pre-update one-step-ahead residual (out-of-sample forecast error)
//   - Train MSE: post-update residual (in-sample, measures overfitting regime)
//
// Expected pattern (Table 2, lambda=1):
//   D < 20:  Train > 0,  Test moderate
//   D = 20:  Train ~ 0,  Test >> 1 (interpolation spike)
//   D > 20:  Train = 0,  Test decreases toward ~0.54 at D=16384

#include "abo/dd_test.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

int main()
{
    // -----------------------------------------------------------------------
    // Generate synthetic time series from paper Eq. (25)
    // -----------------------------------------------------------------------
    const int LAG   = 7;
    const int N     = 20;    // window / init batch size
    const int N_ITS = 10000; // streaming iterations
    const double FF = 1.0;   // Table 2 uses lambda=1

    // Total time steps needed: LAG + N + N_ITS (plus burn-in)
    const int BURNIN  = 1000;
    const int T_TOTAL = BURNIN + LAG + N + N_ITS;

    // Use a fixed seed for reproducibility
    std::srand(12345);
    auto unif = []() -> double {
        return 2.0 * (static_cast<double>(std::rand()) / RAND_MAX) - 1.0; // U(-1,1)
    };

    std::vector<double> series(T_TOTAL);
    series[0] = unif(); // x_0 ~ U(-1,1)
    for (int t = 1; t < T_TOTAL; ++t)
    {
        double x = series[t - 1];
        series[t] = 2.0 * x / (1.0 + 0.8 * x * x) + unif();
    }

    // Discard burn-in; working series starts at index BURNIN
    const int T_USE = T_TOTAL - BURNIN; // = LAG + N + N_ITS
    std::vector<double> s(series.begin() + BURNIN, series.end());

    // Build lag feature matrix and targets
    // Row i: features = [s[i], s[i+1], ..., s[i+LAG-1]], target = s[i+LAG]
    const int NROWS = T_USE - LAG; // = N + N_ITS
    MatrixXd X_full(NROWS, LAG);
    std::vector<double> y_full(NROWS);
    for (int i = 0; i < NROWS; ++i)
    {
        for (int j = 0; j < LAG; ++j) X_full(i, j) = s[i + j];
        y_full[i] = s[i + LAG];
    }

    // Standardize features (mean/std over init batch only, applied to all rows)
    std::vector<double> feat_mean(LAG, 0.0), feat_std(LAG, 1.0);
    for (int j = 0; j < LAG; ++j)
    {
        for (int i = 0; i < N; ++i) feat_mean[j] += X_full(i, j);
        feat_mean[j] /= N;
    }
    for (int j = 0; j < LAG; ++j)
    {
        double var = 0.0;
        for (int i = 0; i < N; ++i)
        {
            double d = X_full(i, j) - feat_mean[j];
            var += d * d;
        }
        feat_std[j] = std::sqrt(var / N);
        if (feat_std[j] < 1e-12) feat_std[j] = 1.0;
    }
    for (int i = 0; i < NROWS; ++i)
        for (int j = 0; j < LAG; ++j)
            X_full(i, j) = (X_full(i, j) - feat_mean[j]) / feat_std[j];

    // Split into init and streaming matrices
    MatrixXd X_init   = X_full.topRows(N);
    MatrixXd X_stream = X_full.bottomRows(N_ITS);

    // -----------------------------------------------------------------------
    // D values to sweep (powers of 2 + interpolation threshold I*=N=20)
    // -----------------------------------------------------------------------
    std::vector<int> D_vals;
    for (int k = 1; k <= 14; ++k) D_vals.push_back(1 << k); // 2,4,8,...,16384
    // Insert D=20 (I*) in sorted order
    {
        auto it = std::lower_bound(D_vals.begin(), D_vals.end(), N);
        if (it == D_vals.end() || *it != N) D_vals.insert(it, N);
    }

    // -----------------------------------------------------------------------
    // Header
    // -----------------------------------------------------------------------
    std::cout << "\nPaper comparison: double-descent (lambda=1, N=20, L=7, n_its=10000)\n";
    std::cout << std::string(72, '-') << "\n";
    std::cout << std::left
              << std::setw(8)  << "log2(D)"
              << std::setw(8)  << "D"
              << std::setw(16) << "Test MSE"
              << std::setw(16) << "Train MSE"
              << "Note\n";
    std::cout << std::string(72, '-') << "\n";

    // -----------------------------------------------------------------------
    // Sweep over D
    // -----------------------------------------------------------------------
    for (int D : D_vals)
    {
        const bool SEED = true;
        GaussianRFF g_rff(LAG, D, 1.0, SEED);

        // Transform init batch
        MatrixXd X_rff_init = g_rff.transform_matrix(X_init);
        std::vector<double> X_c(static_cast<size_t>(N) * D);
        for (int j = 0; j < D; ++j)
            for (int i = 0; i < N; ++i)
                X_c[i + j * N] = X_rff_init(i, j);

        std::vector<double> y_init(y_full.begin(), y_full.begin() + N);
        ABO abo(X_c.data(), y_init.data(), N, FF, D, N);

        // Ring buffer (stores raw standardized features before RFF)
        std::vector<std::vector<double>> X_raw_ring(N, std::vector<double>(LAG));
        std::vector<double> y_ring(N);
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < LAG; ++j) X_raw_ring[i][j] = X_init(i, j);
            y_ring[i] = y_init[i];
        }
        int ring_idx = 0;

        double test_mse  = 0.0; // pre-update (out-of-sample)
        double train_mse = 0.0; // post-update (in-sample)

        std::vector<double> x_rff(D);

        for (int i = 0; i < N_ITS; ++i)
        {
            // Transform current point
            MatrixXd z = g_rff.transform(X_stream.row(i));
            for (int j = 0; j < D; ++j) x_rff[j] = z(0, j);

            double y_true = y_full[N + i];

            // Downdate if window full
            if (abo.n_obs_ == N)
            {
                MatrixXd raw_old(1, LAG);
                for (int j = 0; j < LAG; ++j) raw_old(0, j) = X_raw_ring[ring_idx][j];
                MatrixXd z_old = g_rff.transform(raw_old);
                std::vector<double> z_old_v(D);
                for (int j = 0; j < D; ++j) z_old_v[j] = z_old(0, j);
                abo.downdate(z_old_v.data(), y_ring[ring_idx]);
            }

            // Update ring buffer
            for (int j = 0; j < LAG; ++j) X_raw_ring[ring_idx][j] = X_stream(i, j);
            y_ring[ring_idx] = y_true;
            ring_idx = (ring_idx + 1) % N;

            // Test residual: predict BEFORE update
            double pred_test = abo.pred(x_rff.data());
            double e_test = pred_test - y_true;
            test_mse += e_test * e_test;

            abo.update(x_rff.data(), y_true);

            // Train residual: predict AFTER update
            double pred_train = abo.pred(x_rff.data());
            double e_train = pred_train - y_true;
            train_mse += e_train * e_train;
        }

        test_mse  /= N_ITS;
        train_mse /= N_ITS;

        double log2D = std::log2(static_cast<double>(D));
        std::string note = "";
        if (D == N)       note = "<-- I* (interpolation threshold)";
        else if (D == 16) note = "<-- Table 2: Test~3.17";
        else if (D == 32) note = "<-- Table 2: Test~1.57";
        else if (D == 16384) note = "<-- Table 2: Test~0.54 (best)";

        std::cout << std::left
                  << std::setw(8)  << std::fixed << std::setprecision(2) << log2D
                  << std::setw(8)  << D
                  << std::setw(16) << std::scientific << std::setprecision(4) << test_mse
                  << std::setw(16) << std::scientific << std::setprecision(4) << train_mse
                  << note << "\n";
    }

    std::cout << std::string(72, '-') << "\n";
    std::cout << "\nExpected pattern:\n";
    std::cout << "  D < N=20:  Train > 0,   Test moderate\n";
    std::cout << "  D = 20:    Train ~ 0,   Test >> 1 (spike)\n";
    std::cout << "  D > 20:    Train = 0,   Test decreasing toward ~0.54\n";

    return 0;
}
