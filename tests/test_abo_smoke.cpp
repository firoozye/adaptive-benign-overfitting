// test_abo_smoke.cpp
// Synthetic smoke test for refactored ABO.
// No data files needed. Verifies:
//   1. Constructor + batchInitialize succeed
//   2. update() runs without crash
//   3. downdate() runs without crash when window is full
//   4. Window stays exactly at max_obs_ (never exceeds it)
//   5. pred() returns finite, non-NaN values
//   6. MSE is lower than naive zero-predictor baseline (sanity check on AR(1) series)

#include "abo/dd_test.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

static bool approx_equal(double a, double b, double tol = 1e-8)
{
    return std::fabs(a - b) <= tol * (1.0 + std::fabs(b));
}

int main()
{
    int failures = 0;
    auto FAIL = [&](const char *msg) {
        std::cerr << "FAIL: " << msg << "\n";
        ++failures;
    };
    auto PASS = [&](const char *msg) {
        std::cout << "PASS: " << msg << "\n";
    };

    // -----------------------------------------------------------------------
    // Synthetic AR(1) series: x[t] = 0.8 * x[t-1] + noise
    // -----------------------------------------------------------------------
    const int T      = 500;
    const int lag    = 5;   // raw feature dimension (L)
    const int D      = 16;  // RFF dimension
    const int W      = 30;  // window size (max_obs)
    const double ff  = 1.0;
    const double sig = 1.0; // kernel width

    std::srand(42);
    auto randn = []() {
        // Box-Muller
        double u1 = (std::rand() + 1.0) / (RAND_MAX + 2.0);
        double u2 = (std::rand() + 1.0) / (RAND_MAX + 2.0);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    };

    std::vector<double> series(T);
    series[0] = 0.0;
    for (int t = 1; t < T; ++t)
        series[t] = 0.8 * series[t - 1] + 0.3 * randn();

    // Build lagged feature matrix and targets
    const int N_total = T - lag;
    MatrixXd X_full(N_total, lag);
    std::vector<double> y_full(N_total);
    for (int i = 0; i < N_total; ++i) {
        for (int j = 0; j < lag; ++j) X_full(i, j) = series[i + j];
        y_full[i] = series[i + lag];
    }

    // -----------------------------------------------------------------------
    // Construct ABO with first W rows
    // -----------------------------------------------------------------------
    const bool seed = true;
    GaussianRFF g_rff(lag, D, sig, seed);

    MatrixXd X_init_mat = X_full.topRows(W);
    MatrixXd X_rff_init = g_rff.transform_matrix(X_init_mat);

    // Pack into col-major C array [W x D]
    std::vector<double> X_c(static_cast<size_t>(W) * D);
    for (int j = 0; j < D; ++j)
        for (int i = 0; i < W; ++i)
            X_c[i + j * W] = X_rff_init(i, j);

    std::vector<double> y_init(y_full.begin(), y_full.begin() + W);

    ABO abo(X_c.data(), y_init.data(), W, ff, D, W);

    if (abo.n_obs_ != W)
        FAIL("After construction, n_obs_ != W");
    else
        PASS("Constructor: n_obs_ == W");

    // -----------------------------------------------------------------------
    // Ring buffer
    // -----------------------------------------------------------------------
    std::vector<std::vector<double>> X_raw_ring(W, std::vector<double>(lag));
    std::vector<double> y_ring(W);
    for (int i = 0; i < W; ++i) {
        for (int j = 0; j < lag; ++j) X_raw_ring[i][j] = X_init_mat(i, j);
        y_ring[i] = y_init[i];
    }
    int ring_idx = 0;

    // -----------------------------------------------------------------------
    // In-sample fit check: ABO should beat zero-predictor on training data
    // (tests batchInitialize correctness independently of OOS generalization)
    // -----------------------------------------------------------------------
    {
        double mse_train = 0.0, mse_zero_train = 0.0;
        for (int i = 0; i < W; ++i) {
            MatrixXd z = g_rff.transform(X_init_mat.row(i));
            std::vector<double> xr(D);
            for (int j = 0; j < D; ++j) xr[j] = z(0, j);
            double p = abo.pred(xr.data());
            double e = p - y_init[i];
            mse_train      += e * e;
            mse_zero_train += y_init[i] * y_init[i];
        }
        mse_train      /= W;
        mse_zero_train /= W;
        if (mse_train < mse_zero_train)
            PASS("In-sample MSE: ABO beats zero-predictor on training set");
        else
            FAIL("In-sample MSE: ABO did not beat zero-predictor on training set");
        std::cout << "  Train MSE: " << mse_train << "  Zero-train MSE: " << mse_zero_train << "\n";
    }

    // -----------------------------------------------------------------------
    // Streaming update loop
    // -----------------------------------------------------------------------
    const int n_its = N_total - W;
    int max_obs_exceeded = 0;
    bool pred_non_finite = false;
    double beta_norm_max = 0.0;
    std::vector<double> x_rff(D);

    for (int i = 0; i < n_its; ++i) {
        // Transform current point
        MatrixXd row_mat = X_full.row(W + i);
        MatrixXd z_mat   = g_rff.transform(row_mat);
        for (int j = 0; j < D; ++j) x_rff[j] = z_mat(0, j);

        double y_true = y_full[W + i];

        // Downdate if window full
        if (abo.n_obs_ == W) {
            MatrixXd raw_old(1, lag);
            for (int j = 0; j < lag; ++j) raw_old(0, j) = X_raw_ring[ring_idx][j];
            MatrixXd z_old_mat = g_rff.transform(raw_old);
            std::vector<double> z_old(D);
            for (int j = 0; j < D; ++j) z_old[j] = z_old_mat(0, j);
            abo.downdate(z_old.data(), y_ring[ring_idx]);
        }

        // Update ring buffer
        for (int j = 0; j < lag; ++j) X_raw_ring[ring_idx][j] = X_full(W + i, j);
        y_ring[ring_idx] = y_true;
        ring_idx = (ring_idx + 1) % W;

        // Predict BEFORE updating (forecasting scenario)
        double pred = abo.pred(x_rff.data());
        if (!std::isfinite(pred)) pred_non_finite = true;

        abo.update(x_rff.data(), y_true);

        if (abo.n_obs_ > W) ++max_obs_exceeded;

        // Track beta norm to detect divergence
        double bn = 0.0;
        for (int j = 0; j < D; ++j) bn += abo.beta_[j] * abo.beta_[j];
        beta_norm_max = std::max(beta_norm_max, std::sqrt(bn));
    }

    if (pred_non_finite)
        FAIL("pred() returned non-finite value during streaming");
    else
        PASS("pred() always finite during streaming");

    if (max_obs_exceeded > 0)
        FAIL("n_obs_ exceeded W during streaming");
    else
        PASS("Window size: n_obs_ never exceeded W");

    if (std::isfinite(beta_norm_max) && beta_norm_max < 1e6)
        PASS("Stability: beta norm stayed bounded during streaming");
    else
        FAIL("Stability: beta norm diverged during streaming");

    std::cout << "  max beta norm: " << beta_norm_max << "\n";

    // -----------------------------------------------------------------------
    // Result
    // -----------------------------------------------------------------------
    if (failures == 0)
        std::cout << "\nAll tests passed.\n";
    else
        std::cout << "\n" << failures << " test(s) FAILED.\n";

    return failures == 0 ? 0 : 1;
}
