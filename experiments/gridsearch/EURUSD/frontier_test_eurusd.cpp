// eurusd_frontier_min.cpp
// Minimal frontier runner: hard-coded EURUSD, compares ABO vs KRLS-RBF,
// sweeps W and D, writes a single CSV.
//
// Usage:
//   ./eurusd_frontier_min
//
// Output:
//   eurusd_frontier.csv

#include "abo/dd_test.h"
#include "baselines/KRLS_RBF/krls_rbf.h"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

static inline double bytes_to_mb(double bytes) { return bytes / (1024.0 * 1024.0); }
static inline double mem_mb_proxy_krls(int W) { return bytes_to_mb(8.0 * 1.0 * W * 1.0 * W); }
static inline double mem_mb_proxy_abo(int W, int D) { return bytes_to_mb(8.0 * 1.0 * W * 1.0 * D); }

static void lag_matrix(
    const std::vector<double> &x,
    int lag,
    std::vector<std::vector<double>> &X_lag,
    std::vector<double> &y)
{
    const int T = static_cast<int>(x.size());
    const int N = T - lag;
    X_lag.assign(N, std::vector<double>(lag));
    y.assign(N, 0.0);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < lag; ++j)
            X_lag[i][j] = x[i + j];
        y[i] = x[i + lag];
    }
}

static void dataset_creation(
    const std::vector<std::vector<double>> &data_set,
    const std::vector<double> &target_data,
    int start_row,
    int W,
    int L,
    MatrixXd &initial_matrix,
    MatrixXd &update_matrix,
    std::vector<double> &y_init,
    std::vector<double> &y_update)
{
    const int T = static_cast<int>(target_data.size());
    const int num_elements = T - start_row - W;
    if (num_elements <= 0)
        throw std::runtime_error("Not enough data for fold/start_row/W");

    y_init.resize(W);
    for (int i = 0; i < W; ++i)
        y_init[i] = target_data[start_row + i];

    y_update.resize(num_elements);
    for (int i = 0; i < num_elements; ++i)
        y_update[i] = target_data[start_row + W + i];

    const int len_data_set = static_cast<int>(data_set.size());
    if (start_row + W >= len_data_set)
        throw std::runtime_error("start_row+W out of bounds in data_set");

    MatrixXd close_lag_mat(len_data_set - start_row, L);
    for (int i = 0; i < len_data_set - start_row; ++i)
        for (int j = 0; j < L; ++j)
            close_lag_mat(i, j) = data_set[start_row + i][j];

    initial_matrix = close_lag_mat.block(0, 0, W, L);
    update_matrix = close_lag_mat.block(W, 0, close_lag_mat.rows() - W, L);
}

static inline double variance_of_sqerr(const std::vector<double> &sqerr, double mean)
{
    if (sqerr.size() < 2)
        return std::numeric_limits<double>::quiet_NaN();
    double v = 0.0;
    for (double e : sqerr)
    {
        double t = e - mean;
        v += t * t;
    }
    return v / (static_cast<double>(sqerr.size()) - 1.0);
}

struct FoldResult
{
    double mse = std::numeric_limits<double>::quiet_NaN();
    double var = std::numeric_limits<double>::quiet_NaN();
    double us_update = std::numeric_limits<double>::quiet_NaN();
};

static FoldResult run_fold_ABO(MatrixXd &initial_matrix,
                               MatrixXd &update_matrix,
                               std::vector<double> &y_init,
                               std::vector<double> &y_update,
                               double sigma,
                               int D,
                               int W,
                               int L,
                               int n_its)
{
    FoldResult out;

    const double ff = 1.0;
    const bool seed = true;

    auto g_rff = std::make_unique<GaussianRFF>(L, D, sigma, seed);
    MatrixXd X_old = g_rff->transform_matrix(initial_matrix);

    // keep X_init alive
    std::vector<double> X_init(static_cast<size_t>(W) * static_cast<size_t>(D));
    for (int j = 0; j < D; ++j)
        for (int i = 0; i < W; ++i)
            X_init[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(W)] = X_old(i, j);

    auto abo = std::make_unique<ABO>(X_init.data(),
                                     const_cast<double *>(y_init.data()), // matches your ctor usage pattern
                                     W, ff, D, W);

    std::vector<double> x_rff(static_cast<size_t>(D), 0.0);

    double mse_sum = 0.0;
    std::vector<double> sqerr;
    sqerr.reserve(static_cast<size_t>(n_its));

    double us_sum = 0.0;

    for (int i = 0; i < n_its; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        MatrixXd row_feat = g_rff->transform(update_matrix.row(i));
        for (int j = 0; j < D; ++j)
            x_rff[j] = row_feat(0, j);

        double pred = abo->pred(x_rff.data());
        abo->update(x_rff.data(), y_update[i]);

        auto t1 = std::chrono::high_resolution_clock::now();
        us_sum += std::chrono::duration<double, std::micro>(t1 - t0).count();

        double r2 = (pred - y_update[i]) * (pred - y_update[i]);
        mse_sum += r2;
        sqerr.push_back(r2);
    }

    out.mse = mse_sum / n_its;
    out.var = variance_of_sqerr(sqerr, out.mse);
    out.us_update = us_sum / n_its;
    return out;
}

static FoldResult run_fold_KRLS(MatrixXd &initial_matrix,
                                MatrixXd &update_matrix,
                                std::vector<double> &y_init,
                                std::vector<double> &y_update,
                                double sigma,
                                int W,
                                int L,
                                int n_its)
{
    FoldResult out;

    const double regularizer = 1e-2;
    const double temp_sigma = 1.0 / sigma; // matches your earlier code

    // Build X_no_rff in col-major [W x L]
    std::vector<double> X_no_rff(static_cast<size_t>(W) * static_cast<size_t>(L));
    for (int j = 0; j < L; ++j)
        for (int i = 0; i < W; ++i)
            X_no_rff[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(W)] = initial_matrix(i, j);

    auto krls = std::make_unique<KRLS_RBF>(X_no_rff.data(),
                                           const_cast<double *>(y_init.data()),
                                           W, L, regularizer, temp_sigma, W);

    std::vector<double> x(static_cast<size_t>(L), 0.0);

    double mse_sum = 0.0;
    std::vector<double> sqerr;
    sqerr.reserve(static_cast<size_t>(n_its));

    double us_sum = 0.0;

    for (int i = 0; i < n_its; ++i)
    {
        for (int j = 0; j < L; ++j)
            x[j] = update_matrix(i, j);

        auto t0 = std::chrono::high_resolution_clock::now();

        double pred = 0.0, eps_post = 0.0;
        krls->update(x.data(), y_update[i], pred, eps_post);

        auto t1 = std::chrono::high_resolution_clock::now();
        us_sum += std::chrono::duration<double, std::micro>(t1 - t0).count();

        double r2 = eps_post * eps_post;
        mse_sum += r2;
        sqerr.push_back(r2);
    }

    out.mse = mse_sum / n_its;
    out.var = variance_of_sqerr(sqerr, out.mse);
    out.us_update = us_sum / n_its;
    return out;
}

static void append_csv_row(const std::string &path,
                           const std::string &model,
                           int L, int W, int D,
                           double sigma, int K,
                           double mse, double var,
                           double us_update,
                           double mem_mb)
{
    static bool wrote_header = false;
    std::ofstream f(path, std::ios::app);
    if (!f)
        throw std::runtime_error("Cannot open output CSV");

    if (!wrote_header)
    {
        f << "model,L,W,D,sigma,K,mse,var,us_update,mem_mb\n";
        wrote_header = true;
    }

    f << model << "," << L << "," << W << "," << D << ","
      << sigma << "," << K << ","
      << mse << "," << var << ","
      << us_update << "," << mem_mb << "\n";
}

int main()
{
    // ----------------------------
    // HARD-CODED EXPERIMENT CONFIG
    // ----------------------------
    const std::string csv_path = "eurusd_frontier.csv";

    const int L = 20;         // lags
    const double sigma = 3.0; // kernel width parameter (your convention)
    const int K = 3;          // folds

    const int val_length = 960; // evaluation points per fold

    // Sweep windows W (KRLS uses W; ABO uses W too)
    const std::vector<int> Ws = {256, 512, 1024, 2048};

    // For each W, sweep D multipliers for ABO: D = mult*W
    const std::vector<double> D_mult = {0.25, 0.5, 1.0, 2.0};

    // ----------------------------
    // Load + build lagged dataset
    // ----------------------------
    std::vector<std::vector<std::string>> raw = read_csv_func("data/EURUSD/raw_norm_EURUSD.csv");
    std::vector<double> x;
    x.reserve(raw.size());

    for (size_t i = 1; i < raw.size(); ++i)
        x.push_back(std::stod(raw[i][0]));

    std::vector<std::vector<double>> X_lag;
    std::vector<double> y;
    lag_matrix(x, L, X_lag, y);

    // ----------------------------
    // Run sweeps
    // ----------------------------
    for (int W : Ws)
    {
        // KRLS at this W
        {
            double mse_sum = 0.0, var_sum = 0.0, us_sum = 0.0;

            for (int k = 0; k < K; ++k)
            {
                const int start_row = val_length * k;

                MatrixXd init, upd;
                std::vector<double> y_init, y_upd;
                dataset_creation(X_lag, y, start_row, W, L, init, upd, y_init, y_upd);

                const int n_its = std::min(val_length, static_cast<int>(upd.rows()));
                FoldResult r = run_fold_KRLS(init, upd, y_init, y_upd, sigma, W, L, n_its);

                mse_sum += r.mse;
                var_sum += r.var;
                us_sum += r.us_update;
            }

            append_csv_row(csv_path, "KRLS-RBF", L, W, 0, sigma, K,
                           mse_sum / K, var_sum / K, us_sum / K,
                           mem_mb_proxy_krls(W));
        }

        // ABO for this W and multiple D
        for (double mult : D_mult)
        {
            int D = std::max(1, static_cast<int>(std::llround(mult * W)));

            double mse_sum = 0.0, var_sum = 0.0, us_sum = 0.0;

            for (int k = 0; k < K; ++k)
            {
                const int start_row = val_length * k;

                MatrixXd init, upd;
                std::vector<double> y_init, y_upd;
                dataset_creation(X_lag, y, start_row, W, L, init, upd, y_init, y_upd);

                const int n_its = std::min(val_length, static_cast<int>(upd.rows()));
                FoldResult r = run_fold_ABO(init, upd, y_init, y_upd, sigma, D, W, L, n_its);

                mse_sum += r.mse;
                var_sum += r.var;
                us_sum += r.us_update;
            }

            append_csv_row(csv_path, "ABO", L, W, D, sigma, K,
                           mse_sum / K, var_sum / K, us_sum / K,
                           mem_mb_proxy_abo(W, D));
        }

        std::cerr << "Done W=" << W << "\n";
    }

    std::cerr << "Wrote: " << csv_path << "\n";
    return 0;
}
