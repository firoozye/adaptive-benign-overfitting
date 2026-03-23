// Minimal debug test — checks beta_ divergence step by step
#include "abo/dd_test.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>

using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

static double beta_norm(ABO &abo) {
    double s = 0;
    for (int i = 0; i < abo.dim_; i++) s += abo.beta_[i] * abo.beta_[i];
    return std::sqrt(s);
}

int main()
{
    const int lag = 3;
    const int D   = 2;  // dim < lag, so old regime (n_obs >= dim always)
    const int W   = 5;  // window
    const double ff  = 1.0;
    const double sig = 1.0;
    const bool seed  = true;

    // Simple deterministic AR(1): x[t] = 0.9*x[t-1]
    const int T = 30;
    std::vector<double> series(T);
    series[0] = 1.0;
    for (int t = 1; t < T; t++) series[t] = 0.9 * series[t-1];

    // Lagged features
    const int N = T - lag;
    MatrixXd X_full(N, lag);
    std::vector<double> y_full(N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < lag; j++) X_full(i,j) = series[i+j];
        y_full[i] = series[i+lag];
    }

    GaussianRFF g_rff(lag, D, sig, seed);
    MatrixXd X_init = g_rff.transform_matrix(X_full.topRows(W));

    std::vector<double> X_c(W * D);
    for (int j = 0; j < D; j++)
        for (int i = 0; i < W; i++)
            X_c[i + j*W] = X_init(i,j);
    std::vector<double> y_init(y_full.begin(), y_full.begin()+W);

    ABO abo(X_c.data(), y_init.data(), W, ff, D, W);
    std::cout << "Init: n_obs=" << abo.n_obs_ << " |beta|=" << beta_norm(abo) << "\n";

    // Ring buffer
    std::vector<std::vector<double>> Xring(W, std::vector<double>(lag));
    std::vector<double> yring(W);
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < lag; j++) Xring[i][j] = X_full(i,j);
        yring[i] = y_init[i];
    }
    int ridx = 0;

    std::vector<double> xrff(D);
    for (int it = 0; it < 20; it++) {
        MatrixXd z = g_rff.transform(X_full.row(W+it));
        for (int j = 0; j < D; j++) xrff[j] = z(0,j);

        double beta_before = beta_norm(abo);
        if (abo.n_obs_ == W) {
            MatrixXd raw(1, lag);
            for (int j = 0; j < lag; j++) raw(0,j) = Xring[ridx][j];
            MatrixXd z_old = g_rff.transform(raw);
            std::vector<double> zo(D);
            for (int j = 0; j < D; j++) zo[j] = z_old(0,j);
            abo.downdate(zo.data(), yring[ridx]);
            std::cout << "  after downdate: |beta|=" << beta_norm(abo) << " n_obs=" << abo.n_obs_ << "\n";
        }

        for (int j = 0; j < lag; j++) Xring[ridx][j] = X_full(W+it, j);
        yring[ridx] = y_full[W+it];
        ridx = (ridx+1) % W;

        abo.update(xrff.data(), y_full[W+it]);

        double pred = abo.pred(xrff.data());
        std::cout << "it=" << it << " n_obs=" << abo.n_obs_
                  << " |beta|=" << std::scientific << std::setprecision(3) << beta_norm(abo)
                  << " pred=" << pred << " true=" << y_full[W+it] << "\n";

        if (!std::isfinite(beta_norm(abo))) {
            std::cout << "DIVERGED at it=" << it << "\n";
            break;
        }
    }
    return 0;
}
