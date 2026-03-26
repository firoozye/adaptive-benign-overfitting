#include "baselines/SWKRLS/swkrls.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

SWKRLS::SWKRLS(const double *X_init, const double *y_init,
               int n_obs, int n_features,
               double lambda, double sigma,
               int capacity, double ff, double ald_thresh)
    : dim_(n_features), capacity_(capacity),
      lambda_(lambda), sigma_(sigma), ff_(ff), ald_thresh_(ald_thresh)
{
    // If n_obs > capacity, use only the most recent `capacity` samples.
    int init_size = std::min(n_obs, capacity_);
    int offset    = n_obs - init_size;   // skip oldest samples if trimming

    // Build initial dictionary from the selected samples.
    // X_init is column-major: X_init[sample + feature * n_obs]
    dict_.reserve(init_size);
    for (int i = 0; i < init_size; ++i)
    {
        Eigen::VectorXd xi(dim_);
        for (int j = 0; j < dim_; ++j)
            xi(j) = X_init[(offset + i) + j * n_obs];
        dict_.push_back(xi);
    }

    y_dict_.resize(init_size);
    for (int i = 0; i < init_size; ++i)
        y_dict_(i) = y_init[offset + i];

    // Build kernel matrix K + lambda*I and invert.
    Eigen::MatrixXd K(init_size, init_size);
    for (int i = 0; i < init_size; ++i)
        for (int j = 0; j < init_size; ++j)
        {
            K(i, j) = kernel(dict_[i].data(), dict_[j].data());
            if (i == j) K(i, j) += lambda_;
        }

    P_     = K.inverse();
    alpha_ = P_ * y_dict_;
}

// ---------------------------------------------------------------------------
// Kernel (RBF / Gaussian)
// ---------------------------------------------------------------------------

double SWKRLS::kernel(const double *x1, const double *x2) const
{
    double sq = 0.0;
    for (int i = 0; i < dim_; ++i)
    {
        double d = x1[i] - x2[i];
        sq += d * d;
    }
    return std::exp(-sq / (2.0 * sigma_ * sigma_));
}

// ---------------------------------------------------------------------------
// downdate: Sherman-Morrison removal of the first (oldest) dictionary entry
//
// If P = (K + lambda*I)^{-1} for the m-element dictionary, then after
// removing the first entry the new (m-1)x(m-1) inverse is the Schur
// complement:
//   P_new = P[1:,1:] - (1/P[0,0]) * P[1:,0] * P[0,1:]
// ---------------------------------------------------------------------------

void SWKRLS::downdate()
{
    int m = static_cast<int>(dict_.size());

    double            p00   = P_(0, 0);
    Eigen::VectorXd   p_col = P_.col(0).tail(m - 1);     // P[1:, 0]
    Eigen::RowVectorXd p_row = P_.row(0).tail(m - 1);    // P[0, 1:]

    P_ = P_.bottomRightCorner(m - 1, m - 1)
         - (1.0 / p00) * p_col * p_row;

    dict_.erase(dict_.begin());
    y_dict_ = y_dict_.tail(m - 1).eval();
}

// ---------------------------------------------------------------------------
// update: predict then learn from a new observation
// ---------------------------------------------------------------------------

void SWKRLS::update(const double *new_x, double new_y,
                    double &pred_out, double &err_out)
{
    int m = static_cast<int>(dict_.size());

    // Step 1 — kernel vector between new_x and current dictionary.
    Eigen::VectorXd k_new(m);
    for (int i = 0; i < m; ++i)
        k_new(i) = kernel(new_x, dict_[i].data());

    // Prediction with the current model (prior to any update).
    pred_out = k_new.dot(alpha_);
    err_out  = new_y - pred_out;

    // Step 2 — ALD novelty test.
    double kxx     = kernel(new_x, new_x);        // = 1 for unit-norm RBF
    double ald_delta = kxx - k_new.dot(P_ * k_new);
    bool   novel   = (ald_delta > ald_thresh_);

    if (!novel)
    {
        // Not novel: apply forgetting then recompute alpha.
        // The new point lies in the RKHS span of the dictionary, so no
        // structural change is needed.
        if (ff_ != 1.0)
        {
            P_      *= (1.0 / ff_);
            y_dict_ *= std::sqrt(ff_);
        }
        alpha_ = P_ * y_dict_;
        return;
    }

    // Step 3 — Novel point: if dictionary is full, evict the oldest entry.
    if (m == capacity_)
    {
        downdate();
        --m;
        // Recompute k_new for the smaller dictionary.
        k_new.resize(m);
        for (int i = 0; i < m; ++i)
            k_new(i) = kernel(new_x, dict_[i].data());
    }

    // Step 4 — Apply forgetting factor before incorporating the new sample.
    if (ff_ != 1.0)
    {
        P_      *= (1.0 / ff_);
        y_dict_ *= std::sqrt(ff_);
    }

    // Step 5 — Rank-1 update of P via the matrix inversion lemma.
    // Adding a new row/col (k_new, kxx+lambda) to the kernel matrix:
    //   delta_upd = kxx + lambda - k_new^T * P * k_new
    Eigen::VectorXd P_b     = P_ * k_new;
    double          delta_upd = kxx + lambda_ - k_new.dot(P_b);
    if (delta_upd < 1e-10) delta_upd = lambda_;   // numerical safety floor

    Eigen::MatrixXd P_new(m + 1, m + 1);
    P_new.topLeftCorner(m, m)      = P_ + P_b * P_b.transpose() / delta_upd;
    P_new.topRightCorner(m, 1)     = -P_b / delta_upd;
    P_new.bottomLeftCorner(1, m)   = -P_b.transpose() / delta_upd;
    P_new(m, m)                    = 1.0 / delta_upd;
    P_ = P_new;

    // Step 6 — Append new sample to dictionary and target vector.
    Eigen::VectorXd x_new_vec(dim_);
    for (int j = 0; j < dim_; ++j) x_new_vec(j) = new_x[j];
    dict_.push_back(x_new_vec);

    Eigen::VectorXd y_new(m + 1);
    y_new.head(m) = y_dict_;
    y_new(m)      = new_y;
    y_dict_       = y_new;

    // Step 7 — Recompute dual weights.
    alpha_ = P_ * y_dict_;
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

double SWKRLS::predict(const double *x) const
{
    double f = 0.0;
    for (int i = 0, m = static_cast<int>(dict_.size()); i < m; ++i)
        f += alpha_(i) * kernel(x, dict_[i].data());
    return f;
}
