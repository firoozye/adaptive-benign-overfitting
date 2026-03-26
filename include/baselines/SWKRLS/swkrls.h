#pragma once

#include <vector>
#include <Eigen/Dense>
#include <cmath>

// Sliding-Window Kernel RLS with ALD novelty detection and forgetting factor.
//
// Combines:
//   Van Vaerenbergh et al. (2006) "Sliding-Window Kernel RLS"
//   Guo et al. (2022)            "Improved Sliding Window Kernel RLS" (ALD + ff)
//
// Dictionary grows up to `capacity` samples. New samples that are approximately
// linearly dependent on the dictionary (ALD criterion delta <= ald_thresh) are
// discarded from the dictionary but still contribute to the alpha update via
// forgetting. When a novel sample arrives and the dictionary is full, the oldest
// entry is removed via a Sherman-Morrison rank-1 downdate of P.
//
// Kernel:  k(x1,x2) = exp(-||x1-x2||^2 / (2*sigma^2))
// State:   P = (K_dict + lambda*I)^{-1},  alpha = P * y_dict

class SWKRLS
{
public:
    // Construct and warm-start from a batch of n_obs observations.
    // X_init: column-major array, shape [n_obs x n_features]
    //         X_init[sample + feature * n_obs]
    // y_init: target vector of length n_obs
    // n_obs <= capacity is expected (all initial samples enter the dictionary).
    SWKRLS(const double *X_init, const double *y_init,
           int n_obs, int n_features,
           double lambda, double sigma,
           int capacity,
           double ff = 1.0,
           double ald_thresh = 1e-4);

    // Online update: predict then learn from (new_x, new_y).
    // pred_out: prediction made BEFORE incorporating new_y (prior error metric).
    // err_out:  new_y - pred_out.
    void update(const double *new_x, double new_y,
                double &pred_out, double &err_out);

    // Predict from current model (uses current alpha after last update).
    double predict(const double *x) const;

    // Current dictionary size (0 <= dict_size() <= capacity).
    int dict_size() const { return static_cast<int>(dict_.size()); }

private:
    int    dim_;
    int    capacity_;
    double lambda_;
    double sigma_;
    double ff_;
    double ald_thresh_;

    std::vector<Eigen::VectorXd> dict_;  // dictionary input vectors
    Eigen::VectorXd              y_dict_; // targets for dictionary entries
    Eigen::MatrixXd              P_;     // (K_dict + lambda*I)^{-1}, m x m
    Eigen::VectorXd              alpha_; // dual weights = P_ * y_dict_

    double kernel(const double *x1, const double *x2) const;

    // Remove the oldest (first) dictionary entry from P_ via Sherman-Morrison.
    void downdate();
};
