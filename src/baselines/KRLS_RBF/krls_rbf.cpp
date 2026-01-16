#include "baselines/KRLS_RBF/krls_rbf.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <cstddef>
#include <cmath>

bool isSymmetricColMajor(const double *A, int n, double tol = 1e-12)
{
    for (int j = 0; j < n; ++j)
    {
        for (int i = j + 1; i < n; ++i)
        {
            double diff = std::abs(A[i + j * n] - A[j + i * n]);
            if (diff > tol)
                return false;
        }
    }
    return true;
}

// Constructor
KRLS_RBF::KRLS_RBF(const double *X_init, const double *y_init, int n_obs,
                   int dim_, double delta, double sigma,
                   int window_size)
    : dim_(dim_),
      n_obs_(n_obs),
      window_size_(window_size),
      delta_(delta),
      sigma_(sigma),
      initialized_(false),
      X_init_(nullptr),
      y_init_(nullptr),
      n_init_samples_(0)
{

    // Allocate state arrays
    beta_ = new double[n_obs_];
    P_ = new double[n_obs_ * n_obs_];

    // Allocate working arrays
    h_ = new double[n_obs_];
    X_ = new double[n_obs_ * dim_];
    y_ = new double[n_obs_];
    K_ = new double[n_obs_ * n_obs_];

    // Initialize to zero
    std::memset(X_, 0, n_obs_ * dim_ * sizeof(double));
    std::memset(beta_, 0, n_obs_ * sizeof(double));
    std::memset(P_, 0, n_obs_ * n_obs_ * sizeof(double));

    // Store initial data if provided
    if (X_init != nullptr && y_init != nullptr && n_obs_ > 0)
    {
        n_init_samples_ = n_obs_;
        y_init_ = new double[n_obs_];

        vectorCopy(X_, X_init, n_obs_ * dim_);
        vectorCopy(y_, y_init, n_obs_);

        // Initialize from batch data
        initializeFromBatch(X_, y_, n_obs_);
    }
}

// Destructor
KRLS_RBF::~KRLS_RBF()
{
    delete[] X_;
    delete[] beta_;
    delete[] P_;
    delete[] h_;
    delete[] K_;

    if (X_init_ != nullptr)
    {
        delete[] X_init_;
    }
    if (y_init_ != nullptr)
    {
        delete[] y_init_;
    }
}

// RBF kernel for vectors with a stride (stride=1 means contiguous)
double KRLS_RBF::kernel(const double *x1, int stride1,
                        const double *x2, int stride2) const
{
    double sum_sq = 0.0;
    for (int k = 0; k < dim_; ++k)
    {
        double diff = x1[k * stride1] - x2[k * stride2];
        sum_sq += diff * diff;
    }
    return std::exp(-sum_sq / (2.0 * sigma_ * sigma_));
}

// Copy vector
void KRLS_RBF::vectorCopy(double *dest, const double *src, int n)
{
    std::memcpy(dest, src, n * sizeof(double));
}

// Initialize from batch of data
void KRLS_RBF::initializeFromBatch(const double *X, const double *y, int n_obs_)
{

    // Build kernel matrix K

    for (int j = 0; j < n_obs_; ++j)
    {
        const double *xj0 = &X[j];
        for (int i = 0; i < n_obs_; ++i)
        {
            const double *xi0 = &X[i];

            K_[i + j * n_obs_] = kernel(xi0, n_obs_, xj0, n_obs_);

            if (i == j)
                K_[i + j * n_obs_] += delta_;
        }
    }

    // Compute Q = (K + λI)^(-1)
    pinv(K_, P_, n_obs_, n_obs_);

    cblas_dgemv(CblasColMajor, CblasNoTrans,
                n_obs_, n_obs_, 1.0, P_, n_obs_, y_, 1, 0.0, beta_, 1);

    initialized_ = true;
}

// Update with new sample
void KRLS_RBF::update(const double *new_x, double new_y, double &prediction, double &error)
{
    K_ = addRowAndColumnColMajor(K_, n_obs_, n_obs_);

    // Compute h(i): vector of kernel evaluations
    for (int j = 0; j < n_obs_; ++j)
    {
        const double *xj0 = &X_[j]; // sample j, feature 0
        h_[j] = kernel(new_x, 1, xj0, n_obs_);

        K_[j * (n_obs_ + 1) + n_obs_] = h_[j];
        K_[(n_obs_ + 1) * n_obs_ + j] = h_[j];
    }

    double d_k = kernel(new_x, 1, new_x, 1) + delta_;
    K_[(n_obs_ + 1) * (n_obs_ + 1) - 1] = d_k;

    prediction = cblas_ddot(n_obs_, h_, 1, beta_, 1);
    error = new_y - (prediction);

    X_ = addRowColMajor(X_, n_obs_, dim_);

    for (int i = 0; i < dim_; i++)
    {
        X_[(i + 1) * n_obs_ + i] = new_x[i];
    }

    // Update of inverse matrix

    double P_b[n_obs_];
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                n_obs_, n_obs_, 1.0, P_, n_obs_, h_, 1, 0.0, P_b, 1);
    double g = 1 / (d_k - cblas_ddot(n_obs_, h_, 1, P_b, 1));

    cblas_dger(CblasColMajor, n_obs_, n_obs_, g, P_b, 1, P_b, 1, P_, n_obs_);

    P_ = addRowAndColumnColMajor(P_, n_obs_, n_obs_);

    for (int j = 0; j < n_obs_; ++j)
    {
        P_[j * (n_obs_ + 1) + n_obs_] = -g * P_b[j];
        P_[(n_obs_ + 1) * n_obs_ + j] = -g * P_b[j];
    }

    P_[(n_obs_ + 1) * (n_obs_ + 1) - 1] = g;
    y_ = addRowColMajor(y_, n_obs_, 1);
    y_[n_obs_] = new_y;

    n_obs_++;

    if (n_obs_ > window_size_)
    {
        downdate();
    }
}

void KRLS_RBF::downdate()
{

    K_ = deleteRowColMajor(K_, n_obs_, n_obs_);
    K_ = deleteColColMajor(K_, n_obs_ - 1, n_obs_);

    double f[n_obs_ - 1];

    for (int i = 0; i < n_obs_ - 1; i++)
    {
        f[i] = P_[i + 1];
    }
    double e = P_[0];
    P_ = deleteRowColMajor(P_, n_obs_, n_obs_);
    P_ = deleteColColMajor(P_, n_obs_ - 1, n_obs_);

    y_ = deleteRowColMajor(y_, n_obs_, 1);
    X_ = deleteRowColMajor(X_, n_obs_, dim_);
    n_obs_--;

    cblas_dger(CblasColMajor, n_obs_, n_obs_, -1 / e, f, 1, f, 1, P_, n_obs_);

    cblas_dgemv(CblasColMajor, CblasNoTrans,
                n_obs_, n_obs_, 1.0, P_, n_obs_, y_, 1, 0.0, beta_, 1);
}

// Reset filter
void KRLS_RBF::reset()
{
    if (X_init_ != nullptr && y_init_ != nullptr)
    {
        initializeFromBatch(X_init_, y_init_, n_init_samples_);
    }
    else
    {
        initialized_ = false;
        std::memset(beta_, 0, window_size_ * sizeof(double));
        std::memset(P_, 0, window_size_ * window_size_ * sizeof(double));
    }
}
