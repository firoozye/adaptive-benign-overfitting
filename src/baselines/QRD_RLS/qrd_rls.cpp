#include "baselines/QRD_RLS/qrd_rls.h"
#include "pseudo_inverse.h"
#include "abo/QR_decomposition.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>

extern "C"
{
    void drotg_(double *a, double *b, double *c, double *s);
    void dlartg_(double *a, double *b, double *c, double *s, double *r);
    void drot_(int *n, double *dx, int *incx, double *dy, int *incy,
               double *c, double *s);
}

static int drotg_hyp(double a, double b, double *c, double *s, double *r)
{
    if (a == 0.0)
        return 1;
    double aa = fabs(a), bb = fabs(b);
    if (!(aa > bb))
        return 2; // need a^2 > b^2 for real hyperbolic annihilation

    // scale to reduce overflow/underflow risk
    double scale = aa; // since aa > bb, scaling by aa is decent
    double as = a / scale;
    double bs = b / scale;

    double t = as * as - bs * bs; // > 0
    double rs = sqrt(t);          // sqrt((a/scale)^2 - (b/scale)^2)
    double rr = scale * rs;       // sqrt(a^2 - b^2)

    *c = a / rr;
    *s = -b / rr;
    if (r)
        *r = rr;
    return 0;
}

// Apply hyperbolic rotation to vectors x,y (like drot but with hyp params)
static void drot_hyp(int n, double *x, int incx, double *y, int incy,
                     double c, double s)
{
    for (int i = 0; i < n; ++i)
    {
        double xi = x[i * incx];
        double yi = y[i * incy];
        x[i * incx] = c * xi + s * yi;
        y[i * incy] = s * xi + c * yi;
    }
}

void givens_rot(int p, double *v, double *G)
{
    // p = dim_
    // v has length p+1, last index = p
    // G is (p+1)x(p+1) column-major

    int ldG = p + 1;
    int len = p + 1; // number of columns in G
    int inc = ldG;   // step across columns for fixed row in col-major
    int last = p;

    for (int k = 0; k < p; ++k)
    {
        double c, s, r;
        dlartg_(&v[last], &v[k], &c, &s, &r);

        v[last] = r;
        v[k] = 0.0;

        drot_(&len,
              &G[last], &inc, // row "last"
              &G[k], &inc,    // row "k"
              &c, &s);
    }
}

QRDRLS::QRDRLS(int max_obs, int dim_, double forgetting_factor, double delta)
    : max_obs_(max_obs), ff_(forgetting_factor), sqrt_ff_(std::sqrt(forgetting_factor)), delta_(delta), beta_(nullptr), UT_(nullptr), initialized_(false), dim_(dim_)
{
    if (forgetting_factor <= 0.0 || forgetting_factor > 1.0)
    {
        throw std::invalid_argument("Forgetting factor must be in (0, 1]");
    }
    if (delta <= 0.0)
    {
        throw std::invalid_argument("Delta must be positive");
    }

    // Allocate memory
    beta_ = new double[dim_];

    // Initialize to zero
    std::memset(beta_, 0, dim_ * sizeof(double));
}

QRDRLS::~QRDRLS()
{
    delete[] beta_;
    delete[] UT_;
    delete[] R_;
    delete[] P_;
    delete[] X_;
    delete[] y_;
}

void QRDRLS::batchInitialize(const double *X_batch, const double *y_batch, int batch_size, int dim)
{

    // Form regularized data matrix [sqrt(λ^batch_size * δ) * I; X_batch]
    double reg_term = delta_ * std::pow(ff_, static_cast<double>(batch_size));
    double sqrt_reg = std::sqrt(reg_term);

    n_obs_ = batch_size;

    // Build regularized X matrix (column-major)
    // double *X_reg = new double[n_obs_ * dim_];
    X_ = new double[n_obs_ * dim_]();
    y_ = new double[n_obs_]();

    std::memcpy(y_, y_batch, n_obs_ * sizeof(double));
    // Fill regularization part: sqrt_reg * I in first (N+1) rows
    for (int col = 0; col < dim_; ++col)
    {
        for (int row = 0; row < n_obs_; ++row)
        {
            X_[col * n_obs_ + row] = (row == col) ? sqrt_reg : 0.0;
        }
    }

    // Fill X_batch part (already column-major)
    for (int col = 0; col < dim_; ++col)
    {
        for (int row = 0; row < batch_size; ++row)
        {
            X_[col * n_obs_ + row] += X_batch[col * batch_size + row];
        }
    }

    std::tie(Q_, R_) = Q_R_compute(X_, n_obs_, dim_);
    P_ = new double[n_obs_ * dim_];
    pinv(R_, P_, n_obs_, dim_);
    UT_ = new double[dim_ * (dim_ + 1)]();

    // Compute Q^T * d_reg for right-hand side
    double z[n_obs_];

    cblas_dgemv(CblasColMajor, CblasTrans,
                n_obs_, n_obs_, 1.0, Q_, n_obs_, y_batch, 1, 0.0, z, 1);

    cblas_dgemv(CblasColMajor, CblasNoTrans,
                dim_, n_obs_, 1.0, P_, dim_, z, 1, 0.0, beta_, 1);

    for (int j = 0; j < dim_; ++j)
    {
        for (int i = 0; i < dim_; ++i)
        {
            UT_[j + i * (dim_ + 1)] = P_[i + j * dim_];
        }
    }
}

void QRDRLS::update(const double *new_x, double new_y, double &prediction, double &error)
{
    prediction = cblas_ddot(dim_, new_x, 1, beta_, 1);
    error = new_y - (prediction);

    int dim_p_1 = dim_ + 1;

    double a[dim_p_1];
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                dim_p_1, dim_, -1.0, UT_, dim_p_1, new_x, 1, 0.0, a, 1);

    a[dim_] = 1;
    double G[dim_p_1 * dim_p_1] = {0};
    for (int i = 0; i < dim_p_1; i++)
    {
        G[i * dim_p_1 + i] = 1;
    }

    givens_rot(dim_, a, G);

    double result_1[dim_ * dim_p_1];
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                dim_p_1, dim_, dim_p_1, 1, G, dim_p_1, UT_, dim_p_1, 0, result_1, dim_p_1);
    std::memcpy(UT_, result_1, dim_ * dim_p_1 * sizeof(double));

    double delta = a[dim_];

    for (int i = 0; i < dim_; i++)
    {
        double u = UT_[i * dim_p_1 + dim_];
        beta_[i] -= (new_y - prediction) / delta * u;
        UT_[i * dim_p_1 + dim_] = 0;
    }

    X_ = addRowColMajor(X_, n_obs_, dim_);
    y_ = addRowColMajor(y_, n_obs_, 1);

    y_[n_obs_] = new_y;
    for (int i = 0; i < dim_; i++)
    {
        X_[(i + 1) * n_obs_ + i] = new_x[i];
    }

    n_obs_++;

    if (n_obs_ > max_obs_)
    {
        downdate();
    }
}

void QRDRLS::downdate()
{
    double x_T[dim_];
    for (int i = 0; i < dim_; ++i)
    {
        x_T[i] = X_[n_obs_ * i]; // Copy the first row=
    }

    int dim_p_1 = dim_ + 1;

    double b[dim_p_1];
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                dim_p_1, dim_, -1.0, UT_, dim_p_1, x_T, 1, 0.0, b, 1);

    b[dim_] = 1;

    double G[dim_p_1 * dim_p_1] = {0};
    for (int i = 0; i < dim_p_1; i++)
    {
        G[i * dim_p_1 + i] = 1;
    }

    double c, s, r;

    int p = dim_;
    int ldG = p + 1;
    int len = p + 1;
    int inc = ldG;
    int last = p;

    for (int k = 0; k < p; ++k)
    {
        // Build hyperbolic rotation to zero b[k] using b[last]
        double c, s, r;
        int rc = drotg_hyp(b[last], b[k], &c, &s, &r);

        b[last] = r;
        b[k] = 0.0;

        // Accumulate into G by rotating rows (last, k) across all columns
        drot_hyp(len, &G[last], inc, &G[k], inc, c, s);
    }

    double result_1[dim_ * dim_p_1];

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                dim_p_1, dim_, dim_p_1, 1, G, dim_p_1, UT_, dim_p_1, 0, result_1, dim_p_1);
    std::memcpy(UT_, result_1, dim_ * dim_p_1 * sizeof(double));

    double temp_pred = pred(x_T);
    double delta = b[dim_];

    for (int i = 0; i < dim_; i++)
    {
        double u = UT_[i * dim_p_1 + dim_];
        beta_[i] -= (y_[0] - temp_pred) / delta * u;
        UT_[i * dim_p_1 + dim_] = 0;
    }

    X_ = deleteRowColMajor(X_, n_obs_, dim_);
    y_ = deleteRowColMajor(y_, n_obs_, 1);

    n_obs_--;
}

double QRDRLS::pred(double *x)
{
    double pred_value = cblas_ddot(dim_, x, 1, beta_, 1);
    return pred_value;
}

void QRDRLS::reset()
{
    std::memset(beta_, 0, (n_obs_ + 1) * sizeof(double));
    std::memset(UT_, 0, (n_obs_ + 1) * (n_obs_ + 1) * sizeof(double));
    initialized_ = false;
}
