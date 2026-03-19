#include "abo/ABO.h"

using namespace std;

extern "C"
{
   void dlartg_(double *a, double *b, double *c, double *s, double *r);
   void drot_(int *n, double *dx, int *incx, double *dy, int *incy,
              double *c, double *s);
}

ABO::~ABO()
{
   delete[] y_;
   delete[] R_;
   delete[] R_inv_;
   delete[] beta_;
   delete[] G_e_1_;
   delete[] G_;
   delete[] scratch_n_;
   delete[] scratch_n2_;
   delete[] scratch_dim_;
   delete[] scratch_d_;
   delete[] scratch_d2_;
}

ABO::ABO(double *x_input, double *y_input, int max_obs, double ff, int dim, int n_batch)
    : G_(nullptr),
      R_(nullptr),
      R_inv_(nullptr),
      beta_(nullptr),
      // hyperparameters
      max_obs_(max_obs),
      n_obs_(n_batch),
      dim_(dim),
      ff_(ff),
      sqrt_ff_(sqrt(ff))
{
   // Pre-allocate everything to max capacity — no reallocation in hot path
   y_          = new double[max_obs_]();
   R_          = new double[max_obs_ * dim_]();
   R_inv_      = new double[dim_  * max_obs_]();
   beta_       = new double[dim_]();
   G_e_1_      = new double[max_obs_ + 1];
   G_          = new double[(max_obs_ + 1) * (max_obs_ + 1)];
   scratch_n_  = new double[max_obs_]();
   scratch_n2_ = new double[max_obs_]();
   scratch_dim_ = new double[dim_]();
   scratch_d_  = new double[max_obs_ * dim_]();
   scratch_d2_ = new double[max_obs_ * dim_]();

   std::memcpy(y_, y_input, n_obs_ * sizeof(double));
   batchInitialize(x_input);
}

void ABO::batchInitialize(double *x_input)
{
   // Scale features and targets by forgetting factor
   double *X_scaled = new double[n_obs_ * dim_]();
   std::memcpy(X_scaled, x_input, n_obs_ * dim_ * sizeof(double));

   for (int i = 0; i < n_obs_; i++)
   {
      double scale = std::pow(ff_, (n_obs_ - i - 1) / 2.0);
      for (int j = 0; j < dim_; j++)
         X_scaled[j * n_obs_ + i] *= scale;
      y_[i] *= scale;
   }

   // QR decomposition — Q_local and R_temp have stride n_obs_
   double *Q_local, *R_temp;
   std::tie(Q_local, R_temp) = Q_R_compute(X_scaled, n_obs_, dim_);

   // Copy R_temp (stride n_obs_) into R_ (stride max_obs_) — key stride difference
   for (int j = 0; j < dim_; j++)
      std::memcpy(&R_[j * max_obs_], &R_temp[j * n_obs_], n_obs_ * sizeof(double));

   // Compute pseudo-inverse of R_temp into R_inv_ (stride dim_)
   pinv(R_temp, R_inv_, n_obs_, dim_);

   // z = Q^T * y, then beta = R_inv * z
   double *z_local = new double[n_obs_]();
   cblas_dgemv(CblasColMajor, CblasTrans,
               n_obs_, n_obs_, 1.0, Q_local, n_obs_, y_, 1, 0.0, z_local, 1);
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               dim_, n_obs_, 1.0, R_inv_, dim_, z_local, 1, 0.0, beta_, 1);

   delete[] Q_local;
   delete[] R_temp;
   delete[] z_local;
   delete[] X_scaled;
}

void ABO::update(double *new_x, double &new_y)
{
   if (ff_ != 1.0)
   {
      // Scale y (contiguous)
      cblas_dscal(n_obs_, sqrt_ff_, y_, 1);
      // Scale R_ column by column (stride max_obs_)
      for (int j = 0; j < dim_; ++j)
         cblas_dscal(n_obs_, sqrt_ff_, &R_[j * max_obs_], 1);
      // Scale R_inv_ (contiguous n_obs_ columns of dim_ elements each)
      cblas_dscal(n_obs_ * dim_, 1.0 / sqrt_ff_, R_inv_, 1);
   }

   // Append new target
   y_[n_obs_] = new_y;

   // d = R_inv^T * new_x  -> scratch_n_ (n_obs_-length)
   cblas_dgemv(CblasColMajor, CblasTrans,
               dim_, n_obs_, 1.0, R_inv_, dim_, new_x, 1, 0.0, scratch_n_, 1);

   // temp = R_ * new_x   -> scratch_n2_ (n_obs_-length), lda = max_obs_
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               n_obs_, dim_, 1.0, R_, max_obs_, new_x, 1, 0.0, scratch_n2_, 1);

   // c = new_x - R_inv_ * (R_ * new_x)  -> scratch_dim_ (dim_-length)
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               dim_, n_obs_, 1.0, R_inv_, dim_, scratch_n2_, 1, 0.0, scratch_dim_, 1);
   for (int i = 0; i < dim_; i++)
      scratch_dim_[i] = new_x[i] - scratch_dim_[i];

   if (n_obs_ < dim_)
   {
      // New regime: c = scratch_dim_, c_inv -> scratch_n2_
      pinv(scratch_dim_, scratch_n2_, dim_, 1);
      cblas_dger(CblasColMajor, dim_, n_obs_, -1.0, scratch_n2_, 1, scratch_n_, 1, R_inv_, dim_);
      // Write new column of R_inv_ at position n_obs_
      for (int i = 0; i < dim_; ++i)
         R_inv_[n_obs_ * dim_ + i] = scratch_n2_[i];

      double x_T_w = cblas_ddot(dim_, new_x, 1, beta_, 1);
      for (int i = 0; i < dim_; i++)
         beta_[i] += scratch_n2_[i] * (new_y - x_T_w);
   }
   else
   {
      // Old regime: d = scratch_n_, b_k -> scratch_dim_
      double alpha = cblas_ddot(n_obs_, scratch_n_, 1, scratch_n_, 1);
      alpha = 1.0 / (1.0 + alpha);
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim_, n_obs_, alpha, R_inv_, dim_, scratch_n_, 1, 0.0, scratch_dim_, 1);
      cblas_dger(CblasColMajor, dim_, n_obs_, -1.0, scratch_dim_, 1, scratch_n_, 1, R_inv_, dim_);
      // Write new column of R_inv_ at position n_obs_
      for (int i = 0; i < dim_; ++i)
         R_inv_[n_obs_ * dim_ + i] = scratch_dim_[i];

      double x_T_w = cblas_ddot(dim_, new_x, 1, beta_, 1);
      for (int i = 0; i < dim_; i++)
         beta_[i] += scratch_dim_[i] * (new_y - x_T_w);
   }

   // Write new row to R_ at row index n_obs_ (col j: position n_obs_ + j*max_obs_)
   for (int j = 0; j < dim_; ++j)
      R_[n_obs_ + j * max_obs_] = new_x[j];

   n_obs_++;
   givens::update(this);
}

void ABO::downdate(double *z_old, double y_old)
{
   // Givens rotations to prepare G, G_e_1_, and (in new regime) update R and record giv_rots
   givens::downdate(this, z_old);

   double *x_T = R_;  // pointer to first row; stride between columns is max_obs_

   if (n_obs_ < dim_)
   {
      // New regime
      // k = R_inv_ * G_e_1_  -> scratch_dim_ (dim_-length)
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim_, n_obs_, 1.0, R_inv_, dim_, G_e_1_, 1, 0.0, scratch_dim_, 1);
      // h = R_inv_^T * x_T  -> scratch_n_ (n_obs_-length), x_T stride = max_obs_
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim_, n_obs_, 1.0, R_inv_, dim_, x_T, max_obs_, 0.0, scratch_n_, 1);

      // Lay out temporaries in scratch_d_ to avoid aliasing:
      //   h_inv   at scratch_d_[0 .. n_obs_-1]
      //   P_h_inv at scratch_d_[n_obs_ .. n_obs_+dim_-1]
      double *k_inv    = scratch_n2_;          // dim_-length
      double *h_inv    = scratch_d_;           // n_obs_-length
      double *P_h_inv  = scratch_d_ + n_obs_;  // dim_-length
      double *k_inv_R  = scratch_d2_;          // n_obs_-length

      pinv(scratch_dim_, k_inv, dim_, 1);
      pinv(scratch_n_, h_inv, 1, n_obs_);

      // k_inv_R_inv = R_inv_^T * k_inv  -> k_inv_R
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim_, n_obs_, 1.0, R_inv_, dim_, k_inv, 1, 0.0, k_inv_R, 1);
      double s = cblas_ddot(n_obs_, k_inv_R, 1, h_inv, 1);

      // P_h_inv = R_inv_ * h_inv
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim_, n_obs_, 1.0, R_inv_, dim_, h_inv, 1, 0.0, P_h_inv, 1);

      // R_inv_ -= k * k_inv_R_inv^T
      cblas_dger(CblasColMajor, dim_, n_obs_, -1.0, scratch_dim_, 1, k_inv_R, 1, R_inv_, dim_);
      // R_inv_ -= P_h_inv * h^T
      cblas_dger(CblasColMajor, dim_, n_obs_, -1.0, P_h_inv, 1, scratch_n_, 1, R_inv_, dim_);
      // R_inv_ += s * k * h^T
      cblas_dger(CblasColMajor, dim_, n_obs_, s, scratch_dim_, 1, scratch_n_, 1, R_inv_, dim_);

      // Weight downdate
      double k_inv_w = cblas_ddot(dim_, k_inv, 1, beta_, 1);
      for (int i = 0; i < dim_; ++i)
         beta_[i] -= scratch_dim_[i] * k_inv_w;

      int inc = 1;
      for (const auto &rot : giv_rots)
      {
         drot_(&dim_,
               &R_inv_[rot.j1], &inc,
               &R_inv_[rot.j2], &inc,
               (double *)&rot.c, (double *)&rot.s);
      }
      giv_rots.clear();
   }
   else
   {
      // Old regime
      // h = R_inv_ * G_e_1_  -> scratch_dim_ (dim_-length)
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim_, n_obs_, 1.0, R_inv_, dim_, G_e_1_, 1, 0.0, scratch_dim_, 1);
      // k = R_inv_^T * x_T  -> scratch_n_ (n_obs_-length), x_T stride = max_obs_
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim_, n_obs_, 1.0, R_inv_, dim_, x_T, max_obs_, 0.0, scratch_n_, 1);

      double s = 1.0 - cblas_ddot(n_obs_, scratch_n_, 1, G_e_1_, 1);
      cblas_dger(CblasColMajor, dim_, n_obs_, 1.0 / s, scratch_dim_, 1, scratch_n_, 1, R_inv_, dim_);

      // Weight downdate
      double x_T_B = cblas_ddot(dim_, x_T, max_obs_, beta_, 1);
      for (int i = 0; i < dim_; i++)
         beta_[i] -= (1.0 / s) * (y_old - x_T_B) * scratch_dim_[i];

      // R_inv_ = R_inv_ * G_  (use scratch_d_ to avoid aliasing)
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                  dim_, n_obs_, n_obs_, 1.0,
                  R_inv_, dim_, G_, n_obs_,
                  0.0, scratch_d_, dim_);
      std::memcpy(R_inv_, scratch_d_, n_obs_ * dim_ * sizeof(double));
   }

   // Delete first row of R_ via memmove (Bug 1 fix: stride max_obs_, not n_obs_-1)
   for (int j = 0; j < dim_; ++j)
      std::memmove(&R_[j * max_obs_], &R_[j * max_obs_ + 1], (n_obs_ - 1) * sizeof(double));

   // Delete first column of R_inv_ (stride dim_, contiguous column block)
   std::memmove(R_inv_, R_inv_ + dim_, dim_ * (n_obs_ - 1) * sizeof(double));

   // Delete first element of y_
   std::memmove(y_, y_ + 1, (n_obs_ - 1) * sizeof(double));

   n_obs_--;
}

double ABO::pred(double *x)
{
   return cblas_ddot(dim_, x, 1, beta_, 1);
}

double ABO::get_cond_num()
{
   lapack_int m = dim_, n = n_obs_, lda = m;
   lapack_int ldu = m, ldvt = n;

   double *A_copy = new double[n_obs_ * dim_];
   std::memcpy(A_copy, R_inv_, n_obs_ * dim_ * sizeof(double));
   int min_mn = std::min(n_obs_, dim_);
   double *s = new double[min_mn];
   double *u = new double[ldu * ldu];
   double *vt = new double[ldvt * ldvt];
   LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'N', m, n, A_copy, lda, s, u, ldu, vt, ldvt);

   double maxS = *std::max_element(s, s + min_mn);
   double minS = *std::min_element(s, s + min_mn);

   delete[] A_copy;
   delete[] vt;
   delete[] u;
   delete[] s;

   return maxS / minS;
}
