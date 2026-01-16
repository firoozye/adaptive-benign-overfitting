#include "abo/ABO.h"

using namespace std;

extern "C"
{
   void dlartg_(double *a, double *b, double *c, double *s, double *r);
   void drot_(int *n, double *dx, int *incx, double *dy, int *incy,
              double *c, double *s);
   void dscal_(const int *n,
               const double *da,
               double *dx,
               const int *incx);
}

ABO::~ABO()
{
   delete[] X_;
   delete[] y_;
   delete[] R_;
   delete[] R_inv_;
   delete[] Q_;
   delete[] z_;
   delete[] beta_;
   delete[] G_e_1_;
}

ABO::ABO(double *x_input, double *y_input, int max_obs, double ff, int dim, int n_batch)
    : G_(nullptr),
      R_(nullptr),
      R_inv_(nullptr),
      Q_(nullptr),
      z_(nullptr),
      beta_(nullptr),
      // hyperparameters
      max_obs_(max_obs),
      r_c_size_(n_batch * dim),
      n_obs_(n_batch),
      dim_(dim),
      ff_(ff),
      sqrt_ff_(sqrt(ff))
{

   X_ = new double[n_obs_ * dim_]();
   std::memcpy(X_, x_input, n_obs_ * dim_ * sizeof(double));
   y_ = new double[n_obs_]();
   std::memcpy(y_, y_input, n_obs_ * sizeof(double));
   G_e_1_ = new double[max_obs + 1];

   batchInitialize();
}

void ABO::batchInitialize()
{
   // Forgetting factor matrix

   for (int i = 0; i < n_obs_; i++)
   {
      double pow_n = (n_obs_ - i - 1) / 2.0;
      double scale = pow(sqrt_ff_, pow_n);
      for (int j = 0; j < dim_; j++)
      {
         X_[j * n_obs_ + i] *= scale;
      }
      y_[i] *= scale;
   }

   // initialize the Q, R_ matrices
   std::tie(Q_, R_) = Q_R_compute(X_, n_obs_, dim_);

   R_inv_ = new double[r_c_size_];
   pinv(R_, R_inv_, n_obs_, dim_);

   z_ = new double[n_obs_]();
   beta_ = new double[dim_]();

   cblas_dgemv(CblasColMajor, CblasTrans,
               n_obs_, n_obs_, 1.0, Q_, n_obs_, y_, 1, 0.0, z_, 1);
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               dim_, n_obs_, 1.0, R_inv_, dim_, z_, 1, 0.0, beta_, 1);
}

// Update method
void ABO::update(double *new_x, double &new_y)
{

   if (ff_ != 1.0)
   {
      for (int i = 0; i < n_obs_; ++i)
      {
         y_[i] *= sqrt_ff_;
      }
      for (int i = 0; i < r_c_size_; ++i)
      {
         R_inv_[i] *= (1.0 / sqrt_ff_);
         R_[i] *= sqrt_ff_;
      }
   }

   y_ = addRowColMajor(y_, n_obs_, 1);
   y_[n_obs_] = new_y;

   double d[n_obs_];
   double c[dim_];

   cblas_dgemv(CblasColMajor, CblasTrans,
               dim_, n_obs_, 1.0, R_inv_, dim_, new_x, 1, 0.0, d, 1);

   double temp_c[n_obs_];
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               n_obs_, dim_, 1.0, R_, n_obs_, new_x, 1, 0.0, temp_c, 1);
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               dim_, n_obs_, 1.0, R_inv_, dim_, temp_c, 1, 0.0, c, 1);

   for (int i = 0; i < dim_; i++)
   {
      c[i] = new_x[i] - c[i];
   }

   // Update for new regime
   if (n_obs_ < dim_)
   {
      double c_inv[dim_];
      pinv(c, c_inv, dim_, 1);
      cblas_dger(CblasColMajor, dim_, n_obs_, -1.0, c_inv, 1, d, 1, R_inv_, dim_);
      R_inv_ = addColColMajor(R_inv_, dim_, n_obs_);

      for (int i = 0; i < dim_; ++i)
      {
         R_inv_[n_obs_ * dim_ + i] = c_inv[i];
      }

      // weight update
      double x_T_w = cblas_ddot(dim_, new_x, 1, beta_, 1);
      for (int i = 0; i < dim_; i++)
      {
         beta_[i] += c_inv[i] * (new_y - x_T_w);
      }
   }
   // Update for old regime
   else
   {
      double b_k[dim_];
      double alpha = cblas_ddot(n_obs_, d, 1, d, 1);
      alpha = 1 / (1 + alpha);
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim_, n_obs_, alpha, R_inv_, dim_, d, 1, 0.0, b_k, 1);
      cblas_dger(CblasColMajor, dim_, n_obs_, -1.0, b_k, 1, d, 1, R_inv_, dim_);
      R_inv_ = addColColMajor(R_inv_, dim_, n_obs_);

      for (int i = 0; i < dim_; ++i)
      {
         R_inv_[n_obs_ * dim_ + i] = b_k[i];
      }

      // weight update
      double x_T_w = cblas_ddot(dim_, new_x, 1, beta_, 1);
      for (int i = 0; i < dim_; i++)
      {
         beta_[i] += b_k[i] * (new_y - x_T_w);
      }
   }

   R_ = addRowColMajor(R_, n_obs_, dim_);

   for (int i = 0; i < dim_; ++i)
   {
      R_[(i + 1) * n_obs_ + i] = new_x[i];
   }

   Q_ = addRowAndColumnColMajor(Q_, n_obs_, n_obs_);
   Q_[((n_obs_ + 1) * (n_obs_ + 1)) - 1] = 1;

   n_obs_++;

   givens::update(this);

   r_c_size_ = n_obs_ * dim_;

   if (n_obs_ > max_obs_)
   {
      downdate();
   }
}

void ABO::downdate()
{
   // Update matrices
   givens::downdate(this);
   // double x_T[dim_];
   // for (int i = 0; i < dim_; ++i)
   //{
   //    x_T[i] = R_[n_obs_ * i]; // Copy the first row=
   // }
   double *x_T = R_; // pointer to first row

   // Deletion for new regime
   if (n_obs_ < dim_)
   {

      double k[dim_];
      double h[n_obs_];
      // G_e_1_ is declared in givens downdate. G e_1
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim_, n_obs_, 1.0, R_inv_, dim_, G_e_1_, 1, 0.0, k, 1);
      // cblas_dgemv(CblasColMajor, CblasTrans,
      //             dim_, n_obs_, 1.0, R_inv_, dim_, x_T, 1, 0.0, h, 1);
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim_, n_obs_, 1.0, R_inv_, dim_, x_T, n_obs_, 0.0, h, 1);

      double k_inv[dim_];
      double h_inv[n_obs_];
      pinv(k, k_inv, dim_, 1);
      pinv(h, h_inv, 1, n_obs_);

      double k_inv_R_inv[n_obs_];
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim_, n_obs_, 1.0, R_inv_, dim_, k_inv, 1, 0.0, k_inv_R_inv, 1);
      double s = cblas_ddot(n_obs_, k_inv_R_inv, 1, h_inv, 1);

      double P_h_inv[dim_];
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim_, n_obs_, 1.0, R_inv_, dim_, h_inv, 1, 0.0, P_h_inv, 1);

      cblas_dger(CblasColMajor, dim_, n_obs_, -1.0, k, 1, k_inv_R_inv, 1, R_inv_, dim_);
      cblas_dger(CblasColMajor, dim_, n_obs_, -1.0, P_h_inv, 1, h, 1, R_inv_, dim_);
      cblas_dger(CblasColMajor, dim_, n_obs_, s, k, 1, h, 1, R_inv_, dim_);

      // Weight downdate
      double k_inv_w = cblas_ddot(dim_, k_inv, 1, beta_, 1);
      for (int i = 0; i < dim_; ++i)
      {
         beta_[i] -= k[i] * k_inv_w;
      }
   }
   // Deletion for old regime
   else
   {

      double h[dim_];
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim_, n_obs_, 1.0, R_inv_, dim_, G_e_1_, 1, 0.0, h, 1);

      double k[n_obs_];
      // cblas_dgemv(CblasColMajor, CblasTrans,
      //             dim_, n_obs_, 1.0, R_inv_, dim_, x_T, 1, 0.0, k, 1);
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim_, n_obs_, 1.0, R_inv_, dim_, x_T, n_obs_, 0.0, k, 1);

      double s = 1 - cblas_ddot(n_obs_, k, 1, G_e_1_, 1);
      cblas_dger(CblasColMajor, dim_, n_obs_, 1.0 / s, h, 1, k, 1, R_inv_, dim_);

      // Weight downdate
      double x_T_B = cblas_ddot(dim_, x_T, 1, beta_, 1);
      double y_0 = y_[0];

      for (int i = 0; i < dim_; i++)
      {
         beta_[i] -= (1.0 / s) * (y_0 - x_T_B) * h[i];
      }
   }

   int inc = 1;  // down a column
   int n = dim_; // number of rows in R_inv

   for (const auto &rot : giv_rots)
   {
      drot_(&n,
            &R_inv_[rot.j1], &inc, // column rot.j1
            &R_inv_[rot.j2], &inc, // column rot.j2
            (double *)&rot.c, (double *)&rot.s);
   }
   giv_rots.clear();

   R_ = deleteRowColMajor(R_, n_obs_, dim_);
   R_inv_ = deleteColColMajor(R_inv_, dim_, n_obs_);
   Q_ = deleteRowColMajor(Q_, n_obs_, n_obs_);
   Q_ = deleteColColMajor(Q_, n_obs_ - 1, n_obs_);
   y_ = deleteRowColMajor(y_, n_obs_, 1);
   n_obs_--;
   r_c_size_ = n_obs_ * dim_;
}

double ABO::pred(double *x)
{
   double pred_value = cblas_ddot(dim_, x, 1, beta_, 1);
   return pred_value;
}

double ABO::get_cond_num()
{
   lapack_int m = dim_, n = n_obs_, lda = m;
   lapack_int ldu = m, ldvt = n;

   double *A_copy = new double[n_obs_ * dim_];
   std::memcpy(A_copy, R_inv_, n_obs_ * dim_ * sizeof(double));
   int min_mn = std::min(n_obs_, dim_);
   double s[min_mn];
   double *u = new double[ldu * ldu];
   double *vt = new double[ldvt * ldvt];
   LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'A', m, n, A_copy, lda, s, u, ldu, vt, ldvt);

   double maxS = *std::max_element(s, s + min_mn);
   double minS = *std::min_element(s, s + min_mn);

   delete[] A_copy;
   delete[] vt;
   delete[] u;

   return maxS / minS;
}

double ABO::get_real_cond_num()
{
   lapack_int m = n_obs_, n = dim_, lda = m;
   lapack_int ldu = m, ldvt = n;

   double *A_copy = new double[n_obs_ * dim_];
   std::memcpy(A_copy, X_, n_obs_ * dim_ * sizeof(double));
   int min_mn = std::min(n_obs_, dim_);
   double s[min_mn];
   double *u = new double[ldu * ldu];
   double *vt = new double[ldvt * ldvt];
   LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'A', m, n, A_copy, lda, s, u, ldu, vt, ldvt);

   double maxS = *std::max_element(s, s + min_mn);
   double minS = *std::min_element(s, s + min_mn);

   delete[] A_copy;
   delete[] vt;
   delete[] u;

   return maxS / minS;
}