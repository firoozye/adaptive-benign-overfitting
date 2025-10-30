#include "QR_RLS.h"

using namespace std;

QR_Rls::~QR_Rls()
{
   LOG_INFO("QR_Rls destructor called");
   delete[] X;
   delete[] y;
   delete[] R;
   delete[] R_inv;
   delete[] all_Q;
   delete[] I;
   delete[] z;
   delete[] w;
}

QR_Rls::QR_Rls(double *x_input, double *y_input, int max_obs, double ff, double l, int dim, int n_batch)
    : G(nullptr),
      R(nullptr),
      R_inv(nullptr),
      all_Q(nullptr),
      I(nullptr),
      z(nullptr),
      w(nullptr),
      // hyperparameters
      max_obs(max_obs),
      X_rows(n_batch),
      r_c_size(n_batch * dim),
      l(l),
      n_batch(n_batch),
      n_obs(n_batch),
      dim(dim),
      ff(sqrt(ff)),
      b(1)
{

   X = new double[n_obs * dim]();
   std::memcpy(X, x_input, n_obs * dim * sizeof(double));
   y = new double[n_obs]();
   std::memcpy(y, y_input, n_obs * sizeof(double));

   // too much in the construtor! It should just set thing not make all sorts of calculations!
   LOG_INFO("QR_Rls constructor called with dim: " << dim << " and n_batch: " << n_batch);

   // Forgetting factor matrix

   for (int i = 0; i < n_obs; i++)
   {
      double pow_n = (n_obs - i - 1) / 2;
      double scale = pow(ff, pow_n);
      for (int j = 0; j < dim; j++)
      {
         X[j * n_obs + i] *= scale;
      }
   }

   // initialize the Q, R matrices
   double *temp_Q, *temp_R;
   std::tie(temp_Q, temp_R) = Q_R_compute(this, X, X_rows, dim);

   this->all_Q = temp_Q;
   this->R = temp_R;

   // r_c_size already initialized in constructor list
   R_inv = new double[r_c_size];
   pinv(R, R_inv, X_rows, dim);

   z = new double[n_obs]();
   w = new double[dim]();

   cblas_dgemv(CblasColMajor, CblasTrans,
               X_rows, X_rows, 1.0, all_Q, X_rows, y, 1, 0.0, z, 1);
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               dim, X_rows, 1.0, R_inv, dim, z, 1, 0.0, w, 1);
}

// Update method
void QR_Rls::update(double *new_x, double &new_y)
{
   X = addRowColMajor(X, n_obs, dim);
   for (int i = 0; i < dim; ++i)
   {
      X[(i + 1) * n_obs + i] = new_x[i];
   }

   y = addRowColMajor(y, n_obs, 1);
   z = addRowColMajor(z, n_obs, 1);
   y[n_obs] = new_y;
   z[n_obs] = new_y;

   for (int i = 0; i < r_c_size; ++i)
   {
      R_inv[i] *= (1.0 / pow(ff, 1 / 2));
      R[i] *= pow(ff, 1 / 2);
   }

   double d[n_obs];
   double c[dim];

   cblas_dgemv(CblasColMajor, CblasTrans,
               dim, n_obs, 1.0, R_inv, dim, new_x, 1, 0.0, d, 1);

   double temp_c[n_obs];
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               n_obs, dim, 1.0, R, n_obs, new_x, 1, 0.0, temp_c, 1);
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               dim, n_obs, 1.0, R_inv, dim, temp_c, 1, 0.0, c, 1);

   for (int i = 0; i < dim; i++)
   {
      c[i] = new_x[i] - c[i];
   }

   // Update for new regime
   if (n_obs < dim)
   {
      double c_inv[dim];
      pinv(c, c_inv, dim, 1);
      cblas_dger(CblasColMajor, dim, n_obs, -1.0, c_inv, 1, d, 1, R_inv, dim);
      R_inv = addColColMajor(R_inv, dim, n_obs);

      for (int i = 0; i < dim; ++i)
      {
         R_inv[n_obs * dim + i] = c_inv[i];
      }

      // weight update
      double x_T_w = cblas_ddot(dim, new_x, 1, w, 1);
      for (int i = 0; i < dim; i++)
      {
         w[i] += c_inv[i] * (new_y - x_T_w);
      }
   }
   // Update for old regime
   else
   {
      double b_k[dim];
      double alpha = cblas_ddot(n_obs, d, 1, d, 1);
      alpha = 1 / (1 + alpha);
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim, n_obs, alpha, R_inv, dim, d, 1, 0.0, b_k, 1);
      cblas_dger(CblasColMajor, dim, n_obs, -1.0, b_k, 1, d, 1, R_inv, dim);
      R_inv = addColColMajor(R_inv, dim, n_obs);

      for (int i = 0; i < dim; ++i)
      {
         R_inv[n_obs * dim + i] = b_k[i];
      }

      // weight update
      double x_T_w = cblas_ddot(dim, new_x, 1, w, 1);
      for (int i = 0; i < dim; i++)
      {
         w[i] += b_k[i] * (new_y - x_T_w);
      }
   }

   R = addRowColMajor(R, n_obs, dim);

   for (int i = 0; i < dim; ++i)
   {
      R[(i + 1) * n_obs + i] = new_x[i];
   }

   all_Q = addRowAndColumnColMajor(all_Q, n_obs, n_obs);
   all_Q[((n_obs + 1) * (n_obs + 1)) - 1] = 1;

   n_obs++;

   givens::update(this);

   r_c_size = n_obs * dim;

   double *P_copy = new double[r_c_size];
   cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
               dim, n_obs, n_obs, 1.0, R_inv, dim, G, n_obs, 0.0, P_copy, dim);
   std::memcpy(R_inv, P_copy, n_obs * dim * sizeof(double));
   delete[] P_copy;

   // G will be allocated in givens_update
   if (G != nullptr)
   {
      delete[] G;
      G = nullptr;
   }

   if (n_obs > max_obs)
   {
      downdate();
   }
   LOG_INFO("QR_Rls update finished");
}

void QR_Rls::downdate()
{
   LOG_INFO("QR_Rls downdate called");

   // Update matrices
   givens::downdate(this);
   double *result = new double[r_c_size];
   cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
               n_obs, dim, n_obs, 1, G, n_obs, R, n_obs, 0, result, n_obs);
   std::memcpy(R, result, n_obs * dim * sizeof(double));
   delete[] result; // Free memory

   double x_T[dim];
   for (int i = 0; i < dim; ++i)
   {
      x_T[i] = R[n_obs * i]; // Copy the first row=
      // x_T[i] = X[n_obs * i]; // Copy the first row=
   }

   double c[n_obs] = {0}; // initializes to zero
   c[0] = 1.0;

   // Deletion for new regime
   if (n_obs < dim)
   {

      double G_e_1[n_obs];
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  n_obs, n_obs, 1.0, G, n_obs, c, 1, 0.0, G_e_1, 1);

      double k[dim];
      double h[n_obs];
      LOG_DEBUG("downdate just before fourth and fifth cblas call");
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim, n_obs, 1.0, R_inv, dim, G_e_1, 1, 0.0, k, 1);
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim, n_obs, 1.0, R_inv, dim, x_T, 1, 0.0, h, 1);

      double k_inv[dim];
      double h_inv[n_obs];
      pinv(k, k_inv, dim, 1);
      pinv(h, h_inv, 1, n_obs);

      double k_inv_R_inv[n_obs];
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim, n_obs, 1.0, R_inv, dim, k_inv, 1, 0.0, k_inv_R_inv, 1);
      double s = cblas_ddot(n_obs, k_inv_R_inv, 1, h_inv, 1);

      double P_h_inv[dim];
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim, n_obs, 1.0, R_inv, dim, h_inv, 1, 0.0, P_h_inv, 1);

      cblas_dger(CblasColMajor, dim, n_obs, -1.0, k, 1, k_inv_R_inv, 1, R_inv, dim);
      cblas_dger(CblasColMajor, dim, n_obs, -1.0, P_h_inv, 1, h, 1, R_inv, dim);
      cblas_dger(CblasColMajor, dim, n_obs, s, k, 1, h, 1, R_inv, dim);

      // Weight downdate

      double k_inv_w = cblas_ddot(dim, k_inv, 1, w, 1);
      for (int i = 0; i < dim; ++i)
      {
         w[i] -= k[i] * k_inv_w;
      }
   }
   // Deletion for old regime
   else
   {

      // Pseudo Inverse Matrix downdate
      double G_e_1[n_obs];
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  n_obs, n_obs, 1.0, G, n_obs, c, 1, 0.0, G_e_1, 1);

      double h[dim];
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim, n_obs, 1.0, R_inv, dim, G_e_1, 1, 0.0, h, 1);

      double k[n_obs];
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim, n_obs, 1.0, R_inv, dim, x_T, 1, 0.0, k, 1);

      double s = 1 - cblas_ddot(n_obs, k, 1, G_e_1, 1);
      cblas_dger(CblasColMajor, dim, n_obs, 1.0 / s, h, 1, k, 1, R_inv, dim);

      // Weight downdate
      double x_T_B = cblas_ddot(dim, x_T, 1, w, 1);
      double y_0 = y[0];

      for (int i = 0; i < dim; i++)
      {
         w[i] -= (1.0 / s) * (y_0 - x_T_B) * h[i];
      }
   }
   LOG_DEBUG("downdate just before deleteRows ");

   double *result_1 = new double[r_c_size];
   cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
               dim, n_obs, n_obs, 1, R_inv, dim, G, n_obs, 0, result_1, dim);
   std::memcpy(R_inv, result_1, n_obs * dim * sizeof(double));
   delete[] result_1;

   R = deleteRowColMajor(R, n_obs, dim);
   R_inv = deleteColColMajor(R_inv, dim, n_obs);
   all_Q = deleteRowColMajor(all_Q, n_obs, n_obs);
   all_Q = deleteColColMajor(all_Q, n_obs - 1, n_obs);
   X = deleteRowColMajor(X, n_obs, dim);
   y = deleteRowColMajor(y, n_obs, 1);
   n_obs--;
   r_c_size = n_obs * dim;

   // Cleanup
   LOG_DEBUG("G pointer before final deletion: " << G);
   if (G != nullptr)
   {
      delete[] G;
      G = nullptr;
   }
   LOG_INFO("QR_Rls downdate finished");
}

double QR_Rls::pred(double *x)
{
   double pred_value = cblas_ddot(dim, x, 1, w, 1);
   return pred_value;
}

double QR_Rls::get_cond_num()
{
   lapack_int m = dim, n = n_obs, lda = m;
   lapack_int ldu = m, ldvt = n;

   double *A_copy = new double[n_obs * dim];
   std::memcpy(A_copy, R_inv, n_obs * dim * sizeof(double));
   int min_mn = std::min(n_obs, dim);
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

double QR_Rls::get_real_cond_num()
{
   lapack_int m = n_obs, n = dim, lda = m;
   lapack_int ldu = m, ldvt = n;

   double *A_copy = new double[n_obs * dim];
   std::memcpy(A_copy, X, n_obs * dim * sizeof(double));
   int min_mn = std::min(n_obs, dim);
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
