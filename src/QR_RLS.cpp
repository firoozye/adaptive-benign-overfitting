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
      b(1),
      i(1)
{

   X = new double[n_obs * dim]();
   std::memcpy(X, x_input, n_obs * dim * sizeof(double));
   y = new double[n_obs]();
   std::memcpy(y, y_input, n_obs * sizeof(double));

   // too much in the construtor! It should just set thing not make all sorts of calculations!
   LOG_INFO("QR_Rls constructor called with dim: " << dim << " and n_batch: " << n_batch);
   I = new double[dim * dim]();
   for (int idx = 0; idx < dim; idx++)
   {
      I[idx * dim + idx] = 1;
   }

   // Forgetting factor matrix
   double *B = new double[dim * dim]();
   for (int i = 0; i < dim; i++)
   {
      B[i * dim + i] = pow(ff, dim - i - 1);
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

   i = 1;

   delete[] B;
}

// Update method
void QR_Rls::update(double *new_x, double &new_y)
{
   // LOG_INFO("QR_Rls::update start");
   // LOG_DEBUG("New data size new_x :" << sizeof(new_x) << " new_y: " << sizeof(new_y));
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
      R_inv[i] *= (1.0 / ff);
      R[i] *= ff;
   }

   double d[n_obs];
   double c[dim];

   cblas_dgemv(CblasColMajor, CblasTrans,
               dim, n_obs, 1.0, R_inv, dim, new_x, 1, 0.0, d, 1);

   double *I_P_R = new double[dim * dim]();
   memcpy(I_P_R, I, dim * dim * sizeof(double));
   cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
               dim, dim, n_obs, -1.0, R_inv, dim, R, n_obs, 1.0, I_P_R, dim);
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               dim, dim, 1.0, I_P_R, dim, new_x, 1, 0.0, c, 1);
   delete[] I_P_R;

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

   //  w computation
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               dim, n_obs, 1.0, R_inv, dim, z, 1, 0.0, w, 1);

   r_c_size = n_obs * dim;

   double *P_copy = new double[r_c_size];
   cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
               dim, n_obs, n_obs, 1.0, R_inv, dim, G, n_obs, 0.0, P_copy, dim);
   std::memcpy(R_inv, P_copy, n_obs * dim * sizeof(double));
   delete[] P_copy;

   double temp_z[n_obs];
   cblas_dgemv(CblasColMajor, CblasTrans,
               n_obs, n_obs, 1.0, G, n_obs, z, 1, 0.0, temp_z, 1);
   std::memcpy(z, temp_z, n_obs * sizeof(double));
   i++;

   // G will be allocated in givens_update
   if (G != nullptr)
   {
      delete[] G;
      G = nullptr;
   }

   double sol_R_inv[n_obs * dim];
   double sol_R[n_obs * dim];
   for (i = 0; i < n_obs * dim; i++)
   {
      sol_R_inv[i] = R_inv[i];
      sol_R[i] = R[i];
      // sol_R_inv[i] = 0;
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
   // LOG_DEBUG("downdate just after givens::downdate call");
   // LOG_DEBUG("downdate just before second cblas call");
   double *result = new double[r_c_size];
   // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
   //             dim, n_obs, n_obs, 1, R_inv, dim, G, n_obs, 0, result, dim);
   //  LOG_DEBUG("downdate just after second cblas call");
   // std::memcpy(R_inv, result, n_obs * dim * sizeof(double));

   LOG_DEBUG("downdate just before third cblas call");
   cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
               n_obs, dim, n_obs, 1, G, n_obs, R, n_obs, 0, result, n_obs);
   std::memcpy(R, result, n_obs * dim * sizeof(double));
   LOG_DEBUG("downdate just after third cblas call");

   LOG_DEBUG("downdate just before memory deallocation");
   delete[] result; // Free memory

   double x_T[dim];
   for (int i = 0; i < dim; ++i)
   {
      // x_T[i] = R[n_obs * i]; // Copy the first row=
      x_T[i] = X[n_obs * i]; // Copy the first row=
   }

   double tempi_x_T[dim];
   for (int i = 0; i < dim; ++i)
   {
      tempi_x_T[i] = X[n_obs * i]; // Copy the first row=
   }

   double jjee = 0;
   for (i = 0; i < n_obs * dim; i++)
   {
      jjee += abs(x_T[i] - tempi_x_T[i]);
   }

   double c[n_obs] = {0}; // initializes to zero
   c[0] = 1.0;

   double R_inv_pepin[dim * dim];

   // Deletion for new regime
   if (n_obs < dim)
   {
      // Safety check for allocation size
      if (dim <= 0 || dim > 1E+5)
      { // adjust maximum as needed
         LOG_ERROR("Invalid dimension for allocation: " << dim);
         throw std::runtime_error("Invalid dimension in downdate");
      }
      double k[dim];
      double h[n_obs];
      LOG_DEBUG("downdate just before fourth and fifth cblas call");
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim, n_obs, 1.0, R_inv, dim, c, 1, 0.0, k, 1);
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim, n_obs, 1.0, R_inv, dim, x_T, 1, 0.0, h, 1);

      double k_inv[dim];
      double h_inv[n_obs];
      pinv(k, k_inv, dim, 1);
      pinv(h, h_inv, 1, n_obs);

      double temp_vec[n_obs];
      LOG_DEBUG("downdate just before sixth cblas call");
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim, n_obs, 1.0, R_inv, dim, k_inv, 1, 0.0, temp_vec, 1);
      double s = cblas_ddot(n_obs, temp_vec, 1, h_inv, 1);

      // 1
      double *k_k_inv = new double[dim * dim]();
      double *R_inv_temp = new double[dim * n_obs]();
      LOG_DEBUG("downdate just before seventh and eighth cblas call");
      cblas_dger(CblasColMajor, dim, dim, 1.0, k, 1, k_inv, 1, k_k_inv, dim);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                  dim, n_obs, dim, 1.0, k_k_inv, dim, R_inv, dim, 0.0, R_inv_temp, dim);
      delete[] k_k_inv;

      // 2
      double P_h_inv[dim];
      // LOG_DEBUG("downdate just before ninth and tenth and eleventh cblas call");
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim, n_obs, 1.0, R_inv, dim, h_inv, 1, 0.0, P_h_inv, 1);
      cblas_dger(CblasColMajor, dim, n_obs, -1.0, P_h_inv, 1, h, 1, R_inv, dim);
      //  3
      cblas_dger(CblasColMajor, dim, n_obs, s, k, 1, h, 1, R_inv, dim);

      for (int i = 0; i < dim * n_obs; ++i)
      {
         R_inv[i] -= R_inv_temp[i];
      }

      delete[] R_inv_temp;
   }
   // Deletion for old regime
   else
   {
      double x_neg_T[dim];
      for (int i = 0; i < dim; i++)
      {
         x_neg_T[i] = -1 * x_T[i];
      }

      double G_e_1[n_obs];
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  n_obs, n_obs, 1.0, G, n_obs, c, 1, 0.0, G_e_1, 1);

      double h[dim] = {0};
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim, n_obs, 1.0, R_inv, dim, G_e_1, 1, 0.0, h, 1);
      
      double k[n_obs] = {0};
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim, n_obs, 1.0, R_inv, dim, x_neg_T, 1, 0.0, k, 1);

      double s = 1 + cblas_ddot(n_obs, k, 1, G_e_1, 1);
      cblas_dger(CblasColMajor, dim, n_obs, -1.0/s, h, 1, k, 1, R_inv, dim);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                dim, n_obs, n_obs, 1, R_inv, dim, G, n_obs, 0, result, dim);
      std::memcpy(R_inv, result, n_obs * dim * sizeof(double));

   }
   LOG_DEBUG("downdate just before deleteRows ");

   R = deleteRowColMajor(R, n_obs, dim);
   R_inv = deleteColColMajor(R_inv, dim, n_obs);
   all_Q = deleteRowColMajor(all_Q, n_obs, n_obs);
   all_Q = deleteColColMajor(all_Q, n_obs - 1, n_obs);
   X = deleteRowColMajor(X, n_obs, dim);
   y = deleteRowColMajor(y, n_obs, 1);
   // delete thius
   double anotha_z[n_obs];
   std::memcpy(anotha_z, z, n_obs * sizeof(double));
   z = deleteRowColMajor(z, n_obs, 1);
   n_obs--;
   r_c_size = n_obs * dim;

   double tol = 1e-12;
   for (int i = 0; i < n_obs * dim; ++i)
   {
      if (fabs(R_inv[i]) < tol)
      {
         R_inv[i] = 0;
      }
   }

   // compare both
   // double *temp_R_inv = new double[r_c_size];
   double temp_R_inv[r_c_size];
   double je = 0;
   pinv(R, temp_R_inv, X_rows, dim);

   //double *result = new double[r_c_size];
   

   for (i = 0; i < n_obs * dim; i++)
   {
      je += abs(temp_R_inv[i] - R_inv[i]);
   }

   LOG_DEBUG("downdate just before cblas call");
   cblas_dgemv(CblasColMajor, CblasTrans, n_obs, n_obs, 1.0, all_Q, n_obs, y, 1, 0.0, z, 1);
   double temp_z[n_obs + 1];
   cblas_dgemv(CblasColMajor, CblasTrans,
               n_obs + 1, n_obs + 1, 1.0, G, n_obs + 1, anotha_z, 1, 0.0, temp_z, 1);
   double chacha[n_obs];
   std::memcpy(chacha, z, n_obs * sizeof(double));
   // z = deleteRowColMajor(z, n_obs, 1);
   cblas_dgemv(CblasColMajor, CblasNoTrans, dim, n_obs, 1.0, R_inv, dim, z, 1, 0.0, w, 1);

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
