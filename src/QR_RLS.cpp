#include "QR_RLS.h"

using namespace std;

QR_Rls::~QR_Rls() {
   LOG_INFO("QR_Rls destructor called");
   delete[] R;
   delete[] R_inv;
   delete[] all_Q;
   delete[] I;
   delete[] z;
   delete[] w;
   delete[] G;
   delete[] temp_z;
}

QR_Rls::QR_Rls(double *x, double *y, int max_obs, double ff, double l, int dim, int n_batch)
    : X(x), y(y),  // input data
                   // matrices for QR decomposition
      G(nullptr),
      R(nullptr),
      R_inv(nullptr),
      all_Q(nullptr),
      I(nullptr),
      z(nullptr),
      w(nullptr),
      temp_z(nullptr),
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
      i(1) {
   // too much in the construtor! It should just set thing not make all sorts of calculations!
   LOG_INFO("QR_Rls constructor called with dim: " << dim << " and n_batch: " << n_batch);
   I = new double[dim * dim]();
   for (int idx = 0; idx < dim; idx++) {
      I[idx * dim + idx] = 1;
   }

   // Forgetting factor matrix
   double *B = new double[dim * dim];
   for (int i = 0; i < dim * dim; i++) {
      B[i] = 0.0;
   }
   for (int i = 0; i < dim; i++) {
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

   z = new double[n_obs];
   w = new double[dim];
   temp_z = new double[n_obs];

   cblas_dgemv(CblasColMajor, CblasTrans,
               X_rows, X_rows, 1.0, all_Q, X_rows, y, 1, 0.0, z, 1);
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               dim, X_rows, 1.0, R_inv, dim, z, 1, 0.0, w, 1);

   i = 1;

   delete[] B;
}

// Update method
void QR_Rls::update(double *new_x, double &new_y) {
   LOG_INFO("QR_Rls update called");
   X = addRowColMajor(X, n_obs, dim);
   for (int i = 0; i < dim; ++i) {
      X[(i + 1) * n_obs + i] = new_x[i];
   }

   y = addRowColMajor(y, n_obs, 1);
   z = addRowColMajor(z, n_obs, 1);
   y[n_obs] = new_y;
   z[n_obs] = new_y;

   for (int i = 0; i < r_c_size; ++i) {
      R_inv[i] *= (1.0 / ff);
      R[i] *= ff;
   }

   double *d = new double[n_obs];
   double *c = new double[dim];

   cblas_dgemv(CblasColMajor, CblasTrans,
               dim, n_obs, 1.0, R_inv, dim, new_x, 1, 0.0, d, 1);

   double *I_P_R = new double[dim * dim]();
   memcpy(I_P_R, I, dim * dim * sizeof(double));
   cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
               dim, dim, n_obs, -1.0, R_inv, dim, R, n_obs, 1.0, I_P_R, dim);
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               dim, dim, 1.0, I_P_R, dim, new_x, 1, 0.0, c, 1);
   delete[] I_P_R;

   double tolerance = 1e-10;
   int is_non_zero = 0;
   for (int i = 0; i < dim; i++) {
      if (fabs(c[i]) > tolerance) {
         is_non_zero = 1;
         break;
      }
   }

   // Update for new regime
   if (is_non_zero) {
      double *c_inv = new double[dim];
      pinv(c, c_inv, dim, 1);
      cblas_dger(CblasColMajor, dim, n_obs, -1.0, c_inv, 1, d, 1, R_inv, dim);
      R_inv = addColColMajor(R_inv, dim, n_obs);

      for (int i = 0; i < dim; ++i) {
         R_inv[n_obs * dim + i] = c_inv[i];
      }
   }
   // Update for old regime
   else {
      // VectorXd b_k = (1.0 / (1.0 + d.dot(d))) * P * d.transpose();
      // P = MatrixXd::Zero(P.rows(), P.cols() + 1);
      // P.leftCols(P.cols() - 1) = P - b_k * d.transpose();
      // P.col(P.cols() - 1) = b_k; // Assuming last column is an identity
   }

   R = addRowColMajor(R, n_obs, dim);

   for (int i = 0; i < dim; ++i) {
      R[(i + 1) * n_obs + i] = new_x[i];
   }

   all_Q = addRowAndColumnColMajor(all_Q, n_obs, n_obs);
   all_Q[((n_obs + 1) * (n_obs + 1)) - 1] = 1;

   n_obs++;

   // Reallocate temp_z for the new size
   if (temp_z != nullptr) {
      delete[] temp_z;
   }
   temp_z = new double[n_obs];

   // G will be allocated in givens_update
   if (G != nullptr) {
      delete[] G;
      G = nullptr;
   }
   givens::update(this);
   // w computation
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               dim, n_obs, 1.0, R_inv, dim, z, 1, 0.0, w, 1);

   // Cleanup
   delete[] d;
   delete[] c;

   r_c_size = n_obs * dim;
   double *P_copy = new double[r_c_size];
   cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
               dim, n_obs, n_obs, 1.0, R_inv, dim, G, n_obs, 0.0, P_copy, dim);
   std::memcpy(R_inv, P_copy, n_obs * dim * sizeof(double));
   delete[] P_copy;

   cblas_dgemv(CblasColMajor, CblasTrans,
               n_obs, n_obs, 1.0, G, n_obs, z, 1, 0.0, temp_z, 1);
   std::memcpy(z, temp_z, n_obs * sizeof(double));
   i++;

   delete[] G;

   if (n_obs > max_obs) {
      downdate();
   }
   LOG_INFO("QR_Rls update finished");
}

void QR_Rls::downdate() {
   LOG_INFO("QR_Rls downdate called");
   // Check the condition
   double *temp = new double[dim * dim];
   // A.transpose() * P.transpose()
   cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans,
               dim, dim, n_obs, 1, R, n_obs, R_inv, dim, 0, temp, dim);
   bool bool_st = true;
   double tolerance = 1e-10;
   for (int i = 0; i < dim * dim; ++i) {
      if (fabs(temp[i] - I[i]) > tolerance) {
         bool_st = 0;
         break;
      }
   }

   delete[] temp;

   // Update matrices
   // G will be allocated in givens_downdate
   if (G != nullptr) {
      delete[] G;
      G = nullptr;
   }
   givens::downdate(this);

   double *result = new double[r_c_size];
   cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
               dim, n_obs, n_obs, 1, R_inv, dim, G, n_obs, 0, result, dim);
   std::memcpy(R_inv, result, n_obs * dim * sizeof(double));

   cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
               n_obs, dim, n_obs, 1, G, n_obs, R, n_obs, 0, result, n_obs);
   std::memcpy(R, result, n_obs * dim * sizeof(double));

   delete[] result;  // Free memory
   delete[] G;

   double *x_T = new double[dim];
   for (int i = 0; i < dim; ++i) {
      x_T[i] = R[n_obs * i];  // Copy the first row
   }

   double *c = new double[n_obs]();  // () initializes to zero
   c[0] = 1.0;

   // Deletion for new regime
   if (!bool_st) {
      double *k = new double[dim];
      double *h = new double[n_obs];
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim, n_obs, 1.0, R_inv, dim, c, 1, 0.0, k, 1);
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim, n_obs, 1.0, R_inv, dim, x_T, 1, 0.0, h, 1);

      double *k_inv = new double[dim];
      double *h_inv = new double[n_obs];
      pinv(k, k_inv, dim, 1);
      pinv(h, h_inv, 1, n_obs);

      double *temp_vec = new double[n_obs];
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim, n_obs, 1.0, R_inv, dim, k_inv, 1, 0.0, temp_vec, 1);
      double s = cblas_ddot(n_obs, temp_vec, 1, h_inv, 1);

      // 1
      double *k_k_inv = new double[dim * dim]();
      double *R_inv_temp = new double[dim * n_obs];
      cblas_dger(CblasColMajor, dim, dim, 1.0, k, 1, k_inv, 1, k_k_inv, dim);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                  dim, n_obs, dim, 1.0, k_k_inv, dim, R_inv, dim, 0.0, R_inv_temp, dim);
      delete[] k_k_inv;
      delete[] k;
      delete[] h;
      delete[] k_inv;
      delete[] h_inv;
      delete[] temp_vec;
      delete[] R_inv_temp;

      // 2
      double P_h_inv[dim];
      cblas_dgemv(CblasColMajor, CblasNoTrans,
                  dim, n_obs, 1.0, R_inv, dim, h_inv, 1, 0.0, P_h_inv, 1);
      cblas_dger(CblasColMajor, dim, n_obs, -1.0, P_h_inv, 1, h, 1, R_inv, dim);
      // 3
      cblas_dger(CblasColMajor, dim, n_obs, s, k, 1, h, 1, R_inv, dim);

      for (int i = 0; i < dim * n_obs; ++i) {
         R_inv[i] -= R_inv_temp[i];
      }
   }
   // Deletion for old regime
   else {
      // VectorXd x_neg_T = -x_T;
      // VectorXd h = x_neg_T * P;
      // VectorXd u = (MatrixXld::Identity(P.cols(), P.cols()) - A * P) * c;
      // VectorXd h_T = h.transpose();
      // VectorXd u_T = u.transpose();

      //  VectorXd k = P * c;
      //  double h_mag = h.dot(h_T);
      //  double u_mag = u_T.dot(u);
      //  double S = 1.0 + (x_neg_T * P * c)(0);
      //  MatrixXd p_2 = -((u_mag) / S * P * h_T) - k;
      //  MatrixXd q_2 = -((h_mag) / S * u.transpose() - h);
      //  double sigma_2 = h_mag * u_mag + S * S;
      //  P = P + (1 / S) * P * h.transpose() * u.transpose() - (S / sigma_2) * p_2 * q_2;
   }

   R = deleteRowColMajor(R, n_obs, dim);
   R_inv = deleteColColMajor(R_inv, dim, n_obs);
   all_Q = deleteRowColMajor(all_Q, n_obs, n_obs);
   all_Q = deleteColColMajor(all_Q, n_obs - 1, n_obs);
   X = deleteRowColMajor(X, n_obs, dim);
   y = deleteRowColMajor(y, n_obs, 1);
   z = deleteRowColMajor(z, n_obs, 1);

   n_obs--;
   r_c_size = n_obs * dim;

   cblas_dgemv(CblasColMajor, CblasTrans, n_obs, n_obs, 1.0, all_Q, n_obs, y, 1, 0.0, z, 1);
   cblas_dgemv(CblasColMajor, CblasNoTrans, dim, n_obs, 1.0, R_inv, dim, z, 1, 0.0, w, 1);

   // Cleanup
   // Cleanup
   delete[] x_T;
   delete[] c;
   LOG_INFO("QR_Rls downdate finished");
}

double QR_Rls::pred(double *x) {
   // cout << w << endl;
   // if (x.rows() == 1)
   //{
   double pred_value = cblas_ddot(dim, x, 1, w, 1);
   //}
   // else
   //{
   //     Eigen::MatrixXd pred_matrix = x.transpose() * w;
   //     pred_value = x.transpose() * self.w;
   // }
   return pred_value;
}
