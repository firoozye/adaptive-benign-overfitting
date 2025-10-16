#pragma once
#ifdef __cplusplus
extern "C"
{
#endif

#include <cblas.h>
#include <lapacke.h>

#ifdef __cplusplus
}
#endif

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "QR_decomposition.h"
#include "add_row_col.h"
#include "last_row_givens.h"
#include "logger.h"
#include "pseudo_inverse.h"

class QR_Rls
{
public:
   // Declare destructor in header, define in cpp file
   QR_Rls(double *x, double *y, int max_obs, double ff, double lambda, int dim, int X_rows);
   ~QR_Rls();
   void update(double *new_x, double &new_y);
   void downdate();
   double pred(double *x);
   double get_cond_num();

   // Input data
   double *X; // Input matrix
   double *y; // Target vector

   // Matrices for QR decomposition
   double *G;     // Givens rotation matrix
   double *R;     // R matrix from QR decomposition
   double *R_inv; // Inverse of R matrix
   double *all_Q; // Q matrix from QR decomposition
   double *I;     // Identity matrix
   double *z;     // Intermediate vector
   double *w;     // Weight vector

   // hyperparameters
   int max_obs;  // Maximum number of observations
   int X_rows;   // Number of rows in X matrix
   int r_c_size; // Size of R and C matrices
   double l;     // Lambda parameter
   int n_batch;  // Initial batch size
   int n_obs;    // Current number of observations
   int dim;      // Dimension of input
   double ff;    // Forgetting factor
   double b;     // Beta parameter
};
