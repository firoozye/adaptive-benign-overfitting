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

struct GivensRot
{
   int j1; // first column index (or row index, depending on where applied)
   int j2; // second column index
   double c;
   double s;
};

class ABO
{
public:
   // Declare destructor in header, define in cpp file
   ABO(double *X_batch, double *y_batch, int max_obs, double ff, int dim, int X_rows);
   ~ABO();
   void batchInitialize();
   void update(double *new_x, double &new_y);
   void downdate();
   double pred(double *x);
   double get_cond_num();
   double get_real_cond_num();

   // Input data
   double *X_; // Input matrix
   double *y_; // Target vector

   // Matrices for QR decomposition
   double *G_;     // Givens rotation matrix
   double *R_;     // R matrix from QR decomposition
   double *R_inv_; // Inverse of R matrix
   double *Q_;     // Q matrix from QR decomposition
   double *z_;     // Intermediate vector
   double *beta_;  // Weight vector
   double *G_e_1_;
   std::vector<GivensRot> giv_rots;

   // hyperparameters
   int max_obs_;  // Maximum number of observations
   int r_c_size_; // Size of R and C matrices
   int n_obs_;    // Current number of observations
   int dim_;      // Dimension of input
   double ff_;
   double sqrt_ff_; // Sqrt Forgetting factor
};
