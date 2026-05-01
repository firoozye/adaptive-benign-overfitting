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
#include "last_row_givens.h"
#include "logger.h"
#include "pseudo_inverse.h"

struct GivensRot
{
   int j1; // first column index in R_inv_
   int j2; // second column index in R_inv_
   double c;
   double s;
};

class ABO
{
public:
   ABO(double *X_batch, double *y_batch, int max_obs, double ff, int dim, int X_rows);
   ~ABO();
   void batchInitialize(double *x_input);
   void update(double *new_x, double &new_y);
   void downdate(double *z_old);
   double pred(double *x);
   double get_cond_num();

   // Core matrices — pre-allocated to max_obs_ capacity, never reallocated
   double *y_;      // max_obs_               target vector (scaled by ff)
   double *R_;      // max_obs_ * dim_        col-major, fixed col stride = max_obs_
   double *R_inv_;  // dim_  * max_obs_       col-major, fixed col stride = dim_
   double *beta_;   // dim_                   weight vector
   double *G_;      // (max_obs_+1)^2         Givens accumulation matrix
   double *G_e_1_;  // max_obs_+1             first column of G after downdate rotations
   std::vector<GivensRot> giv_rots;

   // Scratch buffers — eliminate all VLAs and hot-path heap allocations
   double *scratch_n_;   // max_obs_        n_obs_-length temporaries
   double *scratch_n2_;  // max(max_obs_,dim_)  second n_obs_-length region (or dim_ in new-regime downdate)
   double *scratch_dim_; // dim_            dim_-length temporaries
   double *scratch_d_;   // max_obs_ * dim_ compact R copy / large temp
   double *scratch_d2_;  // max_obs_ * dim_ dgemm output (no aliasing)

   // Hyperparameters
   int max_obs_;  // Maximum window size (allocation capacity)
   int n_obs_;    // Current number of observations
   int dim_;      // Feature dimension
   double ff_;
   double sqrt_ff_;
};
