#include <algorithm> // std::fill
#include <cmath>
#include "abo/last_row_givens.h"

extern "C"
{
   void dlartg_(double *a, double *b, double *c, double *s, double *r);
   void drot_(int *n, double *dx, int *incx, double *dy, int *incy,
              double *c, double *s);
}

namespace givens
{

   void update(ABO *abo)
   {
      double c, s, r;
      int n_obs = abo->n_obs_; // current size BEFORE increment in caller
      int dim = abo->dim_;

      double *R = abo->R_;
      double *Q = abo->Q_;
      double *R_inv = abo->R_inv_;

      int limit = std::min(n_obs - 1, dim);
      int row_stride = 1;
      int col_stride = n_obs;
      int last = n_obs - 1;

      for (int j = 0; j < limit; ++j)
      {
         // Zero R(last, j) using Givens on rows (j,last)
         dlartg_(&R[j + j * col_stride],
                 &R[last + j * col_stride],
                 &c, &s, &r);

         R[j + j * col_stride] = r;
         R[last + j * col_stride] = 0.0;

         // Apply to remaining part of those two rows in R
         int temp = dim - j - 1;
         int idx_1 = j + (j + 1) * col_stride;
         int idx_2 = last + (j + 1) * col_stride;

         drot_(&temp, &R[idx_1], &col_stride,
               &R[idx_2], &col_stride,
               &c, &s);

         // Apply same row-rotation to Q (rows j and last)
         int q1 = j * col_stride;
         int q2 = last * col_stride;
         drot_(&n_obs, &Q[q1], &row_stride,
               &Q[q2], &row_stride,
               &c, &s);

         int inc = 1;
         drot_(&dim, &R_inv[j * dim], &inc, &R_inv[last * dim], &inc, &c, &s);
      }
   }
   // DOWNDATE: identical to your old logic for Q (and sign fix),
   // PLUS: compute Ge1 and apply the implied effect on R_inv
   void downdate(ABO *abo)
   {
      int n_obs = abo->n_obs_; // current size BEFORE abo->n_obs_-- in caller
      int dim = abo->dim_;

      double *Q = abo->Q_;
      double *R = abo->R_;

      double G[n_obs * n_obs] = {0};
      for (int i = 0; i < n_obs; i++)
      {
         G[i * n_obs + i] = 1;
      }

      double c, s, r;
      int col_stride = n_obs;
      int row_stride = 1;
      int last = n_obs - 1;

      for (int i = n_obs - 1; i > 0; --i)
      {
         int idx_1 = (i - 1) * col_stride;
         int idx_2 = i * col_stride;

         dlartg_(&Q[idx_1], &Q[idx_2], &c, &s, &r);

         Q[idx_1] = r;
         Q[idx_2] = 0.0;

         int temp = n_obs - 1;
         drot_(&temp, &Q[idx_1 + 1], &row_stride,
               &Q[idx_2 + 1], &row_stride,
               &c, &s);

         drot_(&n_obs, &G[idx_1], &row_stride,
               &G[idx_2], &row_stride, &c, &s);

         int n = dim - i + 1; // rotate across all columns 0..dim-1
         int inc = n_obs;     // step to next column at same row (column-major)
         drot_(&n,
               &R[(i - 1) * n_obs + i - 1], &inc, // R(i-1, 0)
               &R[(i - 1) * n_obs + i], &inc,     // R(i,   0)
               &c, &s);

         inc = 1;
         abo->giv_rots.push_back({(i - 1) * dim, i * dim, c, s});
      }

      for (int t = 0; t < n_obs; t++)
         abo->G_e_1_[t] = G[t];
   }
} // namespace givens
