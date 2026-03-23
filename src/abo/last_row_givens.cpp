#include <algorithm> // std::fill
#include <cmath>
#include <cstring>
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
      int n_obs   = abo->n_obs_;   // current size AFTER n_obs_++ in caller
      int dim     = abo->dim_;
      int max_obs = abo->max_obs_;

      double *R     = abo->R_;
      double *R_inv = abo->R_inv_;
      double *Q     = abo->Q_;

      int col_stride = max_obs;        // R_ column stride
      int q_stride   = max_obs + 1;    // Q_ column stride
      int last = n_obs - 1;            // index of the newly added row

      int limit = std::min(n_obs - 1, dim);
      int row_stride = 1;

      for (int j = 0; j < limit; ++j)
      {
         // Zero R(last, j) using Givens on rows j and last
         dlartg_(&R[j + j * col_stride],
                 &R[last + j * col_stride],
                 &c, &s, &r);

         R[j + j * col_stride]    = r;
         R[last + j * col_stride] = 0.0;

         // Apply rotation to remaining columns j+1..dim-1 of rows j and last
         int temp  = dim - j - 1;
         int idx_1 = j    + (j + 1) * col_stride;
         int idx_2 = last + (j + 1) * col_stride;

         drot_(&temp, &R[idx_1], &col_stride,
               &R[idx_2], &col_stride,
               &c, &s);

         // Apply same rotation to Q_ columns j and last (all n_obs rows)
         drot_(&n_obs, &Q[j * q_stride], &row_stride,
               &Q[last * q_stride], &row_stride, &c, &s);

         // Apply same rotation to R_inv_ columns j and last
         int inc = 1;
         drot_(&dim, &R_inv[j * dim], &inc, &R_inv[last * dim], &inc, &c, &s);
      }
   }

   // Downdate: for old regime (n_obs >= dim), uses Q_ directly (first row).
   //           for new regime (n_obs < dim), uses h = R_inv^T * z_old (Q-less, correct).
   void downdate(ABO *abo, double *z_old)
   {
      int n_obs   = abo->n_obs_;   // current size BEFORE n_obs_-- in caller
      int dim     = abo->dim_;
      int max_obs = abo->max_obs_;

      double *R     = abo->R_;
      double *G     = abo->G_;
      double *R_inv = abo->R_inv_;
      double *Q     = abo->Q_;
      double *h     = abo->scratch_n_;  // n_obs-length working vector

      int q_stride   = max_obs + 1;
      int row_stride = 1;

      // Step 1: populate h
      if (dim > n_obs)
      {
         // New regime: h = R_inv^T * z_old  (mathematically equals Q[0,:] here)
         cblas_dgemv(CblasColMajor, CblasTrans,
                     dim, n_obs, 1.0,
                     R_inv, dim, z_old, 1, 0.0, h, 1);
      }
      else
      {
         // Old regime: use first row of Q_ directly (col-major, stride q_stride)
         for (int j = 0; j < n_obs; j++)
            h[j] = Q[j * q_stride];   // Q_[row=0, col=j]
      }

      // Step 2: G = I
      std::fill(G, G + n_obs * n_obs, 0.0);
      for (int i = 0; i < n_obs; i++)
         G[i * n_obs + i] = 1.0;

      double c, s, r;
      int one = 1;

      // Step 3: Givens rotations to zero h[1..n_obs-1] bottom-up
      for (int i = n_obs - 1; i > 0; --i)
      {
         dlartg_(&h[i - 1], &h[i], &c, &s, &r);
         h[i - 1] = r;
         h[i]     = 0.0;

         // Rotate columns i-1 and i of G
         drot_(&n_obs, &G[(i - 1) * n_obs], &one,
               &G[i * n_obs],       &one,
               &c, &s);

         // Also rotate columns i-1 and i of Q_ (all n_obs rows)
         drot_(&n_obs, &Q[(i - 1) * q_stride], &row_stride,
               &Q[i * q_stride],      &row_stride,
               &c, &s);

         if (dim > n_obs)
         {
            // New regime: rotate rows i-1 and i of R (stride max_obs)
            int n   = dim - i + 1;
            int inc = max_obs;
            drot_(&n,
                  &R[(i - 1) * max_obs + (i - 1)], &inc,
                  &R[(i - 1) * max_obs +  i      ], &inc,
                  &c, &s);

            abo->giv_rots.push_back({(i - 1) * dim, i * dim, c, s});
         }
      }

      // Step 4: sign fix and finalize
      if (dim > n_obs)
      {
         // New regime: record first column of G
         for (int t = 0; t < n_obs; t++)
            abo->G_e_1_[t] = G[t];
      }
      else
      {
         // Old regime: sign fix using Q_[0,0] after rotations (== h[0] now)
         if (Q[0] < 0)
         {
            for (int i = 0; i < n_obs * n_obs; ++i)
               G[i] *= -1;
            // Also negate Q_ columns (apply sign flip consistently)
            for (int j = 0; j < n_obs; j++)
               for (int row = 0; row < n_obs; row++)
                  Q[j * q_stride + row] *= -1;
         }

         for (int t = 0; t < n_obs; t++)
            abo->G_e_1_[t] = G[t];

         // Apply G^T * R, writing result back into R_.
         for (int j = 0; j < dim; j++)
            std::memcpy(&abo->scratch_d_[j * n_obs], &R[j * max_obs],
                        n_obs * sizeof(double));

         cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                     n_obs, dim, n_obs, 1.0,
                     G, n_obs,
                     abo->scratch_d_, n_obs,
                     0.0, abo->scratch_d2_, n_obs);

         for (int j = 0; j < dim; j++)
            std::memcpy(&R[j * max_obs], &abo->scratch_d2_[j * n_obs],
                        n_obs * sizeof(double));
      }
   }

} // namespace givens
