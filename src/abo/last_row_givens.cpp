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
      int n_obs = abo->n_obs_;  // current size AFTER n_obs_++ in caller
      int dim   = abo->dim_;
      int max_obs = abo->max_obs_;

      double *R     = abo->R_;
      double *R_inv = abo->R_inv_;

      int col_stride = max_obs;  // Bug 5 fix: was n_obs (old stride)
      int last = n_obs - 1;      // index of the newly added row

      int limit = std::min(n_obs - 1, dim);

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

         // Apply same rotation to R_inv_ columns j and last
         int inc = 1;
         drot_(&dim, &R_inv[j * dim], &inc, &R_inv[last * dim], &inc, &c, &s);
      }
   }

   // Q-less downdate: h = R_inv^T * z_old replaces reading the first row of Q.
   void downdate(ABO *abo, double *z_old)
   {
      int n_obs   = abo->n_obs_;  // current size BEFORE n_obs_-- in caller
      int dim     = abo->dim_;
      int max_obs = abo->max_obs_;

      double *R     = abo->R_;
      double *G     = abo->G_;
      double *R_inv = abo->R_inv_;
      double *h     = abo->scratch_n_;  // Bug 4: no heap alloc, reuse scratch

      // Step 1: h = R_inv^T * z_old  (replaces reading first row of Q)
      cblas_dgemv(CblasColMajor, CblasTrans,
                  dim, n_obs, 1.0,
                  R_inv, dim, z_old, 1, 0.0, h, 1);

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

         // Rotate columns i-1 and i of G (each column is n_obs elements, stride 1)
         drot_(&n_obs, &G[(i - 1) * n_obs], &one,
               &G[i * n_obs],       &one,
               &c, &s);

         if (dim > n_obs)
         {
            // Bug 5 fix: stride is max_obs (not n_obs)
            // Rotate rows i-1 and i of R, from column i-1 to dim-1
            int n   = dim - i + 1;
            int inc = max_obs;
            drot_(&n,
                  &R[(i - 1) * max_obs + (i - 1)], &inc,
                  &R[(i - 1) * max_obs +  i      ], &inc,
                  &c, &s);

            abo->giv_rots.push_back({(i - 1) * dim, i * dim, c, s});
         }
      }

      // Step 4: Finalize G_e_1_ and (in old regime) apply G^T to R
      if (dim > n_obs)
      {
         // New regime: record first column of G for use in ABO::downdate
         for (int t = 0; t < n_obs; t++)
            abo->G_e_1_[t] = G[t];
      }
      else
      {
         // Old regime: sign fix using h[0] (replaces Q[0] sign check)
         if (h[0] < 0)
            for (int i = 0; i < n_obs * n_obs; ++i)
               G[i] *= -1;

         for (int t = 0; t < n_obs; t++)
            abo->G_e_1_[t] = G[t];

         // Apply G^T * R, writing result back into R_.
         // R_ has col stride max_obs (not contiguous), so copy to compact scratch first.
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
      // No delete[] h — it is scratch_n_, owned by ABO
   }

} // namespace givens
