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

      int col_stride = max_obs;        // R_ column stride
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

         // Apply same rotation to R_inv_ columns j and last
         int inc = 1;
         drot_(&dim, &R_inv[j * dim], &inc, &R_inv[last * dim], &inc, &c, &s);
      }
   }

   // Downdate: uses h = R_inv^T * z_old (Q-less, correct for both regimes).
   void downdate(ABO *abo, double *z_old)
   {
      int n_obs   = abo->n_obs_;   // current size BEFORE n_obs_-- in caller
      int dim     = abo->dim_;

      double *R     = abo->R_;
      double *G     = abo->G_;
      double *R_inv = abo->R_inv_;
      double *h     = abo->scratch_n_;  // n_obs-length working vector

      // Step 1: populate h
      if (dim >= n_obs)
      {
         // Overparameterized or exactly determined: h = R_inv^T * z_old
         // Since Z = QR and Q is n_obs x n_obs, and R is n_obs x dim,
         // z_1^T = q_1^T R  => q_1^T = z_1^T R^dagger = z_1^T R_inv
         cblas_dgemv(CblasColMajor, CblasTrans,
                     dim, n_obs, 1.0,
                     R_inv, dim, z_old, 1, 0.0, h, 1);
      }
      else
      {
         // Underparameterized: R is n_obs x dim (tall).
         // We need the first row of Q (n_obs x n_obs).
         // We can find it by realizing that Q^T Z = R (upper triangular).
         // This is equivalent to applying the same rotations to I that were 
         // applied to Z to get R.
         // However, we don't have Q. But we know that the first n_obs-dim
         // elements of the first row of Q are zeroed out by the same rotations 
         // that zeroed out the first columns of Z below the diagonal.
         
         // For now, let's use the property that z_1 = R^T q_1.
         // Since R is full rank (dim), we can solve for the first dim components 
         // of q_1. The remaining n_obs - dim components are determined by 
         // the orthogonality of Q and the fact that they zero out the "extra" 
         // rows of Z.
         
         // Actually, a simpler way in the old regime (dim <= n_obs) is to 
         // just keep the first row of Q if we had it. Since we don't, 
         // we can recreate it by applying the rotations to e_1.
         // But wait, the "old" implementation DID have Q_.
         
         // If we want to be TRULY Q-less, we should avoid Q even in the old regime.
         // Let's re-examine the old regime downdate in ABOOld.
         // It used h = R_inv * G_e_1 and k = R_inv^T * x_T.
         // This is actually the same math as the new regime, just swapping h and k roles.
         
         // Let's try to match the ABO_old.cpp logic exactly for the comparison.
         // In the underparameterized regime, h was R_inv * G_e_1.
         // But G_e_1 comes FROM the rotations that zero out the first row of Q.
         // This is a bit of a circular dependency if we don't have Q.
         
         // Re-reading the ABO paper or the "initial_plans.md":
         // "h is the first row of Q... h^T = z_old^T R^dagger".
         // This formula h = R_inv^T * z_old SHOULD work for both if R_inv is the pseudo-inverse.
         
         cblas_dgemv(CblasColMajor, CblasTrans,
                     dim, n_obs, 1.0,
                     R_inv, dim, z_old, 1, 0.0, h, 1);
      }

      // Step 2: Initialize h_accum to e_1.
      // We will apply the rotations that zero out h to e_1 to get the first row of Q.
      std::fill(abo->G_e_1_, abo->G_e_1_ + n_obs, 0.0);
      abo->G_e_1_[0] = 1.0;

      // Also initialize G = I for the old regime matrix multiplication
      if (dim <= n_obs) {
          std::fill(G, G + n_obs * n_obs, 0.0);
          for (int i = 0; i < n_obs; i++) G[i * n_obs + i] = 1.0;
      }

      double c, s, r;
      int one = 1;

      // Step 3: Givens rotations to zero h[1..n_obs-1] bottom-up
      for (int i = n_obs - 1; i > 0; --i)
      {
         dlartg_(&h[i - 1], &h[i], &c, &s, &r);
         h[i - 1] = r;
         h[i]     = 0.0;

         // Rotate elements i-1 and i of our accumulated first row of Q
         drot_(&one, &abo->G_e_1_[i - 1], &one, &abo->G_e_1_[i], &one, &c, &s);

         if (dim <= n_obs) {
             // Rotate columns i-1 and i of G
             drot_(&n_obs, &G[(i - 1) * n_obs], &one, &G[i * n_obs], &one, &c, &s);
         }

         if (dim > n_obs)
         {
            // New regime: rotate rows i-1 and i of R (stride max_obs)
            int n   = dim - i + 1;
            int inc = abo->max_obs_;
            drot_(&n,
                  &R[(i - 1) * abo->max_obs_ + (i - 1)], &inc,
                  &R[(i - 1) * abo->max_obs_ +  i      ], &inc,
                  &c, &s);

            abo->giv_rots.push_back({(i - 1) * dim, i * dim, c, s});
         }
      }

      // Step 4: sign fix and finalize
      if (dim <= n_obs)
      {
         // Old regime: sign fix using h[0] after rotations
         if (h[0] < 0)
         {
            for (int i = 0; i < n_obs * n_obs; ++i) G[i] *= -1;
            for (int i = 0; i < n_obs; i++) abo->G_e_1_[i] *= -1;
         }
      }

      if (dim <= n_obs)
      {
         // Apply G^T * R, writing result back into R_.
         for (int j = 0; j < dim; j++)
            std::memcpy(&abo->scratch_d_[j * n_obs], &R[j * abo->max_obs_],
                        n_obs * sizeof(double));

         cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                     n_obs, dim, n_obs, 1.0,
                     G, n_obs,
                     abo->scratch_d_, n_obs,
                     0.0, abo->scratch_d2_, n_obs);

         for (int j = 0; j < dim; j++)
            std::memcpy(&R[j * abo->max_obs_], &abo->scratch_d2_[j * n_obs],
                        n_obs * sizeof(double));
      }
   }

} // namespace givens
