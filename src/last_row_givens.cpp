#include "last_row_givens.h"
extern "C" {
void drotg_(double *a, double *b, double *c, double *s);
void dlartg_(double *a, double *b, double *c, double *s, double *r);
void drot_(int *n, double *dx, int *incx, double *dy, int *incy,
           double *c, double *s);
void dswap_(int *n, double *dx, int *incx, double *dy, int *incy);
}
class QR_Rls;

namespace givens {
void update(QR_Rls *qr_rls) {
   double c, s, r;
   int n_obs = qr_rls->n_obs;
   int dim = qr_rls->dim;
   qr_rls->G = new double[n_obs * n_obs]();
   double *G = qr_rls->G;

   for (int i = 0; i < n_obs; i++) {
      G[i * n_obs + i] = 1;
   }

   double *R = qr_rls->R;
   double *all_Q = qr_rls->all_Q;
   int limit = std::min(n_obs - 1, dim);

   for (int j = 0; j < limit; ++j) {
      int row_stride = 1;
      int col_stride = n_obs;

      dlartg_(&R[j * row_stride + j * col_stride], &R[(n_obs - 1) * row_stride + j * col_stride], &c, &s, &r);

      R[j * row_stride + j * col_stride] = r;
      R[(n_obs - 1) * row_stride + j * col_stride] = 0;

      int temp = dim - j - 1;
      int idx_1 = j * row_stride + (j + 1) * col_stride;
      int idx_2 = (n_obs - 1) * row_stride + (j + 1) * col_stride;
      double *ptr_1 = &R[idx_1];
      double *ptr_2 = &R[idx_2];

      drot_(&temp, &R[idx_1], &col_stride,
            &R[idx_2], &col_stride, &c, &s);

      idx_1 = j * col_stride;
      idx_2 = (n_obs - 1) * col_stride;

      drot_(&n_obs, &all_Q[j * col_stride], &row_stride,
            &all_Q[(n_obs - 1) * col_stride], &row_stride, &c, &s);

      drot_(&n_obs, &G[j * col_stride], &row_stride,
            &G[(n_obs - 1) * col_stride], &row_stride, &c, &s);
   }
}

void downdate(QR_Rls *qr_rls) {
   int n_obs = qr_rls->n_obs;
   int dim = qr_rls->dim;

   qr_rls->G = new double[n_obs * n_obs]();
   double *G = qr_rls->G;
   for (int i = 0; i < n_obs; i++) {
      double idx = i * n_obs + i;
      G[i * n_obs + i] = 1;
   }

   double *R = qr_rls->R;
   double *all_Q = qr_rls->all_Q;
   double c, s, r;
   int col_stride = n_obs;
   int row_stride = 1;

   for (int i = n_obs - 1; i > 0; --i) {
      // Generate Givens rotation

      dlartg_(&all_Q[(i - 1) * col_stride], &all_Q[i * col_stride], &c, &s, &r);

      // Update W
      int idx_1 = (i - 1) * col_stride;
      int idx_2 = i * col_stride;

      all_Q[(i - 1) * col_stride] = r;
      all_Q[i * col_stride] = 0;

      int temp = n_obs - 1;
      drot_(&temp, &all_Q[((i - 1) * col_stride) + 1], &row_stride,
            &all_Q[(i * col_stride) + 1], &row_stride, &c, &s);

      temp = dim - i - 1;

      idx_1 = (i - 1) + (i - 1) * col_stride;
      idx_2 = i + (i - 1) * col_stride;

      drot_(&n_obs, &G[((i - 1) * col_stride)], &row_stride,
            &G[(i * col_stride)], &row_stride, &c, &s);
   }

   if (all_Q[0] < 0) {
      for (int i = 0; i < n_obs * n_obs; ++i) {
         all_Q[i] *= -1;
         G[i] *= -1;
      }
   }
}
}  // namespace givens
