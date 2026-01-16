#include "abo/QR_decomposition.h"

std::pair<double *, double *> Q_R_compute(double *A, int m, int n)
{
   int k = std::min(m, n);
   int info;

   // Allocate memory for Q and R
   double *Q = new double[m * m]();      // Q is m x m
   double *R_temp = new double[m * n](); // R is m x n

   for (int idx = 0; idx < m; idx++)
   {
      Q[idx * m + idx] = 1;
   }

   // Create a copy of A for QR decomposition
   double *A_copy = new double[m * n];
   std::memcpy(A_copy, A, m * n * sizeof(double));

   // Array to store reflectors
   double tau[k];

   // Workspace query to determine optimal workspace size
   double work_query;
   info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n, A_copy, m, tau, &work_query, -1);
   int lwork = (int)work_query;

   // Allocate workspace
   double *workspace = new double[lwork];

   // Perform QR decomposition
   info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n, A_copy, m, tau, workspace, lwork);
   delete[] workspace;

   if (info != 0)
   {
      throw std::runtime_error("LAPACKE_dgeqrf failed");
   }

   // Extract the upper triangular part of A
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         double idx = j * m + i;
         if (j >= i)
         {
            R_temp[j * m + i] = A_copy[j * m + i]; // Copy upper triangular part
         }
         else
         {
            R_temp[j * m + i] = 0.0; // Set lower triangular part to 0
         }
      }
   }

   // Copy Q into all_Q
   if (m <= n)
   {
      int ldm = m;
      info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, ldm, k, A_copy, m, tau);

      if (info != 0)
      {
         throw std::runtime_error("LAPACKE_dorgqr failed");
      }

      for (int i = 0; i < ldm; i++)
      {
         for (int j = 0; j < ldm; j++)
         {
            Q[j * ldm + i] = A_copy[j * ldm + i];
         }
      }
   }
   else
   {
      int ldm = n;
      std::memcpy(Q, A_copy, m * n * sizeof(double));
      info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, ldm, k, Q, m, tau);

      if (info != 0)
      {
         throw std::runtime_error("LAPACKE_dorgqr failed");
      }

      info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'N', m, m - n, n, A_copy, m, tau, Q + n * m, m);
   }

   delete[] A_copy;

   return {Q, R_temp};
}
