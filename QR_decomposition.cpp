#include <cmath>
#include <iostream>
#include <lapacke.h>
#include <Eigen/Dense>
#include "QR_RLS.h"

std::pair<double *, double *> Q_R_compute(QR_Rls *qr_rls, double *A, int m, int n)
{
    int k = std::min(m, n);
    int info;

    // Allocate memory for Q and R
    double *Q = new double[m * m];      // Q is m x m
    double *R_temp = new double[m * n]; // R is m x n

    // Create a copy of A for QR decomposition
    double A_copy[m * n];
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

    double pe[m * n];
    std::memcpy(pe, A_copy, m * n * sizeof(double));

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

    // Generate Q matrix using ORGQR
    info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, m, k, A_copy, m, tau);

    if (info != 0)
    {
        throw std::runtime_error("LAPACKE_dorgqr failed");
    }

    // Copy Q into all_Q
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            Q[j * m + i] = A_copy[j * m + i];
        }
    }

    return {Q, R_temp};
}
