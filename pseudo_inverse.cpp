#include <iostream>
#include <lapacke.h>
#include <cblas.h>

int pinv(double *A, double *P, const int int_m, const int int_n, double tolerance = 1e-16)
{
    // Sample matrix

    // it has to be a C array
    // LDA The leading dimension of array A
    // LDU The leading dimension of the array U

    double A_copy[int_m * int_n];
    std::memcpy(A_copy, A, int_m * int_n * sizeof(double));

    // Dimensions
    lapack_int m = int_m, n = int_n, lda = m;
    lapack_int ldu = m, ldvt = n;
    lapack_int info;

    bool row_vec = int_m == int_m * int_n;
    bool col_vec = int_n == int_m * int_n;

    if (row_vec || col_vec)
    {
        double norm;
        if (row_vec)
        {
            norm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', m, n, A_copy, m);
        }
        else
        {
            norm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, m, A_copy, n);
        }
        int vec_size = int_m * int_n;
        double chch[vec_size];

        for (int i = 0; i < vec_size; i++)
        {
            P[i] = A_copy[i] / (norm * norm);
        }

        return 0;
    }

    double *s = new double[n];
    double *u = new double[ldu * m];
    double *vt = new double[ldvt * n];
    info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'A', m, n, A_copy, lda, s, u, ldu, vt, ldvt);
    double atol = 0.0;
    double rtol = std::max<double>(m, n) * std::numeric_limits<double>::epsilon();

    // Find rank based on singular values
    double maxS = *std::max_element(s, s + m);
    double val = atol + maxS * rtol;
    int rank = 0;
    for (int i = 0; i < m; ++i)
    {
        if (s[i] > val)
            rank++;
    }

    for (int j = 0; j < rank; ++j)
    {
        for (int i = 0; i < m; ++i)
        {
            u[j * m + i] /= s[j];
        }
    }

    double sub_vt[rank * n];
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < rank; ++i)
        {
            sub_vt[j * rank + i] = vt[j * n + i];
        }
    }

    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans,
                n, m, rank, 1.0, sub_vt, rank, u, m, 0.0, P, n);

    delete[] s;
    delete[] vt;
    delete[] u;
    return 0;
}