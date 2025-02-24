#pragma once

// int pinv(double* A, double* A_inv, int m, int n, double tolerance = 1e-10);

#ifdef __cplusplus
extern "C" {
#endif

#include <cblas.h>
#include <lapacke.h>

#ifdef __cplusplus
}
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
int pinv(double *A, double *P, const int int_m, const int int_n, double tolerance = 2e-16);
