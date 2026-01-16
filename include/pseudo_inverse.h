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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
int pinv(double *A, double *P, const int int_m, const int int_n, double tolerance = 2e-16);
