#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

#include <lapacke.h>

#ifdef __cplusplus
}
#endif

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <utility>

#include "last_row_givens.h"

// Function declaration only in header
std::pair<double *, double *> Q_R_compute(double *A, int m, int n);
