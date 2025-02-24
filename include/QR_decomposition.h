#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <lapacke.h>

#ifdef __cplusplus
}
#endif

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <utility>

#include "QR_RLS.h"
#include "last_row_givens.h"
// Forward declare QR_Rls to avoid circular dependency
class QR_Rls;

// Function declaration only in header
std::pair<double*, double*> Q_R_compute(QR_Rls* qr_rls, double* A, int m, int n);
