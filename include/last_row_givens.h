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

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "QR_RLS.h"

class QR_Rls;
namespace givens
{
    void update(QR_Rls *qr_rls);
    void downdate(QR_Rls *qr_rls);
} // namespace givens
