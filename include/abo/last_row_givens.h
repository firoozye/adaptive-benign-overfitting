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

#include "ABO.h"

class ABO;
struct GivensRot;
namespace givens
{
    void update(ABO *abo);
    void downdate(ABO *abo);
} // namespace givens
