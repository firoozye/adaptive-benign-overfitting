#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <random>

using namespace std;

class GaussianRFF
{
public:
   Eigen::MatrixXd A;
   Eigen::RowVectorXd b;
   int D;
   mutable std::mt19937 rng; // Random number generator

   GaussianRFF(int d, int D, double kernel_var, int seed = -1);
   Eigen::MatrixXd transform_matrix(const Eigen::MatrixXd &x);
   Eigen::MatrixXd transform(const Eigen::MatrixXd &x);
};
