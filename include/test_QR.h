
#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <iostream>

#include "read_csv_func.h"
// #include "QR_RLS.cpp"
#include "QR_RLS.h"
#include "gau_rff.h"
using namespace std;

Eigen::MatrixXd lag_matrix(const std::vector<double> &x, int lag);
int main();
