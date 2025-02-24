
#include <cmath>
#include <iostream>
#include <vector>

#include "QR_RLS.h"
#include "gau_rff.h"
#include "read_csv_func.h"
double* create_lag_matrix(const std::vector<double>& data, int n_samples, int n_lags);
int main();
