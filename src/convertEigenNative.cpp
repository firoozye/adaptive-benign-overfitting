// just an examle of 3 sparate ways to convert
/* Eigen::MatrixXd matrix(3, 3);
// Option 1: Get pointer to data (const)
const double* data1 = matrix.data();

// Option 2: Get pointer to data (mutable)
double* data2 = matrix.data();

// Option 3: Copy to new array
double* data3 = new double[matrix.size()];
std::memcpy(data3, matrix.data(), matrix.size() * sizeof(double));
*/
