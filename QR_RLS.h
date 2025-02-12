#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <string>

#ifndef QR_RLS_H // Check if QR_RLS_H is not defined
#define QR_RLS_H // Define QR_RLS_H

using namespace std;

class QR_Rls
{
public:
    double *X, *R, *R_inv, *w, *y, *z, *I, *all_Q, *G;
    int max_obs, dim, n_batch, i, n_obs, r_c_size;
    double ff, l, b;
    double *first_x;

    // QR_Rls(const MatrixXd &x, const VectorXd &y, int max_obs, double ff, double lambda);
    QR_Rls(double *x, double *y, int max_obs, double ff, double lambda,
           int dim, int X_rows);

    // Update method
    void update(double *new_x, double &new_y);

    void downdate();

    double pred(double *x);
};

#endif // QR_RLS_H  // End of the include guard