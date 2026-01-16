#ifndef QRD_RLS_H
#define QRD_RLS_H

#include <cstddef>
#include <Eigen/Dense>
#include "add_row_col.h"

// QRD-RLS with Givens rotations (column-major, C-style arrays)
class QRDRLS
{
public:
    QRDRLS(int max_obs,
           int n_cols,
           double forgetting_factor = 1.0,
           double delta = 1e-2);

    ~QRDRLS();

    void batchInitialize(const double *X_batch,
                         const double *y_batch,
                         int batch_size,
                         int n_cols);

    void update(const double *new_x,
                double new_y,
                double &prediction,
                double &error);

    void downdate();

    void getCoefficients(double *w_out) const;

    size_t getFilterOrder() const { return N_; }
    bool isInitialized() const { return initialized_; }

    double pred(double *x);

    void reset();

private:
    int N_;
    double ff_;
    double sqrt_ff_;
    double delta_;
    int dim_;
    int n_obs_;
    int max_obs_;

    double *x;
    double *UT_;
    double *beta_;
    double *P_;
    double *R_;
    double *Q_;
    double *u;
    double *X_;
    double *y_;

    bool initialized_;

    inline double &U(int row, int col)
    {
        return UT_[col * N_ + row];
    }

    inline const double &U(int row, int col) const
    {
        return UT_[col * N_ + row];
    }
};

#endif // QRD_RLS_H
