#ifndef KRLS_RBF_H
#define KRLS_RBF_H

#include <cstddef>
#include <Eigen/Dense>
#include <cblas.h>
#include <lapacke.h>
#include "pseudo_inverse.h"
#include "add_row_col.h"

// Windowed KRLS with Gaussian (RBF) kernel
class KRLS_RBF
{
public:
    KRLS_RBF(const double *X_init,
             const double *y_init,
             int n_obs,
             int n_features,
             double delta,
             double sigma,
             int window_size = 1000);

    ~KRLS_RBF();

    void update(const double *new_x, double new_y,
                double &prediction,
                double &error);

    void downdate();

    double predict(const double *x) const;

    void reset();

private:
    int dim_;
    int n_obs_;
    int window_size_;
    int dict_size_;

    double delta_;
    double sigma_;
    bool initialized_;

    // Dictionary and KRLS state
    double *X_;
    double *beta_;
    double *P_;
    double *K_;
    double *y_;

    // Workspace
    double *h_;

    // Optional batch initialization
    double *X_init_;
    double *y_init_;
    int n_init_samples_;

    // Gaussian kernel
    double kernel(const double *x1, int stride1,
                  const double *x2, int stride2) const;

    void initializeFromBatch(const double *X_batch,
                             const double *y_batch,
                             int n_samples);

    static void vectorCopy(double *dest,
                           const double *src,
                           int n);

    inline int idx(int row,
                   int col,
                   int n_cols) const
    {
        return row * n_cols + col;
    }

    KRLS_RBF(const KRLS_RBF &) = delete;
    KRLS_RBF &operator=(const KRLS_RBF &) = delete;
};

#endif // KRLS_RBF_H
