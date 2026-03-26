#include "baselines/SWKRLS/swkrls.h"
#include "read_csv_func.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

using namespace std;

int main()
{
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

    vector<vector<string>> data_set;
    vector<vector<string>> target_data;
    data_set    = read_csv_func("data/EURUSD/lags_EURUSD.csv");
    target_data = read_csv_func("data/EURUSD/target_EURUSD.csv");

    vector<double> ret_price;
    int len_data_set = data_set.size() - 1;

    for (int i = 1; i < len_data_set; ++i)
        ret_price.push_back(stod(target_data[i][0]));

    int num_rows = 60;
    int num_cols = 25;

    int num_elements = ret_price.size() - num_rows;
    Eigen::Map<Eigen::VectorXd> y_old(ret_price.data(), num_rows);

    double y[num_rows];
    for (int i = 0; i < num_rows; ++i)
        y[i] = y_old(i);

    double y_update[num_elements];
    for (int i = 0; i < num_elements; ++i)
        y_update[i] = ret_price[num_rows + i];

    Eigen::MatrixXd close_lag_mat(len_data_set, num_cols);
    for (int i = 1; i < len_data_set; ++i)
        for (int j = 0; j < num_cols; ++j)
            close_lag_mat(i - 1, j) = stod(data_set[i][j]);

    MatrixXd initial_matrix = close_lag_mat.block(0, 0, num_rows, num_cols);
    MatrixXd update_matrix  = close_lag_mat.block(num_rows, 0,
                                  close_lag_mat.rows() - num_rows, num_cols);

    int d = num_cols;

    // X in column-major order for SWKRLS constructor
    double *X = new double[num_rows * d];
    for (int j = 0; j < d; ++j)
        for (int i = 0; i < num_rows; ++i)
            X[i + j * num_rows] = initial_matrix(i, j);

    double lambda = 1e-2;
    double sigma  = 1.0;
    int    capacity = num_rows;   // sliding window size = 60
    double ff       = 1.0;  // TODO: fix forgetting factor implementation before using ff<1

    int    n_its = 12000;

    // Sweep over ALD novelty thresholds
    double ald_thresholds[] = {1e-5, 1e-4, 1e-3};

    for (double ald_thresh : ald_thresholds)
    {
        SWKRLS sw_krls(X, y, num_rows, num_cols, lambda, sigma,
                       capacity, ff, ald_thresh);

        vector<double> mse;
        double all_mse    = 0.0;
        double X_update[d];
        double pred, err;
        long long dict_size_sum = 0;

        for (int i = 0; i < n_its; ++i)
        {
            for (int j = 0; j < d; ++j)
                X_update[j] = update_matrix(i, j);

            sw_krls.update(X_update, y_update[i], pred, err);

            double sq_err = err * err;
            mse.push_back(sq_err);
            all_mse += sq_err;
            dict_size_sum += sw_krls.dict_size();
        }

        double real_mse = all_mse / n_its;
        double var = 0.0;
        for (int i = 0; i < n_its; ++i)
        {
            double tmp = mse[i] - real_mse;
            var += tmp * tmp;
        }
        var /= (n_its - 1);

        cout << "ald_thresh=" << ald_thresh
             << "  MSE=" << real_mse
             << "  VAR=" << var
             << "  avg_dict_size=" << (double)dict_size_sum / n_its
             << endl;
    }

    delete[] X;
    return 0;
}
