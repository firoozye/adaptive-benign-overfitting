#include "test_QR.h"

using namespace std;

Eigen::MatrixXd lag_matrix(const std::vector<double> &x, int lag) {
   int n = x.size();
   int num_rows = n - lag + 1;

   // Initialize the output matrix with `num_rows` rows and `lag` columns
   Eigen::MatrixXd result(num_rows, lag);

   // Fill the matrix with lagged values
   for (int i = 0; i < num_rows; ++i) {
      for (int j = 0; j < lag; ++j) {
         result(i, j) = x[i + j];
      }
   }

   return result;
}

int main() {
   typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
   typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;

   vector<vector<string>> data_set;
   data_set = read_csv_func("daily_vix.csv");
   vector<double> close_price;
   vector<double> ret_price;

   for (int i = 2; i < data_set.size(); i++)
   // for (auto i :: )
   {
      // cout << data_set[i][4] << endl;
      double temp_num = stod(data_set[i][4]);
      close_price.push_back(temp_num);

      // double ret = stod(data_set[i][4]) / stod(data_set[i - 1][4]) - 1;
      double ret = log(stod(data_set[i][4]) / stod(data_set[i - 1][4]));
      ret_price.push_back(ret);
   }

   int n_lags = 14;
   MatrixXd close_lag_mat = lag_matrix(ret_price, n_lags);

   int start_row = 0;
   int start_col = 0;
   int num_rows = 10;
   int num_cols = n_lags;

   int num_elements = close_price.size() - n_lags - num_rows;
   Eigen::Map<Eigen::VectorXd> y_old(close_price.data() + n_lags + 1, num_rows);
   Eigen::Map<Eigen::VectorXd> y_update_old(close_price.data() + n_lags + num_rows + 1, num_elements);

   double *y = new double[num_rows];
   for (int i = 0; i < num_rows; ++i) {
      y[i] = y_old(i);
   }

   // double y_update[num_elements];
   double *y_update = new double[num_elements];
   for (int i = 0; i < num_elements; ++i) {
      y_update[i] = y_update_old(i);
   }

   // cout << y.row(y.rows() - 1) << endl;
   // cout << y_update.row(0) << endl;

   MatrixXd initial_matrix = close_lag_mat.block(0, 0, num_rows, num_cols);
   MatrixXd update_matrix = close_lag_mat.block(num_rows, 0, close_lag_mat.rows() - num_rows, num_cols);

   // cout << initial_matrix.row(initial_matrix.rows() - 1) << endl;
   // cout << update_matrix.row(0) << endl;

   int d = num_cols;
   // int D = 1000;
   int D = 4000;
   double kernel_var = 1.0;
   bool seed = true;

   GaussianRFF g_rff(d, D, kernel_var, seed);

   MatrixXd X_old = g_rff.transform_matrix(initial_matrix);

   double *X = new double[num_rows * D];

   // Copy elements from Eigen Matrix to C-style array (column-major order)
   for (int j = 0; j < D; ++j) {
      for (int i = 0; i < num_rows; ++i) {
         X[i + j * num_rows] = X_old(i, j);  // C-style column-major order
      }
   }

   // cout << X.rows() << " " << X.cols() << endl;
   //  cout << X << endl;

   int max_obs = 25;
   double ff = 1;
   double lambda = 0.1;
   QR_Rls qr_rls(X, y, max_obs, ff, lambda, D, num_rows);

   vector<double> preds;
   vector<double> mse;

   double X_update[D];

   int n_its = 1000;
   for (int i = 0; i < n_its; i++) {
      MatrixXd X_update_old = g_rff.transform(update_matrix.row(i));
      for (int i = 0; i < D; ++i) {
         X_update[i] = X_update_old(0, i);
      }

      qr_rls.update(X_update, y_update[i]);
      cout << y_update[i] << endl;

      preds.push_back(qr_rls.pred(X_update));
      mse.push_back(pow(preds[i] - y_update[i], 2));
   }

   return 0;
};
