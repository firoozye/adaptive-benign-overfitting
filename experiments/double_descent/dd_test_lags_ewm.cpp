#include "dd_test.h"

using namespace std;

void saveVectorToCSV(const std::vector<double> &vec, const std::string &filename, bool asRow = true)
{
   std::ofstream file(filename);
   if (!file)
   {
      std::cerr << "Error opening file!" << std::endl;
      return;
   }

   for (size_t i = 0; i < vec.size(); ++i)
   {
      file << vec[i];
      if (asRow)
      {
         if (i < vec.size() - 1)
            file << ","; // Separate with commas
      }
      else
      {
         file << "\n"; // Newline for column format
      }
   }
   file.close();
}

int main()
{
   typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

   vector<vector<string>> data_set;
   vector<vector<string>> target_data;
   // data_set = read_csv_func("data/daily_vix.csv");
   data_set = read_csv_func("data/SPX_lags_ewm.csv");
   target_data = read_csv_func("data/target_SPX.csv");
   vector<double> close_price;
   vector<double> ret_price;
   int num_it_samples = 1000;
   int len_data_set = data_set.size() - 1;
   // int idx_close_col = 4;

   for (int i = 1; i < len_data_set; ++i)
   {
      ret_price.push_back(stod(target_data[i][0]));
   }

   int start_row = 0;
   int start_col = 0;
   int num_rows = 50;
   int num_cols = 24;

   int num_elements = ret_price.size() - num_rows;
   Eigen::Map<Eigen::VectorXd> y_old(ret_price.data(), num_rows);
   Eigen::Map<Eigen::VectorXd> y_update_old(ret_price.data() + num_rows, num_elements);

   // double *y = new double[num_rows];
   double y[num_rows];
   for (int i = 0; i < num_rows; ++i)
   {
      y[i] = y_old(i);
   }

   // double *y_update = new double[num_elements];
   double y_update[num_elements];
   for (int i = 0; i < num_elements; ++i)
   {
      y_update[i] = y_update_old(i);
   }

   Eigen::MatrixXd close_lag_mat(len_data_set, num_cols);
   for (int i = 1; i < len_data_set; ++i)
   {
      for (int j = 0; j < num_cols; ++j)
      {
         close_lag_mat(i - 1, j) = stod(data_set[i][j]);
      }
   }

   MatrixXd initial_matrix = close_lag_mat.block(0, 0, num_rows, num_cols);
   MatrixXd update_matrix = close_lag_mat.block(num_rows, 0, close_lag_mat.rows() - num_rows, num_cols);

   int d = num_cols;
   vector<double> all_mse_array;

   for (int idx_rff = 1; idx_rff < 15; idx_rff++)
   {
      // int D = pow(2, idx_rff) + 5;
      int D = pow(2, idx_rff);
      // int D = 2;
      double kernel_var = 1.0;
      bool seed = true;

      GaussianRFF g_rff(d, D, kernel_var, seed);
      MatrixXd X_old = g_rff.transform_matrix(initial_matrix);
      double *X = new double[num_rows * D];
      // double X[num_rows * D];

      // Copy elements from Eigen Matrix to C-style array (column-major order)
      for (int j = 0; j < D; ++j)
      {
         for (int i = 0; i < num_rows; ++i)
         {
            X[i + j * num_rows] = X_old(i, j); // C-style column-major order
         }
      }

      int max_obs = num_rows;
      double ff = 1;
      ABO abo(X, y, max_obs, ff, D, num_rows);

      // Ring buffer: stores raw (pre-RFF) features and y values for downdate
      int N = max_obs;
      std::vector<std::vector<double>> X_raw_ring(N, std::vector<double>(num_cols));
      std::vector<double> y_ring(N, 0.0);
      for (int ri = 0; ri < N; ri++)
      {
         for (int j = 0; j < num_cols; j++) X_raw_ring[ri][j] = initial_matrix(ri, j);
         y_ring[ri] = y[ri];
      }
      int ring_idx = 0;

      vector<double> preds;
      vector<double> mse;
      double all_mse = 0;

      std::vector<double> X_update(D);

      int n_its = 50;
      // int n_its = 2;
      for (int i = 0; i < n_its; i++)
      {
         MatrixXd X_update_old = g_rff.transform(update_matrix.row(i));
         for (int ii = 0; ii < D; ++ii)
         {
            X_update[ii] = X_update_old(0, ii);
         }

         // Downdate oldest observation before adding new one
         if (abo.n_obs_ == N)
         {
            MatrixXd raw_old_mat(1, num_cols);
            for (int j = 0; j < num_cols; j++) raw_old_mat(0, j) = X_raw_ring[ring_idx][j];
            MatrixXd z_old_mat = g_rff.transform(raw_old_mat);
            std::vector<double> z_old_arr(D);
            for (int j = 0; j < D; j++) z_old_arr[j] = z_old_mat(0, j);
            abo.downdate(z_old_arr.data(), y_ring[ring_idx]);
         }

         // Update ring buffer with current new point
         for (int j = 0; j < num_cols; j++) X_raw_ring[ring_idx][j] = update_matrix(i, j);
         y_ring[ring_idx] = y_update[i];
         ring_idx = (ring_idx + 1) % N;

         abo.update(X_update.data(), y_update[i]);

         preds.push_back(abo.pred(X_update.data()));
         mse.push_back(pow(preds[i] - y_update[i], 2));
         all_mse += pow(preds[i] - y_update[i], 2);
      }

      cout << "Number of RFF: " << D << endl;
      cout << "MSE: " << all_mse << endl;
      all_mse_array.push_back(all_mse / n_its);
      delete[] X;
   }
   // save as column
   saveVectorToCSV(all_mse_array, "dd_train_mse.csv", false);
   // saveVectorToCSV(all_mse_array, "dd_test_mse.csv", false);

   return 0;
};
