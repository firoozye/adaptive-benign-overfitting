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
   data_set = read_csv_func("data/non_linear_ts_lags.csv");
   target_data = read_csv_func("data/target_non_linear_ts.csv");
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
   int num_rows = 20;
   int num_cols = 7;

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
         close_lag_mat(i, j) = stod(data_set[i][j]);
      }
   }

   MatrixXd initial_matrix = close_lag_mat.block(0, 0, num_rows, num_cols);
   MatrixXd update_matrix = close_lag_mat.block(num_rows, 0, close_lag_mat.rows() - num_rows, num_cols);

   int d = num_cols;
   vector<double> all_mse_array;
   vector<double> all_var_array;
   vector<double> all_cond_num_mean_array;
   vector<double> all_cond_num_var_array;

   for (int idx_rff = 1; idx_rff < 15; idx_rff++)
   {
      int D = pow(2, idx_rff);
      // int D = pow(2, 14);
      double kernel_var = 1.0;
      // double kernel_var = 1.0 / 4.0;
      // double kernel_var = 1.0 / 2.0;
      bool seed = true;

      GaussianRFF g_rff(d, D, kernel_var, seed);
      MatrixXd X_old = g_rff.transform_matrix(initial_matrix);
      double *X = new double[num_rows * D];

      // Copy elements from Eigen Matrix to C-style array (column-major order)
      for (int j = 0; j < D; ++j)
      {
         for (int i = 0; i < num_rows; ++i)
         {
            X[i + j * num_rows] = X_old(i, j); // C-style column-major order
         }
      }

      int max_obs = num_rows;
      double ff = 1.0;
      double lambda = 0.1;
      QR_Rls qr_rls(X, y, max_obs, ff, lambda, D, num_rows);

      vector<double> preds;
      vector<double> mse;
      vector<double> cond_nums_mse;
      double all_mse = 0;
      double all_cond_nums = 0;

      double X_update[D];

      const std::string &filename = "res_real_cond_num_false_cond_num/cond_nums_" + std::to_string(D) + ".csv";
      std::ifstream file(filename);
      std::string line;
      std::vector<double> pertur_cond_num;

      while (std::getline(file, line))
      {
         if (!line.empty())
         {
            pertur_cond_num.push_back(std::stod(line));
         }
      }

      int n_its = 1000;
      for (int i = 0; i < n_its; i++)
      {
         MatrixXd X_update_old = g_rff.transform(update_matrix.row(i));
         for (int i = 0; i < D; ++i)
         {
            X_update[i] = X_update_old(0, i);
         }

         double temp_cond_num = qr_rls.get_real_cond_num();
         double temp_res = pow(pertur_cond_num[i] - temp_cond_num, 2);

         cond_nums_mse.push_back(temp_res);
         all_cond_nums += pow(pertur_cond_num[i] - temp_cond_num, 2);
         ;
      }

      double var_cond_nums = 0;
      double mean_cond_num = all_cond_nums / n_its;
      for (int i = 0; i < n_its; i++)
      {
         double temp_1 = cond_nums_mse[i] - mean_cond_num;
         var_cond_nums += temp_1 * temp_1;
      }

      cout << "Number of RFF: " << D << endl;
      cout << "RealCondNumMSE: " << mean_cond_num << endl;
      cout << "RealCondNumVAR: " << var_cond_nums / (n_its - 1) << endl;

      all_cond_num_mean_array.push_back(mean_cond_num);
      all_cond_num_var_array.push_back(var_cond_nums / (n_its - 1));

      // saveVectorToCSV(cond_nums, "res_real_cond_num_false_cond_num/cond_nums_" + std::to_string(D) + ".csv", false);

      delete[] X;
   }
   // save as column
   saveVectorToCSV(all_cond_num_mean_array, "real_res_mse.csv", false);
   saveVectorToCSV(all_cond_num_var_array, "real_res_mse.csv", false);

   return 0;
};
