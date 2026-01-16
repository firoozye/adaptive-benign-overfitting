#include "baselines/QRD_RLS/qrd_rls_test.h"

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
   data_set = read_csv_func("data/EURUSD/lags_EURUSD.csv");
   target_data = read_csv_func("data/EURUSD/target_EURUSD.csv");
   vector<double> close_price;
   vector<double> ret_price;
   int len_data_set = data_set.size() - 1;

   for (int i = 1; i < len_data_set; ++i)
   {
      ret_price.push_back(stod(target_data[i][0]));
   }

   int start_row = 0;
   int start_col = 0;
   int num_rows = 60;
   int num_cols = 25;

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
   vector<double> all_var_array;
   vector<double> all_cond_num_mean_array;
   vector<double> all_cond_num_var_array;

   bool seed = true;
   double *X = new double[num_rows * d];

   // Copy elements from Eigen Matrix to C-style array (column-major order)
   for (int j = 0; j < d; ++j)
   {
      for (int i = 0; i < num_rows; ++i)
      {
         X[i + j * num_rows] = initial_matrix(i, j); // C-style column-major order
      }
   }

   int max_obs = num_rows;
   // double ff = 1.0;
   double ff = .97;

   vector<double> preds;
   vector<double> mse;
   vector<double> cond_nums;
   double all_mse = 0;
   double all_cond_nums = 0;

   double X_update[d];
   double delta = 1e-2;
   double lambda = 1;

   // QRDRLS qrd_rls(X, num_rows, num_cols, delta, lambda);
   QRDRLS qrd_rls(num_rows, num_cols, ff, delta);

   // qrd_rls.reset();
   qrd_rls.batchInitialize(X, y, num_rows, num_cols);

   qrd_rls.reset();
   double eps_post;
   double eps_pri;

   int n_its = 12000;
   for (int i = 0; i < n_its; i++)
   {
      for (int j = 0; j < d; ++j)
      {
         // X_update[i] = X_update_old(0, i);
         X_update[j] = update_matrix(i, j);
      }

      // preds.push_back(qr_rls.pred(X_update));
      // void QRDRLS::update(const double *x, double d, double *eps_post, double *eps_pri)

      // qrd_rls.update(X_update, y_update[i], &eps_post, &eps_pri);
      double pred = 0;
      qrd_rls.update(X_update, y_update[i], pred, eps_post);

      // preds.push_back(qr_rls.pred(X_update));
      // double temp_res = pow(preds[i] - y_update[i], 2);
      double temp_res = eps_post * eps_post;
      mse.push_back(temp_res);
      all_mse += temp_res;

      // double temp_cond_num = qr_rls.get_cond_num();
      // cond_nums.push_back(temp_cond_num);
      // all_cond_nums += temp_cond_num;
   }

   double var = 0;
   double var_cond_nums = 0;
   double real_mse = all_mse / n_its;
   // double mean_cond_num = all_cond_nums / n_its;
   for (int i = 0; i < n_its; i++)
   {
      double temp = mse[i] - real_mse;
      var += temp * temp;

      // double temp_1 = cond_nums[i] - mean_cond_num;
      // var_cond_nums += temp_1 * temp_1;
   }

   cout << "ResMSE: " << real_mse << endl;
   cout << "ResVAR: " << var / (n_its - 1) << endl;
   // cout << "CondNumMSE: " << mean_cond_num << endl;
   // cout << "CondNumVAR: " << var_cond_nums / (n_its - 1) << endl;

   // all_mse_array.push_back(real_mse);
   // all_var_array.push_back(var / (n_its - 1));
   //  all_cond_num_mean_array.push_back(mean_cond_num);
   //  all_cond_num_var_array.push_back(var_cond_nums / (n_its - 1));

   // saveVectorToCSV(cond_nums, "res_real_cond_num_false_cond_num/cond_nums_" + std::to_string(D) + ".csv", false);

   delete[] X;
   //  save as column
   //  saveVectorToCSV(all_mse_array, "results/EURUSD/dd_test_mse.csv", false);
   //  saveVectorToCSV(all_var_array, "results/EURUSD/dd_test_var.csv", false);
   //  saveVectorToCSV(all_mse_array, "results/EURUSD/dd_test_mse_ff_97.csv", false);
   //  saveVectorToCSV(all_var_array, "results/EURUSD/dd_test_var_ff_97.csv", false);

   return 0;
};
