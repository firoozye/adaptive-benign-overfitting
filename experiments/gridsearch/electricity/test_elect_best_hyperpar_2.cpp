#include "abo/dd_test.h"
#include "baselines/QRD_RLS/qrd_rls.h"
#include "baselines/KRLS_RBF/krls_rbf.h"

using namespace std;

#include <vector>

struct ValMetrics
{
   double mse_abo, var_abo;
   double mse_qrd, var_qrd;
   double mse_krls, var_krls;
};

struct RunResult
{
   int k;
   double s;
   int L, W;
   double mse_abo, var_abo;
   double mse_qrd, var_qrd;
   double mse_krls, var_krls;
};

void lag_matrix(
    const std::vector<double> &x,
    int lag,
    std::vector<std::vector<double>> &X_lag,
    std::vector<double> &y)
{
   const int T = x.size();
   const int N = T - lag;

   X_lag.resize(N, std::vector<double>(lag));
   y.resize(N);

   for (int i = 0; i < N; ++i)
   {
      for (int j = 0; j < lag; ++j)
         X_lag[i][j] = x[i + j];

      y[i] = x[i + lag];
   }
}

void saveRunResultsToCSV(const std::vector<RunResult> &results,
                         const std::string &filename)
{
   std::ofstream file(filename);
   if (!file)
   {
      std::cerr << "Error opening file!" << std::endl;
      return;
   }

   // ---- header ----
   file << "k,s,L,W,"
        << "mse_abo,var_abo,"
        << "mse_qrd,var_qrd,"
        << "mse_krls,var_krls\n";

   // ---- rows ----
   for (const auto &r : results)
   {
      file << r.k << "," << r.s << "," << r.L << "," << r.W << ","
           << r.mse_abo << "," << r.var_abo << ","
           << r.mse_qrd << "," << r.var_qrd << ","
           << r.mse_krls << "," << r.var_krls << "\n";
   }

   file.close();
}

void dataset_creation(vector<vector<double>> &data_set, vector<double> &target_data,
                      Eigen::MatrixXd &initial_matrix, Eigen::MatrixXd &update_matrix, int &d,
                      double *y, double *&y_update, int num_rows, int num_cols, int start_row)
{

   // int start_row = 0;
   int start_col = 0;

   int num_elements = target_data.size() - num_rows;
   Eigen::Map<Eigen::VectorXd> y_old(target_data.data() + start_row, num_rows);
   Eigen::Map<Eigen::VectorXd> y_update_old(target_data.data() + start_row + num_rows, num_elements);

   for (int i = 0; i < num_rows; ++i)
   {
      y[i] = y_old(i);
   }

   y_update = new double[num_elements];
   for (int i = 0; i < num_elements; ++i)
   {
      y_update[i] = y_update_old(i);
   }

   int len_data_set = data_set.size();
   Eigen::MatrixXd close_lag_mat(len_data_set - start_row, num_cols);
   for (int i = 0; i < len_data_set - start_row; ++i)
   {
      for (int j = 0; j < num_cols; ++j)
      {
         close_lag_mat(i, j) = data_set[i + start_row][j];
      }
   }

   initial_matrix = close_lag_mat.block(0, 0, num_rows, num_cols);
   update_matrix = close_lag_mat.block(num_rows, 0, close_lag_mat.rows() - num_rows, num_cols);
}

void get_var(vector<double> mse, double real_mse, double &var, int n_its)
{
   for (int i = 0; i < n_its; i++)
   {
      double temp = mse[i] - real_mse;
      var += temp * temp;
   }
};

ValMetrics cross_val(int first_date, int num_rows, int num_cols, double sigma, int k_fold)
{
   typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

   vector<vector<double>> data_set;
   vector<double> target_data;
   vector<vector<string>> raw_data;
   // data_set = read_csv_func("data/electricity/lags_LD2011_2014.csv");
   // target_data = read_csv_func("data/electricity/target_LD2011_2014.csv");
   raw_data = read_csv_func("data/electricity/raw_norm_LD2011_2014.csv");
   vector<double> close_price;
   vector<double> raw_data_dob;
   int len_raw_data = raw_data.size() - 1;

   for (int i = 1; i < len_raw_data; ++i)
   {
      // ret_price.push_back(stod(target_data[i][0]));
      raw_data_dob.push_back(stod(raw_data[i][0]));
   }

   int lags = num_cols;

   // std::vector<std::vector<double>> X_lag;
   // std::vector<double> y_tar;

   lag_matrix(raw_data_dob, lags, data_set, target_data);

   Eigen::MatrixXd initial_matrix;
   Eigen::MatrixXd update_matrix;
   // int num_cols = 48;
   // int num_rows = 96;
   int d = num_cols;
   double y[num_rows];
   double *y_update;
   double val_length = 672 * 2;
   // double start_date = (val_length + num_rows) * k_fold;
   double start_date = first_date + val_length * k_fold;

   dataset_creation(data_set, target_data,
                    initial_matrix, update_matrix, d, y, y_update, num_rows, num_cols,
                    start_date);

   vector<double> all_mse_array;
   vector<double> all_var_array;
   vector<double> all_cond_num_mean_array;
   vector<double> all_cond_num_var_array;

   int D = pow(2, 13);
   // int D = pow(2, 12);
   double kernel_var = 1.0;
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
   // double ff = .98;
   double regularizer = 1e-2;
   // double lambda = 1;

   // our model
   ABO abo(X, y, max_obs, ff, D, num_rows);

   double X_no_rff[num_rows * d];
   for (int j = 0; j < d; ++j)
   {
      for (int i = 0; i < num_rows; ++i)
      {
         X_no_rff[i + j * num_rows] = initial_matrix(i, j); // C-style column-major order
      }
   }

   // windowed qr rls
   QRDRLS qrd_rls(num_rows, num_cols, ff, regularizer);
   qrd_rls.batchInitialize(X_no_rff, y, num_rows, num_cols);

   double temp_sigma = 1.0 / sigma;

   // windowed kernel rls
   KRLS_RBF krls_rbf(X_no_rff, y, num_rows, num_cols, regularizer, temp_sigma, num_rows);

   vector<double> preds_abo;
   vector<double> mse_abo;
   vector<double> preds_qrd_rls;
   vector<double> mse_qrd_rls;
   vector<double> preds_k_rls;
   vector<double> mse_k_rls;
   double all_mse_abo = 0;
   double all_mse_qrd_rls = 0;
   double all_mse_k_rls = 0;
   ValMetrics m;

   double X_update[D];

   // int n_its = 35100;
   int n_its = val_length;
   for (int i = 0; i < n_its; i++)
   {
      MatrixXd X_update_old = g_rff.transform(update_matrix.row(i));
      for (int j = 0; j < D; ++j)
      {
         X_update[j] = X_update_old(0, j);
      }

      double temp_pred = abo.pred(X_update);
      preds_abo.push_back(temp_pred);
      abo.update(X_update, y_update[i]);

      // double temp_res = pow(preds_abo[i] - y_update[i], 2);
      double temp_res = pow(temp_pred - y_update[i], 2);
      mse_abo.push_back(temp_res);
      all_mse_abo += temp_res;

      double x_no_rff[d];
      for (int j = 0; j < d; ++j)
      {
         x_no_rff[j] = update_matrix(i, j);
      }

      double pred = 0;
      double eps_post = 0;
      qrd_rls.update(x_no_rff, y_update[i], pred, eps_post);
      temp_res = eps_post * eps_post;
      mse_qrd_rls.push_back(temp_res);
      all_mse_qrd_rls += temp_res;

      krls_rbf.update(x_no_rff, y_update[i], pred, eps_post);
      preds_k_rls.push_back(pred);
      temp_res = eps_post * eps_post;
      mse_k_rls.push_back(temp_res);
      all_mse_k_rls += temp_res;
   }

   double var_abo = 0;
   double real_mse_abo = all_mse_abo / n_its;

   double var_qrd_rls = 0;
   double real_mse_qrd_rls = all_mse_qrd_rls / n_its;

   double var_k_rls = 0;
   double real_mse_k_rls = all_mse_k_rls / n_its;

   get_var(mse_abo, real_mse_abo, var_abo, n_its);
   get_var(mse_qrd_rls, real_mse_qrd_rls, var_qrd_rls, n_its);
   get_var(mse_k_rls, real_mse_k_rls, var_k_rls, n_its);

   cout << "Number of RFF: " << D << endl;
   cout << "ResMSE_abo: " << real_mse_abo << endl;
   cout << "ResVAR_abo: " << var_abo / (n_its - 1) << endl;
   cout << "ResMSE_qrd_rls: " << real_mse_qrd_rls << endl;
   cout << "ResVAR_qrd_rls: " << var_qrd_rls / (n_its - 1) << endl;
   cout << "ResMSE_k_rls: " << real_mse_k_rls << endl;
   cout << "ResVAR_k_rls: " << var_k_rls / (n_its - 1) << endl;

   m.mse_abo = real_mse_abo;
   m.var_abo = var_abo / (n_its - 1);

   // ---- QRD-RLS ----
   m.mse_qrd = real_mse_qrd_rls;
   m.var_qrd = var_qrd_rls / (n_its - 1);

   // ---- K-RLS ----
   m.mse_krls = real_mse_k_rls;
   m.var_krls = var_k_rls / (n_its - 1);

   delete[] X;
   delete[] y_update;
   return m;
}

int main()
{
   // we used 3 k folds, for test we will use 3
   int start = 0;
   int end = 3; // exclusive
   // there were 0,1,2 k folds
   int first_date = 672 * 3;

   std::vector<int> k_folds(end - start);
   std::iota(k_folds.begin(), k_folds.end(), start);
   std::vector<int> lags = {96};
   std::vector<int> windows = {192};
   double sigma = 6.9;
   int c = 0;
   vector<RunResult> results;
   // vector<ValMetrics> m;
   ValMetrics m;

   for (int k : k_folds)
   {

      for (int L : lags)
      {

         for (int W : windows)
         {
            cout << "Experiment number: " << c << endl;
            cout << "Test window: " << k - start << endl;
            cout << "Lag size: " << L << endl;
            cout << "Window size: " << W << endl;
            m = cross_val(first_date, W, L, sigma, k);
            results.push_back({k - start, sigma, L, W, m.mse_abo, m.var_abo, m.mse_qrd, m.var_qrd, m.mse_krls, m.var_krls});
            c++;
         }
      }
   }

   saveRunResultsToCSV(results, "results/gridsearch/electricity/mse_var_grid_1_best_hyp.csv");

   return 0;
};
