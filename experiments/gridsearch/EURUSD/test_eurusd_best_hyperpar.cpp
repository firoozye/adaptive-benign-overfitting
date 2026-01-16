#include "abo/dd_test.h"
#include "baselines/QRD_RLS/qrd_rls.h"
#include "baselines/KRLS_RBF/krls_rbf.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

using Clock = std::chrono::steady_clock;

static inline double ns_to_us(double ns) { return ns / 1000.0; }
static inline double ns_to_s(double ns) { return ns / 1e9; }

struct ValMetrics
{
   double mse_abo = 0.0, var_abo = 0.0;
   double mse_qrd = 0.0, var_qrd = 0.0;
   double mse_krls = 0.0, var_krls = 0.0;

   // timing (microseconds per update)
   double us_rff_transform = 0.0;
   double us_abo_pred_update = 0.0; // ABO only (excluding RFF transform)
   double us_qrd_update = 0.0;
   double us_krls_update = 0.0;

   // totals (seconds)
   double s_rff_transform = 0.0;
   double s_abo_pred_update = 0.0;
   double s_qrd_update = 0.0;
   double s_krls_update = 0.0;
};

struct RunResult
{
   int k = 0;
   double s = 0.0;
   int L = 0, W = 0;

   double mse_abo = 0.0, var_abo = 0.0;
   double mse_qrd = 0.0, var_qrd = 0.0;
   double mse_krls = 0.0, var_krls = 0.0;

   // timing (avg microseconds/update)
   double us_rff = 0.0;
   double us_abo = 0.0;
   double us_qrd = 0.0;
   double us_krls = 0.0;
};

void lag_matrix(
    const std::vector<double> &x,
    int lag,
    std::vector<std::vector<double>> &X_lag,
    std::vector<double> &y)
{
   const int T = static_cast<int>(x.size());
   const int N = T - lag;
   if (N <= 0)
   {
      X_lag.clear();
      y.clear();
      return;
   }

   X_lag.assign(N, std::vector<double>(lag));
   y.assign(N, 0.0);

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
      std::cerr << "Error opening file: " << filename << std::endl;
      return;
   }

   // ---- header ----
   file << "k,s,L,W,"
        << "mse_abo,var_abo,"
        << "mse_qrd,var_qrd,"
        << "mse_krls,var_krls,"
        << "us_rff,us_abo,us_qrd,us_krls\n";

   // ---- rows ----
   for (const auto &r : results)
   {
      file << r.k << "," << r.s << "," << r.L << "," << r.W << ","
           << r.mse_abo << "," << r.var_abo << ","
           << r.mse_qrd << "," << r.var_qrd << ","
           << r.mse_krls << "," << r.var_krls << ","
           << r.us_rff << "," << r.us_abo << "," << r.us_qrd << "," << r.us_krls
           << "\n";
   }

   file.close();
}

void dataset_creation(std::vector<std::vector<double>> &data_set,
                      std::vector<double> &target_data,
                      Eigen::MatrixXd &initial_matrix,
                      Eigen::MatrixXd &update_matrix,
                      int &d,
                      double *y,
                      double *&y_update,
                      int num_rows,
                      int num_cols,
                      int start_row)
{
   (void)d; // d is redundant here, but kept to match your signature

   const int num_elements = static_cast<int>(target_data.size()) - num_rows - start_row;
   if (num_elements <= 0)
   {
      std::cerr << "dataset_creation: not enough target_data for the requested split.\n";
      y_update = nullptr;
      initial_matrix.resize(0, 0);
      update_matrix.resize(0, 0);
      return;
   }

   Eigen::Map<Eigen::VectorXd> y_old(target_data.data() + start_row, num_rows);
   Eigen::Map<Eigen::VectorXd> y_update_old(target_data.data() + start_row + num_rows, num_elements);

   for (int i = 0; i < num_rows; ++i)
      y[i] = y_old(i);

   y_update = new double[num_elements];
   for (int i = 0; i < num_elements; ++i)
      y_update[i] = y_update_old(i);

   const int len_data_set = static_cast<int>(data_set.size());
   const int n_rows_mat = len_data_set - start_row;
   if (n_rows_mat <= num_rows)
   {
      std::cerr << "dataset_creation: not enough rows in data_set after start_row.\n";
      initial_matrix.resize(0, 0);
      update_matrix.resize(0, 0);
      return;
   }

   Eigen::MatrixXd close_lag_mat(n_rows_mat, num_cols);
   for (int i = 0; i < n_rows_mat; ++i)
      for (int j = 0; j < num_cols; ++j)
         close_lag_mat(i, j) = data_set[i + start_row][j];

   initial_matrix = close_lag_mat.block(0, 0, num_rows, num_cols);
   update_matrix = close_lag_mat.block(num_rows, 0, close_lag_mat.rows() - num_rows, num_cols);
}

void get_var(const std::vector<double> &mse, double real_mse, double &var, int n_its)
{
   var = 0.0;
   for (int i = 0; i < n_its; i++)
   {
      const double temp = mse[i] - real_mse;
      var += temp * temp;
   }
}

ValMetrics cross_val(int first_date, int num_rows, int num_cols, double sigma, int k_fold)
{
   using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

   std::vector<std::vector<double>> data_set;
   std::vector<double> target_data;

   // ---- read raw data (expects your dd_test.h provides read_csv_func) ----
   std::vector<std::vector<std::string>> raw_data = read_csv_func("data/EURUSD/raw_norm_EURUSD.csv");

   std::vector<double> raw_data_dob;
   const int len_raw_data = static_cast<int>(raw_data.size()) - 1;
   raw_data_dob.reserve(std::max(0, len_raw_data - 1));

   for (int i = 1; i < len_raw_data; ++i)
      raw_data_dob.push_back(std::stod(raw_data[i][0]));

   const int lags = num_cols;
   lag_matrix(raw_data_dob, lags, data_set, target_data);

   Eigen::MatrixXd initial_matrix;
   Eigen::MatrixXd update_matrix;

   const int d = num_cols;

   // NOTE: variable-length arrays are non-standard in C++ -> use vector
   std::vector<double> y_vec(num_rows);
   double *y = y_vec.data();

   double *y_update = nullptr;

   const int val_length = 960 * 2; // MUST be int for loop bounds
   const int start_date = first_date + val_length * k_fold;

   dataset_creation(data_set, target_data,
                    initial_matrix, update_matrix,
                    (int &)d, y, y_update,
                    num_rows, num_cols,
                    start_date);

   ValMetrics m;
   if (y_update == nullptr || update_matrix.rows() < val_length)
   {
      std::cerr << "cross_val: dataset_creation failed or update_matrix too short.\n";
      delete[] y_update;
      return m;
   }

   // ---- RFF setup ----
   const int D = static_cast<int>(std::pow(2.0, 11.0));
   // int D = 4;
   const double kernel_var = sigma;
   const bool seed = true;

   GaussianRFF g_rff(d, D, kernel_var, seed);

   MatrixXd X_old = g_rff.transform_matrix(initial_matrix);

   std::vector<double> X(static_cast<size_t>(num_rows) * static_cast<size_t>(D));
   for (int j = 0; j < D; ++j)
      for (int i = 0; i < num_rows; ++i)
         X[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(num_rows)] = X_old(i, j);

   // ---- models ----
   const int max_obs = num_rows;
   const double ff = 1.0;
   const double regularizer = 1e-2;

   // ABO
   ABO abo(X.data(), y, max_obs, ff, D, num_rows);

   // No-RFF input for windowed methods
   std::vector<double> X_no_rff(static_cast<size_t>(num_rows) * static_cast<size_t>(d));
   for (int j = 0; j < d; ++j)
      for (int i = 0; i < num_rows; ++i)
         X_no_rff[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(num_rows)] = initial_matrix(i, j);

   // QRD-RLS
   QRDRLS qrd_rls(num_rows, num_cols, ff, regularizer);
   qrd_rls.batchInitialize(X_no_rff.data(), y, num_rows, num_cols);

   // KRLS-RBF
   const double temp_sigma = 1.0 / kernel_var;
   KRLS_RBF krls_rbf(X_no_rff.data(), y, num_rows, num_cols, regularizer, temp_sigma, num_rows);

   // ---- metrics storage ----
   std::vector<double> mse_abo;
   std::vector<double> mse_qrd_rls;
   std::vector<double> mse_k_rls;
   mse_abo.reserve(val_length);
   mse_qrd_rls.reserve(val_length);
   mse_k_rls.reserve(val_length);

   double all_mse_abo = 0.0;
   double all_mse_qrd_rls = 0.0;
   double all_mse_k_rls = 0.0;

   // ---- timing accumulators ----
   double ns_rff = 0.0;
   double ns_abo = 0.0;
   double ns_qrd = 0.0;
   double ns_krls = 0.0;

   const int warmup = 50;

   std::vector<double> X_update(D);
   std::vector<double> x_no_rff(d);

   const int n_its = val_length;

   for (int i = 0; i < n_its; i++)
   {
      // (A) RFF transform timing (warmup-consistent)
      auto t0 = Clock::now();
      MatrixXd X_update_old = g_rff.transform(update_matrix.row(i));
      auto t1 = Clock::now();
      if (i >= warmup)
         ns_rff += std::chrono::duration<double, std::nano>(t1 - t0).count();

      for (int j = 0; j < D; ++j)
         X_update[j] = X_update_old(0, j);

      // (B) ABO pred+update timing (excluding transform)
      double temp_pred = 0.0;
      t0 = Clock::now();
      temp_pred = abo.pred(X_update.data());
      abo.update(X_update.data(), y_update[i]);
      t1 = Clock::now();
      if (i >= warmup)
         ns_abo += std::chrono::duration<double, std::nano>(t1 - t0).count();

      double temp_res = (temp_pred - y_update[i]);
      temp_res *= temp_res;
      all_mse_abo += temp_res;
      mse_abo.push_back(temp_res);

      // build x_no_rff (not timed)
      for (int j = 0; j < d; ++j)
         x_no_rff[j] = update_matrix(i, j);

      // (C) QRD-RLS update timing
      double pred = 0.0;
      double eps_post = 0.0;

      t0 = Clock::now();
      qrd_rls.update(x_no_rff.data(), y_update[i], pred, eps_post);
      t1 = Clock::now();
      if (i >= warmup)
         ns_qrd += std::chrono::duration<double, std::nano>(t1 - t0).count();

      temp_res = eps_post * eps_post;
      all_mse_qrd_rls += temp_res;
      mse_qrd_rls.push_back(temp_res);

      // (D) KRLS update timing
      t0 = Clock::now();
      krls_rbf.update(x_no_rff.data(), y_update[i], pred, eps_post);
      t1 = Clock::now();
      if (i >= warmup)
         ns_krls += std::chrono::duration<double, std::nano>(t1 - t0).count();

      temp_res = eps_post * eps_post;
      all_mse_k_rls += temp_res;
      mse_k_rls.push_back(temp_res);
   }

   const int effective_its = std::max(1, n_its - warmup);

   m.s_rff_transform = ns_to_s(ns_rff);
   m.s_abo_pred_update = ns_to_s(ns_abo);
   m.s_qrd_update = ns_to_s(ns_qrd);
   m.s_krls_update = ns_to_s(ns_krls);

   m.us_rff_transform = ns_to_us(ns_rff / effective_its);
   m.us_abo_pred_update = ns_to_us(ns_abo / effective_its);
   m.us_qrd_update = ns_to_us(ns_qrd / effective_its);
   m.us_krls_update = ns_to_us(ns_krls / effective_its);

   cout << "\n--- Timing (avg us/update) ---\n";
   cout << "RFF transform:        " << m.us_rff_transform << " us\n";
   cout << "ABO pred+update:      " << m.us_abo_pred_update << " us\n";
   cout << "ABO end-to-end:       " << (m.us_rff_transform + m.us_abo_pred_update) << " us\n";
   cout << "QRD-RLS update:       " << m.us_qrd_update << " us\n";
   cout << "KRLS-RBF update:      " << m.us_krls_update << " us\n";
   cout << "----------------------------\n";

   // ---- compute MSE/VAR ----
   const double real_mse_abo = all_mse_abo / n_its;
   const double real_mse_qrd_rls = all_mse_qrd_rls / n_its;
   const double real_mse_k_rls = all_mse_k_rls / n_its;

   double var_abo = 0.0, var_qrd_rls = 0.0, var_k_rls = 0.0;
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

   m.mse_qrd = real_mse_qrd_rls;
   m.var_qrd = var_qrd_rls / (n_its - 1);

   m.mse_krls = real_mse_k_rls;
   m.var_krls = var_k_rls / (n_its - 1);

   delete[] y_update;
   return m;
}

int main()
{
   // we used 3 k folds, for test we will use 3
   const int start = 0;
   const int end = 3;              // exclusive
   const int first_date = 960 * 3; // baseline offset

   std::vector<int> k_folds(end - start);
   std::iota(k_folds.begin(), k_folds.end(), start);

   std::vector<int> lags = {30};
   std::vector<int> windows = {120};
   // std::vector<int> windows = {2};
   const double sigma = 4.5;

   int c = 0;
   std::vector<RunResult> results;

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

            ValMetrics m = cross_val(first_date, W, L, sigma, k);

            results.push_back(RunResult{
                k - start, sigma, L, W,
                m.mse_abo, m.var_abo,
                m.mse_qrd, m.var_qrd,
                m.mse_krls, m.var_krls,
                m.us_rff_transform,
                m.us_abo_pred_update,
                m.us_qrd_update,
                m.us_krls_update});

            c++;
         }
      }
   }

   saveRunResultsToCSV(results, "results/gridsearch/EURUSD/mse_var_grid_1_best_hyp.csv");
   return 0;
}
