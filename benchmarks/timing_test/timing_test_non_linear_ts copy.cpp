#include "dd_test.h"
#include <benchmark/benchmark.h>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

using namespace std;

namespace
{

   // Data prepared once and reused across iterations
   struct DataCache
   {
      int num_rows;
      int num_cols;
      std::vector<double> y_vec;        // size num_rows
      std::vector<double> y_update_vec; // size num_elements
      Eigen::MatrixXd initial_matrix;   // num_rows x num_cols
      Eigen::MatrixXd update_matrix;    // (len-num_rows) x num_cols
   };

   // Load & prep all inputs once
   const DataCache &getData()
   {
      static bool inited = false;
      static DataCache cache;

      if (!inited)
      {
         vector<vector<string>> data_set = read_csv_func("data/non_linear_ts_lags.csv");
         vector<vector<string>> target_data = read_csv_func("data/target_non_linear_ts.csv");

         int len_data_set = static_cast<int>(data_set.size()) - 1;
         int num_cols = 7;  // number of lags when the time series was created
         int num_rows = 14; // initial batch, number of observations

         vector<double> ret_price;
         ret_price.reserve(static_cast<size_t>(len_data_set - 1));
         for (int i = 1; i < len_data_set; ++i)
         {
            ret_price.push_back(stod(target_data[i][0]));
         }

         int num_elements = static_cast<int>(ret_price.size()) - num_rows;

         // y and y_update (as vectors)
         cache.y_vec.resize(num_rows);
         for (int i = 0; i < num_rows; ++i)
            cache.y_vec[i] = ret_price[i];
         cache.y_update_vec.resize(num_elements);
         for (int i = 0; i < num_elements; ++i)
            cache.y_update_vec[i] = ret_price[num_rows + i];

         // Build lag matrix
         Eigen::MatrixXd close_lag_mat(len_data_set, num_cols);
         for (int i = 1; i < len_data_set; ++i)
         {
            for (int j = 0; j < num_cols; ++j)
            {
               close_lag_mat(i, j) = stod(data_set[i][j]);
            }
         }

         cache.initial_matrix = close_lag_mat.block(0, 0, num_rows, num_cols);
         cache.update_matrix = close_lag_mat.block(num_rows, 0, close_lag_mat.rows() - num_rows, num_cols);
         cache.num_rows = num_rows;
         cache.num_cols = num_cols;

         inited = true;
      }
      return cache;
   }

   // One benchmarked run for a given D (RFF count)
   void run_once_for_D(int D, QR_Rls qr_rls, DataCache data, GaussianRFF g_rff)
   {

      for (int t = 0; t < n_its; ++t)
      {
         Eigen::MatrixXd X_update_old = g_rff.transform(data.update_matrix.row(t));
         for (int i = 0; i < D; ++i)
            X_update[i] = X_update_old(0, i);

         double y_new = data.y_update_vec[t];
         double p = qr_rls.pred(X_update.data());
         qr_rls.update(X_update.data(), y_new);

         preds.push_back(p);
         mse.push_back((p - data.y_update_vec[t]) * (p - data.y_update_vec[t]));
      }

      // Prevent the compiler from optimizing away the work
      benchmark::DoNotOptimize(preds.data());
      benchmark::DoNotOptimize(mse.data());
      benchmark::ClobberMemory();
   }

} // namespace

// ---------------- Google Benchmark entry ----------------

static void BM_QR_RLS_RFF(benchmark::State &state)
{
   const auto &data = getData();
   const int num_rows = data.num_rows;

   // Build features
   int d = data.num_cols;
   double kernel_var = 1.0;
   bool seed = true;

   GaussianRFF g_rff(d, D, kernel_var, seed);
   Eigen::MatrixXd X_old = g_rff.transform_matrix(data.initial_matrix);

   // Column-major raw array as required by your QR_Rls
   std::vector<double> X(static_cast<size_t>(num_rows) * D);
   for (int j = 0; j < D; ++j)
      for (int i = 0; i < num_rows; ++i)
         X[i + j * num_rows] = X_old(i, j);

   // y array
   std::vector<double> y = data.y_vec;

   int max_obs = num_rows;
   double ff = 1.0;
   double lambda = 0.1;

   ABO abo(X.data(), y.data(), max_obs, ff, D, num_rows);

   std::vector<double> preds;
   std::vector<double> mse;
   int n_its = 1000;
   preds.reserve(n_its);
   mse.reserve(n_its);

   std::vector<double> X_update(static_cast<size_t>(D));

   int D = static_cast<int>(state.range(0)); // number of random Fourier features
   // Warm up data (outside timing)
   (void)getData();

   for (auto _ : state)
   {
      run_once_for_D(D, qr_rls, data, g_rff);
   }
   // Optionally report items processed, custom counters, etc.
   state.counters["D"] = D;
}

// Sweep D = 2^1, 2^2, ..., 2^14 (same spirit as your loop)
BENCHMARK(BM_QR_RLS_RFF)
    ->RangeMultiplier(2)
    ->Range(2, 1 << 14);

BENCHMARK_MAIN();
