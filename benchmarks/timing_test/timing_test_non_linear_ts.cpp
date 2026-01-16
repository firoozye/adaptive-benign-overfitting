#include "abo/dd_test.h"
#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

namespace
{

   // Cached inputs reused across runs
   struct DataCache
   {
      int num_rows{};
      int num_cols{};
      std::vector<double> y_vec;        // size = num_rows
      std::vector<double> y_update_vec; // size = (#samples - num_rows)
      Eigen::MatrixXd initial_matrix;   // num_rows x num_cols
      Eigen::MatrixXd update_matrix;    // (len - num_rows) x num_cols
   };

   const DataCache &getData()
   {
      static bool inited = false;
      static DataCache cache;
      if (!inited)
      {
         // Load CSVs once (adjust paths to your dataset)
         std::vector<std::vector<std::string>> data_set =
             read_csv_func("data/non_linear_ts_lags.csv");
         std::vector<std::vector<std::string>> target_data =
             read_csv_func("data/target_non_linear_ts.csv");

         const int len_data_set = static_cast<int>(data_set.size()) - 1;
         const int num_cols = 7;  // lags used when creating the TS
         const int num_rows = 20; // initial batch size

         std::vector<double> y_all;
         y_all.reserve(static_cast<size_t>(len_data_set - 1));
         for (int i = 1; i < len_data_set; ++i)
         {
            y_all.push_back(std::stod(target_data[i][0]));
         }

         const int num_elements = static_cast<int>(y_all.size()) - num_rows;

         cache.y_vec.resize(num_rows);
         for (int i = 0; i < num_rows; ++i)
            cache.y_vec[i] = y_all[i];

         cache.y_update_vec.resize(num_elements);
         for (int i = 0; i < num_elements; ++i)
            cache.y_update_vec[i] = y_all[num_rows + i];

         Eigen::MatrixXd lag_mat(len_data_set, num_cols);
         for (int i = 1; i < len_data_set; ++i)
            for (int j = 0; j < num_cols; ++j)
               lag_mat(i - 1, j) = std::stod(data_set[i][j]);

         cache.initial_matrix = lag_mat.block(0, 0, num_rows, num_cols);
         cache.update_matrix =
             lag_mat.block(num_rows, 0, lag_mat.rows() - num_rows, num_cols);

         cache.num_rows = num_rows;
         cache.num_cols = num_cols;
         inited = true;
      }
      return cache;
   }

   // One timed pass of update+pred repeated n_its times
   void run_once_for_D(
       int D,
       ABO &abo,
       const DataCache &data,
       GaussianRFF &g_rff,
       std::vector<double> &preds,
       std::vector<double> &mse,
       std::vector<double> &X_update,
       int n_its,
       benchmark::State &state) // <— pass state reference
   {
      preds.clear();
      mse.clear();
      preds.reserve(n_its);
      mse.reserve(n_its);

      for (int t = 0; t < n_its; ++t)
      {
         state.PauseTiming();
         Eigen::MatrixXd X_update_old = g_rff.transform(data.update_matrix.row(t));
         for (int i = 0; i < D; ++i)
            X_update[i] = X_update_old(0, i);
         state.ResumeTiming();

         double y_new = data.y_update_vec[t];
         double p = abo.pred(X_update.data());
         abo.update(X_update.data(), y_new);

         preds.push_back(p);
         double err = p - y_new;
         mse.push_back(err * err);
      }

      benchmark::DoNotOptimize(preds.data());
      benchmark::DoNotOptimize(mse.data());
      benchmark::ClobberMemory();
   }

} // namespace

// ---------------- Google Benchmark entry ----------------

static void BM_ABO_RFF(benchmark::State &state)
{
   const int D = static_cast<int>(state.range(0)); // RFF dimension
   const auto &data = getData();                   // warm inputs
   const int num_rows = data.num_rows;

   // --- Setup (outside timing) ----------------------------------------
   const int d = data.num_cols;
   const double kernel_var = 1.0;
   const bool seed = true;

   GaussianRFF g_rff(d, D, kernel_var, seed);
   Eigen::MatrixXd X0 = g_rff.transform_matrix(data.initial_matrix);

   // Column-major raw storage for QR_Rls constructor
   std::vector<double> X(static_cast<size_t>(num_rows) * D);
   for (int j = 0; j < D; ++j)
      for (int i = 0; i < num_rows; ++i)
         X[i + j * num_rows] = X0(i, j);

   std::vector<double> y = data.y_vec;

   const int max_obs = num_rows;
   const double ff = 1.0;

   ABO abo(X.data(), y.data(), max_obs, ff, D, num_rows);

   std::vector<double> preds;
   std::vector<double> mse;
   std::vector<double> X_update(static_cast<size_t>(D));
   const int n_its = 1000;

   // --- Timed section --------------------------------------------------
   for (auto _ : state)
   {
      run_once_for_D(D, abo, data, g_rff, preds, mse, X_update, n_its, state);
   }

   state.counters["D"] = D;
}

// Sweep D = 2^1, 2^2, ..., 2^14
BENCHMARK(BM_ABO_RFF)->RangeMultiplier(2)->Range(2, 1 << 14);
BENCHMARK_MAIN();
