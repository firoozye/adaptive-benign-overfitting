#include "abo/dd_test.h"
#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstdlib>

namespace
{

   // Synthetic data from paper Eq.(25): x_t = 2*x_{t-1}/(1+0.8*x_{t-1}^2) + U(-1,1)
   // Generated once and cached. No LFS CSV files needed.
   struct DataCache
   {
      int num_rows{};  // init batch size (N)
      int num_cols{};  // lag dimension (L)
      std::vector<double> y_vec;         // init targets, size = num_rows
      std::vector<double> y_update_vec;  // streaming targets
      Eigen::MatrixXd initial_matrix;    // num_rows x num_cols (raw, standardized)
      Eigen::MatrixXd update_matrix;     // n_its x num_cols  (raw, standardized)
   };

   const DataCache &getData()
   {
      static bool inited = false;
      static DataCache cache;
      if (inited) return cache;

      const int LAG     = 7;
      const int N       = 20;
      const int N_ITS   = 1000;  // streaming steps in benchmark
      const int BURNIN  = 500;
      const int T_TOTAL = BURNIN + LAG + N + N_ITS;

      std::srand(42);
      auto unif = []() -> double {
         return 2.0 * (static_cast<double>(std::rand()) / RAND_MAX) - 1.0;
      };

      std::vector<double> series(T_TOTAL);
      series[0] = unif();
      for (int t = 1; t < T_TOTAL; ++t)
      {
         double x = series[t - 1];
         series[t] = 2.0 * x / (1.0 + 0.8 * x * x) + unif();
      }

      // Discard burn-in
      const int T_USE = T_TOTAL - BURNIN;
      std::vector<double> s(series.begin() + BURNIN, series.end());

      // Build lag matrix: NROWS = N + N_ITS rows
      const int NROWS = T_USE - LAG;
      Eigen::MatrixXd X_full(NROWS, LAG);
      std::vector<double> y_full(NROWS);
      for (int i = 0; i < NROWS; ++i)
      {
         for (int j = 0; j < LAG; ++j) X_full(i, j) = s[i + j];
         y_full[i] = s[i + LAG];
      }

      // Standardize over init batch
      std::vector<double> feat_mean(LAG, 0.0), feat_std(LAG, 1.0);
      for (int j = 0; j < LAG; ++j)
      {
         for (int i = 0; i < N; ++i) feat_mean[j] += X_full(i, j);
         feat_mean[j] /= N;
         double var = 0.0;
         for (int i = 0; i < N; ++i)
         {
            double d = X_full(i, j) - feat_mean[j];
            var += d * d;
         }
         feat_std[j] = std::sqrt(var / N);
         if (feat_std[j] < 1e-12) feat_std[j] = 1.0;
      }
      for (int i = 0; i < NROWS; ++i)
         for (int j = 0; j < LAG; ++j)
            X_full(i, j) = (X_full(i, j) - feat_mean[j]) / feat_std[j];

      cache.num_rows = N;
      cache.num_cols = LAG;
      cache.initial_matrix = X_full.topRows(N);
      cache.update_matrix  = X_full.middleRows(N, N_ITS);
      cache.y_vec.assign(y_full.begin(), y_full.begin() + N);
      cache.y_update_vec.assign(y_full.begin() + N, y_full.begin() + N + N_ITS);

      inited = true;
      return cache;
   }

   // One timed pass: downdate + update + pred, repeated n_its times.
   // Maintains a sliding window of exactly max_obs observations.
   void run_once_for_D(
       int D,
       ABO &abo,
       const DataCache &data,
       GaussianRFF &g_rff,
       std::vector<std::vector<double>> &X_raw_ring,
       std::vector<double> &y_ring,
       int &ring_idx,
       std::vector<double> &X_update,
       int n_its,
       benchmark::State &state)
   {
      const int N = data.num_rows;

      for (int t = 0; t < n_its; ++t)
      {
         state.PauseTiming();
         Eigen::MatrixXd z_new = g_rff.transform(data.update_matrix.row(t));
         for (int i = 0; i < D; ++i) X_update[i] = z_new(0, i);
         double y_new = data.y_update_vec[t];
         state.ResumeTiming();

         // Downdate oldest (keep window exactly at N)
         if (abo.n_obs_ == N)
         {
            state.PauseTiming();
            Eigen::MatrixXd raw_old(1, data.num_cols);
            for (int j = 0; j < data.num_cols; ++j)
               raw_old(0, j) = X_raw_ring[ring_idx][j];
            Eigen::MatrixXd z_old = g_rff.transform(raw_old);
            std::vector<double> z_old_v(D);
            for (int j = 0; j < D; ++j) z_old_v[j] = z_old(0, j);
            state.ResumeTiming();
            abo.downdate(z_old_v.data(), y_ring[ring_idx]);
         }

         // Ring buffer update (untimed)
         state.PauseTiming();
         for (int j = 0; j < data.num_cols; ++j)
            X_raw_ring[ring_idx][j] = data.update_matrix(t, j);
         y_ring[ring_idx] = y_new;
         ring_idx = (ring_idx + 1) % N;
         state.ResumeTiming();

         abo.pred(X_update.data());
         abo.update(X_update.data(), y_new);
      }

      benchmark::ClobberMemory();
   }

} // namespace

// ---------------- Google Benchmark entry ----------------

static void BM_ABO_RFF(benchmark::State &state)
{
   const int D = static_cast<int>(state.range(0));
   const auto &data = getData();
   const int N = data.num_rows;
   const int d = data.num_cols;

   // --- Setup (outside timing) ----------------------------------------
   GaussianRFF g_rff(d, D, 1.0, /*seed=*/true);
   Eigen::MatrixXd X0 = g_rff.transform_matrix(data.initial_matrix);

   std::vector<double> X_init(static_cast<size_t>(N) * D);
   for (int j = 0; j < D; ++j)
      for (int i = 0; i < N; ++i)
         X_init[i + j * N] = X0(i, j);

   std::vector<double> y_init = data.y_vec;
   ABO abo(X_init.data(), y_init.data(), N, /*ff=*/1.0, D, N);

   // Ring buffer
   std::vector<std::vector<double>> X_raw_ring(N, std::vector<double>(d));
   std::vector<double> y_ring(N);
   for (int i = 0; i < N; ++i)
   {
      for (int j = 0; j < d; ++j) X_raw_ring[i][j] = data.initial_matrix(i, j);
      y_ring[i] = y_init[i];
   }
   int ring_idx = 0;

   std::vector<double> X_update(static_cast<size_t>(D));

   // --- Timed section --------------------------------------------------
   for (auto _ : state)
   {
      run_once_for_D(D, abo, data, g_rff,
                     X_raw_ring, y_ring, ring_idx,
                     X_update, static_cast<int>(data.y_update_vec.size()), state);
   }

   state.counters["D"] = D;
}

// Sweep D = 2^1, 2^2, ..., 2^14
BENCHMARK(BM_ABO_RFF)->RangeMultiplier(2)->Range(2, 1 << 14);
BENCHMARK_MAIN();
