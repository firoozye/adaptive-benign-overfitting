#include "abo/dd_test.h"
#include "baselines/QRD_RLS/qrd_rls.h"
#include "baselines/KRLS_RBF/krls_rbf.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static inline double ns_to_us(double ns) { return ns / 1000.0; }
static inline double ns_to_s(double ns) { return ns / 1e9; }

struct ModelParamsABO
{
   int L = -1, W = -1;
   double sigma = -1.0;
   int D = 2048;
   int log2D = -1;
   double ff = 1.0;
   double regularizer = 1e-2;
};

struct ModelParamsQRD
{
   int L = -1, W = -1;
   double ff = 1.0;
   double regularizer = 1e-2;
};

struct ModelParamsKRLS
{
   int L = -1, W = -1;
   double sigma = -1.0;
   double ff = 1.0;
   double regularizer = 1e-2;
};

struct CommonParams
{
   int first_date = 960 * 8;
   int start_k = 0;
   int end_k = 5; // exclusive
   int val_length = 960 * 2;
   int warmup = 50;
   std::string out_csv = "results/gridsearch/EURUSD/best_test.csv";

   bool run_abo = true;
   bool run_qrd = true;
   bool run_krls = true;
};

struct Args
{
   CommonParams common;
   ModelParamsABO abo;
   ModelParamsQRD qrd;
   ModelParamsKRLS krls;
};

struct FoldResultRow
{
   int fold = 0;
   std::string model;

   int L = 0, W = 0;
   double sigma = std::numeric_limits<double>::quiet_NaN();
   int D = 0;

   double mse = std::numeric_limits<double>::quiet_NaN();
   double var = std::numeric_limits<double>::quiet_NaN();

   double us_rff = 0.0; // ABO only
   double us_update = 0.0;

   double s_rff = 0.0;
   double s_update = 0.0;
};

static inline bool is_flag(const char *a, const char *b)
{
   return std::strcmp(a, b) == 0;
}

static inline void parse_args(int argc, char **argv, Args &a)
{
   for (int i = 1; i < argc; ++i)
   {
      auto need = [&](const char *flag)
      {
         if (i + 1 >= argc)
            throw std::runtime_error(std::string("Missing value for ") + flag);
      };

      // ---- common ----
      if (is_flag(argv[i], "--first_date"))
      {
         need("--first_date");
         a.common.first_date = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--start_k"))
      {
         need("--start_k");
         a.common.start_k = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--end_k"))
      {
         need("--end_k");
         a.common.end_k = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--val_length"))
      {
         need("--val_length");
         a.common.val_length = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--warmup"))
      {
         need("--warmup");
         a.common.warmup = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--out_csv"))
      {
         need("--out_csv");
         a.common.out_csv = argv[++i];
      }
      else if (is_flag(argv[i], "--run"))
      {
         need("--run");
         std::string s = argv[++i];
         auto has = [&](const std::string &key)
         { return s.find(key) != std::string::npos; };
         a.common.run_abo = has("abo");
         a.common.run_qrd = has("qrd");
         a.common.run_krls = has("krls");
      }

      // ---- ABO ----
      else if (is_flag(argv[i], "--abo_lags"))
      {
         need("--abo_lags");
         a.abo.L = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--abo_window"))
      {
         need("--abo_window");
         a.abo.W = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--abo_sigma"))
      {
         need("--abo_sigma");
         a.abo.sigma = std::stod(argv[++i]);
      }
      else if (is_flag(argv[i], "--abo_D"))
      {
         need("--abo_D");
         a.abo.D = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--abo_log2D"))
      {
         need("--abo_log2D");
         a.abo.log2D = std::stoi(argv[++i]);
      }

      // ---- QRD ----
      else if (is_flag(argv[i], "--qrd_lags"))
      {
         need("--qrd_lags");
         a.qrd.L = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--qrd_window"))
      {
         need("--qrd_window");
         a.qrd.W = std::stoi(argv[++i]);
      }

      // ---- KRLS ----
      else if (is_flag(argv[i], "--krls_lags"))
      {
         need("--krls_lags");
         a.krls.L = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--krls_window"))
      {
         need("--krls_window");
         a.krls.W = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--krls_sigma"))
      {
         need("--krls_sigma");
         a.krls.sigma = std::stod(argv[++i]);
      }
   }

   if (a.abo.log2D >= 0)
      a.abo.D = 1 << a.abo.log2D;

   if (a.common.end_k <= a.common.start_k)
      throw std::runtime_error("--end_k must be > --start_k");

   if (!a.common.run_abo && !a.common.run_qrd && !a.common.run_krls)
      throw std::runtime_error("--run must include at least one of: abo,qrd,krls");

   if (a.common.run_abo)
   {
      if (a.abo.L <= 0 || a.abo.W <= 0 || a.abo.sigma <= 0.0 || a.abo.D <= 0)
         throw std::runtime_error("ABO needs --abo_lags --abo_window --abo_sigma and --abo_D/--abo_log2D");
   }
   if (a.common.run_qrd)
   {
      if (a.qrd.L <= 0 || a.qrd.W <= 0)
         throw std::runtime_error("QRD needs --qrd_lags --qrd_window");
   }
   if (a.common.run_krls)
   {
      if (a.krls.L <= 0 || a.krls.W <= 0 || a.krls.sigma <= 0.0)
         throw std::runtime_error("KRLS needs --krls_lags --krls_window --krls_sigma");
   }
}

// ---- helpers (same logic as your code) ----

static inline void lag_matrix(
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

static inline void dataset_creation(
    std::vector<std::vector<double>> &data_set,
    std::vector<double> &target_data,
    Eigen::MatrixXd &initial_matrix,
    Eigen::MatrixXd &update_matrix,
    double *y,
    double *&y_update,
    int num_rows,
    int num_cols,
    int start_row)
{
   const int remaining = static_cast<int>(target_data.size()) - start_row;
   const int num_elements = remaining - num_rows;

   if (num_elements <= 0)
   {
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

static inline void get_var(const std::vector<double> &se, double mean_se, double &var, int n)
{
   var = 0.0;
   for (int i = 0; i < n; ++i)
   {
      double t = se[i] - mean_se;
      var += t * t;
   }
}

// ---- data loader (shared) ----

struct RawSeries
{
   std::vector<double> x;
};

static inline RawSeries load_series()
{
   RawSeries s;
   std::vector<std::vector<std::string>> raw_data =
       read_csv_func("data/EURUSD/raw_norm_EURUSD.csv");

   const int len_raw_data = static_cast<int>(raw_data.size()) - 1;
   s.x.reserve(std::max(0, len_raw_data - 1));
   for (int i = 1; i < len_raw_data; ++i)
      s.x.push_back(std::stod(raw_data[i][0]));
   return s;
}

// ---- per-model fold runners ----

static inline FoldResultRow run_fold_abo(
    const RawSeries &series,
    int first_date,
    int fold_k,
    int W, int L,
    double sigma,
    int D,
    int val_length,
    int warmup,
    double ff,
    double regularizer)
{
   using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

   FoldResultRow row;
   row.fold = fold_k;
   row.model = "ABO";
   row.L = L;
   row.W = W;
   row.sigma = sigma;
   row.D = D;

   std::vector<std::vector<double>> data_set;
   std::vector<double> target_data;
   lag_matrix(series.x, L, data_set, target_data);

   MatrixXd initial_matrix, update_matrix;
   std::vector<double> y_vec(W);
   double *y = y_vec.data();
   double *y_update = nullptr;

   const int start_row = first_date + val_length * fold_k;

   dataset_creation(data_set, target_data, initial_matrix, update_matrix,
                    y, y_update, W, L, start_row);

   if (y_update == nullptr || update_matrix.rows() < val_length)
   {
      delete[] y_update;
      return row;
   }

   const bool seed = true;
   GaussianRFF g_rff(L, D, sigma, seed);

   MatrixXd X_old = g_rff.transform_matrix(initial_matrix);
   std::vector<double> X(static_cast<size_t>(W) * static_cast<size_t>(D));
   for (int j = 0; j < D; ++j)
      for (int i = 0; i < W; ++i)
         X[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(W)] = X_old(i, j);

   ABO abo(X.data(), y, W, ff, D, W);

   std::vector<double> se;
   se.reserve(val_length);

   std::vector<double> X_update(static_cast<size_t>(D));

   double ns_rff = 0.0;
   double ns_update = 0.0;
   double se_sum = 0.0;

   const int n_its = val_length;
   const int eff_its = std::max(1, n_its - warmup);

   for (int i = 0; i < n_its; ++i)
   {
      auto t0 = Clock::now();
      MatrixXd X_up = g_rff.transform(update_matrix.row(i));
      auto t1 = Clock::now();
      if (i >= warmup)
         ns_rff += std::chrono::duration<double, std::nano>(t1 - t0).count();

      for (int j = 0; j < D; ++j)
         X_update[static_cast<size_t>(j)] = X_up(0, j);

      double pred = 0.0;
      t0 = Clock::now();
      pred = abo.pred(X_update.data());
      abo.update(X_update.data(), y_update[i]);
      t1 = Clock::now();
      if (i >= warmup)
         ns_update += std::chrono::duration<double, std::nano>(t1 - t0).count();

      double e = pred - y_update[i];
      double r2 = e * e;
      se_sum += r2;
      se.push_back(r2);
   }

   row.s_rff = ns_to_s(ns_rff);
   row.s_update = ns_to_s(ns_update);
   row.us_rff = ns_to_us(ns_rff / eff_its);
   row.us_update = ns_to_us(ns_update / eff_its);

   const double mean_se = se_sum / n_its;
   double var_se = 0.0;
   get_var(se, mean_se, var_se, n_its);

   row.mse = mean_se;
   row.var = var_se / (n_its - 1);

   delete[] y_update;
   return row;
}

static inline FoldResultRow run_fold_qrd(
    const RawSeries &series,
    int first_date,
    int fold_k,
    int W, int L,
    int val_length,
    int warmup,
    double ff,
    double regularizer)
{
   FoldResultRow row;
   row.fold = fold_k;
   row.model = "QRD-RLS";
   row.L = L;
   row.W = W;
   row.D = 0;

   std::vector<std::vector<double>> data_set;
   std::vector<double> target_data;
   lag_matrix(series.x, L, data_set, target_data);

   Eigen::MatrixXd initial_matrix, update_matrix;
   std::vector<double> y_vec(W);
   double *y = y_vec.data();
   double *y_update = nullptr;

   const int start_row = first_date + val_length * fold_k;

   dataset_creation(data_set, target_data, initial_matrix, update_matrix,
                    y, y_update, W, L, start_row);

   if (y_update == nullptr || update_matrix.rows() < val_length)
   {
      delete[] y_update;
      return row;
   }

   std::vector<double> X_no_rff(static_cast<size_t>(W) * static_cast<size_t>(L));
   for (int j = 0; j < L; ++j)
      for (int i = 0; i < W; ++i)
         X_no_rff[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(W)] = initial_matrix(i, j);

   QRDRLS qrd(W, L, ff, regularizer);
   qrd.batchInitialize(X_no_rff.data(), y, W, L);

   std::vector<double> x_no_rff(static_cast<size_t>(L));
   std::vector<double> se;
   se.reserve(val_length);

   double ns_update = 0.0;
   double se_sum = 0.0;

   const int n_its = val_length;
   const int eff_its = std::max(1, n_its - warmup);

   for (int i = 0; i < n_its; ++i)
   {
      for (int j = 0; j < L; ++j)
         x_no_rff[static_cast<size_t>(j)] = update_matrix(i, j);

      double pred = 0.0, eps_post = 0.0;
      auto t0 = Clock::now();
      qrd.update(x_no_rff.data(), y_update[i], pred, eps_post);
      auto t1 = Clock::now();
      if (i >= warmup)
         ns_update += std::chrono::duration<double, std::nano>(t1 - t0).count();

      double r2 = eps_post * eps_post;
      se_sum += r2;
      se.push_back(r2);
   }

   row.s_update = ns_to_s(ns_update);
   row.us_update = ns_to_us(ns_update / eff_its);

   const double mean_se = se_sum / n_its;
   double var_se = 0.0;
   get_var(se, mean_se, var_se, n_its);

   row.mse = mean_se;
   row.var = var_se / (n_its - 1);

   delete[] y_update;
   return row;
}

static inline FoldResultRow run_fold_krls(
    const RawSeries &series,
    int first_date,
    int fold_k,
    int W, int L,
    double sigma,
    int val_length,
    int warmup,
    double ff,
    double regularizer)
{
   FoldResultRow row;
   row.fold = fold_k;
   row.model = "KRLS-RBF";
   row.L = L;
   row.W = W;
   row.sigma = sigma;
   row.D = 0;

   std::vector<std::vector<double>> data_set;
   std::vector<double> target_data;
   lag_matrix(series.x, L, data_set, target_data);

   Eigen::MatrixXd initial_matrix, update_matrix;
   std::vector<double> y_vec(W);
   double *y = y_vec.data();
   double *y_update = nullptr;

   const int start_row = first_date + val_length * fold_k;

   dataset_creation(data_set, target_data, initial_matrix, update_matrix,
                    y, y_update, W, L, start_row);

   if (y_update == nullptr || update_matrix.rows() < val_length)
   {
      delete[] y_update;
      return row;
   }

   std::vector<double> X_no_rff(static_cast<size_t>(W) * static_cast<size_t>(L));
   for (int j = 0; j < L; ++j)
      for (int i = 0; i < W; ++i)
         X_no_rff[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(W)] = initial_matrix(i, j);

   const double temp_sigma = 1.0 / sigma;
   KRLS_RBF krls(X_no_rff.data(), y, W, L, regularizer, temp_sigma, W);

   std::vector<double> x_no_rff(static_cast<size_t>(L));
   std::vector<double> se;
   se.reserve(val_length);

   double ns_update = 0.0;
   double se_sum = 0.0;

   const int n_its = val_length;
   const int eff_its = std::max(1, n_its - warmup);

   for (int i = 0; i < n_its; ++i)
   {
      for (int j = 0; j < L; ++j)
         x_no_rff[static_cast<size_t>(j)] = update_matrix(i, j);

      double pred = 0.0, eps_post = 0.0;
      auto t0 = Clock::now();
      krls.update(x_no_rff.data(), y_update[i], pred, eps_post);
      auto t1 = Clock::now();
      if (i >= warmup)
         ns_update += std::chrono::duration<double, std::nano>(t1 - t0).count();

      double r2 = eps_post * eps_post;
      se_sum += r2;
      se.push_back(r2);
   }

   row.s_update = ns_to_s(ns_update);
   row.us_update = ns_to_us(ns_update / eff_its);

   const double mean_se = se_sum / n_its;
   double var_se = 0.0;
   get_var(se, mean_se, var_se, n_its);

   row.mse = mean_se;
   row.var = var_se / (n_its - 1);

   delete[] y_update;
   return row;
}

// ---- CSV output ----

static inline void save_rows_csv(const std::vector<FoldResultRow> &rows, const std::string &path)
{
   std::ofstream f(path);
   if (!f)
   {
      std::cerr << "Error opening file: " << path << "\n";
      return;
   }

   f << "fold,model,L,W,sigma,D,mse,var,us_rff,us_update,s_rff,s_update\n";
   for (const auto &r : rows)
   {
      f << r.fold << ","
        << r.model << ","
        << r.L << ","
        << r.W << ","
        << r.sigma << ","
        << r.D << ","
        << r.mse << ","
        << r.var << ","
        << r.us_rff << ","
        << r.us_update << ","
        << r.s_rff << ","
        << r.s_update
        << "\n";
   }
}

// ---- main ----

int main(int argc, char **argv)
{
   Args args;
   try
   {
      parse_args(argc, argv, args);
   }
   catch (const std::exception &e)
   {
      std::cerr << "Arg error: " << e.what() << "\n\n";
      std::cerr
          << "Example:\n"
          << "  ./best_test \\\n"
          << "    --run abo,qrd,krls \\\n"
          << "    --first_date 5376 --start_k 0 --end_k 5 --val_length 1344 --warmup 50 \\\n"
          << "    --abo_lags 19 --abo_window 20 --abo_sigma 6.50586 --abo_log2D 11 \\\n"
          << "    --qrd_lags 48 --qrd_window 128 \\\n"
          << "    --krls_lags 25 --krls_window 261 --krls_sigma 4.2 \\\n"
          << "    --out_csv results/gridsearch/EURUSD/best_test.csv\n";
      return 1;
   }

   RawSeries series = load_series();

   std::vector<int> folds(args.common.end_k - args.common.start_k);
   std::iota(folds.begin(), folds.end(), args.common.start_k);

   std::vector<FoldResultRow> rows;
   rows.reserve(static_cast<size_t>(folds.size()) * 3);

   // model summaries (means over folds)
   struct Agg
   {
      int n = 0;
      double mse_sum = 0.0, var_sum = 0.0;
      double us_rff_sum = 0.0, us_update_sum = 0.0;
   };
   Agg agg_abo, agg_qrd, agg_krls;

   for (int k : folds)
   {
      const int fold_idx = k - args.common.start_k;

      if (args.common.run_abo)
      {
         std::cout << "\n[ABO] fold=" << fold_idx
                   << " L=" << args.abo.L
                   << " W=" << args.abo.W
                   << " sigma=" << args.abo.sigma
                   << " D=" << args.abo.D
                   << "\n";

         FoldResultRow r = run_fold_abo(series,
                                        args.common.first_date, k,
                                        args.abo.W, args.abo.L,
                                        args.abo.sigma, args.abo.D,
                                        args.common.val_length, args.common.warmup,
                                        args.abo.ff, args.abo.regularizer);

         rows.push_back(r);
         agg_abo.n++;
         agg_abo.mse_sum += r.mse;
         agg_abo.var_sum += r.var;
         agg_abo.us_rff_sum += r.us_rff;
         agg_abo.us_update_sum += r.us_update;
      }

      if (args.common.run_qrd)
      {
         std::cout << "\n[QRD-RLS] fold=" << fold_idx
                   << " L=" << args.qrd.L
                   << " W=" << args.qrd.W
                   << "\n";

         FoldResultRow r = run_fold_qrd(series,
                                        args.common.first_date, k,
                                        args.qrd.W, args.qrd.L,
                                        args.common.val_length, args.common.warmup,
                                        args.qrd.ff, args.qrd.regularizer);

         rows.push_back(r);
         agg_qrd.n++;
         agg_qrd.mse_sum += r.mse;
         agg_qrd.var_sum += r.var;
         agg_qrd.us_update_sum += r.us_update;
      }

      if (args.common.run_krls)
      {
         std::cout << "\n[KRLS-RBF] fold=" << fold_idx
                   << " L=" << args.krls.L
                   << " W=" << args.krls.W
                   << " sigma=" << args.krls.sigma
                   << "\n";

         FoldResultRow r = run_fold_krls(series,
                                         args.common.first_date, k,
                                         args.krls.W, args.krls.L,
                                         args.krls.sigma,
                                         args.common.val_length, args.common.warmup,
                                         args.krls.ff, args.krls.regularizer);

         rows.push_back(r);
         agg_krls.n++;
         agg_krls.mse_sum += r.mse;
         agg_krls.var_sum += r.var;
         agg_krls.us_update_sum += r.us_update;
      }
   }

   save_rows_csv(rows, args.common.out_csv);

   auto mean_or_nan = [](double s, int n)
   { return (n > 0) ? (s / n) : std::numeric_limits<double>::quiet_NaN(); };

   std::cout << "\nSUMMARY "
             << "folds=" << (args.common.end_k - args.common.start_k) << " "
             << "out_csv=" << args.common.out_csv << " ";

   if (args.common.run_abo)
   {
      std::cout << "abo(L=" << args.abo.L << ",W=" << args.abo.W << ",sigma=" << args.abo.sigma << ",D=" << args.abo.D << ") "
                << "mse=" << mean_or_nan(agg_abo.mse_sum, agg_abo.n) << " "
                << "var=" << mean_or_nan(agg_abo.var_sum, agg_abo.n) << " "
                << "us_rff=" << mean_or_nan(agg_abo.us_rff_sum, agg_abo.n) << " "
                << "us_update=" << mean_or_nan(agg_abo.us_update_sum, agg_abo.n) << " ";
   }

   if (args.common.run_qrd)
   {
      std::cout << "qrd(L=" << args.qrd.L << ",W=" << args.qrd.W << ") "
                << "mse=" << mean_or_nan(agg_qrd.mse_sum, agg_qrd.n) << " "
                << "var=" << mean_or_nan(agg_qrd.var_sum, agg_qrd.n) << " "
                << "us_update=" << mean_or_nan(agg_qrd.us_update_sum, agg_qrd.n) << " ";
   }

   if (args.common.run_krls)
   {
      std::cout << "krls(L=" << args.krls.L << ",W=" << args.krls.W << ",sigma=" << args.krls.sigma << ") "
                << "mse=" << mean_or_nan(agg_krls.mse_sum, agg_krls.n) << " "
                << "var=" << mean_or_nan(agg_krls.var_sum, agg_krls.n) << " "
                << "us_update=" << mean_or_nan(agg_krls.us_update_sum, agg_krls.n) << " ";
   }

   std::cout << "\n";
   return 0;
}
