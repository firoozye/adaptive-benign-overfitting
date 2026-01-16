#include "abo/dd_test.h"
#include "baselines/QRD_RLS/qrd_rls.h"
#include "baselines/KRLS_RBF/krls_rbf.h"

using namespace std;

#include <vector>
#include <cstring>

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

void parse_args(int argc, char **argv,
                int &L, int &W, double &sigma, int &K)
{
   for (int i = 1; i < argc; ++i)
   {
      if (strcmp(argv[i], "--lags") == 0)
         L = std::stoi(argv[++i]);
      else if (strcmp(argv[i], "--window") == 0)
         W = std::stoi(argv[++i]);
      else if (strcmp(argv[i], "--sigma") == 0)
         sigma = std::stod(argv[++i]);
      else if (strcmp(argv[i], "--kfolds") == 0)
         K = std::stoi(argv[++i]);
   }
}

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
   file << "k,L,W,"
        << "mse_abo,var_abo,"
        << "mse_qrd,var_qrd,"
        << "mse_krls,var_krls\n";

   // ---- rows ----
   for (const auto &r : results)
   {
      file << r.k << "," << r.L << "," << r.W << ","
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

ValMetrics cross_val(int num_rows, int num_cols, double sigma, int k_fold)
{
   typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

   vector<vector<double>> data_set;
   vector<double> target_data;
   vector<vector<string>> raw_data;
   // data_set = read_csv_func("data/electricity/lags_LD2011_2014.csv");
   // target_data = read_csv_func("data/electricity/target_LD2011_2014.csv");
   raw_data = read_csv_func("data/EURUSD/raw_norm_EURUSD.csv");
   vector<double> close_price;
   vector<double> ret_price;
   vector<double> raw_data_dob;
   int len_raw_data = raw_data.size() - 1;

   for (int i = 1; i < len_raw_data; ++i)
   {
      // ret_price.push_back(stod(target_data[i][0]));
      raw_data_dob.push_back(stod(raw_data[i][0]));
   }

   int lags = num_cols;
   lag_matrix(raw_data_dob, lags, data_set, target_data);

   Eigen::MatrixXd initial_matrix;
   Eigen::MatrixXd update_matrix;
   int d = num_cols;
   double y[num_rows];
   double *y_update;
   // double val_length = 2880;
   // double val_length = 1440;
   double val_length = 960;
   double start_date = val_length * k_fold;
   int len_data_set = data_set.size() - 1;

   dataset_creation(data_set, target_data,
                    initial_matrix, update_matrix, d, y, y_update, num_rows, num_cols,
                    start_date);

   vector<double> all_mse_array;
   vector<double> all_var_array;
   vector<double> all_cond_num_mean_array;
   vector<double> all_cond_num_var_array;

   // int D = pow(2, 13);
   // int D = pow(2, 12);
   int D = pow(2, 11);
   double kernel_var = sigma;
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
   double regularizer = 1e-2;

   // our model
   ABO abo(X, y, max_obs, ff, D, num_rows);

   vector<double> preds_abo;
   vector<double> mse_abo;
   double all_mse_abo = 0;
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

      double temp_res = pow(temp_pred - y_update[i], 2);
      mse_abo.push_back(temp_res);
      all_mse_abo += temp_res;
   }

   double var_abo = 0;
   double real_mse_abo = all_mse_abo / n_its;

   get_var(mse_abo, real_mse_abo, var_abo, n_its);

   m.mse_abo = real_mse_abo;
   m.var_abo = var_abo / (n_its - 1);

   delete[] X;
   delete[] y_update;
   return m;
}

int main(int argc, char **argv)
{
   // -----------------------------
   // Hyperparameters (from Optuna)
   // -----------------------------
   int L = -1, W = -1, K = -1;
   double sigma = -1.0;

   parse_args(argc, argv, L, W, sigma, K);

   // -----------------------------
   // Accumulators
   // -----------------------------
   double mse_abo_sum = 0.0;
   double var_abo_sum = 0.0;

   // -----------------------------
   // Cross-validation loop
   // -----------------------------
   for (int k = 0; k < K; ++k)
   {
      ValMetrics m = cross_val(W, L, sigma, k);

      mse_abo_sum += m.mse_abo;
      var_abo_sum += m.var_abo;
   }

   // -----------------------------
   // Means across folds
   // -----------------------------
   double mean_mse_abo = mse_abo_sum / K;
   double mean_var_abo = var_abo_sum / K;

   // -----------------------------
   // Output for Optuna
   // -----------------------------
   // stdout: machine-readable
   // stderr: logs if needed
   std::cout << mean_mse_abo << " " << mean_var_abo << std::endl;

   return 0;
}
