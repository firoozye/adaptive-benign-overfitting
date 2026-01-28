#include "abo/dd_test.h"
#include "baselines/QRD_RLS/qrd_rls.h"
#include "baselines/KRLS_RBF/krls_rbf.h"

using namespace std;

#include <vector>
#include <cstring>

enum class ModelType
{
   ABO,
   QRD,
   KRLS,
   ALL
};

inline const char *model_name_to_string(ModelType model)
{
   switch (model)
   {
   case ModelType::ABO:
      return "ABO";
   case ModelType::QRD:
      return "QRD-RLS";
   case ModelType::KRLS:
      return "KRLS-RBF";
   case ModelType::ALL:
      return "ALL";
   default:
      return "UNKNOWN";
   }
}

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

struct FullResults
{
   vector<double> resid_abo;
   vector<double> resid_qrd;
   vector<double> resid_krls;
};

ModelType parse_model(const std::string &name)
{
   if (name == "abo")
      return ModelType::ABO;
   if (name == "qrd")
      return ModelType::QRD;
   if (name == "krls")
      return ModelType::KRLS;
   if (name == "all")
      return ModelType::ALL;

   throw std::runtime_error("Unknown model_name");
}

void parse_args(int argc, char **argv,
                int &L, int &W, double &sigma, int &K,
                std::string &model_name)
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
      else if (strcmp(argv[i], "--model_name") == 0)
         model_name = argv[++i];
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

ValMetrics cross_val(int num_rows, int num_cols, double sigma, int k_fold, ModelType model)
{
   using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

   // ---- load/prepare data (unchanged) ----
   std::vector<std::vector<std::string>> raw_data = read_csv_func("data/EURUSD/raw_norm_EURUSD.csv");
   std::vector<double> raw_data_dob;
   int len_raw_data = static_cast<int>(raw_data.size()) - 1;
   raw_data_dob.reserve(static_cast<size_t>(len_raw_data));

   for (int i = 1; i < len_raw_data; ++i)
      raw_data_dob.push_back(std::stod(raw_data[i][0]));

   std::vector<std::vector<double>> data_set;
   std::vector<double> target_data;
   int lags = num_cols;
   lag_matrix(raw_data_dob, lags, data_set, target_data);

   MatrixXd initial_matrix, update_matrix;

   int d = num_cols;
   std::vector<double> y(num_rows);
   double *y_update = nullptr;

   int W_max = 600; // maximum batch size before anchor
   int val_length = 960;

   // anchor: the first time index you start forecasting for this fold
   int t0 = W_max + val_length * k_fold;

   // model-specific initial start (so you get exactly num_rows history before t0)
   int start_row = t0 - num_rows;
   int start_date = val_length * k_fold;

   // std::cout << "hola"
   ///          << std::endl;

   dataset_creation(data_set, target_data,
                    initial_matrix, update_matrix, d,
                    y.data(), y_update, num_rows, num_cols,
                    start_date);

   // std::cout << "addios"
   //           << std::endl;

   // ---- shared params ----
   int D = static_cast<int>(std::pow(2, 11));
   double kernel_var = sigma;
   bool seed = true;

   double ff = 1.0;
   double regularizer = 1e-2;

   // No-RFF input for windowed methods (only build if needed)
   std::vector<double> X_no_rff;
   if (model == ModelType::QRD || model == ModelType::KRLS || model == ModelType::ALL)
   {
      X_no_rff.resize(static_cast<size_t>(num_rows) * static_cast<size_t>(d));
      for (int j = 0; j < d; ++j)
         for (int i = 0; i < num_rows; ++i)
            X_no_rff[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(num_rows)] = initial_matrix(i, j);
   }

   // ---- instantiate only what you need ----
   std::unique_ptr<GaussianRFF> g_rff;
   std::unique_ptr<ABO> abo;

   std::unique_ptr<QRDRLS> qrd_rls;
   std::unique_ptr<KRLS_RBF> krls_rbf;

   if (model == ModelType::ABO || model == ModelType::ALL)
   {
      g_rff = std::make_unique<GaussianRFF>(d, D, kernel_var, seed);

      MatrixXd X_old = g_rff->transform_matrix(initial_matrix);

      std::vector<double> X(static_cast<size_t>(num_rows) * static_cast<size_t>(D));
      for (int j = 0; j < D; ++j)
         for (int i = 0; i < num_rows; ++i)
            X[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(num_rows)] = X_old(i, j);

      abo = std::make_unique<ABO>(X.data(), y.data(), num_rows, ff, D, num_rows);
      // NOTE: This assumes ABO copies X internally or only uses it during ctor.
      // If ABO stores X pointer, keep X alive (make X a member / static / move into abo wrapper).
   }

   if (model == ModelType::QRD || model == ModelType::ALL)
   {
      qrd_rls = std::make_unique<QRDRLS>(num_rows, num_cols, ff, regularizer);
      qrd_rls->batchInitialize(X_no_rff.data(), y.data(), num_rows, num_cols);
   }

   if (model == ModelType::KRLS || model == ModelType::ALL)
   {
      const double temp_sigma = 1.0 / kernel_var;
      krls_rbf = std::make_unique<KRLS_RBF>(X_no_rff.data(), y.data(),
                                            num_rows, num_cols, regularizer,
                                            temp_sigma, num_rows);
   }

   // ---- update loop ----
   FullResults all_resid;
   double mse_abo = 0.0, mse_qrd = 0.0, mse_krls = 0.0;

   std::vector<double> X_update(D);
   std::vector<double> x_no_rff(d);

   int n_its = val_length;
   for (int i = 0; i < n_its; ++i)
   {
      if (abo)
      {
         MatrixXd X_update_old = g_rff->transform(update_matrix.row(i));
         for (int j = 0; j < D; ++j)
            X_update[j] = X_update_old(0, j);

         double pred = abo->pred(X_update.data());
         abo->update(X_update.data(), y_update[i]);

         double r2 = (pred - y_update[i]) * (pred - y_update[i]);
         mse_abo += r2;
         all_resid.resid_abo.push_back(r2);
      }

      if (qrd_rls || krls_rbf)
      {
         for (int j = 0; j < d; ++j)
            x_no_rff[j] = update_matrix(i, j);
      }

      if (qrd_rls)
      {
         double pred = 0.0, eps_post = 0.0;
         qrd_rls->update(x_no_rff.data(), y_update[i], pred, eps_post);

         double r2 = eps_post * eps_post;
         mse_qrd += r2;
         all_resid.resid_qrd.push_back(r2);
      }

      if (krls_rbf)
      {
         double pred = 0.0, eps_post = 0.0;
         krls_rbf->update(x_no_rff.data(), y_update[i], pred, eps_post);

         double r2 = eps_post * eps_post;
         mse_krls += r2;
         all_resid.resid_krls.push_back(r2);
      }
   }

   // ---- finalize only what ran ----
   ValMetrics m{};
   auto nan = std::numeric_limits<double>::quiet_NaN();
   m.mse_abo = m.var_abo = nan;
   m.mse_qrd = m.var_qrd = nan;
   m.mse_krls = m.var_krls = nan;

   if (abo)
   {
      mse_abo /= n_its;
      double var_abo = 0.0;
      get_var(all_resid.resid_abo, mse_abo, var_abo, n_its);
      m.mse_abo = mse_abo;
      m.var_abo = var_abo / (n_its - 1);
   }

   if (qrd_rls)
   {
      mse_qrd /= n_its;
      double var_qrd = 0.0;
      get_var(all_resid.resid_qrd, mse_qrd, var_qrd, n_its);
      m.mse_qrd = mse_qrd;
      m.var_qrd = var_qrd / (n_its - 1);
   }

   if (krls_rbf)
   {
      mse_krls /= n_its;
      double var_krls = 0.0;
      get_var(all_resid.resid_krls, mse_krls, var_krls, n_its);
      m.mse_krls = mse_krls;
      m.var_krls = var_krls / (n_its - 1);
   }

   delete[] y_update;
   return m;
}

int main(int argc, char **argv)
{
   int L = -1, W = -1, K = -1;
   double sigma = -1.0;
   std::string model_name = "none";

   parse_args(argc, argv, L, W, sigma, K, model_name);

   // std::cerr << "L=" << L << " W=" << W << " K=" << K << " sigma=" << sigma
   //           << " model=" << model_name << "\n";

   ModelType model = parse_model(model_name);

   // std::cout << "Running model: "
   //          << model_name_to_string(model)
   //          << std::endl;

   double mse_sum = 0.0;
   double var_sum = 0.0;

   int count = 0;

   for (int k = 0; k < K; ++k)
   {
      ValMetrics m = cross_val(W, L, sigma, k, model);

      if (model == ModelType::ABO || model == ModelType::ALL)
      {
         mse_sum += m.mse_abo;
         var_sum += m.var_abo;
         count++;
      }

      if (model == ModelType::QRD || model == ModelType::ALL)
      {
         mse_sum += m.mse_qrd;
         var_sum += m.var_qrd;
         count++;
      }

      if (model == ModelType::KRLS || model == ModelType::ALL)
      {
         mse_sum += m.mse_krls;
         var_sum += m.var_krls;
         count++;
      }
   }

   double mean_mse = mse_sum / count;
   double mean_var = var_sum / count;

   std::cout << mean_mse << " " << mean_var << std::endl;
}
