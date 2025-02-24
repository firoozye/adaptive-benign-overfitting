#include "main_noeigen.h"

// Create lag matrix using native arrays
double* create_lag_matrix(const std::vector<double>& data, int n_samples, int n_lags) {
   double* matrix = new double[n_samples * n_lags];

   // Column-major order (for BLAS/LAPACK compatibility)
   for (int col = 0; col < n_lags; col++) {
      for (int row = 0; row < n_samples; row++) {
         matrix[row + col * n_samples] = data[row + col];
      }
   }
   return matrix;
}

int main() {
   // Example parameters
   int feature_dim = 14;  // Number of features
   int rff_dim = 4000;    // Random Fourier Feature dimension
   double kernel_var = 1.0;
   int roll_window = 100;
   double forgetting_factor = 1.0;
   double lambda = 0.1;

   // Read data
   std::vector<std::vector<std::string>> data_set = read_csv_func("daily_vix.csv");
   std::vector<double> returns;

   // Process data - calculate log returns
   for (int i = 3; i < data_set.size(); i++) {
      double price = std::stod(data_set[i][4]);
      double prev_price = std::stod(data_set[i - 1][4]);
      returns.push_back(std::log(price / prev_price));
   }

   int n_samples = returns.size() - feature_dim;

   // Create feature matrix using lags
   double* features = create_lag_matrix(returns, n_samples, feature_dim);

   // Initialize RFF transformer (still uses Eigen internally, but we'll get raw output)
   GaussianRFF rff(feature_dim, rff_dim, kernel_var, true);

   // Transform features using RFF
   // We'll need to convert our native array to Eigen temporarily
   Eigen::Map<Eigen::MatrixXd> features_eigen(features, n_samples, feature_dim);
   Eigen::MatrixXd rff_features_eigen = rff.transform_matrix(features_eigen);

   // Convert back to native array
   double* rff_features = new double[n_samples * rff_dim];
   std::memcpy(rff_features, rff_features_eigen.data(),
               n_samples * rff_dim * sizeof(double));

   // Prepare initial training data
   double* X = new double[roll_window * rff_dim];
   double* y = new double[roll_window];

   // Fill initial training data (in column-major order)
   for (int j = 0; j < rff_dim; j++) {
      for (int i = 0; i < roll_window; i++) {
         X[i + j * roll_window] = rff_features[i + j * n_samples];
      }
   }

   for (int i = 0; i < roll_window; i++) {
      y[i] = returns[i + feature_dim];
   }

   // Initialize QR_RLS
   QR_Rls model(X, y, roll_window, forgetting_factor, lambda, rff_dim, roll_window);

   // Online learning loop
   std::vector<double> predictions;
   std::vector<double> actual_values;

   for (int t = roll_window; t < n_samples - 1; t++) {
      // Prepare new sample
      double* new_x = new double[rff_dim];
      for (int j = 0; j < rff_dim; j++) {
         new_x[j] = rff_features[t + j * n_samples];
      }
      double new_y = returns[t + feature_dim];

      // Make prediction before update
      double pred = model.pred(new_x);
      predictions.push_back(pred);
      actual_values.push_back(new_y);

      // Update model
      model.update(new_x, new_y);

      if (t % 100 == 0) {
         std::cout << "Step " << t << " completed" << std::endl;
      }

      delete[] new_x;
   }

   // Calculate MSE
   double mse = 0.0;
   for (size_t i = 0; i < predictions.size(); i++) {
      double error = predictions[i] - actual_values[i];
      mse += error * error;
   }
   mse /= predictions.size();

   std::cout << "Final MSE: " << mse << std::endl;

   // Cleanup
   delete[] features;
   delete[] rff_features;
   delete[] X;
   delete[] y;

   return 0;
}
