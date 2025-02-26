#include "main.h"

int main()
{
   // Example parameters
   int feature_dim = 14; // Number of features
   int rff_dim = 4000;   // Random Fourier Feature dimension
   double kernel_var = 1.0;
   int roll_window = 100;
   double forgetting_factor = 1.0;
   double lambda = 0.1;
   bool seed = true;

   // Read data
   std::vector<std::vector<std::string>> data_set = read_csv_func("data/daily_vix.csv");
   std::vector<double> close_price;
   std::vector<double> returns;

   // Process data
   for (int i = 2; i < static_cast<int>(data_set.size()); i++)
   {
      double price = std::stod(data_set[i][4]);
      close_price.push_back(price);

      if (i > 2)
      {
         double ret = std::log(price / std::stod(data_set[i - 1][4]));
         returns.push_back(ret);
      }
   }

   // Create lag matrix for features
   int n_samples = returns.size() - feature_dim;
   Eigen::MatrixXd features(n_samples, feature_dim);

   for (int i = 0; i < n_samples; i++)
   {
      for (int j = 0; j < feature_dim; j++)
      {
         features(i, j) = returns[i + j];
      }
   }

   // Initialize RFF transformer
   GaussianRFF rff(feature_dim, rff_dim, kernel_var, seed);

   // Transform features using RFF
   Eigen::MatrixXd rff_features = rff.transform_matrix(features);

   // Convert to C-style arrays for QR_RLS
   double *X = new double[roll_window * rff_dim];
   double *y = new double[roll_window];

   // Fill initial training data
   for (int j = 0; j < rff_dim; j++)
   {
      for (int i = 0; i < roll_window; i++)
      {
         X[i + j * roll_window] = rff_features(i, j);
      }
   }

   for (int i = 0; i < roll_window; i++)
   {
      y[i] = returns[i + feature_dim];
   }

   // Initialize QR_RLS
   QR_Rls model(X, y, roll_window, forgetting_factor, lambda, rff_dim, roll_window);

   // Online learning loop
   std::vector<double> predictions;
   std::vector<double> actual_values;

   for (int t = roll_window; t < n_samples - 1; t++)
   {
      // Prepare new sample
      double *new_x = new double[rff_dim];
      for (int j = 0; j < rff_dim; j++)
      {
         new_x[j] = rff_features(t, j);
      }
      double new_y = returns[t + feature_dim];

      // Make prediction before update
      double pred = model.pred(new_x);
      predictions.push_back(pred);
      actual_values.push_back(new_y);

      if (t % 100 == 0)
      {
         std::cout << "Step " << t << " completed" << std::endl;
      }

      // Update model
      model.update(new_x, new_y);

      delete[] new_x;
   }

   // Calculate MSE
   double mse = 0.0;
   for (size_t i = 0; i < predictions.size(); i++)
   {
      double error = predictions[i] - actual_values[i];
      mse += error * error;
   }
   mse /= predictions.size();

   std::cout << "Final MSE: " << mse << std::endl;

   // Cleanup
   delete[] X;
   delete[] y;

   return 0;
}
