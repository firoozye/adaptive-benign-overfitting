Adaptive Benign Overfitting (ABO) C++ Optimization Plan

1. Architectural Overview & State ReductionThe current implementation suffers from severe memory bloat and heap-allocation bottlenecks. This optimization plan strips the algorithm to its theoretical minimum state by eliminating redundant matrices and replacing dynamic heap allocations with pre-allocated, in-place memory shifts.Key Mathematical Reductions:Q-Less Downdate: We drop the $N \times N$ orthogonal matrix $Q$. The downdating rotation target $h$ (which corresponds to the first row of $Q$) is computed directly via $h^\top = z_{old}^\top R^\dagger$.Z-Less Feature History: We drop the $N \times D$ expanded feature matrix $Z$. The main loop will instead maintain a tiny $N \times d$ ring buffer of the raw features and compute $z_{old}$ using Random Fourier Features (RFF) on the fly during a downdate.Key Memory Management Reductions:Eliminate add_row_col.cpp completely. No more new double[] or delete[] in the streaming loop.Matrices $R$, $R^\dagger$, and $y$ will be statically pre-allocated to their maximum possible sizes (max_obs_ * dim_).Rows/columns are "deleted" using fast, contiguous std::memmove operations.

2. Header ModificationsUpdate ABO.hRemove all references to $Q$, the historical features $X$, and $z$. Pre-allocate sizes based on max_obs_.C++// Remove these completely:
// double *X_;
// double *Q_;
// double *z_;

// Modify the downdate signature:
void downdate(double* z_old, double y_old);

// Retained State (Statically sized in constructor up to max_obs_):
double *y_;       // Size: max_obs_ 
double *R_;       // Size: max_obs_ * dim_ (Column-major)
double *R_inv_;   // Size: dim_ * max_obs_ (Column-major)
double *beta_;    // Size: dim_
double *G_e_1_;   // Size: max_obs_ + 1
double *G_;       // Size: (max_obs_ + 1) * (max_obs_ + 1)
Update last_row_givens.hModify the downdate signature to accept the expired feature vector.C++namespace givens {
    void update(ABO *abo);
    void downdate(ABO *abo, double *z_old); 
}

3. Core Implementation PatchesPatch 1: Pre-allocation & In-Place Shifts (ABO.cpp)Instruct the agent to modify the ABO constructor to allocate arrays to their maximum sizes once. Then, rewrite downdate to use std::memmove instead of the helper functions from add_row_col.cpp.C++

// 1. Constructor: Pre-allocate to maximum capacity
ABO::ABO(double *x_input, double *y_input, int max_obs, double ff, int dim, int n_batch)
    // ... initializers ...
{
    // Allocate to MAXIMUM capacity so we never call 'new' again
    R_ = new double[max_obs_ * dim_]();
    R_inv_ = new double[dim_ * max_obs_]();
    y_ = new double[max_obs_]();
    
    // ... copy initial batch data and run batchInitialize()
}

// 2. Downdate: Shift memory in-place
void ABO::downdate(double* z_old, double y_old)
{
   // Execute the Q-less Givens downdate
   givens::downdate(this, z_old);

   double *x_T = R_; // Pointer to the first row of R
   
   if (n_obs_ < dim_) {
       // ... existing new regime math (unchanged) ...
   } else {
       // ... existing old regime math ...
       
       // Ensure weight update uses the y_old parameter passed in
       double x_T_B = cblas_ddot(dim_, x_T, n_obs_, beta_, 1);
       for (int i = 0; i < dim_; i++) {
           beta_[i] -= (1.0 / s) * (y_old - x_T_B) * h[i];
       }
       // ... rest of old regime math ...
   }

   // --- IN-PLACE MEMORY SHIFTING (Replaces deleteRowColMajor) ---
   
   // 1. Delete first row of R (Size: n_obs_ x dim_, Col-Major)
   // Shift each column up by 1 element
   for (int j = 0; j < dim_; ++j) {
       std::memmove(&R_[j * (n_obs_ - 1)], &R_[j * n_obs_ + 1], (n_obs_ - 1) * sizeof(double));
   }

   // 2. Delete first column of R_inv (Size: dim_ x n_obs_, Col-Major)
   // Shift all subsequent columns left by 1 column
   std::memmove(R_inv_, R_inv_ + dim_, dim_ * (n_obs_ - 1) * sizeof(double));

   // 3. Delete first element of y
   std::memmove(y_, y_ + 1, (n_obs_ - 1) * sizeof(double));

   n_obs_--;
   r_c_size_ = n_obs_ * dim_;
}
Patch 2: Q-Less Downdate Logic (last_row_givens.cpp)Instruct the agent to replace the givens::downdate function entirely. This implementation computes $h$ directly and drops all rotations of the $Q$ matrix.C++void givens::downdate(ABO *abo, double *z_old)
{
   int n_obs = abo->n_obs_;
   int dim = abo->dim_;
   double *R = abo->R_;
   double *G = abo->G_;
   double *R_inv = abo->R_inv_;

   // 1. Compute h = R_inv^T * z_old (This effectively recreates the target row of Q)
   double *h = new double[n_obs];
   cblas_dgemv(CblasColMajor, CblasTrans,
               dim, n_obs, 1.0, R_inv, dim, z_old, 1, 0.0, h, 1);

   std::fill(G, G + n_obs * n_obs, 0.0);
   for (int i = 0; i < n_obs; i++) { G[i * n_obs + i] = 1.0; }

   double c, s, r;
   int row_stride = 1;

   // 2. Compute Givens rotations to zero out h from bottom up
   for (int i = n_obs - 1; i > 0; --i)
   {
      dlartg_(&h[i - 1], &h[i], &c, &s, &r);
      h[i - 1] = r;
      h[i] = 0.0;

      // Apply to G matrix (columns i-1 and i)
      int g_idx_1 = (i - 1) * n_obs; 
      int g_idx_2 = i * n_obs;       
      drot_(&n_obs, &G[g_idx_1], &row_stride, &G[g_idx_2], &row_stride, &c, &s);

      // Apply directly to R in the overparameterized regime
      if (dim > n_obs)
      {
         int n = dim - i + 1; 
         int inc = n_obs;     // Step to next column
         drot_(&n,
               &R[(i - 1) * n_obs + i - 1], &inc, 
               &R[(i - 1) * n_obs + i], &inc,     
               &c, &s);

         inc = 1;
         abo->giv_rots.push_back({(i - 1) * dim, i * dim, c, s});
      }
   }

   // 3. Finalize G_e_1 mapping
   if (dim > n_obs)
   {
      for (int t = 0; t < n_obs; t++) { abo->G_e_1_[t] = G[t]; }
   }
   else
   {
      if (h[0] < 0) {
         for (int i = 0; i < n_obs * n_obs; ++i) { G[i] *= -1; }
      }
      for (int t = 0; t < n_obs; t++) { abo->G_e_1_[t] = G[t]; }
      
      double *result = new double[n_obs * dim];
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                  n_obs, dim, n_obs, 1, G, n_obs, R, n_obs, 0, result, n_obs);
      std::memcpy(R, result, n_obs * dim * sizeof(double));
      delete[] result;
   }

   delete[] h; 
}
Patch 3: The Main Data Stream LoopThe main loop (e.g., dd_test.cpp) must take responsibility for tracking the raw features ($x_{old}$) and computing the expired RFF vector ($z_{old}$) on the fly.C++// --- Inside the main streaming execution loop ---

int d = raw_feature_dim; // e.g., 30
int N = max_obs;         // e.g., 300

// 1. Maintain lightweight circular buffers for raw inputs
std::vector<Eigen::VectorXd> X_raw_buffer(N, Eigen::VectorXd(d));
std::vector<double> y_buffer(N, 0.0);
int ring_idx = 0;

// ... For each incoming data point ...

// 2. Expand new point
Eigen::MatrixXd z_new_mat = rff.transform(raw_new_x.transpose());
double* z_new = z_new_mat.data();

// 3. Update the ABO filter
abo.update(z_new, new_y);

// 4. On-the-fly Downdate Trigger
if (abo.n_obs_ > N) 
{
    // Retrieve the oldest RAW data
    Eigen::VectorXd raw_old_x = X_raw_buffer[ring_idx];
    double old_y = y_buffer[ring_idx];

    // Compute the expired expanded vector ON THE FLY
    Eigen::MatrixXd z_old_mat = rff.transform(raw_old_x.transpose());
    double* z_old = z_old_mat.data();

    // Pass explicitly to ABO
    abo.downdate(z_old, old_y);
}

// 5. Overwrite oldest buffer data
X_raw_buffer[ring_idx] = raw_new_x;
y_buffer[ring_idx] = new_y;
ring_idx = (ring_idx + 1) % N; 
Cleanup Tasks for the AgentRemove add_row_col.cpp and add_row_col.h entirely from the build system.Modify QR_decomposition.cpp: Strip out all LAPACKE_dorgqr and LAPACKE_dormqr calls. The function should strictly return the $R$ matrix and clean up the workspace. Update the Q_R_compute signature to just double* R_compute(double *A, int m, int n).
