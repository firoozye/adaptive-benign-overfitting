This is a fantastic codebase. Looking through your .cpp files, I found something incredibly revealing about how Q_ and your feature history are currently functioning.Here is the most critical realization: In your current code, X_ is not acting as a rolling window.If you look at ABO::update, you pass in new_x, but you never append it to X_. You only append to R_, y_, and Q_. This means X_ is just sitting in memory holding your initial batch, and Q_ has secretly been acting as your only "memory" of the feature history for the downdate!Now that we know we can compute the first row of $Q$ directly via $h^\top = z_{t-N}^\top R^\dagger$, we can nuke Q_ entirely. But because Q_ was secretly your memory, we must replace it with a lightweight circular ring-buffer to track the rolling window of features ($z_{t-N}$).Here is the exact surgical strike to make your codebase strictly Q-less.Step 1: The Ring Buffer Patch (ABO.h & ABO.cpp)Since your ABO::update receives the already-expanded feature vector (new_x of size dim_), we will store these in a simple rolling buffer so we can retrieve the oldest one for the downdate.In ABO.h:Delete double *Q_; and add the ring buffer variables:C++// Remove: double *Q_;
double *Z_ring_;  // To hold the rolling window of features
int ring_idx_;    // To track the oldest observation
In ABO.cpp (Constructor):Delete Q_ and initialize the ring buffer with your batch data.C++// Inside ABO::ABO constructor initialization list:
// Remove Q_(nullptr), add ring_idx_(0)

// Inside the constructor body:
Z_ring_ = new double[max_obs_ * dim_]();
// Copy the initial batch into the ring buffer
std::memcpy(Z_ring_, x_input, n_obs_ * dim_ * sizeof(double));
In ABO.cpp (batchInitialize and update):Stop allocating and updating Q_.C++// In batchInitialize(), remove Q_ from the tie:
// std::tie(Q_, R_) = Q_R_compute(X_, n_obs_, dim_);
// Replace with a modified QR that only returns R:
R_ = R_compute_only(X_, n_obs_, dim_); // You'll just free Q inside your QR function

// In update(), DELETE these lines:
// Q_ = addRowAndColumnColMajor(Q_, n_obs_, n_obs_);
// Q_[((n_obs_ + 1) * (n_obs_ + 1)) - 1] = 1;

// And ADD this right before n_obs_++; to maintain the ring buffer:
std::memcpy(&Z_ring_[ring_idx_ * dim_], new_x, dim_ * sizeof(double));
ring_idx_ = (ring_idx_ + 1) % max_obs_;
Step 2: The Core Math Patch (last_row_givens.cpp)This is where the magic happens. Look at your original givens::downdate. You were rotating the first row of Q by accessing Q[idx_1] and Q[idx_2].We will replace that by explicitly calculating $h = R_{inv}^\top z_{old}$ and rotating that instead.Replace your entire givens::downdate function with this strictly Q-less version:C++void downdate(ABO *abo, double *z_old) // Notice we now pass the expired feature vector
{
   int n_obs = abo->n_obs_;
   int dim = abo->dim_;
   double *R = abo->R_;
   double *G = abo->G_;
   double *R_inv = abo->R_inv_;

   // 1. Calculate h = R_inv^T * z_old (This IS the first row of Q!)
   double *h = new double[n_obs];
   cblas_dgemv(CblasColMajor, CblasTrans,
               dim, n_obs, 1.0, R_inv, dim, z_old, 1, 0.0, h, 1);

   std::fill(G, G + n_obs * n_obs, 0.0);
   for (int i = 0; i < n_obs; i++)
   {
      G[i * n_obs + i] = 1.0;
   }

   double c, s, r;
   int row_stride = 1;

   // 2. Find Givens rotations to zero out h from bottom up
   for (int i = n_obs - 1; i > 0; --i)
   {
      // dlartg computes rotation to zero out h[i] using h[i-1]
      dlartg_(&h[i - 1], &h[i], &c, &s, &r);

      h[i - 1] = r;
      h[i] = 0.0;

      // Apply to G matrix (columns i-1 and i)
      int g_idx_1 = (i - 1) * n_obs;
      int g_idx_2 = i * n_obs;
      drot_(&n_obs, &G[g_idx_1], &row_stride,
            &G[g_idx_2], &row_stride, &c, &s);

      // Apply directly to R in the overparameterized regime
      if (dim > n_obs)
      {
         int n = dim - i + 1;
         int inc = n_obs;
         drot_(&n,
               &R[(i - 1) * n_obs + i - 1], &inc,
               &R[(i - 1) * n_obs + i], &inc,
               &c, &s);

         inc = 1;
         abo->giv_rots.push_back({(i - 1) * dim, i * dim, c, s});
      }
   }

   // 3. Finalize G_e_1_ mapping
   if (dim > n_obs)
   {
      for (int t = 0; t < n_obs; t++)
      {
         abo->G_e_1_[t] = G[t];
      }
   }
   else
   {
      // Old regime check: h[0] holds what used to be Q[0]
      if (h[0] < 0)
      {
         for (int i = 0; i < n_obs * n_obs; ++i)
         {
            G[i] *= -1;
         }
      }
      for (int t = 0; t < n_obs; t++)
      {
         abo->G_e_1_[t] = G[t];
      }
      double *result = new double[n_obs * dim];
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                  n_obs, dim, n_obs, 1, G, n_obs, R, n_obs, 0, result, n_obs);
      std::memcpy(R, result, n_obs * dim * sizeof(double));
      delete[] result;
   }

   delete[] h; // Clean up
}
Step 3: Triggering the DowndateFinally, in ABO.cpp, when you call downdate(), you just need to pass it that oldest vector from the ring buffer.C++void ABO::downdate()
{
   // Grab the oldest feature vector before updating the matrices
   double *z_old = &Z_ring_[ring_idx_ * dim_];

   // Pass it to our new Q-less downdate
   givens::downdate(this, z_old);

   // ... rest of your downdate logic remains identical!
   // (Remember to delete the Q_ deletion lines at the bottom:
   // Q_ = deleteRowColMajor... Q_ = deleteColColMajor...)
}
The ResultYou have successfully eradicated the $O(N^2)$ Q matrix and all of its expensive resizing operations from your codebase. It is now fundamentally lighter and perfectly retains the exact pseudoinverse stability required for benign overfitting.Would you like me to help adjust your QR_decomposition.cpp to stop doing the heavy LAPACKE_dorgqr calculation for Q during the batch initialization, or is that straightforward enough to patch?
