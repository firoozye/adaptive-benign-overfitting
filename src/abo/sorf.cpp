#include "abo/sorf.h"

int SORF::next_pow2(int n)
{
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

SORF::SORF(int d, int D, double sigma, bool seed)
    : d_in_(d), D_(D), sigma_(sigma)
{
    d_pad_ = next_pow2(d);
    int n_blocks = (D + d_pad_ - 1) / d_pad_;

    std::mt19937 rng;
    if (seed) rng.seed(0);
    else      rng.seed(std::random_device{}());

    std::uniform_int_distribution<int>     sign_dist(0, 1);
    std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
    std::normal_distribution<double>       gauss_dist(0.0, 1.0);

    // ±1 sign vectors for 3 HD products per block
    signs_.assign(n_blocks,
        std::vector<std::vector<double>>(3, std::vector<double>(d_pad_)));
    for (int b = 0; b < n_blocks; ++b)
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < d_pad_; ++j)
                signs_[b][k][j] = sign_dist(rng) ? 1.0 : -1.0;

    // chi(d_in_) norm samples: r_{b,j} = ||g|| where g ~ N(0, I_{d_in_}).
    // Multiplying the unit-norm SORF direction by r/sigma reproduces the
    // Gaussian spectral density exactly in expectation (ORF paper, Thm 1).
    chi_scales_.assign(n_blocks, std::vector<double>(d_pad_));
    for (int b = 0; b < n_blocks; ++b)
        for (int j = 0; j < d_pad_; ++j)
        {
            double r2 = 0.0;
            for (int k = 0; k < d_in_; ++k) { double g = gauss_dist(rng); r2 += g * g; }
            chi_scales_[b][j] = std::sqrt(r2);
        }

    // Random phase shifts b_j ~ Uniform[0, 2π]
    bias_.resize(D_);
    for (int j = 0; j < D_; ++j)
        bias_[j] = phase_dist(rng);
}

// Unnormalized in-place Fast Walsh-Hadamard Transform.
// After this call ||a_out|| = sqrt(n) * ||a_in||.
void SORF::fwht(double* a, int n)
{
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = 0; j < len; ++j)
            {
                double u = a[i + j], v = a[i + j + len];
                a[i + j]       = u + v;
                a[i + j + len] = u - v;
            }
}

// One HD product: apply random ±1 signs then normalized WHT.
// Norm-preserving: ||v_out|| == ||v_in||.
void SORF::apply_hd(double* v, const std::vector<double>& signs) const
{
    for (int j = 0; j < d_pad_; ++j) v[j] *= signs[j];
    fwht(v, d_pad_);
    double inv_sqrt = 1.0 / std::sqrt(static_cast<double>(d_pad_));
    for (int j = 0; j < d_pad_; ++j) v[j] *= inv_sqrt;
}

// Apply three HD products to get unit-norm directions, then scale each
// component j by chi_scales_[block][j] / sigma_ to reproduce Gaussian norms.
// After 3 norm-preserving HD products: ||v|| = ||x_pad||.
// Dividing by sigma and re-scaling per-component by chi(d_in_) gives
// each effective frequency the same distribution as N(0, I/sigma^2).
void SORF::apply_sorf_block(double* v, int block) const
{
    apply_hd(v, signs_[block][0]);
    apply_hd(v, signs_[block][1]);
    apply_hd(v, signs_[block][2]);
    // After 3 HD products: v_j = (W_row_j)^T x_pad where W is orthogonal.
    // The j-th frequency is ω_j = (chi_j/sigma) * W_row_j, so
    // ω_j^T x_pad = (chi_j/sigma) * v_j.
    for (int j = 0; j < d_pad_; ++j)
        v[j] *= chi_scales_[block][j] / sigma_;
}

// Transform a single row vector x (1 × d_in_) → z (1 × D_).
Eigen::MatrixXd SORF::transform(const Eigen::MatrixXd& x) const
{
    Eigen::MatrixXd z(1, D_);
    const double rff_scale = std::sqrt(2.0 / static_cast<double>(D_));
    int n_blocks = static_cast<int>(signs_.size());

    std::vector<double> v(d_pad_);
    int out = 0;
    for (int b = 0; b < n_blocks && out < D_; ++b)
    {
        for (int j = 0;      j < d_in_;  ++j) v[j] = x(0, j);
        for (int j = d_in_;  j < d_pad_; ++j) v[j] = 0.0;

        apply_sorf_block(v.data(), b);

        int take = std::min(d_pad_, D_ - out);
        for (int j = 0; j < take; ++j)
            z(0, out + j) = rff_scale * std::cos(v[j] + bias_[out + j]);
        out += take;
    }
    return z;
}

// Transform a batch matrix X (N × d_in_) → Z (N × D_).
Eigen::MatrixXd SORF::transform_matrix(const Eigen::MatrixXd& X) const
{
    int N = static_cast<int>(X.rows());
    Eigen::MatrixXd Z(N, D_);
    for (int i = 0; i < N; ++i)
        Z.row(i) = transform(X.row(i));
    return Z;
}
