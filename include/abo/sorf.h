#pragma once
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>

// Structured Orthogonal Random Features (SORF)
// Yu et al. (2016) "Orthogonal Random Features", NeurIPS
//
// Drop-in replacement for GaussianRFF. Same interface, same Gaussian kernel,
// but uses O(d log d) structured Walsh-Hadamard transform instead of O(dD)
// random matrix multiply.
//
// Construction per feature j:
//   ω_j = (r_j / sigma) * v_j
//   where v_j is a unit-norm direction from 3 HD blocks (SORF rotation)
//   and r_j ~ chi(d_in) is an independent norm sample matching Gaussian norms.
//
// This matches the ORF paper's construction: directions from orthogonal
// structure, norms from chi(d) to reproduce the Gaussian spectral density.
// The chi correction is essential for small d where ||N(0,I)|| has
// significant spread and the fixed-norm approximation introduces bias.

class SORF
{
public:
    int    d_in_;   // original input dimension
    int    d_pad_;  // padded to next power of 2 >= d_in_
    int    D_;      // output RFF dimension
    double sigma_;  // kernel bandwidth (same meaning as kernel_var in GaussianRFF)

    // signs_[block][hd_idx][j] = ±1 for block b, HD product k=0..2, dim j
    std::vector<std::vector<std::vector<double>>> signs_;
    // chi_scales_[b][j] = r_{b,j} ~ chi(d_in_): norm correction per feature
    std::vector<std::vector<double>> chi_scales_;
    std::vector<double> bias_;  // D_ phases in [0, 2π]

    SORF(int d, int D, double sigma, int seed = -1);
    Eigen::MatrixXd transform(const Eigen::MatrixXd& x) const;
    Eigen::MatrixXd transform_matrix(const Eigen::MatrixXd& X) const;

private:
    void apply_hd(double* v, const std::vector<double>& signs) const;
    void apply_sorf_block(double* v, int block) const;
    static void fwht(double* a, int n);
    static int  next_pow2(int n);
};
