#include "ips_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;

namespace {

const char* IPS_KERNEL_CLASS = "ips_kernel";

class DenseKernel : public IPSKernel {
public:
  explicit DenseKernel(const arma::mat& weights) : weights_(weights) {}

  arma::uword n_rows() const {
    return weights_.n_rows;
  }

  arma::mat multiply(const arma::mat& rhs) const {
    return weights_ * rhs;
  }

  double quad(const arma::vec& rhs) const {
    return arma::dot(rhs, weights_ * rhs);
  }

  arma::rowvec crossprod(const arma::vec& lhs,
                         const arma::mat& rhs) const {
    return lhs.t() * (weights_ * rhs);
  }

  arma::mat dense() const {
    return weights_;
  }

private:
  arma::mat weights_;
};

class CompactDenseKernel : public IPSKernel {
public:
  CompactDenseKernel(const arma::mat& compact_weights,
                     const arma::uvec& groups,
                     arma::uword n_rows)
    : compact_weights_(compact_weights), groups_(groups), n_rows_(n_rows) {}

  arma::uword n_rows() const {
    return n_rows_;
  }

  arma::mat multiply(const arma::mat& rhs) const {
    const arma::uword n_groups = compact_weights_.n_rows;
    arma::mat grouped_rhs(n_groups, rhs.n_cols, fill::zeros);

    for (arma::uword i = 0; i < n_rows_; ++i) {
      grouped_rhs.row(groups_[i]) += rhs.row(i);
    }

    arma::mat compact_result = compact_weights_ * grouped_rhs;
    arma::mat out(n_rows_, rhs.n_cols);

    for (arma::uword i = 0; i < n_rows_; ++i) {
      out.row(i) = compact_result.row(groups_[i]);
    }

    return out;
  }

  double quad(const arma::vec& rhs) const {
    arma::vec grouped_rhs(compact_weights_.n_rows, fill::zeros);

    for (arma::uword i = 0; i < n_rows_; ++i) {
      grouped_rhs[groups_[i]] += rhs[i];
    }

    return arma::dot(grouped_rhs, compact_weights_ * grouped_rhs);
  }

  arma::rowvec crossprod(const arma::vec& lhs,
                         const arma::mat& rhs) const {
    const arma::uword n_groups = compact_weights_.n_rows;
    arma::vec grouped_lhs(n_groups, fill::zeros);
    arma::mat grouped_rhs(n_groups, rhs.n_cols, fill::zeros);

    for (arma::uword i = 0; i < n_rows_; ++i) {
      const arma::uword group = groups_[i];
      grouped_lhs[group] += lhs[i];
      grouped_rhs.row(group) += rhs.row(i);
    }

    return grouped_lhs.t() * (compact_weights_ * grouped_rhs);
  }

  arma::mat dense() const {
    arma::mat out(n_rows_, n_rows_);

    for (arma::uword i = 0; i < n_rows_; ++i) {
      for (arma::uword j = 0; j < n_rows_; ++j) {
        out(i, j) = compact_weights_(groups_[i], groups_[j]);
      }
    }

    return out;
  }

private:
  arma::mat compact_weights_;
  arma::uvec groups_;
  arma::uword n_rows_;
};

struct CompactRows {
  arma::mat X;
  arma::uvec groups;
  arma::vec counts;
};

inline double clean_zero(double x) {
  return std::abs(x) < 1e-12 ? 0.0 : x;
}

std::string row_key(const arma::mat& X, arma::uword row) {
  const arma::uword k = X.n_cols;
  std::string key(k * sizeof(double), '\0');

  for (arma::uword col = 0; col < k; ++col) {
    double value = X(row, col);
    if (value == 0.0) {
      value = 0.0;
    }
    std::memcpy(&key[col * sizeof(double)], &value, sizeof(double));
  }

  return key;
}

arma::vec validate_case_weights(const arma::mat& X,
                                const arma::vec& case_weights) {
  if (case_weights.n_elem != X.n_rows) {
    Rcpp::stop("case.weights must have one entry per row in xbal");
  }

  if (!case_weights.is_finite()) {
    Rcpp::stop("case.weights must be finite");
  }

  if (arma::any(case_weights < 0.0)) {
    Rcpp::stop("case.weights must be nonnegative");
  }

  if (arma::accu(case_weights) <= 0.0) {
    Rcpp::stop("case.weights must have positive total weight");
  }

  return case_weights;
}

CompactRows compact_rows(const arma::mat& X,
                         const arma::vec& case_weights) {
  const arma::uword n = X.n_rows;
  const arma::uword k = X.n_cols;
  std::unordered_map<std::string, arma::uword> seen;
  seen.reserve(static_cast<std::size_t>(n * 1.3) + 1);

  arma::mat unique_rows(n, k);
  arma::uvec groups(n);
  std::vector<double> counts;
  counts.reserve(n);

  arma::uword n_unique = 0;
  for (arma::uword i = 0; i < n; ++i) {
    std::string key = row_key(X, i);
    std::unordered_map<std::string, arma::uword>::const_iterator found =
      seen.find(key);

    if (found == seen.end()) {
      seen.emplace(key, n_unique);
      groups[i] = n_unique;
      unique_rows.row(n_unique) = X.row(i);
      counts.push_back(case_weights[i]);
      ++n_unique;
    } else {
      groups[i] = found->second;
      counts[found->second] += case_weights[i];
    }
  }

  unique_rows.resize(n_unique, k);
  arma::vec count_vec(counts);

  CompactRows out;
  out.X = unique_rows;
  out.groups = groups;
  out.counts = count_vec;
  return out;
}

CompactRows compact_rows(const arma::mat& X) {
  return compact_rows(X, arma::vec(X.n_rows, fill::ones));
}

arma::mat pairwise_exp_from_gram(const arma::mat& gram,
                                 const arma::vec& norms) {
  arma::mat dist = -2.0 * gram;
  dist.each_col() += norms;
  dist.each_row() += norms.t();

  for (arma::uword j = 0; j < dist.n_cols; ++j) {
    dist(j, j) = 0.0;
  }

  return arma::exp(-0.5 * dist);
}

arma::mat clean_distance_from_gram(const arma::mat& gram,
                                   const arma::vec& norms) {
  arma::mat dist = -2.0 * gram;
  dist.each_col() += norms;
  dist.each_row() += norms.t();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long jj = 0; jj < static_cast<long>(dist.n_cols); ++jj) {
    const arma::uword j = static_cast<arma::uword>(jj);
    for (arma::uword i = 0; i < dist.n_rows; ++i) {
      dist(i, j) = clean_zero(dist(i, j));
    }
    dist(j, j) = 0.0;
  }

  return dist;
}

arma::mat build_projection_compact_weights(const arma::mat& X,
                                           const arma::vec& counts,
                                           double n_total) {
  const arma::uword n_unique = X.n_rows;
  arma::mat gram = X * X.t();
  arma::vec norms = gram.diag();
  arma::mat distances = clean_distance_from_gram(gram, norms);
  arma::mat inv_sqrt_distances(n_unique, n_unique, fill::zeros);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long jj = 0; jj < static_cast<long>(n_unique); ++jj) {
    const arma::uword j = static_cast<arma::uword>(jj);
    double* inv_col = inv_sqrt_distances.colptr(j);
    const double* dist_col = distances.colptr(j);
    for (arma::uword i = 0; i < n_unique; ++i) {
      if (dist_col[i] != 0.0) {
        inv_col[i] = 1.0 / std::sqrt(dist_col[i]);
      }
    }
  }

  arma::mat weights(n_unique, n_unique, fill::zeros);
  const double* count_ptr = counts.memptr();
  bool only_diagonal_zero = true;
  for (arma::uword j = 0; j < n_unique && only_diagonal_zero; ++j) {
    const double* dist_j = distances.colptr(j);
    for (arma::uword i = 0; i < n_unique; ++i) {
      if (i != j && dist_j[i] == 0.0) {
        only_diagonal_zero = false;
        break;
      }
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (long jj = 0; jj < static_cast<long>(n_unique); ++jj) {
    const arma::uword j = static_cast<arma::uword>(jj);
    const double* dist_j = distances.colptr(j);
    const double* inv_j = inv_sqrt_distances.colptr(j);

    for (arma::uword l = 0; l <= j; ++l) {
      const double idxjl = dist_j[l];
      const double* dist_l = distances.colptr(l);
      const double* inv_l = inv_sqrt_distances.colptr(l);
      double branch_out = 0.0;
      double acos_out = 0.0;

      if (only_diagonal_zero) {
        if (j == l) {
          branch_out = M_PI * (n_total + count_ptr[j]);
        } else {
          branch_out = M_PI * (count_ptr[j] + count_ptr[l]);
          for (arma::uword r = 0; r < n_unique; ++r) {
            if (r == j || r == l) {
              continue;
            }
            const double xjr = dist_j[r];
            const double xlr = dist_l[r];
            const double xjl = 0.5 * (xjr + xlr - idxjl);
            double arg = xjl * inv_j[r] * inv_l[r];
            arg = std::max(-1.0, std::min(1.0, arg));
            acos_out += (M_PI - std::acos(arg)) * count_ptr[r];
          }
        }
      } else {
        for (arma::uword r = 0; r < n_unique; ++r) {
          const double xjr = dist_j[r];
          const double xlr = dist_l[r];

          if ((xlr == 0.0) && (xjr == 0.0)) {
            branch_out += 2.0 * M_PI * count_ptr[r];
          } else if (((xlr != 0.0) && (xjr != 0.0)) && (idxjl != 0.0)) {
            const double xjl = 0.5 * (xjr + xlr - idxjl);
            double arg = xjl * inv_j[r] * inv_l[r];
            arg = std::max(-1.0, std::min(1.0, arg));
            acos_out += (M_PI - std::acos(arg)) * count_ptr[r];
          } else {
            branch_out += M_PI * count_ptr[r];
          }
        }
      }

      const double out = branch_out + acos_out;
      weights(j, l) = out / n_total;
      weights(l, j) = weights(j, l);
    }
  }

  return weights;
}

arma::vec weighted_mean_by_col(const arma::mat& X,
                               const arma::vec& counts,
                               double n_original) {
  return (X.t() * counts) / n_original;
}

arma::vec weighted_sd_by_col(const arma::mat& X,
                             const arma::vec& counts,
                             const arma::vec& means,
                             double n_original) {
  arma::vec out(X.n_cols, fill::zeros);
  const double denom = n_original - 1.0;

  for (arma::uword col = 0; col < X.n_cols; ++col) {
    arma::vec centered = X.col(col) - means[col];
    out[col] = std::sqrt(arma::dot(counts, centered % centered) / denom);
  }

  return out;
}

arma::mat weighted_cov(const arma::mat& X,
                       const arma::vec& counts,
                       double n_original) {
  arma::vec means = weighted_mean_by_col(X, counts, n_original);
  arma::mat centered = X;
  centered.each_row() -= means.t();
  centered.each_col() %= arma::sqrt(counts);
  return centered.t() * centered / (n_original - 1.0);
}

arma::mat build_exp_compact_weights(const arma::mat& X,
                                    const arma::vec& counts,
                                    double n_total,
                                    const std::string& X_trans) {
  const arma::uword npar = X.n_cols;

  if (X_trans == "normal") {
    arma::mat xprob(X.n_rows, X.n_cols);
    arma::vec means = weighted_mean_by_col(X, counts, n_total);
    arma::vec sds = weighted_sd_by_col(X, counts, means, n_total);

    for (arma::uword l = 0; l < npar; ++l) {
      xprob.col(l) = arma::normcdf((X.col(l) - means[l]) / sds[l]);
    }

    arma::mat gram = xprob * xprob.t();
    arma::vec norms = arma::sum(xprob % xprob, 1);
    return pairwise_exp_from_gram(gram, norms);
  }

  if (X_trans == "arctan") {
    arma::mat xprob(X.n_rows, X.n_cols);
    arma::vec means = weighted_mean_by_col(X, counts, n_total);
    arma::vec sds = weighted_sd_by_col(X, counts, means, n_total);
    const double v = 0.0;

    for (arma::uword l = 0; l < npar; ++l) {
      xprob.col(l) = arma::atan((X.col(l) - v * means[l]) / sds[l]);
    }

    arma::mat gram = xprob * xprob.t();
    arma::vec norms = arma::sum(xprob % xprob, 1);
    return pairwise_exp_from_gram(gram, norms);
  }

  arma::mat vcinv = arma::pinv(weighted_cov(X, counts, n_total));
  arma::mat transformed = X * vcinv;
  arma::mat gram = transformed * X.t();
  arma::vec norms = arma::sum(transformed % X, 1);
  return pairwise_exp_from_gram(gram, norms);
}

arma::mat build_indicator_compact_weights(const arma::mat& X,
                                          const arma::vec& counts,
                                          double n_total) {
  arma::mat factor = ips_build_indicator_factor(X);
  arma::mat weighted_factor = factor;
  weighted_factor.each_row() %= counts.t();
  return weighted_factor * factor.t() / n_total;
}

CompactDenseKernel* make_projection_kernel_from_rows(const arma::mat& X) {
  CompactRows compact = compact_rows(X);
  arma::mat compact_weights =
    build_projection_compact_weights(compact.X, compact.counts,
                                     static_cast<double>(X.n_rows));

  return new CompactDenseKernel(compact_weights, compact.groups, X.n_rows);
}

CompactDenseKernel* make_projection_kernel_from_rows(
    const arma::mat& X,
    const arma::vec& case_weights) {
  arma::vec weights = validate_case_weights(X, case_weights);
  CompactRows compact = compact_rows(X, weights);
  arma::mat compact_weights =
    build_projection_compact_weights(compact.X, compact.counts,
                                     arma::accu(weights));

  return new CompactDenseKernel(compact_weights, compact.groups, X.n_rows);
}

CompactDenseKernel* make_projection_kernel_from_unique(const arma::mat& X,
                                                       const arma::vec& counts) {
  arma::uword n_original = 0;
  for (arma::uword i = 0; i < counts.n_elem; ++i) {
    n_original += static_cast<arma::uword>(counts[i]);
  }

  arma::uvec groups(n_original);
  arma::uword pos = 0;
  for (arma::uword group = 0; group < counts.n_elem; ++group) {
    const arma::uword repetitions = static_cast<arma::uword>(counts[group]);
    for (arma::uword r = 0; r < repetitions; ++r) {
      groups[pos++] = group;
    }
  }

  arma::mat compact_weights =
    build_projection_compact_weights(X, counts,
                                     static_cast<double>(n_original));

  return new CompactDenseKernel(compact_weights, groups, n_original);
}

IPSKernel* make_exp_kernel_from_rows(const arma::mat& X,
                                     const std::string& X_trans) {
  CompactRows compact = compact_rows(X);

  if (compact.X.n_rows == X.n_rows) {
    return new DenseKernel(ips_build_exp_weights(X, X_trans));
  }

  arma::mat compact_weights =
    build_exp_compact_weights(compact.X, compact.counts,
                              static_cast<double>(X.n_rows), X_trans);

  return new CompactDenseKernel(compact_weights, compact.groups, X.n_rows);
}

IPSKernel* make_exp_kernel_from_rows(const arma::mat& X,
                                     const std::string& X_trans,
                                     const arma::vec& case_weights) {
  arma::vec weights = validate_case_weights(X, case_weights);
  CompactRows compact = compact_rows(X, weights);
  arma::mat compact_weights =
    build_exp_compact_weights(compact.X, compact.counts,
                              arma::accu(weights), X_trans);

  return new CompactDenseKernel(compact_weights, compact.groups, X.n_rows);
}

IPSKernel* make_indicator_kernel_from_rows(const arma::mat& X) {
  CompactRows compact = compact_rows(X);

  if (compact.X.n_rows == X.n_rows) {
    arma::mat factor = ips_build_indicator_factor(X);
    return new DenseKernel(factor * factor.t() / static_cast<double>(X.n_rows));
  }

  arma::mat compact_weights =
    build_indicator_compact_weights(compact.X, compact.counts,
                                    static_cast<double>(X.n_rows));

  return new CompactDenseKernel(compact_weights, compact.groups, X.n_rows);
}

IPSKernel* make_indicator_kernel_from_rows(const arma::mat& X,
                                           const arma::vec& case_weights) {
  arma::vec weights = validate_case_weights(X, case_weights);
  CompactRows compact = compact_rows(X, weights);
  arma::mat compact_weights =
    build_indicator_compact_weights(compact.X, compact.counts,
                                    arma::accu(weights));

  return new CompactDenseKernel(compact_weights, compact.groups, X.n_rows);
}

SEXP wrap_kernel(IPSKernel* kernel) {
  Rcpp::XPtr<IPSKernel> ptr(kernel, true);
  ptr.attr("class") = IPS_KERNEL_CLASS;
  return ptr;
}

} // namespace

bool ips_is_kernel(SEXP kernel) {
  if (TYPEOF(kernel) != EXTPTRSXP) {
    return false;
  }

  Rcpp::RObject object(kernel);
  return object.inherits(IPS_KERNEL_CLASS);
}

arma::mat ips_kernel_multiply(SEXP kernel, const arma::mat& rhs) {
  if (ips_is_kernel(kernel)) {
    Rcpp::XPtr<IPSKernel> ptr(kernel);
    if (ptr->n_rows() != rhs.n_rows) {
      Rcpp::stop("kernel and right-hand side have incompatible dimensions");
    }
    return ptr->multiply(rhs);
  }

  arma::mat dense_kernel = Rcpp::as<arma::mat>(kernel);
  return dense_kernel * rhs;
}

arma::vec ips_kernel_multiply(SEXP kernel, const arma::vec& rhs) {
  arma::mat rhs_mat(rhs.n_elem, 1);
  rhs_mat.col(0) = rhs;
  return ips_kernel_multiply(kernel, rhs_mat).col(0);
}

double ips_kernel_quad(SEXP kernel, const arma::vec& rhs) {
  if (ips_is_kernel(kernel)) {
    Rcpp::XPtr<IPSKernel> ptr(kernel);
    if (ptr->n_rows() != rhs.n_elem) {
      Rcpp::stop("kernel and vector have incompatible dimensions");
    }
    return ptr->quad(rhs);
  }

  arma::mat dense_kernel = Rcpp::as<arma::mat>(kernel);
  return arma::as_scalar(rhs.t() * dense_kernel * rhs);
}

arma::rowvec ips_kernel_crossprod(SEXP kernel, const arma::vec& lhs,
                                  const arma::mat& rhs) {
  if (ips_is_kernel(kernel)) {
    Rcpp::XPtr<IPSKernel> ptr(kernel);
    if (ptr->n_rows() != lhs.n_elem || ptr->n_rows() != rhs.n_rows) {
      Rcpp::stop("kernel, left-hand side, and right-hand side have incompatible dimensions");
    }
    return ptr->crossprod(lhs, rhs);
  }

  arma::mat dense_kernel = Rcpp::as<arma::mat>(kernel);
  return lhs.t() * (dense_kernel * rhs);
}

arma::mat ips_kernel_dense(SEXP kernel) {
  if (ips_is_kernel(kernel)) {
    Rcpp::XPtr<IPSKernel> ptr(kernel);
    return ptr->dense();
  }

  return Rcpp::as<arma::mat>(kernel);
}

arma::mat ips_build_exp_weights(const arma::mat& X,
                                const std::string& X_trans) {
  const arma::uword npar = X.n_cols;

  if (X_trans == "normal") {
    arma::mat xprob(X.n_rows, X.n_cols);
    for (arma::uword l = 0; l < npar; ++l) {
      xprob.col(l) =
        arma::normcdf((X.col(l) - arma::mean(X.col(l))) /
                      arma::stddev(X.col(l)));
    }

    arma::mat gram = xprob * xprob.t();
    arma::vec norms = arma::sum(xprob % xprob, 1);
    return pairwise_exp_from_gram(gram, norms);
  }

  if (X_trans == "arctan") {
    arma::mat xprob(X.n_rows, X.n_cols);
    const double v = 0.0;
    for (arma::uword l = 0; l < npar; ++l) {
      xprob.col(l) =
        arma::atan((X.col(l) - v * arma::mean(X.col(l))) /
                   arma::stddev(X.col(l)));
    }

    arma::mat gram = xprob * xprob.t();
    arma::vec norms = arma::sum(xprob % xprob, 1);
    return pairwise_exp_from_gram(gram, norms);
  }

  arma::mat vcinv = arma::pinv(arma::cov(X));
  arma::mat transformed = X * vcinv;
  arma::mat gram = transformed * X.t();
  arma::vec norms = arma::sum(transformed % X, 1);
  return pairwise_exp_from_gram(gram, norms);
}

arma::mat ips_build_indicator_factor(const arma::mat& X) {
  const arma::uword nobj = X.n_rows;
  const arma::uword npar = X.n_cols;
  arma::mat factor(nobj, nobj, fill::zeros);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long ll = 0; ll < static_cast<long>(nobj); ++ll) {
    const arma::uword l = static_cast<arma::uword>(ll);
    for (arma::uword j = 0; j < nobj; ++j) {
      bool dominates = true;
      for (arma::uword s = 0; s < npar; ++s) {
        if (X(j, s) > X(l, s)) {
          dominates = false;
          break;
        }
      }
      factor(j, l) = dominates ? 1.0 : 0.0;
    }
  }

  return factor;
}

arma::mat ips_build_projection_dense(const arma::mat& X) {
  Rcpp::XPtr<IPSKernel> ptr(make_projection_kernel_from_rows(X), true);
  return ptr->dense();
}

arma::mat ips_build_projection_dense_unique(const arma::mat& X,
                                            const arma::vec& counts) {
  Rcpp::XPtr<IPSKernel> ptr(make_projection_kernel_from_unique(X, counts), true);
  return ptr->dense();
}

// [[Rcpp::export]]
SEXP kernelIPSdense(const arma::mat& weights) {
  return wrap_kernel(new DenseKernel(weights));
}

// [[Rcpp::export]]
SEXP kernelIPSexp(const arma::mat& X, std::string X_trans) {
  return wrap_kernel(make_exp_kernel_from_rows(X, X_trans));
}

// [[Rcpp::export]]
SEXP kernelIPSexpWeighted(const arma::mat& X, std::string X_trans,
                          const arma::vec& case_weights) {
  return wrap_kernel(make_exp_kernel_from_rows(X, X_trans, case_weights));
}

// [[Rcpp::export]]
SEXP kernelIPSind(const arma::mat& X) {
  return wrap_kernel(make_indicator_kernel_from_rows(X));
}

// [[Rcpp::export]]
SEXP kernelIPSindWeighted(const arma::mat& X,
                          const arma::vec& case_weights) {
  return wrap_kernel(make_indicator_kernel_from_rows(X, case_weights));
}

// [[Rcpp::export]]
SEXP kernelIPSproj(const arma::mat& X) {
  return wrap_kernel(make_projection_kernel_from_rows(X));
}

// [[Rcpp::export]]
SEXP kernelIPSprojWeighted(const arma::mat& X,
                           const arma::vec& case_weights) {
  return wrap_kernel(make_projection_kernel_from_rows(X, case_weights));
}

// [[Rcpp::export]]
arma::mat kernelIPSDense(SEXP kernel) {
  return ips_kernel_dense(kernel);
}
