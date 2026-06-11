#include <RcppArmadillo.h>
#include "ips_kernel.h"

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

namespace {

struct IPSEval {
  double value;
  arma::vec grad;
  bool finite;
};

IPSEval eval_ips(const arma::vec& b,
                 const arma::vec& d,
                 const arma::mat& X,
                 SEXP w,
                 double treated_flag,
                 const arma::vec& whs) {
  const arma::uword nobj = X.n_rows;
  const arma::uword npar = X.n_cols;
  const double n2 = static_cast<double>(nobj) * static_cast<double>(nobj);
  const bool include_h1 = (treated_flag != 1.0);

  arma::vec psfit = X * b;
  psfit = 1.0 / (1.0 + arma::exp(-psfit));

  arma::vec w0_raw = whs % (1.0 - d) / (1.0 - psfit);
  const double mean_w0 = arma::mean(w0_raw);
  arma::vec w0 = w0_raw / mean_w0;
  arma::vec h0 = w0 - 1.0;

  arma::vec base0 = w0 % psfit;
  arma::mat h0dot = X;
  h0dot.each_col() %= base0;
  h0dot -= w0 * arma::mean(h0dot, 0);

  double value = 0.0;
  arma::rowvec grad_row(npar, fill::zeros);

  if (include_h1) {
    arma::vec w1_raw = whs % d / psfit;
    const double mean_w1 = arma::mean(w1_raw);
    arma::vec w1 = w1_raw / mean_w1;
    arma::vec h1 = w1 - 1.0;

    arma::vec base1 = -w1 % (1.0 - psfit);
    arma::mat h1dot = X;
    h1dot.each_col() %= base1;
    h1dot -= w1 * arma::mean(h1dot, 0);

    arma::mat h(nobj, 2);
    h.col(0) = h1;
    h.col(1) = h0;
    arma::mat Wh = ips_kernel_multiply(w, h);
    value = (1.0 - treated_flag) * arma::dot(h1, Wh.col(0)) +
      arma::dot(h0, Wh.col(1));
    grad_row = (1.0 - treated_flag) * ips_kernel_crossprod(w, h1, h1dot) +
      ips_kernel_crossprod(w, h0, h0dot);
  } else {
    value = ips_kernel_quad(w, h0);
    grad_row = ips_kernel_crossprod(w, h0, h0dot);
  }

  IPSEval out;
  out.value = value / n2;
  out.grad = arma::trans(grad_row / n2);
  out.finite = std::isfinite(out.value) && out.grad.is_finite();
  return out;
}

} // namespace

// [[Rcpp::export]]
Rcpp::List optimIPSCpp(const arma::vec& par,
                       const arma::vec& d,
                       const arma::mat& X,
                       SEXP w,
                       double treated_flag,
                       const arma::vec& whs,
                       int maxit,
                       double abstol,
                       double reltol) {
  const arma::uword npar = par.n_elem;
  arma::vec current_par = par;
  IPSEval current = eval_ips(current_par, d, X, w, treated_flag, whs);
  int fn_count = 1;
  int gr_count = 1;
  int convergence = 1;

  arma::mat H(npar, npar, fill::eye);
  const double c1 = 1e-4;
  const int max_line_search = 30;
  int iter = 0;

  if (!current.finite) {
    return Rcpp::List::create(
      Rcpp::_["par"] = current_par,
      Rcpp::_["value"] = current.value,
      Rcpp::_["counts"] = Rcpp::IntegerVector::create(fn_count, gr_count),
      Rcpp::_["convergence"] = 52,
      Rcpp::_["message"] = "initial objective or gradient is not finite"
    );
  }

  for (iter = 0; iter < maxit; ++iter) {
    arma::vec direction = -H * current.grad;
    double directional_derivative = arma::dot(current.grad, direction);

    if (!direction.is_finite() || directional_derivative >= 0.0) {
      H.eye();
      direction = -current.grad;
      directional_derivative = -arma::dot(current.grad, current.grad);
    }

    if (std::sqrt(arma::dot(current.grad, current.grad)) <= reltol) {
      convergence = 0;
      break;
    }

    bool accepted = false;
    double step = 1.0;
    arma::vec next_par = current_par;
    IPSEval next = current;

    for (int ls = 0; ls < max_line_search; ++ls) {
      next_par = current_par + step * direction;
      next = eval_ips(next_par, d, X, w, treated_flag, whs);
      ++fn_count;
      ++gr_count;

      if (next.finite &&
          next.value <= current.value + c1 * step * directional_derivative) {
        accepted = true;
        break;
      }

      step *= 0.5;
    }

    if (!accepted) {
      convergence = 51;
      break;
    }

    arma::vec s = next_par - current_par;
    arma::vec y = next.grad - current.grad;
    const double ys = arma::dot(y, s);
    const double scale = std::sqrt(arma::dot(s, s) * arma::dot(y, y));
    const double previous_value = current.value;

    current_par = next_par;
    current = next;

    if (std::abs(previous_value - current.value) <=
        reltol * (std::abs(previous_value) + reltol) ||
        current.value <= abstol) {
      convergence = 0;
      break;
    }

    if (ys > 1e-12 * scale) {
      const double rho = 1.0 / ys;
      arma::mat I(npar, npar, fill::eye);
      arma::mat V = I - rho * s * y.t();
      H = V * H * V.t() + rho * s * s.t();
    }
  }

  return Rcpp::List::create(
    Rcpp::_["par"] = current_par,
    Rcpp::_["value"] = current.value,
    Rcpp::_["counts"] = Rcpp::IntegerVector::create(fn_count, gr_count),
    Rcpp::_["convergence"] = convergence,
    Rcpp::_["message"] = R_NilValue
  );
}
