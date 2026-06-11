#include <RcppArmadillo.h>
#include "ips_kernel.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
double objIPS(const arma::vec& b, const arma::vec& d, const arma::mat& X,
              SEXP w, double treated_flag, const arma::vec& whs) {
  int nobj = X.n_rows;
  double obj = 0.0;

  
  arma::vec psfit = X * b;
  psfit = 1.0/(1.0 + exp(-psfit));
  
  
  const bool include_h1 = (treated_flag != 1.0);
  arma::vec h0_raw = whs % (1.0 - d) / (1.0 - psfit);
  arma::vec h0 = h0_raw / mean(h0_raw) - 1.0;
  
  const double n2 = static_cast<double>(nobj) * static_cast<double>(nobj);

  if (include_h1) {
    arma::vec h1_raw = whs % d / psfit;
    arma::vec h1 = h1_raw / mean(h1_raw) - 1.0;
    arma::mat h(nobj, 2);
    h.col(0) = h1;
    h.col(1) = h0;
    arma::mat Wh = ips_kernel_multiply(w, h);
    obj = (1.0 - treated_flag) * arma::dot(h1, Wh.col(0)) +
      arma::dot(h0, Wh.col(1));
  } else {
    obj = ips_kernel_quad(w, h0);
  }

  obj /= n2;
  
  return obj;
}
