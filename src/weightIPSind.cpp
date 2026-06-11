#include <RcppArmadillo.h>
#include "ips_kernel.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::mat weightIPSind(const arma::mat& X) {
  arma::mat wind = ips_build_indicator_factor(X);
  return wind * wind.t() / static_cast<double>(X.n_rows);
}
