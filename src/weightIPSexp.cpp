#include <RcppArmadillo.h>
#include "ips_kernel.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;



// [[Rcpp::export]]
arma::mat weightIPSexp(const arma::mat& X, std::string X_trans) {
  return ips_build_exp_weights(X, X_trans);
}
