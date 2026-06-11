#include <RcppArmadillo.h>
#include "ips_kernel.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat weightIPSproj_uniq(const arma::mat& X, const arma::vec& wgt){
  return ips_build_projection_dense_unique(X, wgt);
}
