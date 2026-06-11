#include <RcppArmadillo.h>
#include "ips_kernel.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat weightIPSproj(const arma::mat& X){
  return ips_build_projection_dense(X);
}
