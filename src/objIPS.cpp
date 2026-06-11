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
  
  
  arma::vec h1 = (whs % d/psfit) / mean(whs % d/psfit) - 1.0;
  arma::vec h0 = (whs % (1.0 - d)/(1.0 - psfit))/mean(whs % (1.0 - d)/(1.0 - psfit)) - 1.0;
  
  const double n2 = static_cast<double>(nobj) * static_cast<double>(nobj);
  obj = ((1.0 - treated_flag) * ips_kernel_quad(w, h1) +
         ips_kernel_quad(w, h0)) / n2;
  
  return obj;
}
