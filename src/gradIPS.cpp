#include <RcppArmadillo.h>
#include "ips_kernel.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::vec gradIPS(const arma::vec& b, const arma::vec& d, const arma::mat& X,
                  SEXP w, double treated_flag, const arma::vec& whs) {
  int nobj = X.n_rows, npar = X.n_cols;
  
  
  arma::vec psfit = X * b;
  psfit = 1.0/(1.0 + exp(-psfit));
 
  const bool include_h1 = (treated_flag != 1.0);
  arma::vec w0_raw = whs % (1.0 - d) / (1.0 - psfit);
  arma::vec w0 = w0_raw / mean(w0_raw);
  
  // calculate h0.dot
  arma::mat h0dot(nobj, npar);
  
  for (int j = 0;  j<npar; j++){
    h0dot.col(j) =  w0 % psfit % X.col(j);
    h0dot.col(j) =  h0dot.col(j) - w0 * mean(h0dot.col(j));
  }

  arma::vec h0 = w0 - 1.0;
  arma::rowvec Qd_row;

  if (include_h1) {
    arma::vec w1_raw = whs % d / psfit;
    arma::vec w1 = w1_raw / mean(w1_raw);
    arma::mat h1dot(nobj, npar);

    for (int j = 0;  j<npar; j++){
      h1dot.col(j) = - w1 % (1.0 - psfit) % X.col(j);
      h1dot.col(j) =  h1dot.col(j) - w1 * mean(h1dot.col(j));
    }

    arma::vec h1 = w1 - 1.0;
    Qd_row = (1.0 - treated_flag) * ips_kernel_crossprod(w, h1, h1dot) +
      ips_kernel_crossprod(w, h0, h0dot);
  } else {
    Qd_row = ips_kernel_crossprod(w, h0, h0dot);
  }

  const double n2 = static_cast<double>(nobj) * static_cast<double>(nobj);
  return arma::trans(Qd_row / n2);
}
