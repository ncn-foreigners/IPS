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
 
  arma::vec w1 = (whs % d/psfit) / mean(whs % d/psfit);
  arma::vec w0 = (whs % (1.0 - d)/(1.0 - psfit)) / mean(whs % (1.0 - d)/(1.0 - psfit));
  
  // calculate h1.dot and h0.dot
  arma::mat h1dot(nobj, npar);
  arma::mat h0dot(nobj, npar);
  
  for (int j = 0;  j<npar; j++){
    h1dot.col(j) = - w1 % (1.0 - psfit) % X.col(j);
    h1dot.col(j) =  h1dot.col(j) - w1 * mean(h1dot.col(j));

    h0dot.col(j) =  w0 % psfit % X.col(j);
    h0dot.col(j) =  h0dot.col(j) - w0 * mean(h0dot.col(j));
  }

  arma::vec h1 = w1 - 1.0;
  arma::vec h0 = w0 - 1.0;
  arma::mat Qdot =
    (1.0 - treated_flag) * ips_kernel_multiply(w, h1dot);
  Qdot.each_col() %= h1;

  arma::mat Qdot0 = ips_kernel_multiply(w, h0dot);
  Qdot0.each_col() %= h0;
  Qdot += Qdot0;
  Qdot /= nobj;

  arma::vec Qd = arma::trans(mean(Qdot, 0));
  return Qd;
}
