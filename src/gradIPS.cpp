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
  arma::mat Qdot;

  if (include_h1) {
    arma::vec w1_raw = whs % d / psfit;
    arma::vec w1 = w1_raw / mean(w1_raw);
    arma::mat h1dot(nobj, npar);

    for (int j = 0;  j<npar; j++){
      h1dot.col(j) = - w1 % (1.0 - psfit) % X.col(j);
      h1dot.col(j) =  h1dot.col(j) - w1 * mean(h1dot.col(j));
    }

    arma::vec h1 = w1 - 1.0;
    arma::mat hdot(nobj, 2 * npar);
    hdot.cols(0, npar - 1) = h1dot;
    hdot.cols(npar, 2 * npar - 1) = h0dot;

    arma::mat Whdot = ips_kernel_multiply(w, hdot);
    arma::mat Qdot1 = Whdot.cols(0, npar - 1);
    Qdot1.each_col() %= h1;
    arma::mat Qdot0 = Whdot.cols(npar, 2 * npar - 1);
    Qdot0.each_col() %= h0;
    Qdot = (1.0 - treated_flag) * Qdot1 + Qdot0;
  } else {
    Qdot = ips_kernel_multiply(w, h0dot);
    Qdot.each_col() %= h0;
  }

  Qdot /= nobj;

  arma::vec Qd = arma::trans(mean(Qdot, 0));
  return Qd;
}
