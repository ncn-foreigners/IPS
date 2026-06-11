#include <RcppArmadillo.h>
#include "ips_kernel.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;


// [[Rcpp::export]]
arma::mat linIPS(const arma::vec& bhat, const arma::vec& d,
                 const arma::vec& pshat, const arma::mat& X,
                 SEXP w, double treated_flag, const arma::vec& whs) {
  int nobj = X.n_rows, npar = X.n_cols;
  
  arma::mat Cinv = zeros<arma::mat>(npar,npar);
  arma::mat Cmat = zeros<arma::mat>(npar,npar);
  arma::mat linrep = zeros<arma::mat>(nobj,npar);
  
  const bool include_h1 = (treated_flag != 1.0);
  arma::vec w0_raw = whs % (1.0 - d) / (1.0 - pshat);
  arma::vec w0 = w0_raw / mean(w0_raw);
  arma::mat h0dot= zeros<arma::mat>(nobj, npar);
  
  // calculate h0.dot
  for (int j = 0;  j< npar; j++){
    h0dot.col(j) =  w0 % pshat % X.col(j);
    h0dot.col(j) =  h0dot.col(j) - w0 * mean(h0dot.col(j));
  }

  arma::vec h0 = w0 - 1.0;
  arma::mat lin_core;

  if (include_h1) {
    arma::vec w1_raw = whs % d / pshat;
    arma::vec w1 = w1_raw / mean(w1_raw);
    arma::mat h1dot = zeros<arma::mat>(nobj, npar);

    for (int j = 0;  j< npar; j++){
      h1dot.col(j) = - w1 % (1.0 - pshat) % X.col(j);
      h1dot.col(j) =  h1dot.col(j) - w1 * mean(h1dot.col(j));
    }

    arma::mat hdot(nobj, 2 * npar);
    hdot.cols(0, npar - 1) = h1dot;
    hdot.cols(npar, 2 * npar - 1) = h0dot;

    arma::mat Whdot = ips_kernel_multiply(w, hdot);
    arma::mat Wh1dot = Whdot.cols(0, npar - 1);
    arma::mat Wh0dot = Whdot.cols(npar, 2 * npar - 1);
    Cmat = trans(h0dot) * Wh0dot;
    Cmat += (1.0 - treated_flag) * trans(h1dot) * Wh1dot;

    arma::vec h1 = w1 - 1.0;
    Wh1dot.each_col() %= h1;
    Wh0dot.each_col() %= h0;
    lin_core = Wh0dot + (1.0 - treated_flag) * Wh1dot;
  } else {
    arma::mat Wh0dot = ips_kernel_multiply(w, h0dot);
    Cmat = trans(h0dot) * Wh0dot;
    lin_core = Wh0dot;
    lin_core.each_col() %= h0;
  }

  Cmat /= nobj;
  bool flag = arma::inv(Cinv, Cmat);
  
  if (!flag) {
    Cinv = pinv(Cmat);
  }
  
  linrep = - lin_core * Cinv;
  return linrep;
}
