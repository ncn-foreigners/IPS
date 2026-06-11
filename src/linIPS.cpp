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
  
  arma::vec w1   = (whs % d/pshat) / mean(whs % d/pshat);
  arma::vec w0   = (whs % (1.0 - d)/(1.0 - pshat) )/ mean(whs % (1.0 - d)/(1.0 - pshat)) ;
  arma::mat h1dot = zeros<arma::mat>(nobj, npar);
  arma::mat h0dot= zeros<arma::mat>(nobj, npar);
  
  // calculate h11.dot
  for (int j = 0;  j< npar; j++){
    h1dot.col(j) = - w1 % (1.0 - pshat) % X.col(j);
    h1dot.col(j) =  h1dot.col(j) - w1 * mean(h1dot.col(j));

    h0dot.col(j) =  w0 % pshat % X.col(j);
    h0dot.col(j) =  h0dot.col(j) - w0 * mean(h0dot.col(j));
  }

  arma::mat Wh1dot = ips_kernel_multiply(w, h1dot);
  arma::mat Wh0dot = ips_kernel_multiply(w, h0dot);

  Cmat = ((1.0 - treated_flag ) * trans(h1dot) * Wh1dot +
          trans(h0dot) * Wh0dot) / nobj;
  bool flag = arma::inv(Cinv, Cmat);
  
  if (!flag) {
    Cinv = pinv(Cmat);
  }
  
  arma::vec h1 = w1 - 1.0;
  arma::vec h0 = w0 - 1.0;
  arma::mat lin_core = (1.0 - treated_flag) * Wh1dot;
  lin_core.each_col() %= h1;

  Wh0dot.each_col() %= h0;
  lin_core += Wh0dot;

  linrep = - lin_core * Cinv;
  return linrep;
}
