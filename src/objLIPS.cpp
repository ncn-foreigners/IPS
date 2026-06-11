#include <RcppArmadillo.h>
#include "ips_kernel.h"
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double objLIPS(const arma::vec& b, const arma::vec& d, const arma::vec& z,
               const arma::mat& X, SEXP w, const arma::vec& whs) {
  int nobj = X.n_rows;
  double obj =  0;
  
  arma::vec ipsfit = X * b;
  ipsfit = 1.0/(1.0 + exp(-ipsfit));
  
  
  arma::vec hlte1(nobj, fill::zeros);
  arma::vec hlte0(nobj, fill::zeros);
  
  hlte1 =  (whs % d % ( z/ipsfit -  (1.0-z)/(1.0-ipsfit)) )/mean(whs % d % ( z/ipsfit -  (1.0-z)/(1.0-ipsfit)) )  -  ( whs % (  1.0 - (1.0 - d)%z/ipsfit - d%(1.0 - z)/(1.0 - ipsfit) ) ) / mean( whs % (  1.0 - (1.0 - d)%z/ipsfit - d%(1.0 - z)/(1.0 - ipsfit) ) );
  hlte0 =  (whs % (1.0-d) % ( (1.0-z)/(1.0-ipsfit) - z/ipsfit) )/mean(whs % (1.0-d) % ( (1.0-z)/(1.0-ipsfit) - z/ipsfit) )  -  ( whs % (  1.0 - (1.0 - d)%z/ipsfit - d%(1.0 - z)/(1.0 - ipsfit) ) )/ mean( whs % (  1.0 - (1.0 - d)%z/ipsfit - d%(1.0 - z)/(1.0 - ipsfit) ) );
  
  
  const double n2 = static_cast<double>(nobj) * static_cast<double>(nobj);
  obj = (ips_kernel_quad(w, hlte1) + ips_kernel_quad(w, hlte0)) / n2;
  
  
  return obj;
}
