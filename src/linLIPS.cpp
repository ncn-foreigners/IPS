#include <RcppArmadillo.h>
#include "ips_kernel.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;


// [[Rcpp::export]]
arma::mat linLIPS(const arma::vec& bhat, const arma::vec& d,
                  const arma::vec& z, const arma::vec& ipshat,
                  const arma::mat& X, SEXP w, const arma::vec& whs) {
  int nobj = X.n_rows, npar = X.n_cols;
  
  arma::mat Cinv = zeros<arma::mat>(npar,npar);
  arma::mat Cmat = zeros<arma::mat>(npar,npar);
  arma::mat linrep = zeros<arma::mat>(nobj,npar);
  
  
  arma::vec hlte1(nobj, fill::zeros);
  arma::vec hlte0(nobj, fill::zeros);
  
  hlte1 =  (whs % d % ( z/ipshat -  (1.0-z)/(1.0-ipshat)) )/mean(whs % d % ( z/ipshat -  (1.0-z)/(1.0-ipshat)) )  -  ( whs % (  1.0 - (1.0 - d)%z/ipshat - d%(1.0 - z)/(1.0 - ipshat) ) ) / mean( whs % (  1.0 - (1.0 - d)%z/ipshat - d%(1.0 - z)/(1.0 - ipshat) ) );
  hlte0 =  (whs % (1.0-d) % ( (1.0-z)/(1.0-ipshat) - z/ipshat) )/mean(whs % (1.0-d) % ( (1.0-z)/(1.0-ipshat) - z/ipshat) )  -  ( whs % (  1.0 - (1.0 - d)%z/ipshat - d%(1.0 - z)/(1.0 - ipshat) ) )/ mean( whs % (  1.0 - (1.0 - d)%z/ipshat - d%(1.0 - z)/(1.0 - ipshat) ) );
  
  
  // calculate hlte dot
  arma::mat hltedot1(nobj, npar, fill::zeros);
  arma::mat hltedot0(nobj, npar, fill::zeros);
  arma::mat hltedotw1(nobj, npar, fill::zeros);
  arma::mat hltedotw(nobj, npar, fill::zeros);
  arma::mat hltedotw0(nobj, npar, fill::zeros);
  
  
  
  for (int j = 0;  j<npar; j++){
    hltedotw1.col(j) = - whs% d % (z % (1.0 - ipshat)/ipshat + (1.0 - z) % ipshat /(1.0 - ipshat)) /mean(whs % d % ( z/ipshat -  (1.0-z)/(1.0-ipshat)) ) %  X.col(j);
    hltedotw1.col(j) =  hltedotw1.col(j) +  whs % d % ( z/ipshat -  (1.0-z)/(1.0-ipshat))/mean(whs % d % ( z/ipshat -  (1.0-z)/(1.0-ipshat)) ) * mean(whs% d % (z % (1.0 - ipshat)/ipshat + (1.0 - z) % ipshat /(1.0 - ipshat)) /mean(whs % d % ( z/ipshat -  (1.0-z)/(1.0-ipshat)) ) %  X.col(j));
    
    hltedotw0.col(j) =  whs% (1.0-d) % (z % (1.0 - ipshat)/ipshat + (1.0 - z) % ipshat /(1.0 - ipshat)) /mean(whs % (1.0-d) % ( (1.0-z)/(1.0-ipshat) - z/ipshat) ) %  X.col(j);
    hltedotw0.col(j) =  hltedotw0.col(j) - whs % (1.0-d) % ( (1.0-z)/(1.0-ipshat) - z/ipshat)/mean(whs % (1.0-d) % ( (1.0-z)/(1.0-ipshat) - z/ipshat)) * mean( whs% (1.0-d) % (z % (1.0 - ipshat)/ipshat + (1.0 - z) % ipshat /(1.0 - ipshat)) /mean(whs % (1.0-d) % ( (1.0-z)/(1.0-ipshat) - z/ipshat) ) %  X.col(j));
    
    
    hltedotw.col(j) =   whs % ( (1.0 - d)% z % (1.0 - ipshat)/ipshat - d%(1.0 - z) % ipshat/(1.0 - ipshat) ) / mean(whs % (1.0 - (1.0 - d)%z/ipshat - d%(1.0 - z)/(1.0 - ipshat))) %  X.col(j); 
    hltedotw.col(j) =   hltedotw.col(j) -  whs%(1.0 - (1.0 - d)%z/ipshat - d%(1.0 - z)/(1.0 - ipshat))/mean(whs % (1.0 - (1.0 - d)%z/ipshat - d%(1.0 - z)/(1.0 - ipshat))) * mean(whs % ( (1.0 - d)% z % (1.0 - ipshat)/ipshat - d%(1.0 - z) % ipshat/(1.0 - ipshat) ) / mean(whs % (1.0 - (1.0 - d)%z/ipshat - d%(1.0 - z)/(1.0 - ipshat)))%  X.col(j));
    
    hltedot1.col(j) =  hltedotw1.col(j) - hltedotw.col(j);
    hltedot0.col(j) =  hltedotw0.col(j) - hltedotw.col(j);
    
  }
  
  
  
  
  arma::mat Whltedot1 = ips_kernel_multiply(w, hltedot1);
  arma::mat Whltedot0 = ips_kernel_multiply(w, hltedot0);

  Cmat = (trans(hltedot1) * Whltedot1 +
          trans(hltedot0) * Whltedot0) / nobj;
  bool flag = arma::inv(Cinv, Cmat);
  
  if (!flag) {
    Cinv = pinv(Cmat);
    ///Rcout << "The Variance-Covariance Matrix is close to singular - proceed with caution!\n";
  }
  
  arma::mat lin_core = Whltedot1;
  lin_core.each_col() %= hlte1;

  Whltedot0.each_col() %= hlte0;
  lin_core += Whltedot0;

  linrep = - lin_core * Cinv;
  
  
  
  
  
  
  
  return linrep;
}
