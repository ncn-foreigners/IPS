#ifndef IPS_KERNEL_H
#define IPS_KERNEL_H

#include <RcppArmadillo.h>

class IPSKernel {
public:
  virtual ~IPSKernel() {}
  virtual arma::uword n_rows() const = 0;
  virtual arma::mat multiply(const arma::mat& rhs) const = 0;
  virtual arma::mat dense() const = 0;
};

bool ips_is_kernel(SEXP kernel);
arma::mat ips_kernel_multiply(SEXP kernel, const arma::mat& rhs);
arma::vec ips_kernel_multiply(SEXP kernel, const arma::vec& rhs);
double ips_kernel_quad(SEXP kernel, const arma::vec& rhs);
arma::mat ips_kernel_dense(SEXP kernel);

arma::mat ips_build_exp_weights(const arma::mat& X,
                                const std::string& X_trans);
arma::mat ips_build_indicator_factor(const arma::mat& X);
arma::mat ips_build_projection_dense(const arma::mat& X);
arma::mat ips_build_projection_dense_unique(const arma::mat& X,
                                            const arma::vec& counts);

#endif
