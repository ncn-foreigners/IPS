###################################################################################
# Integrated Propensity Score estimator based on indicator weighting function
#' Integrated Propensity Score estimator based on indicator weighting function
#'
#' @param d An \eqn{n} x \eqn{1} vector of binary treatment adoption indicators.
#' @param x An \eqn{n} x \eqn{k}  matrix of covariates (potentially including interactions) to be used in the propensity score. First element must be a vector of 1's.
#' @param xbal An \eqn{n} x \eqn{l}, \eqn{l\leq k}, matrix of ``raw'' covariares to be balanced (does not need to include interaction terms). Default is \code{NULL}, which will use the same as x. 
#' @param Treated Default is FALSE, which aims to achieve covariate distribution balance among treated, untreated and overall subpopulations.
#' If TRUE, then the estimator aims to achieve covariate distribution balance for the treated subpopulation.
#' @param kernel Optional precomputed IPS kernel from \code{\link{ips_kernel}}.
#' If supplied, \code{xbal} is not used to build a new kernel.
#' @param beta.initial An optional \eqn{k} x \eqn{1} vector of initial values for the parameters to be optimized over.
#' @param init.method Method used to compute starting values when
#' \code{beta.initial = NULL}. Default is \code{"glm"}; \code{"CBPS"} uses the
#' optional \pkg{CBPS} package.
#' @param optim.engine Optimizer backend. \code{"R"} uses \code{stats::optim}
#' and preserves historical behavior; \code{"cpp"} uses an internal C++ BFGS
#' implementation that evaluates the objective and gradient together.
#' @param lin.rep Logical argument to whether an estimator for the asymptotic linear representation of the IPS
#' parameters should be provided. Deafault is TRUE.
#' @param whs An optional \eqn{n} x \eqn{1} vector of weights to be used. If NULL, then every observation has the same weights.
#' @param maxit The maximum number of iterations. Defaults to 50000.  = FALSE). Deafault is 999 if boot = TRUE
#' @param x_keep Default is FALSE. If TRUE, we return covariate matrix in the output.
#' 
#' @return A list containing the following components:
#' \item{coefficients}{The estimated IPS_ind coefficients}
#' \item{fitted.values}{The IPS_ind fitted probabilities}
#' \item{linear.predictors}{The IPS_ind estimated index (X'beta)}
#' \item{lin.rep}{An estimator of the IPS_ind coefficients' asymptotic linear representation}
#' \item{converged}{An integer code. 0 indicates successful completion}
#' \item{x}{The model matrix (i.e. the matrix of covariates used to estimate the IPS_ind parameters). Only returned if \code{x_keep = TRUE}.}
#'
#'
#' @references
#'       Sant'Anna, Pedro H. C., Song, Xiaojun, and Xu, Qi (2022), \emph{Covariate distribution balance via propensity scores},
#'       \emph{Journal of Applied Econometrics}, 37(6), 1093-1120. <https://doi.org/10.1002/jae.2909>.
#'       
#' @export

 
# Estimate IPS using logit link function
#-------------------------------------------------------------------------------
IPS_ind = function(d, x, xbal = NULL, Treated = FALSE,
                   beta.initial = NULL, lin.rep = TRUE,
                   whs = NULL, x_keep = FALSE,
                   maxit = 50000,
                   kernel = NULL,
                   init.method = c("glm", "CBPS"),
                   optim.engine = c("R", "cpp")) {
  #-----------------------------------------------------------------------------
  # Define some underlying variables
  d <-  base::as.matrix(d)
  x <- base::as.matrix(x)
  n <- base::dim(x)[1]
  k <- base::dim(x)[2]
  treated.flag <- base::as.numeric(base::isTRUE(Treated))
  optim.engine <- base::match.arg(optim.engine)
  if(is.null(whs)) whs <- rep(1, n)
  whs <- .ips_validate_case_weights(whs, n, "whs")
  #-----------------------------------------------------------------------------
  # FIRST ELEMENT OF X MUST BE A CONSTANT
  if(all.equal(as.numeric(x[,1]), rep(1,n)) == FALSE) {
    stop(" first element of x must be a vector of 1's")
  }
  #-----------------------------------------------------------------------------
  #Weight function based on exponential: exp{iu'phi(X)}
  #w.ind <- weightIPSind(x) 
  w.ind <- .ips_validate_kernel(kernel, n, "ind")
  if (base::is.null(w.ind)) {
    if(is.null(xbal)) {
      xbal.kernel <- x
    } else {
      xbal.kernel <- base::as.matrix(xbal)
    }
    w.ind <- if (.ips_has_case_weights(whs)) {
      kernelIPSindWeighted(xbal.kernel, whs)
    } else {
      kernelIPSind(xbal.kernel)
    }
  }
  #-----------------------------------------------------------------------------
  # initial parameter value for IPS
  if (is.null(beta.initial)==TRUE){
    beta.initial <- .ips_initial_beta(d, x, init.method)
  }
  #-----------------------------------------------------------------------------
  # Define the Objective function for exponential weights
  #-----------------------------------------------------------------------------
  # Define the gradient of the objective function
  #-----------------------------------------------------------------------------
  # Now we are ready to estimate the pscore parameters
  ips.est.ind <- .ips_optim(par = beta.initial,
                            d = d,
                            X = x,
                            w = w.ind,
                            treated.flag = treated.flag,
                            whs = whs,
                            maxit = maxit,
                            optim.engine = optim.engine)
  
  beta.hat.ips <- ips.est.ind$par
  converged <- ips.est.ind$convergence
  linear.predictors <- x %*% beta.hat.ips
  ps.hat <- as.numeric(1/(1 + exp(-linear.predictors)))
  probs.min <- 1e-8
  if(base::any(ps.hat<probs.min)) {
    base::message("IPS.ind: fitted probabilities smaller than 1e-8 occurred. We truncate these.")
  }
  if(base::any(ps.hat>(1-probs.min))) {
    base::message("IPS.ind: fitted probabilities bigger than 1 - 1e-8 occurred. We truncate these.")
  }
  ps.hat <- base::pmin(1 - probs.min, ps.hat)
  ps.hat <- base::pmax(probs.min, ps.hat)
  
  #-----------------------------------------------------------------------------
  #-----------------------------------------------------------------------------
  # Next, we compute an estimate of the asymptotic linear representation of
  # beta.hat - beta
  lin.rep.hat <- NULL
  if (lin.rep == TRUE){
    lin.rep.hat <- linIPS(beta.hat.ips, d, ps.hat, x, w.ind, treated.flag, whs)
    covSing <- .ips_is_full_rank(base::crossprod(lin.rep.hat))
    if(covSing==FALSE) base::message("IPS.ind: The variance-Covariance matrix is close to singular. Used Generalized-Inverse to compute std. errors.")
    
  }
  
  if(converged!=0) base::warning("IPS.ind: IPS optmization did not converge.")
  
  if(x_keep != TRUE){
    x = NULL
  }
  
  out <- list(coefficients = beta.hat.ips,
              fitted.values = ps.hat,
              linear.predictors = linear.predictors,
              lin.rep = lin.rep.hat,
              converged = converged,
              x = x,
              treated.flag = Treated)
  return(out)
}
