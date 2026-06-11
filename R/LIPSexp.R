###################################################################################
# 'Local' Integrated Propensity Score estimator based on exponential weighting function
#' 'Local' Integrated Propensity Score estimator based on exponential weighting function
#'
#' @param z An \eqn{n} x \eqn{1} vector of binary instruments.
#' @param d An \eqn{n} x \eqn{1} vector of binary treatment adoption indicators.
#' @param x An \eqn{n} x \eqn{k}  matrix of covariates to be used in the propensity score. First element must be a vector of 1's.
#' @param xbal An \eqn{n} x \eqn{l}, \eqn{l\leq k}, matrix of ``raw'' covariares to be balanced (does not need to include interaction terms). Default is \code{NULL}, which will use the same as x. 
#' @param X.trans description of which transformation of covariates is used to enforce compactness.
#'            The alternatives are 'normal' (default), and 'arctan'.
#' @param kernel Optional precomputed IPS kernel from \code{\link{ips_kernel}}.
#' If supplied, \code{xbal} and \code{X.trans} are not used to build a new kernel.
#' @param beta.initial An optional \eqn{k} x \eqn{1} vector of initial values for the parameters to be optimized over.
#' @param init.method Method used to compute starting values when
#' \code{beta.initial = NULL}. Default is \code{"glm"}; \code{"CBPS"} uses the
#' optional \pkg{CBPS} package.
#' @param lin.rep Logical argument to whether an estimator for the asymptotic linear representation of the LIPS
#' parameters should be provided. Deafault is TRUE.
#' @param whs An optional \eqn{n} x \eqn{1} vector of weights to be used. If NULL, then every observation has the same weights.
#' @param maxit The maximum number of iterations. Defaults to 50000.  = FALSE). Deafault is 999 if boot = TRUE
#' @param x_keep Default is FALSE. If TRUE, we return covariate matrix in the output.
#' 
#' @return A list containing the following components:
#' \item{coefficients}{The estimated LIPS_exp coefficients}
#' \item{fitted.values}{The LIPS_exp fitted probabilities}
#' \item{linear.predictors}{The LIPS_exp estimated index (X'beta)}
#' \item{lin.rep}{An estimator of the LIPS_exp coefficients' asymptotic linear representation}
#' \item{converged}{An integer code. 0 indicates successful completion}
#' \item{x}{The model matrix (i.e. the matrix of covariates used to estimate the LIPS_exp parameters). Only returned if \code{x_keep = TRUE}.}
#'
#'
#' @references
#'       Sant'Anna, Pedro H. C., Song, Xiaojun, and Xu, Qi (2022), \emph{Covariate distribution balance via propensity scores},
#'       \emph{Journal of Applied Econometrics}, 37(6), 1093-1120. <https://doi.org/10.1002/jae.2909>.
#' @export


# Estimate Local IPS using logit link function
#-------------------------------------------------------------------------------
LIPS_exp = function(z, d, x, xbal = NULL,
                    X.trans = "normal", 
                   beta.initial = NULL, lin.rep = TRUE, 
                   whs = NULL,  x_keep = FALSE,
                   maxit = 50000,
                   kernel = NULL,
                   init.method = c("glm", "CBPS")) {
  #-----------------------------------------------------------------------------
  # Define some underlying variables
  d <-  base::as.matrix(d)
  z <- base::as.matrix(z)
  x <- base::as.matrix(x)
  n <- base::dim(x)[1]
  k <- base::dim(x)[2]
 
  if(is.null(whs)) whs <- rep(1, n)
  whs <- .ips_validate_case_weights(whs, n, "whs")
  #-----------------------------------------------------------------------------
  # FIRST ELEMENT OF X MUST BE A CONSTANT
  if(all.equal(x[,1], rep(1,n)) == FALSE) {
    stop(" first element of x must be a vector of 1's")
  }
  #-----------------------------------------------------------------------------
  #Weight function based on exponential: exp{iu'phi(X)}
  w.exp <- .ips_validate_kernel(kernel, n, "exp")
  if (base::is.null(w.exp)) {
    if(is.null(xbal)) {
      xbal.kernel <- x[, -1, drop = FALSE]
    } else {
      xbal.kernel <- base::as.matrix(xbal)
    }
    w.exp <- if (.ips_has_case_weights(whs)) {
      kernelIPSexpWeighted(xbal.kernel, X.trans, whs)
    } else {
      kernelIPSexp(xbal.kernel, X.trans)
    }
  }
  
  #-----------------------------------------------------------------------------
  # initial parameter value for IPS ???
  if (is.null(beta.initial)==TRUE){
    beta.initial <- .ips_initial_beta(z, x, init.method)
  }
  #-----------------------------------------------------------------------------
  # Define the Objective function for exponential weights
  #-----------------------------------------------------------------------------
  # Define the gradient of the objective function
  #-----------------------------------------------------------------------------
  # Now we are ready to estimate the pscore parameters
  ips.est.exp <- stats::optim(par = beta.initial,
                              fn = objLIPS,
                              gr = gradLIPS,
                              method = "BFGS",
                              control =  list(maxit = maxit, abstol = 1e-8, reltol=1e-8),
                              d = d,
                              z = z,
                              X = x,
                              w = w.exp,
                              whs = whs)
  
  beta.hat.ips <- ips.est.exp$par
  converged <- ips.est.exp$convergence
  linear.predictors <- x %*% beta.hat.ips
  ips.hat <- as.numeric(1/(1 + exp(-linear.predictors)))
  probs.min <- 1e-8
  if(base::any(ips.hat<probs.min)) {
    base::message("LIPS.proj: fitted probabilities smaller than 1e-8 occurred. We truncate these.")
  }
  if(base::any(ips.hat>(1-probs.min))) {
    base::message("LIPS.proj: fitted probabilities bigger than 1 - 1e-8 occurred. We truncate these.")
  }
  ips.hat <- base::pmin(1 - probs.min, ips.hat)
  ips.hat <- base::pmax(probs.min, ips.hat)
  
  #-----------------------------------------------------------------------------
  #-----------------------------------------------------------------------------
  # Next, we compute an estimate of the asymptotic linear representation of
  # beta.hat - beta
  lin.rep.hat <- NULL
  if (lin.rep == TRUE){
    lin.rep.hat <- linLIPS(beta.hat.ips, d, z, ips.hat, x, w.exp, whs)
    covSing <- .ips_is_full_rank(base::crossprod(lin.rep.hat))
    if(covSing==FALSE) base::message("LIPS.exp: The variance-Covariance matrix is close to singular. Used Generalized-Inverse to compute std. errors.")
    
  }
  if(converged!=0) base::warning("LIPS.exp: IPS optmization did not converge.")
  
  if(x_keep != TRUE){
    x = NULL
  }
  
  out <- list(coefficients = beta.hat.ips,
              fitted.values = ips.hat,
              linear.predictors = linear.predictors,
              lin.rep = lin.rep.hat,
              converged = converged,
              x = x
  )
  return(out)
}
