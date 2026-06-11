#' Estimate IPS weights in a WeightIt-compatible object
#'
#' @param formula A formula with a binary treatment on the left-hand side and
#' propensity score covariates on the right-hand side.
#' @param data Optional data frame containing the variables in \code{formula}.
#' @param method IPS kernel type. One of \code{"proj"}, \code{"exp"}, or
#' \code{"ind"}.
#' @param estimand Target estimand. \code{"ATE"} estimates inverse probability
#' weights for the full population; \code{"ATT"} estimates weights for the
#' treated population.
#' @param balance Optional balance covariates. If \code{NULL}, the model matrix
#' from \code{formula}, excluding the intercept, is used. A one-sided formula,
#' matrix, or data frame may be supplied.
#' @param X.trans For \code{method = "exp"}, the transformation passed to
#' \code{\link{IPS_exp}}.
#' @param s.weights Optional nonnegative sampling or frequency weights. These
#' are passed to \code{whs}, used when building the IPS kernel, and returned in
#' the \pkg{WeightIt} object.
#' @param stabilize Logical; if \code{TRUE}, use stabilized IPW weights.
#' @param ps Optional propensity scores. If supplied, IPS fitting is skipped and
#' only the \pkg{WeightIt} object is constructed.
#' @param kernel Optional precomputed IPS kernel from \code{\link{ips_kernel}}.
#' @param beta.initial Optional initial coefficient vector passed to the IPS
#' estimator.
#' @param init.method Initializer used when \code{beta.initial = NULL}. See
#' \code{\link{IPS_exp}}.
#' @param optim.engine Optimizer backend passed to the IPS estimator. Use
#' \code{"R"} for \code{stats::optim()} or \code{"cpp"} for the internal C++
#' BFGS optimizer.
#' @param lin.rep Logical; passed to the IPS estimator.
#' @param maxit Maximum optimizer iterations passed to the IPS estimator.
#' @param include.obj Logical; if \code{TRUE}, include the IPS fit and kernel in
#' the returned object.
#' @param ... Additional named arguments passed to \code{WeightIt::as.weightit()}.
#'
#' @return A \code{weightit} object from \pkg{WeightIt}.
#' @export
weightit_ips <- function(formula, data = NULL,
                         method = c("proj", "exp", "ind"),
                         estimand = c("ATE", "ATT"),
                         balance = NULL,
                         X.trans = "normal",
                         s.weights = NULL,
                         stabilize = FALSE,
                         ps = NULL,
                         kernel = NULL,
                         beta.initial = NULL,
                         init.method = c("glm", "CBPS"),
                         optim.engine = c("R", "cpp"),
                         lin.rep = FALSE,
                         maxit = 50000,
                         include.obj = FALSE,
                         ...) {
  if (!base::requireNamespace("WeightIt", quietly = TRUE)) {
    base::stop("weightit_ips() requires the optional package 'WeightIt'")
  }

  method <- base::match.arg(method)
  estimand <- base::match.arg(estimand)
  init.method <- base::match.arg(init.method)
  optim.engine <- base::match.arg(optim.engine)

  mf <- stats::model.frame(formula, data = data, na.action = stats::na.fail)
  treat <- stats::model.response(mf)
  d <- .ips_encode_binary_treat(treat)
  n <- base::length(d)
  x <- stats::model.matrix(formula, data = mf)

  if (!base::all(x[, 1] == 1)) {
    base::stop("the propensity score model matrix must include an intercept")
  }

  s.weights <- .ips_get_s_weights(s.weights, data, n)
  xbal <- .ips_balance_matrix(balance, formula, data, mf, x, n)
  covs <- .ips_weightit_covariates(balance, formula, data, mf, n)

  fit <- NULL
  ps.hat <- .ips_validate_ps(ps, n)
  if (base::is.null(ps.hat)) {
    if (base::is.null(kernel)) {
      kernel <- ips_kernel(xbal, type = method, X.trans = X.trans,
                           case.weights = s.weights)
    }

    fit <- switch(method,
                  exp = IPS_exp(d, x, xbal = xbal, X.trans = X.trans,
                                Treated = estimand == "ATT",
                                beta.initial = beta.initial,
                                lin.rep = lin.rep,
                                whs = s.weights,
                                maxit = maxit,
                                kernel = kernel,
                                init.method = init.method,
                                optim.engine = optim.engine),
                  ind = IPS_ind(d, x, xbal = xbal,
                                Treated = estimand == "ATT",
                                beta.initial = beta.initial,
                                lin.rep = lin.rep,
                                whs = s.weights,
                                maxit = maxit,
                                kernel = kernel,
                                init.method = init.method,
                                optim.engine = optim.engine),
                  proj = IPS_proj(d, x, xbal = xbal,
                                  Treated = estimand == "ATT",
                                  beta.initial = beta.initial,
                                  lin.rep = lin.rep,
                                  whs = s.weights,
                                  maxit = maxit,
                                  kernel = kernel,
                                  init.method = init.method,
                                  optim.engine = optim.engine))
    ps.hat <- fit$fitted.values
  }

  weights <- .ips_weightit_weights(d, ps.hat, estimand, stabilize, s.weights)
  extra <- base::list(...)
  args <- base::c(base::list(x = weights,
                             treat = d,
                             covs = covs,
                             estimand = estimand,
                             s.weights = s.weights,
                             ps = ps.hat,
                             method = base::paste0("ips_", method)),
                  extra)

  out <- base::do.call(WeightIt::as.weightit, args)
  if (include.obj) {
    out$obj <- base::list(fit = fit,
                          kernel = kernel,
                          call = base::match.call())
  }

  out
}

.ips_encode_binary_treat <- function(treat) {
  if (base::is.factor(treat)) {
    if (base::nlevels(treat) != 2) {
      base::stop("formula must contain a binary treatment")
    }
    return(base::as.numeric(treat == base::levels(treat)[2]))
  }

  if (base::is.logical(treat)) {
    return(base::as.numeric(treat))
  }

  values <- base::sort(base::unique(treat))
  if (base::length(values) != 2) {
    base::stop("formula must contain a binary treatment")
  }

  if (base::all(values == c(0, 1))) {
    return(base::as.numeric(treat))
  }

  base::as.numeric(treat == values[2])
}

.ips_get_s_weights <- function(s.weights, data, n) {
  if (base::is.null(s.weights)) {
    return(NULL)
  }

  if (base::is.character(s.weights) && base::length(s.weights) == 1) {
    if (base::is.null(data) || base::is.null(data[[s.weights]])) {
      base::stop("s.weights was supplied as a name but was not found in data")
    }
    s.weights <- data[[s.weights]]
  }

  .ips_validate_case_weights(s.weights, n, "s.weights")
}

.ips_balance_matrix <- function(balance, formula, data, mf, x, n) {
  if (base::is.null(balance)) {
    return(x[, -1, drop = FALSE])
  }

  if (base::inherits(balance, "formula")) {
    bmf <- stats::model.frame(balance, data = data, na.action = stats::na.fail)
    out <- stats::model.matrix(balance, data = bmf)
    if (base::ncol(out) > 0 && base::all(out[, 1] == 1)) {
      out <- out[, -1, drop = FALSE]
    }
  } else {
    out <- base::as.matrix(balance)
  }

  if (base::nrow(out) != n) {
    base::stop("balance must have one row per observation")
  }

  out
}

.ips_weightit_covariates <- function(balance, formula, data, mf, n) {
  if (base::is.null(balance)) {
    return(mf[-1])
  }

  if (base::inherits(balance, "formula")) {
    covs <- stats::model.frame(balance, data = data, na.action = stats::na.fail)
  } else {
    covs <- base::as.data.frame(balance)
  }

  if (base::nrow(covs) != n) {
    base::stop("balance must have one row per observation")
  }

  covs
}

.ips_validate_ps <- function(ps, n) {
  if (base::is.null(ps)) {
    return(NULL)
  }

  ps <- base::as.numeric(ps)
  if (base::length(ps) != n) {
    base::stop("ps must have one entry per observation")
  }
  if (base::any(!base::is.finite(ps))) {
    base::stop("ps must be finite")
  }

  probs.min <- 1e-8
  base::pmin(1 - probs.min, base::pmax(probs.min, ps))
}

.ips_weightit_weights <- function(d, ps, estimand, stabilize, s.weights) {
  treated_share <- if (base::is.null(s.weights)) {
    base::mean(d)
  } else {
    stats::weighted.mean(d, s.weights)
  }

  if (estimand == "ATT") {
    weights <- base::ifelse(d == 1, 1, ps / (1 - ps))
    if (base::isTRUE(stabilize)) {
      weights[d == 0] <- weights[d == 0] * (1 - treated_share) / treated_share
    }
    return(weights)
  }

  weights <- base::ifelse(d == 1, 1 / ps, 1 / (1 - ps))
  if (base::isTRUE(stabilize)) {
    weights[d == 1] <- weights[d == 1] * treated_share
    weights[d == 0] <- weights[d == 0] * (1 - treated_share)
  }

  weights
}
