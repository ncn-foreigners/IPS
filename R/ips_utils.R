.ips_initial_beta <- function(response, x, init.method = c("glm", "CBPS")) {
  init.method <- base::match.arg(init.method)

  if (init.method == "CBPS") {
    if (!base::requireNamespace("CBPS", quietly = TRUE)) {
      base::stop("init.method = \"CBPS\" requires the optional package 'CBPS'")
    }
    return(base::suppressWarnings(CBPS::CBPS(response ~ x[, -1],
                                             ATT = 0)$coefficients))
  }

  fit <- stats::glm.fit(x = x, y = base::as.numeric(response),
                        family = stats::binomial())
  coefficients <- fit$coefficients
  coefficients[!base::is.finite(coefficients)] <- 0
  coefficients
}

.ips_is_full_rank <- function(mat) {
  base::qr(mat)$rank == base::ncol(mat)
}

.ips_validate_case_weights <- function(case.weights, n,
                                       arg = "case.weights") {
  if (base::is.null(case.weights)) {
    return(NULL)
  }

  case.weights <- base::as.numeric(case.weights)
  if (base::length(case.weights) != n) {
    base::stop(arg, " must have one entry per observation")
  }
  if (base::any(!base::is.finite(case.weights))) {
    base::stop(arg, " must be finite")
  }
  if (base::any(case.weights < 0)) {
    base::stop(arg, " must be nonnegative")
  }
  if (base::sum(case.weights) <= 0) {
    base::stop(arg, " must have positive total weight")
  }

  case.weights
}

.ips_has_case_weights <- function(case.weights) {
  !base::is.null(case.weights) &&
    !base::all(case.weights == 1)
}

.ips_optim <- function(par, d, X, w, treated.flag, whs, maxit,
                       optim.engine = c("R", "cpp")) {
  optim.engine <- base::match.arg(optim.engine)

  if (optim.engine == "cpp") {
    return(optimIPSCpp(par, d, X, w, treated.flag, whs,
                       base::as.integer(maxit), 1e-8, 1e-8))
  }

  stats::optim(par = par,
               fn = objIPS,
               gr = gradIPS,
               method = "BFGS",
               control = list(maxit = maxit, abstol = 1e-8, reltol = 1e-8),
               d = d,
               X = X,
               w = w,
               treated_flag = treated.flag,
               whs = whs)
}
