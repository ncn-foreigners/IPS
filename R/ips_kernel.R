#' Build an IPS kernel once and reuse it across estimators
#'
#' @param xbal An \eqn{n} x \eqn{l} matrix of covariates to be balanced.
#' @param type The kernel type: \code{"exp"}, \code{"ind"}, or \code{"proj"}.
#' @param X.trans For \code{type = "exp"}, the transformation used to enforce
#' compactness. The alternatives are \code{"normal"} and \code{"arctan"};
#' other values use the Mahalanobis fallback.
#' @param case.weights Optional nonnegative frequency or sampling weights used
#' to build the empirical balancing kernel. If \code{NULL}, every row receives
#' weight one.
#'
#' @return An external pointer to an internal IPS kernel. Kernel pointers are
#' session-local and should be rebuilt after saving/loading an R session.
#' @export
ips_kernel <- function(xbal, type = c("exp", "ind", "proj"),
                       X.trans = "normal", case.weights = NULL) {
  type <- base::match.arg(type)
  xbal <- base::as.matrix(xbal)
  case.weights <- .ips_validate_case_weights(case.weights, base::nrow(xbal))

  kernel <- if (.ips_has_case_weights(case.weights)) {
    switch(type,
           exp = kernelIPSexpWeighted(xbal, X.trans, case.weights),
           ind = kernelIPSindWeighted(xbal, case.weights),
           proj = kernelIPSprojWeighted(xbal, case.weights))
  } else {
    switch(type,
           exp = kernelIPSexp(xbal, X.trans),
           ind = kernelIPSind(xbal),
           proj = kernelIPSproj(xbal))
  }

  base::attr(kernel, "type") <- type
  base::attr(kernel, "n") <- base::nrow(xbal)
  base::attr(kernel, "weighted") <- .ips_has_case_weights(case.weights)
  if (!base::is.null(case.weights)) {
    base::attr(kernel, "weight.sum") <- base::sum(case.weights)
  }
  if (type == "exp") {
    base::attr(kernel, "X.trans") <- X.trans
  }

  kernel
}

#' @export
print.ips_kernel <- function(x, ...) {
  type <- base::attr(x, "type", exact = TRUE)
  n <- base::attr(x, "n", exact = TRUE)
  weighted <- base::attr(x, "weighted", exact = TRUE)
  weight.sum <- base::attr(x, "weight.sum", exact = TRUE)
  if (base::is.null(type)) type <- "unknown"
  if (base::is.null(n)) n <- "unknown"
  if (base::is.null(weighted)) weighted <- FALSE

  base::cat("IPS kernel\n")
  base::cat("  type:", type, "\n")
  base::cat("  rows:", n, "\n")
  base::cat("  weighted:", base::ifelse(weighted, "yes", "no"), "\n")
  if (!base::is.null(weight.sum)) {
    base::cat("  weight sum:", weight.sum, "\n")
  }
  base::invisible(x)
}

.ips_validate_kernel <- function(kernel, n, type) {
  if (base::is.null(kernel)) {
    return(NULL)
  }

  if (base::is.matrix(kernel)) {
    if (!base::is.numeric(kernel)) {
      base::stop("kernel matrix must be numeric")
    }
    if (!base::identical(base::dim(kernel), base::c(n, n))) {
      base::stop("kernel matrix must be n x n, where n is the number of rows in x")
    }
    return(kernel)
  }

  if (!base::inherits(kernel, "ips_kernel")) {
    base::stop("kernel must be created by ips_kernel() or be an n x n numeric matrix")
  }

  kernel_n <- base::attr(kernel, "n", exact = TRUE)
  if (!base::is.null(kernel_n) && !base::identical(base::as.integer(kernel_n),
                                                   base::as.integer(n))) {
    base::stop("kernel row count does not match x")
  }

  kernel_type <- base::attr(kernel, "type", exact = TRUE)
  if (!base::is.null(kernel_type) && !base::identical(kernel_type, type)) {
    base::stop("kernel type does not match estimator")
  }

  kernel
}
