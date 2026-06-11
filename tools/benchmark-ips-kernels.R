#!/usr/bin/env Rscript

if (!requireNamespace("pkgload", quietly = TRUE)) {
  stop("Package 'pkgload' is required to run this benchmark.")
}

pkgload::load_all(".", quiet = TRUE)

make_data <- function(n, k) {
  set.seed(20260611 + n + k)
  x <- cbind(1, matrix(rnorm(n * (k - 1)), ncol = k - 1))
  xbal <- round(x[, -1, drop = FALSE], 1)
  p <- as.numeric(plogis(x %*% seq(0.05, 0.05 * k, length.out = k)))
  d <- rbinom(n, 1, p)
  z <- rbinom(n, 1, 0.55)
  whs <- runif(n, 0.75, 1.25)
  list(x = x, xbal = xbal, d = d, z = z, whs = whs, beta = rep(0, k))
}

time_call <- function(label, expr) {
  gc()
  elapsed <- system.time(force(expr))[["elapsed"]]
  data.frame(label = label, elapsed = elapsed)
}

run_one <- function(n, k) {
  dat <- make_data(n, k)
  rbind(
    time_call(
      sprintf("IPS_exp n=%d k=%d", n, k),
      IPS_exp(dat$d, dat$x, xbal = dat$xbal, beta.initial = dat$beta,
              lin.rep = FALSE, whs = dat$whs, maxit = 10)
    ),
    time_call(
      sprintf("IPS_ind n=%d k=%d", n, k),
      IPS_ind(dat$d, dat$x, xbal = dat$xbal, beta.initial = dat$beta,
              lin.rep = FALSE, whs = dat$whs, maxit = 10)
    ),
    time_call(
      sprintf("IPS_proj n=%d k=%d", n, k),
      IPS_proj(dat$d, dat$x, xbal = dat$xbal, beta.initial = dat$beta,
               lin.rep = FALSE, whs = dat$whs, maxit = 10)
    ),
    time_call(
      sprintf("LIPS_exp n=%d k=%d", n, k),
      LIPS_exp(dat$z, dat$d, dat$x, xbal = dat$xbal,
               beta.initial = dat$beta, lin.rep = FALSE,
               whs = dat$whs, maxit = 10)
    ),
    time_call(
      sprintf("LIPS_ind n=%d k=%d", n, k),
      LIPS_ind(dat$z, dat$d, dat$x, xbal = dat$xbal,
               beta.initial = dat$beta, lin.rep = FALSE,
               whs = dat$whs, maxit = 10)
    ),
    time_call(
      sprintf("LIPS_proj n=%d k=%d", n, k),
      LIPS_proj(dat$z, dat$d, dat$x, xbal = dat$xbal,
                beta.initial = dat$beta, lin.rep = FALSE,
                whs = dat$whs, maxit = 10)
    )
  )
}

sizes <- list(c(100, 4), c(250, 4), c(500, 5))
results <- do.call(rbind, lapply(sizes, function(size) run_one(size[1], size[2])))
print(results, row.names = FALSE)
