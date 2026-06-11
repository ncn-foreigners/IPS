test_that("internal kernels match dense weight matrices", {
  set.seed(20260611)
  n <- 14
  x <- cbind(1, matrix(rnorm(n * 2), ncol = 2))
  xbal <- round(x[, -1, drop = FALSE], 1)
  d <- c(rep(1, 8), rep(0, 6))
  z <- c(rep(1, 5), rep(0, 3), rep(1, 2), rep(0, 4))
  whs <- seq(0.75, 1.25, length.out = n)
  b <- c(0.1, -0.2, 0.15)
  p <- as.numeric(1 / (1 + exp(-x %*% b)))

  kernels <- list(
    exp = list(
      dense = IPS:::weightIPSexp(xbal, "normal"),
      kernel = IPS:::kernelIPSexp(xbal, "normal")
    ),
    ind = list(
      dense = IPS:::weightIPSind(xbal),
      kernel = IPS:::kernelIPSind(xbal)
    ),
    proj = list(
      dense = IPS:::weightIPSproj(xbal),
      kernel = IPS:::kernelIPSproj(xbal)
    )
  )

  for (item in kernels) {
    expect_equal(IPS:::kernelIPSDense(item$kernel), item$dense,
                 tolerance = 1e-10)
    for (treated in c(0, 1)) {
      expect_equal(
        IPS:::objIPS(b, d, x, item$kernel, treated, whs),
        IPS:::objIPS(b, d, x, item$dense, treated, whs),
        tolerance = 1e-10
      )
      expect_equal(
        IPS:::gradIPS(b, d, x, item$kernel, treated, whs),
        IPS:::gradIPS(b, d, x, item$dense, treated, whs),
        tolerance = 1e-10
      )
      expect_equal(
        IPS:::linIPS(b, d, p, x, item$kernel, treated, whs),
        IPS:::linIPS(b, d, p, x, item$dense, treated, whs),
        tolerance = 1e-8
      )
    }
    expect_equal(
      IPS:::objLIPS(b, d, z, x, item$kernel, whs),
      IPS:::objLIPS(b, d, z, x, item$dense, whs),
      tolerance = 1e-10
    )
    expect_equal(
      IPS:::gradLIPS(b, d, z, x, item$kernel, whs),
      IPS:::gradLIPS(b, d, z, x, item$dense, whs),
      tolerance = 1e-10
    )
    expect_equal(
      IPS:::linLIPS(b, d, z, p, x, item$kernel, whs),
      IPS:::linLIPS(b, d, z, p, x, item$dense, whs),
      tolerance = 1e-8
    )
  }
})

test_that("compact exp and indicator kernels match dense weights with duplicates", {
  set.seed(20260612)
  base <- matrix(round(rnorm(24), 1), ncol = 3)
  idx <- c(2, 5, 2, 7, 1, 5, 8, 3, 3, 3, 4, 6, 1, 8, 7, 2)
  x <- base[idx, , drop = FALSE]

  for (x_trans in c("normal", "arctan", "mahalanobis")) {
    expect_equal(
      IPS:::kernelIPSDense(IPS:::kernelIPSexp(x, x_trans)),
      IPS:::weightIPSexp(x, x_trans),
      tolerance = 1e-10
    )
  }

  expect_equal(
    IPS:::kernelIPSDense(IPS:::kernelIPSind(x)),
    IPS:::weightIPSind(x),
    tolerance = 1e-10
  )
})

test_that("weighted kernels match explicit frequency replication", {
  set.seed(20260616)
  x <- matrix(round(rnorm(18), 2), ncol = 3)
  freq <- c(2, 1, 3, 2, 1, 4)
  idx <- rep(seq_len(nrow(x)), freq)
  expanded <- x[idx, , drop = FALSE]
  first <- match(seq_len(nrow(x)), idx)

  for (x_trans in c("normal", "arctan", "mahalanobis")) {
    weighted <- IPS::ips_kernel(x, type = "exp", X.trans = x_trans,
                                case.weights = freq)
    expanded_kernel <- IPS::ips_kernel(expanded, type = "exp",
                                       X.trans = x_trans)
    expect_equal(
      IPS:::kernelIPSDense(weighted),
      IPS:::kernelIPSDense(expanded_kernel)[first, first],
      tolerance = 1e-10
    )
  }

  for (type in c("ind", "proj")) {
    weighted <- IPS::ips_kernel(x, type = type, case.weights = freq)
    expanded_kernel <- IPS::ips_kernel(expanded, type = type)
    expect_equal(
      IPS:::kernelIPSDense(weighted),
      IPS:::kernelIPSDense(expanded_kernel)[first, first],
      tolerance = 1e-10
    )
  }
})

test_that("weighted kernels validate case weights", {
  x <- matrix(1:6, ncol = 2)

  expect_error(
    IPS::ips_kernel(x, type = "exp", case.weights = c(1, 2)),
    "one entry"
  )
  expect_error(
    IPS::ips_kernel(x, type = "ind", case.weights = c(1, -1, 1)),
    "nonnegative"
  )
  expect_error(
    IPS::ips_kernel(x, type = "proj", case.weights = c(0, 0, 0)),
    "positive total"
  )
})

test_that("projection kernel handles sorted and unsorted duplicate rows", {
  x_sorted <- matrix(c(
    0, 0,
    0, 0,
    0, 0,
    1, 0,
    1, 0,
    1, 1
  ), ncol = 2, byrow = TRUE)

  x_unique <- matrix(c(
    0, 0,
    1, 0,
    1, 1
  ), ncol = 2, byrow = TRUE)
  counts <- c(3, 2, 1)

  expect_equal(
    IPS:::weightIPSproj(x_sorted),
    IPS:::weightIPSproj_uniq(x_unique, counts),
    tolerance = 1e-10
  )

  perm <- c(4, 1, 6, 2, 5, 3)
  x_unsorted <- x_sorted[perm, , drop = FALSE]
  expect_equal(
    IPS:::kernelIPSDense(IPS:::kernelIPSproj(x_unsorted)),
    IPS:::weightIPSproj(x_sorted)[perm, perm],
    tolerance = 1e-10
  )
})

test_that("estimators build weighted kernels from whs when no kernel is supplied", {
  set.seed(20260617)
  n <- 18
  x <- cbind(1, matrix(rnorm(n * 2), ncol = 2))
  xbal <- round(matrix(rnorm(n * 2), ncol = 2), 1)
  d <- c(rep(1, 10), rep(0, 8))
  whs <- rep(c(1, 2, 3), length.out = n)
  b <- c(0, 0, 0)

  exp_kernel <- IPS::ips_kernel(xbal, type = "exp", case.weights = whs)
  ind_kernel <- IPS::ips_kernel(xbal, type = "ind", case.weights = whs)
  proj_kernel <- IPS::ips_kernel(xbal, type = "proj", case.weights = whs)

  exp_implicit <- suppressWarnings(IPS:::IPS_exp(
    d, x, xbal = xbal, whs = whs, beta.initial = b,
    lin.rep = FALSE, maxit = 3
  ))
  exp_explicit <- suppressWarnings(IPS:::IPS_exp(
    d, x, whs = whs, beta.initial = b, lin.rep = FALSE,
    maxit = 3, kernel = exp_kernel
  ))
  expect_equal(exp_implicit$coefficients, exp_explicit$coefficients,
               tolerance = 1e-10)

  ind_implicit <- suppressWarnings(IPS:::IPS_ind(
    d, x, xbal = xbal, whs = whs, beta.initial = b,
    lin.rep = FALSE, maxit = 3
  ))
  ind_explicit <- suppressWarnings(IPS:::IPS_ind(
    d, x, whs = whs, beta.initial = b, lin.rep = FALSE,
    maxit = 3, kernel = ind_kernel
  ))
  expect_equal(ind_implicit$coefficients, ind_explicit$coefficients,
               tolerance = 1e-10)

  proj_implicit <- suppressWarnings(IPS:::IPS_proj(
    d, x, xbal = xbal, whs = whs, beta.initial = b,
    lin.rep = FALSE, maxit = 3
  ))
  proj_explicit <- suppressWarnings(IPS:::IPS_proj(
    d, x, whs = whs, beta.initial = b, lin.rep = FALSE,
    maxit = 3, kernel = proj_kernel
  ))
  expect_equal(proj_implicit$coefficients, proj_explicit$coefficients,
               tolerance = 1e-10)
})

test_that("lin.rep = FALSE skips linear representation for all estimators", {
  set.seed(20260611)
  n <- 18
  x <- cbind(1, matrix(rnorm(n * 2), ncol = 2))
  xbal <- round(x[, -1, drop = FALSE], 1)
  d <- c(rep(1, 10), rep(0, 8))
  z <- c(rep(1, 7), rep(0, 3), rep(1, 2), rep(0, 6))
  whs <- seq(0.8, 1.2, length.out = n)
  b <- c(0, 0, 0)

  fits <- list(
    suppressWarnings(IPS:::IPS_exp(d, x, xbal = xbal, beta.initial = b,
                                   lin.rep = FALSE, whs = whs, maxit = 2)),
    suppressWarnings(IPS:::IPS_ind(d, x, xbal = xbal, beta.initial = b,
                                   lin.rep = FALSE, whs = whs, maxit = 2)),
    suppressWarnings(IPS:::IPS_proj(d, x, xbal = xbal, beta.initial = b,
                                    lin.rep = FALSE, whs = whs, maxit = 2)),
    suppressWarnings(IPS:::LIPS_exp(z, d, x, xbal = xbal, beta.initial = b,
                                    lin.rep = FALSE, whs = whs, maxit = 2)),
    suppressWarnings(IPS:::LIPS_ind(z, d, x, xbal = xbal, beta.initial = b,
                                    lin.rep = FALSE, whs = whs, maxit = 2)),
    suppressWarnings(IPS:::LIPS_proj(z, d, x, xbal = xbal, beta.initial = b,
                                     lin.rep = FALSE, whs = whs, maxit = 2))
  )

  for (fit in fits) {
    expect_null(fit$lin.rep)
  }
})

test_that("precomputed kernels can be reused by IPS and LIPS estimators", {
  set.seed(20260613)
  n <- 24
  x <- cbind(1, matrix(rnorm(n * 2), ncol = 2))
  xbal <- round(matrix(rnorm(n * 3), ncol = 3), 1)
  d <- c(rep(1, 13), rep(0, 11))
  z <- c(rep(1, 9), rep(0, 4), rep(1, 3), rep(0, 8))
  whs <- seq(0.8, 1.2, length.out = n)
  b <- c(0, 0, 0)

  kernels <- list(
    exp = IPS::ips_kernel(xbal, type = "exp", X.trans = "normal",
                          case.weights = whs),
    ind = IPS::ips_kernel(xbal, type = "ind", case.weights = whs),
    proj = IPS::ips_kernel(xbal, type = "proj", case.weights = whs)
  )

  expect_s3_class(kernels$proj, "ips_kernel")
  expect_equal(attr(kernels$proj, "n"), n)

  ips_exp_ref <- suppressWarnings(IPS:::IPS_exp(
    d, x, xbal = xbal, beta.initial = b, lin.rep = FALSE,
    whs = whs, maxit = 3
  ))
  ips_exp_reuse <- suppressWarnings(IPS:::IPS_exp(
    d, x, beta.initial = b, lin.rep = FALSE, whs = whs,
    maxit = 3, kernel = kernels$exp
  ))
  expect_equal(ips_exp_reuse$coefficients, ips_exp_ref$coefficients,
               tolerance = 1e-10)

  ips_ind_ref <- suppressWarnings(IPS:::IPS_ind(
    d, x, xbal = xbal, beta.initial = b, lin.rep = FALSE,
    whs = whs, maxit = 3
  ))
  ips_ind_reuse <- suppressWarnings(IPS:::IPS_ind(
    d, x, beta.initial = b, lin.rep = FALSE, whs = whs,
    maxit = 3, kernel = kernels$ind
  ))
  expect_equal(ips_ind_reuse$coefficients, ips_ind_ref$coefficients,
               tolerance = 1e-10)

  ips_proj_ref <- suppressWarnings(IPS:::IPS_proj(
    d, x, xbal = xbal, beta.initial = b, lin.rep = FALSE,
    whs = whs, maxit = 3
  ))
  ips_proj_reuse <- suppressWarnings(IPS:::IPS_proj(
    d, x, beta.initial = b, lin.rep = FALSE, whs = whs,
    maxit = 3, kernel = kernels$proj
  ))
  expect_equal(ips_proj_reuse$coefficients, ips_proj_ref$coefficients,
               tolerance = 1e-10)

  lips_exp_ref <- suppressWarnings(IPS:::LIPS_exp(
    z, d, x, xbal = xbal, beta.initial = b, lin.rep = FALSE,
    whs = whs, maxit = 3
  ))
  lips_exp_reuse <- suppressWarnings(IPS:::LIPS_exp(
    z, d, x, beta.initial = b, lin.rep = FALSE, whs = whs,
    maxit = 3, kernel = kernels$exp
  ))
  expect_equal(lips_exp_reuse$coefficients, lips_exp_ref$coefficients,
               tolerance = 1e-10)

  lips_ind_ref <- suppressWarnings(IPS:::LIPS_ind(
    z, d, x, xbal = xbal, beta.initial = b, lin.rep = FALSE,
    whs = whs, maxit = 3
  ))
  lips_ind_reuse <- suppressWarnings(IPS:::LIPS_ind(
    z, d, x, beta.initial = b, lin.rep = FALSE, whs = whs,
    maxit = 3, kernel = kernels$ind
  ))
  expect_equal(lips_ind_reuse$coefficients, lips_ind_ref$coefficients,
               tolerance = 1e-10)

  lips_proj_ref <- suppressWarnings(IPS:::LIPS_proj(
    z, d, x, xbal = xbal, beta.initial = b, lin.rep = FALSE,
    whs = whs, maxit = 3
  ))
  lips_proj_reuse <- suppressWarnings(IPS:::LIPS_proj(
    z, d, x, beta.initial = b, lin.rep = FALSE, whs = whs,
    maxit = 3, kernel = kernels$proj
  ))
  expect_equal(lips_proj_reuse$coefficients, lips_proj_ref$coefficients,
               tolerance = 1e-10)
})

test_that("precomputed kernels are validated before reuse", {
  set.seed(20260614)
  n <- 12
  x <- cbind(1, matrix(rnorm(n * 2), ncol = 2))
  xbal <- matrix(rnorm(n * 2), ncol = 2)
  d <- c(rep(1, 6), rep(0, 6))
  b <- c(0, 0, 0)

  short_kernel <- IPS::ips_kernel(xbal[-1, , drop = FALSE], type = "exp")
  expect_error(
    IPS:::IPS_exp(d, x, beta.initial = b, lin.rep = FALSE,
                 maxit = 1, kernel = short_kernel),
    "row count"
  )

  exp_kernel <- IPS::ips_kernel(xbal, type = "exp")
  expect_error(
    IPS:::IPS_ind(d, x, beta.initial = b, lin.rep = FALSE,
                 maxit = 1, kernel = exp_kernel),
    "type"
  )
})

test_that("estimators use glm starting values without CBPS by default", {
  set.seed(20260615)
  n <- 20
  x <- cbind(1, matrix(rnorm(n * 2), ncol = 2))
  xbal <- round(matrix(rnorm(n * 2), ncol = 2), 1)
  d <- c(rep(1, 11), rep(0, 9))
  z <- c(rep(1, 7), rep(0, 4), rep(1, 3), rep(0, 6))
  whs <- seq(0.85, 1.15, length.out = n)

  fits <- list(
    suppressWarnings(IPS:::IPS_exp(d, x, xbal = xbal, lin.rep = FALSE,
                                   whs = whs, maxit = 1)),
    suppressWarnings(IPS:::IPS_ind(d, x, xbal = xbal, lin.rep = FALSE,
                                   whs = whs, maxit = 1)),
    suppressWarnings(IPS:::IPS_proj(d, x, xbal = xbal, lin.rep = FALSE,
                                    whs = whs, maxit = 1)),
    suppressWarnings(IPS:::LIPS_exp(z, d, x, xbal = xbal, lin.rep = FALSE,
                                    whs = whs, maxit = 1)),
    suppressWarnings(IPS:::LIPS_ind(z, d, x, xbal = xbal, lin.rep = FALSE,
                                    whs = whs, maxit = 1)),
    suppressWarnings(IPS:::LIPS_proj(z, d, x, xbal = xbal, lin.rep = FALSE,
                                     whs = whs, maxit = 1))
  )

  for (fit in fits) {
    expect_length(fit$coefficients, ncol(x))
    expect_true(all(is.finite(fit$coefficients)))
  }
})

test_that("C++ optimizer engine improves IPS objectives", {
  set.seed(20260620)
  n <- 28
  x <- cbind(1, matrix(rnorm(n * 2), ncol = 2))
  xbal <- round(matrix(rnorm(n * 2), ncol = 2), 1)
  d <- c(rep(1, 15), rep(0, 13))
  whs <- seq(0.75, 1.25, length.out = n)
  b <- c(0, 0, 0)

  estimators <- list(
    exp = list(
      fit = function(kernel) IPS:::IPS_exp(
        d, x, xbal = xbal, beta.initial = b, lin.rep = FALSE,
        whs = whs, maxit = 25, kernel = kernel, optim.engine = "cpp"
      ),
      kernel = IPS::ips_kernel(xbal, type = "exp", case.weights = whs)
    ),
    ind = list(
      fit = function(kernel) IPS:::IPS_ind(
        d, x, xbal = xbal, beta.initial = b, lin.rep = FALSE,
        whs = whs, maxit = 25, kernel = kernel, optim.engine = "cpp"
      ),
      kernel = IPS::ips_kernel(xbal, type = "ind", case.weights = whs)
    ),
    proj = list(
      fit = function(kernel) IPS:::IPS_proj(
        d, x, xbal = xbal, beta.initial = b, lin.rep = FALSE,
        whs = whs, maxit = 25, kernel = kernel, optim.engine = "cpp"
      ),
      kernel = IPS::ips_kernel(xbal, type = "proj", case.weights = whs)
    )
  )

  for (item in estimators) {
    start_obj <- IPS:::objIPS(b, d, x, item$kernel, 0, whs)
    fit <- suppressWarnings(item$fit(item$kernel))
    final_obj <- IPS:::objIPS(fit$coefficients, d, x, item$kernel, 0, whs)

    expect_length(fit$coefficients, ncol(x))
    expect_true(all(is.finite(fit$coefficients)))
    expect_true(is.finite(final_obj))
    expect_lte(final_obj, start_obj + 1e-8)
  }
})

test_that("weightit_ips returns a WeightIt object", {
  testthat::skip_if_not_installed("WeightIt")

  set.seed(20260618)
  n <- 20
  dat <- data.frame(
    d = c(rep(1, 11), rep(0, 9)),
    x1 = rnorm(n),
    x2 = rnorm(n),
    sw = rep(c(1, 2), length.out = n)
  )

  fit <- suppressWarnings(IPS::weightit_ips(d ~ x1 + x2,
                                            data = dat,
                                            method = "exp",
                                            s.weights = "sw",
                                            beta.initial = c(0, 0, 0),
                                            lin.rep = FALSE,
                                            maxit = 2))

  expect_s3_class(fit, "weightit")
  expect_length(fit$weights, n)
  expect_equal(fit$s.weights, dat$sw)
  expect_equal(fit$estimand, "ATE")
})
