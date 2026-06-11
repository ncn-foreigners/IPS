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
    expect_equal(
      IPS:::objIPS(b, d, x, item$kernel, 0, whs),
      IPS:::objIPS(b, d, x, item$dense, 0, whs),
      tolerance = 1e-10
    )
    expect_equal(
      IPS:::gradIPS(b, d, x, item$kernel, 0, whs),
      IPS:::gradIPS(b, d, x, item$dense, 0, whs),
      tolerance = 1e-10
    )
    expect_equal(
      IPS:::linIPS(b, d, p, x, item$kernel, 0, whs),
      IPS:::linIPS(b, d, p, x, item$dense, 0, whs),
      tolerance = 1e-8
    )
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
