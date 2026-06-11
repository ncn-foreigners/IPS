test_that("weightit_ips computes exact ATE weights from supplied ps", {
  testthat::skip_if_not_installed("WeightIt")

  dat <- data.frame(
    d = c(1, 0, 1, 0, 1, 0),
    x1 = seq(-1, 1, length.out = 6),
    x2 = c(2, 1, 3, 2, 4, 3),
    sw = c(1, 2, 1, 3, 2, 4)
  )
  ps <- c(0.8, 0.2, 0.6, 0.3, 0.7, 0.4)

  fit <- IPS::weightit_ips(d ~ x1 + x2, data = dat, ps = ps,
                           s.weights = "sw")
  expected <- ifelse(dat$d == 1, 1 / ps, 1 / (1 - ps))

  expect_s3_class(fit, "weightit")
  expect_equal(as.numeric(fit$weights), expected)
  expect_equal(fit$ps, ps)
  expect_equal(fit$s.weights, dat$sw)
  expect_equal(fit$method, "ips_proj")

  stabilized <- IPS::weightit_ips(d ~ x1 + x2, data = dat, ps = ps,
                                  s.weights = dat$sw,
                                  stabilize = TRUE)
  treated_share <- stats::weighted.mean(dat$d, dat$sw)
  expected_stabilized <- expected
  expected_stabilized[dat$d == 1] <- expected_stabilized[dat$d == 1] *
    treated_share
  expected_stabilized[dat$d == 0] <- expected_stabilized[dat$d == 0] *
    (1 - treated_share)

  expect_equal(as.numeric(stabilized$weights), expected_stabilized)
})

test_that("weightit_ips computes exact ATT weights from supplied ps", {
  testthat::skip_if_not_installed("WeightIt")

  dat <- data.frame(
    tr = factor(c("treated", "control", "treated", "control",
                  "treated", "control"),
                levels = c("control", "treated")),
    x1 = c(1, 2, 1, 3, 2, 4),
    x2 = c(0, 1, 1, 0, 2, 2),
    sw = c(1, 2, 1, 3, 2, 4)
  )
  d <- as.numeric(dat$tr == "treated")
  ps <- c(0.75, 0.25, 0.65, 0.35, 0.7, 0.2)

  fit <- IPS::weightit_ips(tr ~ x1 + x2, data = dat, ps = ps,
                           estimand = "ATT", s.weights = "sw")
  expected <- ifelse(d == 1, 1, ps / (1 - ps))

  expect_equal(as.numeric(fit$treat), d)
  expect_equal(as.numeric(fit$weights), expected)
  expect_equal(fit$estimand, "ATT")

  stabilized <- IPS::weightit_ips(tr ~ x1 + x2, data = dat, ps = ps,
                                  estimand = "ATT", s.weights = dat$sw,
                                  stabilize = TRUE)
  treated_share <- stats::weighted.mean(d, dat$sw)
  expected_stabilized <- expected
  expected_stabilized[d == 0] <- expected_stabilized[d == 0] *
    (1 - treated_share) / treated_share

  expect_equal(as.numeric(stabilized$weights), expected_stabilized)
})

test_that("weightit_ips uses requested balance covariates", {
  testthat::skip_if_not_installed("WeightIt")

  dat <- data.frame(
    d = c(1, 0, 1, 0, 1, 0),
    x1 = 1:6,
    x2 = c(2, 1, 3, 2, 4, 3),
    z = c(0, 1, 0, 1, 0, 1)
  )
  ps <- c(0.8, 0.2, 0.6, 0.3, 0.7, 0.4)

  formula_balance <- IPS::weightit_ips(d ~ x1 + x2, data = dat, ps = ps,
                                       balance = ~ x2 + I(z + x1))
  expect_named(formula_balance$covs, c("x2", "I(z + x1)"))
  expect_equal(formula_balance$covs[["I(z + x1)"]], dat$z + dat$x1)

  balance_matrix <- cbind(z = dat$z, x1_squared = dat$x1^2)
  matrix_balance <- IPS::weightit_ips(d ~ x1 + x2, data = dat, ps = ps,
                                      balance = balance_matrix)
  expect_equal(matrix_balance$covs, as.data.frame(balance_matrix))
})

test_that("weightit_ips validates user-facing inputs", {
  testthat::skip_if_not_installed("WeightIt")

  dat <- data.frame(
    d = c(1, 0, 1, 0),
    x = 1:4,
    sw = c(1, 2, 3, 4)
  )

  expect_error(
    IPS::weightit_ips(d ~ x, data = transform(dat, d = c(0, 1, 2, 1)),
                      ps = rep(0.5, 4)),
    "binary treatment"
  )
  expect_error(
    IPS::weightit_ips(d ~ x, data = dat, ps = c(0.5, 0.6)),
    "ps must have one entry"
  )
  expect_error(
    IPS::weightit_ips(d ~ x, data = dat, ps = rep(0.5, 4),
                      s.weights = "missing"),
    "not found"
  )
  expect_error(
    IPS::weightit_ips(d ~ x, data = dat, ps = rep(0.5, 4),
                      s.weights = c(1, -1, 1, 1)),
    "nonnegative"
  )
  expect_error(
    IPS::weightit_ips(d ~ x, data = dat, ps = rep(0.5, 4),
                      balance = matrix(1, nrow = 3)),
    "one row"
  )
})

test_that("weightit_ips stores fitted IPS object and weighted kernel", {
  testthat::skip_if_not_installed("WeightIt")

  set.seed(20260619)
  n <- 16
  dat <- data.frame(
    d = c(rep(1, 9), rep(0, 7)),
    x1 = rnorm(n),
    x2 = rnorm(n),
    sw = rep(c(1, 2, 3, 1), length.out = n)
  )

  fit <- suppressWarnings(IPS::weightit_ips(
    d ~ x1 + x2,
    data = dat,
    method = "exp",
    s.weights = "sw",
    beta.initial = c(0, 0, 0),
    lin.rep = FALSE,
    maxit = 2,
    include.obj = TRUE
  ))

  expect_s3_class(fit$obj$kernel, "ips_kernel")
  expect_true(attr(fit$obj$kernel, "weighted", exact = TRUE))
  expect_equal(attr(fit$obj$kernel, "weight.sum", exact = TRUE), sum(dat$sw))
  expect_equal(fit$ps, fit$obj$fit$fitted.values)
  expect_equal(fit$s.weights, dat$sw)

  expected <- ifelse(dat$d == 1, 1 / fit$ps, 1 / (1 - fit$ps))
  expect_equal(as.numeric(fit$weights), expected)
})

test_that("weightit_ips forwards the C++ optimizer engine", {
  testthat::skip_if_not_installed("WeightIt")

  set.seed(20260621)
  n <- 18
  dat <- data.frame(
    d = c(rep(1, 10), rep(0, 8)),
    x1 = rnorm(n),
    x2 = rnorm(n)
  )

  fit <- suppressWarnings(IPS::weightit_ips(
    d ~ x1 + x2,
    data = dat,
    method = "ind",
    beta.initial = c(0, 0, 0),
    lin.rep = FALSE,
    maxit = 10,
    optim.engine = "cpp",
    include.obj = TRUE
  ))

  expect_s3_class(fit, "weightit")
  expect_true(all(is.finite(fit$ps)))
  expect_equal(fit$ps, fit$obj$fit$fitted.values)
  expect_s3_class(fit$obj$kernel, "ips_kernel")
})
