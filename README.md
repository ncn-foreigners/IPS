# IPS: Covariate Distribution Balance via Integrated Propensity Scores

## Overview 


This `R` package implements the different integrated propensity score (IPS) estimators proposed in Sant'Anna, Song and Xu (2022), [Covariate distribution balance via propensity scores](https://doi.org/10.1002/jae.2909), and also the inverse probabily weigthed (IPW) estimators for the average, quantile and distributional treatment effects that build on these IPS estimators.

The IPS is estimated by fully exploiting the covariate balancing of the propensity score, i.e., by maximing the entire covariate distribution balance between the treated, untreated, and combined groups. The IPS estimators are data-driven, do not rely on tuning parameters such as bandwidths, and admit an asymptotic linear representation, which, in turn, facilitates the statistical analysis of IPW estimators for the average, quantile and distributional treatment effects.

We emphasize that the IPS can be used under different "research designs", including not only the unconfounded treatment assignment setup, but also the "local treatment effect" setup, where selection into treatment is possibly endogenous but a binary instrumental variable is available, see [Sant'Anna, Song and Xu (2022)](https://doi.org/10.1002/jae.2909) for further details.


At the moment, The `IPS` package implements three IPS estimators and three local IPS (LIPS) estimators, the latter aiming to balancing covariate distribution among compliers: 

**IPS ESTIMATORS**
        
* `IPS_exp` - This implements the IPS estimator with the exponential weigthing function.

* `IPS_proj` - This implements the IPS estimator with the projection weigthing function.

* `IPS_ind` - This implements the IPS estimator with the indicator weigthing function --- we do not recommend using this estimator when the number of covariates is moderate or large.

**LOCAL IPS ESTIMATORS (suitable for setups with treatment noncompliance)**

      
* `LIPS_exp` - This implements the local IPS estimator with the exponential weigthing function.

* `LIPS_proj` - This implements the local IPS estimator with the projection weigthing function.

* `LIPS_ind` - This implements the local IPS estimator with the indicator weigthing function --- we do not recommend using this estimator, but included it here for completeness and transparency.


On top of the aforementioned propensity score estimators, the `IPS` package also implements IPW estimators for the average, distributional and quantile treatment effects: Check out the commands `ATE`, `ATT`, `QTE`, `QTT`, `DTE`, `DTT` for treatment effect measures under unconfoundedness, and `LATE`, `LQTE`, and `LDTE` for treatment effect measures under the local treatment effect setup.

## Quick Usage

The low-level IPS estimators take a binary treatment vector `d`, a propensity score model matrix `x` whose first column is an intercept, and optional balance covariates `xbal`.

```r
fit <- IPS_proj(
  d = d,
  x = cbind(1, x1, x2),
  xbal = cbind(x1, x2),
  lin.rep = FALSE,
  optim.engine = "cpp"
)

ps <- fit$fitted.values
```

The package also provides a formula interface for users who want a `WeightIt`-compatible object. This requires the optional `WeightIt` package.

```r
w <- weightit_ips(
  treat ~ x1 + x2,
  data = dat,
  method = "proj",
  estimand = "ATE"
)

summary(w)
```

Use `method = "exp"`, `"ind"`, or `"proj"` to select the IPS kernel. `method = "proj"` is often the most direct implementation of the projection weighting approach; `method = "ind"` is included for completeness and may be slower or less attractive with many covariates. The low-level IPS estimators also accept `optim.engine = "cpp"` for an opt-in C++ BFGS optimizer that evaluates the objective and gradient together.

## Sampling or Frequency Weights

The argument `whs` in the IPS/LIPS estimators can be used for nonnegative sampling or frequency weights. These weights now enter both the IPS objective and the empirical balancing kernel.

```r
fit <- IPS_exp(
  d = d,
  x = cbind(1, x1, x2),
  xbal = cbind(x1, x2),
  whs = survey_weights,
  lin.rep = FALSE
)
```

The same weights can be supplied through the formula interface with `s.weights`.

```r
w <- weightit_ips(
  treat ~ x1 + x2,
  data = dat,
  method = "exp",
  s.weights = "survey_weights"
)
```

For frequency weights, the weighted kernels are equivalent to explicitly replicating rows according to the supplied counts, up to floating-point tolerance.

## Reusing Kernels

Kernel construction can be the dominant cost, especially for `IPS_proj`. If you run multiple estimators on the same balance covariates, build the kernel once and pass it through `kernel`.

```r
k <- ips_kernel(
  xbal = cbind(x1, x2),
  type = "proj",
  case.weights = survey_weights
)

fit1 <- IPS_proj(d, cbind(1, x1, x2), whs = survey_weights,
                 kernel = k, lin.rep = FALSE)
fit2 <- IPS_proj(d, cbind(1, x1, x2, x1 * x2), whs = survey_weights,
                 kernel = k, lin.rep = FALSE)
```

Kernel objects are external pointers and are valid only in the current R session. Rebuild them after saving/loading an R session.


For further details, please see Sant'Anna, Song and Xu (2022), [Covariate distribution balance via propensity scores](https://doi.org/10.1002/jae.2909). ***This is still a work in progress, so in case you have any comments and/or questions, please contact Pedro Sant'Anna (see email below)***.

## Installing IPS
This GitHub repository hosts the source code, and it always has the most updated version of the package.

To install the most recent version of the `IPS` package from GitHub (this is what we recommend):

        library(devtools)
        devtools::install_github("ncn-foreigners/IPS")

 If you are a macOS user and are facing issues installing our package, make sure you have Xcode installed in your machine. [Here is a detailed guidelines on how to compile Rcpp codes in macOS](https://thecoatlessprofessor.com/programming/cpp/r-compiler-tools-for-rcpp-on-macos/).
## Authors 

Pedro H. C. Sant'Anna, Microsoft (Seattle, WA) and Vanderbilt University (Nashville, TN). E-mail: pedro.h.santanna [at] vanderbilt [dot] edu or psantanna[at] microsoft.com

Xiaojun Song, Peking University, Beijing, China. E-mail: sxj [at] gsm [dot] pku [dot] edu [dot] cn.

Qi Xu, Vanderbilt University, Nashville, TN. E-mail: qi.xu.1 [at] vanderbilt [dot] edu.


## References

* Sant'Anna, Pedro H. C., Song, Xiaojun, and Xu, Qi. (2022), [Covariate distribution balance via propensity scores](https://doi.org/10.1002/jae.2909), *Journal of Applied Econometrics*, 37(6), 1093-1120.
