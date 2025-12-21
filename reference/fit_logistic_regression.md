# Fit Logistic Regression Model

This function fits a logistic regression model to binary classification
data. It supports both dense and sparse matrix inputs for the predictor
variables. The optimization is performed using the L-BFGS algorithm.

## Usage

``` r
fit_logistic_regression(X_r, y_r, maxit = 300L, eps_f = 1e-08, eps_g = 1e-05)
```

## Arguments

- X_r:

  A numeric matrix or sparse matrix (dgCMatrix) of predictor variables.
  Rows represent observations and columns represent features.

- y_r:

  A numeric vector of binary outcome values (0 or 1). Must have the same
  number of observations as rows in `X_r`.

- maxit:

  Integer. Maximum number of iterations for the optimizer. Default is
  300.

- eps_f:

  Numeric. Convergence tolerance for the function value. Default is
  1e-8.

- eps_g:

  Numeric. Convergence tolerance for the gradient norm. Default is 1e-5.

## Value

A list containing the results of the logistic regression fit:

- `coefficients`:

  Numeric vector of estimated regression coefficients.

- `se`:

  Numeric vector of standard errors for the coefficients.

- `z_scores`:

  Numeric vector of z-statistics for testing coefficient significance.

- `p_values`:

  Numeric vector of p-values associated with the z-statistics.

- `loglikelihood`:

  Scalar. The maximized log-likelihood value.

- `gradient`:

  Numeric vector. The gradient at the solution.

- `hessian`:

  Matrix. The Hessian matrix evaluated at the solution.

- `convergence`:

  Logical. Whether the algorithm converged successfully.

- `iterations`:

  Integer. Number of iterations performed.

- `message`:

  Character. Convergence message.

## Details

The logistic regression model estimates the probability of the binary
outcome \\y_i \in \\0, 1\\\\ given predictors \\x_i\\: \$\$P(y_i = 1 \|
x_i) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x\_{i1} + ... + \beta_p
x\_{ip})}}\$\$

The function maximizes the log-likelihood: \$\$\ell(\beta) =
\sum\_{i=1}^n \[y_i \cdot (\beta^T x_i) - \ln(1 + e^{\beta^T x_i})\]\$\$

Standard errors are computed from the inverse of the Hessian matrix
evaluated at the estimated coefficients. Z-scores and p-values are
derived under the assumption of asymptotic normality.

## Note

- An intercept term is not automatically included. Users should add a
  column of ones to `X_r` if an intercept is desired.

- If the Hessian matrix is singular (determinant is zero), standard
  errors, z-scores, and p-values will be returned as `NA`.

- The function uses the L-BFGS quasi-Newton optimization method.

## Examples

``` r
# Generate sample data
set.seed(123)
n <- 100
p <- 3
X <- matrix(rnorm(n * p), n, p)
# Add intercept column
X <- cbind(1, X)
colnames(X) <- c("(Intercept)", "X1", "X2", "X3")

# True coefficients
beta_true <- c(0.5, 1.2, -0.8, 0.3)

# Generate linear predictor
eta <- X %*% beta_true

# Generate binary outcome
prob <- 1 / (1 + exp(-eta))
y <- rbinom(n, 1, prob)

# Fit logistic regression
result <- fit_logistic_regression(X, y)

# View coefficients and statistics
print(data.frame(
  Coefficient = result$coefficients,
  Std_Error = result$se,
  Z_score = result$z_scores,
  P_value = result$p_values
))
#>   Coefficient Std_Error   Z_score     P_value
#> 1   0.4677691 0.2424352  1.929460 0.053673775
#> 2   1.2559101 0.3432823  3.658534 0.000253662
#> 3  -0.7060973 0.2735459 -2.581275 0.009843607
#> 4   0.5184608 0.2689094  1.928013 0.053853568

# Check convergence
cat("Converged:", result$convergence, "\n")
#> Converged: TRUE 
cat("Log-Likelihood:", result$loglikelihood, "\n")
#> Log-Likelihood: -51.99246 
```
