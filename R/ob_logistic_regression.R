#' Fit Logistic Regression Model
#'
#' This function fits a logistic regression model to binary classification data.
#' It supports both dense and sparse matrix inputs for the predictor variables.
#' The optimization is performed using the L-BFGS algorithm.
#'
#' The logistic regression model estimates the probability of the binary outcome
#' \eqn{y_i \in \{0, 1\}} given predictors \eqn{x_i}:
#' \deqn{P(y_i = 1 | x_i) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip})}}}
#'
#' The function maximizes the log-likelihood:
#' \deqn{\ell(\beta) = \sum_{i=1}^n [y_i \cdot (\beta^T x_i) - \ln(1 + e^{\beta^T x_i})]}
#'
#' Standard errors are computed from the inverse of the Hessian matrix evaluated
#' at the estimated coefficients. Z-scores and p-values are derived under the
#' assumption of asymptotic normality.
#'
#' @param X_r A numeric matrix or sparse matrix (dgCMatrix) of predictor variables.
#'   Rows represent observations and columns represent features.
#' @param y_r A numeric vector of binary outcome values (0 or 1). Must have the
#'   same number of observations as rows in \code{X_r}.
#' @param maxit Integer. Maximum number of iterations for the optimizer.
#'   Default is 300.
#' @param eps_f Numeric. Convergence tolerance for the function value.
#'   Default is 1e-8.
#' @param eps_g Numeric. Convergence tolerance for the gradient norm.
#'   Default is 1e-5.
#'
#' @return A list containing the results of the logistic regression fit:
#' \describe{
#'   \item{\code{coefficients}}{Numeric vector of estimated regression coefficients.}
#'   \item{\code{se}}{Numeric vector of standard errors for the coefficients.}
#'   \item{\code{z_scores}}{Numeric vector of z-statistics for testing coefficient significance.}
#'   \item{\code{p_values}}{Numeric vector of p-values associated with the z-statistics.}
#'   \item{\code{loglikelihood}}{Scalar. The maximized log-likelihood value.}
#'   \item{\code{gradient}}{Numeric vector. The gradient at the solution.}
#'   \item{\code{hessian}}{Matrix. The Hessian matrix evaluated at the solution.}
#'   \item{\code{convergence}}{Logical. Whether the algorithm converged successfully.}
#'   \item{\code{iterations}}{Integer. Number of iterations performed.}
#'   \item{\code{message}}{Character. Convergence message.}
#' }
#'
#' @note
#' \itemize{
#'   \item An intercept term is not automatically included. Users should add a column
#'         of ones to \code{X_r} if an intercept is desired.
#'   \item If the Hessian matrix is singular (determinant is zero), standard errors,
#'         z-scores, and p-values will be returned as \code{NA}.
#'   \item The function uses the L-BFGS quasi-Newton optimization method.
#' }
#'
#' @examples
#' # Generate sample data
#' set.seed(123)
#' n <- 100
#' p <- 3
#' X <- matrix(rnorm(n * p), n, p)
#' # Add intercept column
#' X <- cbind(1, X)
#' colnames(X) <- c("(Intercept)", "X1", "X2", "X3")
#'
#' # True coefficients
#' beta_true <- c(0.5, 1.2, -0.8, 0.3)
#'
#' # Generate linear predictor
#' eta <- X %*% beta_true
#'
#' # Generate binary outcome
#' prob <- 1 / (1 + exp(-eta))
#' y <- rbinom(n, 1, prob)
#'
#' # Fit logistic regression
#' result <- fit_logistic_regression(X, y)
#'
#' # View coefficients and statistics
#' print(data.frame(
#'   Coefficient = result$coefficients,
#'   Std_Error = result$se,
#'   Z_score = result$z_scores,
#'   P_value = result$p_values
#' ))
#'
#' # Check convergence
#' cat("Converged:", result$convergence, "\n")
#' cat("Log-Likelihood:", result$loglikelihood, "\n")
#'
#' @export
fit_logistic_regression <- function(X_r, y_r, maxit = 300L, eps_f = 1e-8, eps_g = 1e-5) {
  # Input validation
  if (!is.matrix(X_r) && !inherits(X_r, "dgCMatrix")) {
    stop("X_r must be a matrix or a dgCMatrix (sparse matrix).")
  }

  if (!is.numeric(y_r)) {
    stop("y_r must be a numeric vector.")
  }

  # Check dimensions
  n_obs_X <- if (inherits(X_r, "dgCMatrix")) nrow(X_r) else nrow(X_r)
  if (length(y_r) != n_obs_X) {
    stop("Number of rows in X_r must match the length of y_r.")
  }

  # Check binary outcome
  if (!all(y_r %in% c(0, 1))) {
    stop("y_r must contain only 0 and 1 values.")
  }

  # Ensure integer types for parameters
  maxit <- as.integer(maxit)

  # Call C++ function
  .Call("_OptimalBinningWoE_fit_logistic_regression",
    X_r = X_r,
    y_r = y_r,
    maxit = maxit,
    eps_f = eps_f,
    eps_g = eps_g,
    PACKAGE = "OptimalBinningWoE"
  )
}
