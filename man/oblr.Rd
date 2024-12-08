% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/oblr.R
\name{oblr}
\alias{oblr}
\title{Optimized Logistic Regression}
\usage{
oblr(formula, data, max_iter = 1000, tol = 1e-06)
}
\arguments{
\item{formula}{An object of class \code{formula} describing the model to be fitted.}

\item{data}{A data frame or data.table containing the model data.}

\item{max_iter}{Maximum number of iterations for the optimization algorithm. Default is 1000.}

\item{tol}{Convergence tolerance for the optimization algorithm. Default is 1e-6.}
}
\value{
An object of class \code{oblr} containing the results of the logistic regression fit, including:
\describe{
\item{coefficients}{Vector of estimated coefficients.}
\item{se}{Standard errors of the coefficients.}
\item{z_scores}{Z-statistics for the coefficients.}
\item{p_values}{P-values for the coefficients.}
\item{loglikelihood}{Log-likelihood of the model.}
\item{convergence}{Convergence indicator.}
\item{iterations}{Number of iterations performed.}
\item{message}{Convergence message.}
\item{data}{List containing the design matrix X, response y, and the function call.}
}
}
\description{
Fits logistic regression models using an optimized C++ implementation via Rcpp.
}
\details{
The \code{oblr} function fits a logistic regression model using an optimized C++
implementation via Rcpp. This implementation is designed to be efficient, especially
for large or sparse datasets.

The logistic regression model is defined as:

\deqn{P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + ... + \beta_p X_p)}}}

where \eqn{\beta} are the coefficients to be estimated.

The optimization method used is L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno),
a variant of the BFGS method that uses a limited amount of memory. This method is
particularly effective for optimization problems with many variables.

The estimation process involves the following steps:
\enumerate{
\item Data preparation: The design matrix X is created using \code{sparse.model.matrix}
from the Matrix package, which is efficient for sparse data.
\item Optimization: The C++ function \code{fit_logistic_regression} is called to perform
the optimization using L-BFGS.
\item Statistics calculation: Standard errors, z-statistics, and p-values are calculated
using the Hessian matrix returned by the optimization function.
}

Convergence is determined by the relative change in the objective function
(log-likelihood) between successive iterations, compared to the specified tolerance.
}
\examples{
\dontrun{
library(data.table)

# Create example data
set.seed(123)
n <- 10000
X1 <- rnorm(n)
X2 <- rnorm(n)
Y <- rbinom(n, 1, plogis(1 + 0.5 * X1 - 0.5 * X2))
dt <- data.table(Y, X1, X2)

# Fit logistic regression model
model <- oblr(Y ~ X1 + X2, data = dt, max_iter = 1000, tol = 1e-6)

# View results
print(model)
}
}
