% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{optimal_binning_numerical_ldb}
\alias{optimal_binning_numerical_ldb}
\title{Optimal Binning for Numerical Variables using Local Density Binning (LDB)}
\usage{
optimal_binning_numerical_ldb(
  target,
  feature,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L,
  convergence_threshold = 1e-06,
  max_iterations = 1000L
)
}
\arguments{
\item{target}{An integer binary vector (0 or 1) representing the response variable.}

\item{feature}{A numeric vector representing the feature to be binned.}

\item{min_bins}{Minimum number of bins to be created (default: 3).}

\item{max_bins}{Maximum number of bins allowed (default: 5).}

\item{bin_cutoff}{Minimum frequency proportion for retaining a bin (default: 0.05).}

\item{max_n_prebins}{Maximum number of pre-bins before optimization (default: 20).}

\item{convergence_threshold}{Convergence threshold for IV optimization (default: 1e-6).}

\item{max_iterations}{Maximum number of iterations allowed for optimization (default: 1000).}
}
\value{
A list containing the following elements:
\itemize{
\item \code{bins}: A vector of bin intervals in the format "[lower;upper)".
\item \code{woe}: A numeric vector of WoE values for each bin.
\item \code{iv}: A numeric vector of IV contributions for each bin.
\item \code{count}: An integer vector of the total number of observations per bin.
\item \code{count_pos}: An integer vector of the number of positive cases per bin.
\item \code{count_neg}: An integer vector of the number of negative cases per bin.
\item \code{cutpoints}: A numeric vector of the cutpoints defining the bin edges.
\item \code{converged}: A boolean indicating whether the algorithm converged.
\item \code{iterations}: An integer indicating the number of iterations executed.
}
}
\description{
Implements the Local Density Binning (LDB) algorithm for optimal binning of numerical variables.
The method adjusts binning to maximize predictive power while maintaining monotonicity in Weight of Evidence (WoE),
handling rare bins, and ensuring numerical stability.
}
\details{
\subsection{Key Features:}{
\itemize{
\item \strong{Weight of Evidence (WoE)}: Ensures interpretability by calculating the WoE for each bin, useful for logistic regression and risk models.
\item \strong{Information Value (IV)}: Evaluates the predictive power of the binned feature.
\item \strong{Monotonicity}: Ensures WoE values are either strictly increasing or decreasing across bins.
\item \strong{Rare Bin Handling}: Merges bins with low frequencies to maintain statistical reliability.
\item \strong{Numerical Stability}: Prevents log(0) issues through smoothing (Laplace adjustment).
\item \strong{Dynamic Adjustments}: Supports constraints on minimum and maximum bins, convergence thresholds, and iteration limits.
}
}

\subsection{Mathematical Framework:}{
\itemize{
\item \strong{Weight of Evidence (WoE)}: For a bin \( i \):
\deqn{WoE_i = \ln\left(\frac{\text{Distribution of positives}_i}{\text{Distribution of negatives}_i}\right)}
\item \strong{Information Value (IV)}: Aggregates predictive power across all bins:
\deqn{IV = \sum_{i=1}^{N} (\text{Distribution of positives}_i - \text{Distribution of negatives}_i) \times WoE_i}
}
}

\subsection{Algorithm Steps:}{
\enumerate{
\item \strong{Input Validation}: Ensures the feature and target vectors are valid and properly formatted.
\item \strong{Pre-Binning}: Divides the feature into pre-bins based on quantile cuts or unique values.
\item \strong{Rare Bin Merging}: Combines bins with frequencies below \code{bin_cutoff} to maintain statistical stability.
\item \strong{WoE and IV Calculation}: Computes the WoE and IV values for each bin based on the target distribution.
\item \strong{Monotonicity Enforcement}: Adjusts bins to ensure WoE values are monotonic (either increasing or decreasing).
\item \strong{Bin Optimization}: Iteratively merges bins to respect constraints on \code{min_bins} and \code{max_bins}.
\item \strong{Result Validation}: Ensures bins cover the entire range of the feature without overlap and adhere to constraints.
}
}

\subsection{Parameters:}{
\itemize{
\item \code{min_bins}: Minimum number of bins to be created (default: 3).
\item \code{max_bins}: Maximum number of bins allowed (default: 5).
\item \code{bin_cutoff}: Minimum proportion of total observations required for a bin to be retained as standalone (default: 0.05).
\item \code{max_n_prebins}: Maximum number of pre-bins before optimization (default: 20).
\item \code{convergence_threshold}: Threshold for determining convergence in terms of IV changes (default: 1e-6).
\item \code{max_iterations}: Maximum number of iterations allowed for optimization (default: 1000).
}
}
}
\examples{
\dontrun{
set.seed(123)
target <- sample(0:1, 1000, replace = TRUE)
feature <- rnorm(1000)
result <- optimal_binning_numerical_ldb(target, feature, min_bins = 3, max_bins = 6)
print(result$bins)
print(result$woe)
print(result$iv)
}

}
