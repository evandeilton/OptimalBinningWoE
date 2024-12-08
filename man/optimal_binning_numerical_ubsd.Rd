% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{optimal_binning_numerical_ubsd}
\alias{optimal_binning_numerical_ubsd}
\title{Optimal Binning for Numerical Variables using Unsupervised Binning with Standard Deviation}
\usage{
optimal_binning_numerical_ubsd(
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
\item{target}{A numeric vector of binary target values (should contain exactly two unique values: 0 and 1).}

\item{feature}{A numeric vector of feature values to be binned.}

\item{min_bins}{Minimum number of bins (default: 3).}

\item{max_bins}{Maximum number of bins (default: 5).}

\item{bin_cutoff}{Minimum frequency of observations in each bin (default: 0.05).}

\item{max_n_prebins}{Maximum number of pre-bins for initial standard deviation-based discretization (default: 20).}

\item{convergence_threshold}{Threshold for convergence of the total IV (default: 1e-6).}

\item{max_iterations}{Maximum number of iterations for the algorithm (default: 1000).}
}
\value{
A list containing the following elements:
\item{bins}{A character vector of bin names.}
\item{woe}{A numeric vector of Weight of Evidence values for each bin.}
\item{iv}{A numeric vector of Information Value for each bin.}
\item{count}{An integer vector of the total count of observations in each bin.}
\item{count_pos}{An integer vector of the count of positive observations in each bin.}
\item{count_neg}{An integer vector of the count of negative observations in each bin.}
\item{cutpoints}{A numeric vector of cut points used to generate the bins.}
\item{converged}{A logical value indicating whether the algorithm converged.}
\item{iterations}{An integer value indicating the number of iterations run.}
}
\description{
This function implements an optimal binning algorithm for numerical variables using an
Unsupervised Binning approach based on Standard Deviation (UBSD) with Weight of Evidence (WoE)
and Information Value (IV) criteria.
}
\details{
The optimal binning algorithm for numerical variables uses an Unsupervised Binning approach
based on Standard Deviation (UBSD) with Weight of Evidence (WoE) and Information Value (IV)
to create bins that maximize the predictive power of the feature while maintaining interpretability.

The algorithm follows these steps:
\enumerate{
\item Initial binning based on standard deviations around the mean
\item Assignment of data points to bins
\item Merging of rare bins based on the bin_cutoff parameter
\item Calculation of WoE and IV for each bin
\item Enforcement of monotonicity in WoE across bins
\item Further merging of bins to ensure the number of bins is within the specified range
}

The algorithm iterates until convergence is reached or the maximum number of iterations is hit.
}
\examples{
\dontrun{
# Generate sample data
set.seed(123)
n <- 10000
feature <- rnorm(n)
target <- rbinom(n, 1, plogis(0.5 * feature))

# Apply optimal binning
result <- optimal_binning_numerical_ubsd(target, feature, min_bins = 3, max_bins = 5)

# View binning results
print(result)
}

}
