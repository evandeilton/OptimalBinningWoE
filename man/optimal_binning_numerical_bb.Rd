% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{optimal_binning_numerical_bb}
\alias{optimal_binning_numerical_bb}
\title{Optimal Binning for Numerical Variables using Branch and Bound}
\usage{
optimal_binning_numerical_bb(
  target,
  feature,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L,
  is_monotonic = TRUE
)
}
\arguments{
\item{target}{An integer vector of binary target values (0 or 1).}

\item{feature}{A numeric vector of feature values to be binned.}

\item{min_bins}{Minimum number of bins (default: 3).}

\item{max_bins}{Maximum number of bins (default: 5).}

\item{bin_cutoff}{Minimum frequency of observations in each bin (default: 0.05).}

\item{max_n_prebins}{Maximum number of pre-bins for initial quantile-based discretization (default: 20).}

\item{is_monotonic}{Boolean indicating whether to enforce monotonicity of WoE across bins (default: TRUE).}
}
\value{
A list containing two elements:
\item{woefeature}{A numeric vector of WoE-transformed feature values.}
\item{woebin}{A data frame with binning details, including bin boundaries, WoE, IV, and count statistics.}
}
\description{
This function implements an optimal binning algorithm for numerical variables using a Branch and Bound approach with Weight of Evidence (WoE) and Information Value (IV) criteria.
}
\details{
The optimal binning algorithm for numerical variables uses a Branch and Bound approach with Weight of Evidence (WoE) and Information Value (IV) to create bins that maximize the predictive power of the feature while maintaining interpretability.

The algorithm follows these steps:
\enumerate{
\item Initial discretization using quantile-based binning
\item Merging of rare bins
\item Calculation of WoE and IV for each bin
\item Enforcing monotonicity of WoE across bins (if is_monotonic is TRUE)
\item Adjusting the number of bins to be within the specified range using a Branch and Bound approach
}

Weight of Evidence (WoE) is calculated for each bin as:

\deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}

where \eqn{P(X_i|Y=1)} is the proportion of positive cases in bin i, and \eqn{P(X_i|Y=0)} is the proportion of negative cases in bin i.

Information Value (IV) for each bin is calculated as:

\deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}

The total IV for the feature is the sum of IVs across all bins:

\deqn{IV_{total} = \sum_{i=1}^{n} IV_i}

The Branch and Bound approach iteratively merges bins with the lowest IV contribution while respecting the constraints on the number of bins and minimum bin frequency. This process ensures that the resulting binning maximizes the total IV while maintaining the desired number of bins.
}
\examples{
\dontrun{
# Generate sample data
set.seed(123)
n <- 10000
feature <- rnorm(n)
target <- rbinom(n, 1, plogis(0.5 * feature))

# Apply optimal binning
result <- optimal_binning_numerical_bb(target, feature, min_bins = 3, max_bins = 5)

# View binning results
print(result$woebin)

# Plot WoE transformation
plot(feature, result$woefeature, main = "WoE Transformation",
xlab = "Original Feature", ylab = "WoE")
}

}
\references{
\itemize{
\item Farooq, B., & Miller, E. J. (2015). Optimal Binning for Continuous Variables in Credit Scoring. Journal of Risk Model Validation, 9(1), 1-21.
\item Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization Techniques: A Recent Survey. GESTS International Transactions on Computer Science and Engineering, 32(1), 47-58.
}
}
\author{
Lopes, J. E.
}
