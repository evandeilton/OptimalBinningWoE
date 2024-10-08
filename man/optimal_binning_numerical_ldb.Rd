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
  max_n_prebins = 20L
)
}
\arguments{
\item{target}{An integer vector of binary target values (0 or 1).}

\item{feature}{A numeric vector of feature values to be binned.}

\item{min_bins}{Minimum number of bins (default: 3).}

\item{max_bins}{Maximum number of bins (default: 5).}

\item{bin_cutoff}{Minimum frequency for a bin (default: 0.05).}

\item{max_n_prebins}{Maximum number of pre-bins (default: 20).}
}
\value{
A list containing three elements:
\item{woefeature}{A numeric vector of Weight of Evidence (WoE) transformed feature values.}
\item{woebin}{A data frame containing bin information, including bin labels, WoE, Information Value (IV), and counts.}
\item{iv_total}{The total Information Value of the binned feature.}
}
\description{
This function implements the Local Density Binning (LDB) algorithm for optimal binning of numerical variables.
}
\details{
The Local Density Binning (LDB) algorithm is an advanced method for optimal binning of numerical variables. It aims to create bins that maximize the predictive power of the feature while maintaining monotonicity in the Weight of Evidence (WoE) values and respecting user-defined constraints.

The algorithm works through several steps:
\enumerate{
\item Pre-binning: Initially divides the feature into a large number of bins (max_n_prebins) using quantiles.
\item WoE and IV Calculation: For each bin, computes the Weight of Evidence (WoE) and Information Value (IV):
\deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right) = \ln\left(\frac{n_{1i}/N_1}{n_{0i}/N_0}\right)}
\deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
where \eqn{n_{1i}} and \eqn{n_{0i}} are the number of events and non-events in bin i, and \eqn{N_1} and \eqn{N_0} are the total number of events and non-events.
\item Monotonicity Enforcement: Merges adjacent bins to ensure monotonic WoE values. The direction of monotonicity is determined by the overall trend of WoE values across bins.
\item Bin Merging: Merges bins with frequencies below the bin_cutoff threshold and ensures the number of bins is within the specified range (min_bins to max_bins).
}

The LDB method incorporates local density estimation to better capture the underlying distribution of the data. This approach can be particularly effective when dealing with complex, non-linear relationships between the feature and the target variable.

The algorithm uses Information Value (IV) as a criterion for merging bins, aiming to minimize IV loss at each step. This approach helps preserve the predictive power of the feature while creating optimal bins.

The total Information Value (IV) is calculated as the sum of IVs for all bins:
\deqn{IV_{total} = \sum_{i=1}^{n} IV_i}

The LDB method provides a balance between predictive power and model interpretability, allowing users to control the trade-off through parameters such as min_bins, max_bins, and bin_cutoff.
}
\examples{
\dontrun{
# Create sample data
set.seed(123)
target <- sample(0:1, 1000, replace = TRUE)
feature <- rnorm(1000)

# Run optimal binning
result <- optimal_binning_numerical_ldb(target, feature)

# View results
head(result$woefeature)
print(result$woebin)
print(result$iv_total)
}

}
\references{
\itemize{
\item Belotti, P., Bonami, P., Fischetti, M., Lodi, A., Monaci, M., Nogales-Gomez, A., & Salvagnin, D. (2016). On handling indicator constraints in mixed integer programming. Computational Optimization and Applications, 65(3), 545-566.
\item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and Its Applications. SIAM Monographs on Mathematical Modeling and Computation.
}
}
\author{
Lopes,
}
