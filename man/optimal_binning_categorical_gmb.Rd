% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{optimal_binning_categorical_gmb}
\alias{optimal_binning_categorical_gmb}
\title{Categorical Optimal Binning with Greedy Merge Binning}
\usage{
optimal_binning_categorical_gmb(
  target,
  feature,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L
)
}
\arguments{
\item{target}{Integer vector of binary target values (0 or 1).}

\item{feature}{Character vector of categorical feature values.}

\item{min_bins}{Minimum number of bins (default: 3).}

\item{max_bins}{Maximum number of bins (default: 5).}

\item{bin_cutoff}{Minimum frequency for a separate bin (default: 0.05).}

\item{max_n_prebins}{Maximum number of pre-bins before merging (default: 20).}
}
\value{
A list with two elements:
\itemize{
\item woefeature: Numeric vector of WoE values for each input feature value.
\item woebin: Data frame with binning results (bin names, WoE, IV, counts).
}
}
\description{
Implements optimal binning for categorical variables using a Greedy Merge approach,
calculating Weight of Evidence (WoE) and Information Value (IV).
}
\details{
The algorithm uses a greedy merge approach to find an optimal binning solution.
It starts with each unique category as a separate bin and iteratively merges
bins to maximize the overall Information Value (IV) while respecting the
constraints on the number of bins.

Weight of Evidence (WoE) for each bin is calculated as:

\deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}

Information Value (IV) for each bin is calculated as:

\deqn{IV = (P(X|Y=1) - P(X|Y=0)) \times WoE}

The algorithm includes the following key steps:
\enumerate{
\item Initialize bins with each unique category.
\item Merge rare categories based on bin_cutoff.
\item Iteratively merge adjacent bins that result in the highest IV.
\item Stop merging when the number of bins reaches min_bins or max_bins.
\item Calculate final WoE and IV for each bin.
}

The algorithm handles zero counts by using a small constant (epsilon) to avoid
undefined logarithms and division by zero.
}
\examples{
\dontrun{
# Sample data
target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")

# Run optimal binning
result <- optimal_binning_categorical_gmb(target, feature, min_bins = 2, max_bins = 4)

# View results
print(result$woebin)
print(result$woefeature)
}

}
\references{
\itemize{
\item Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. Risks, 9(3), 58.
\item Siddiqi, N. (2006). Credit risk scorecards: developing and implementing intelligent credit scoring (Vol. 3). John Wiley & Sons.
}
}
\author{
Lopes, J. E.
}
