% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{optimal_binning_categorical_fetb}
\alias{optimal_binning_categorical_fetb}
\title{Categorical Optimal Binning with Fisher's Exact Test}
\usage{
optimal_binning_categorical_fetb(
  target,
  feature,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L,
  convergence_threshold = 1e-06,
  max_iterations = 1000L,
  bin_separator = "\%;\%"
)
}
\arguments{
\item{target}{Integer vector of binary target values (0 or 1).}

\item{feature}{Character vector of categorical feature values.}

\item{min_bins}{Minimum number of bins (default: 3).}

\item{max_bins}{Maximum number of bins (default: 5).}

\item{bin_cutoff}{Minimum frequency for a separate bin (default: 0.05).}

\item{max_n_prebins}{Maximum number of pre-bins before merging (default: 20).}

\item{convergence_threshold}{Threshold for convergence (default: 1e-6).}

\item{max_iterations}{Maximum number of iterations (default: 1000).}

\item{bin_separator}{Separator for bin labels (default: "\%;\%").}
}
\value{
A list containing:
\itemize{
\item bin: Character vector of bin labels (merged categories).
\item woe: Numeric vector of Weight of Evidence values for each bin.
\item iv: Numeric vector of Information Value for each bin.
\item count: Integer vector of total count in each bin.
\item count_pos: Integer vector of positive class count in each bin.
\item count_neg: Integer vector of negative class count in each bin.
\item converged: Logical indicating whether the algorithm converged.
\item iterations: Integer indicating the number of iterations performed.
}
}
\description{
Implements optimal binning for categorical variables using Fisher's Exact Test,
calculating Weight of Evidence (WoE) and Information Value (IV).
}
\details{
The algorithm uses Fisher's Exact Test to iteratively merge bins, maximizing
the statistical significance of the difference between adjacent bins. It ensures
monotonicity in the resulting bins and respects the minimum number of bins specified.
}
\examples{
\dontrun{
target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
result <- optimal_binning_categorical_fetb(target, feature, min_bins = 2,
max_bins = 4, bin_separator = "|")
print(result)
}

}
