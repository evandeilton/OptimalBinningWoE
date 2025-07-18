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
  max_n_prebins = 20L,
  bin_separator = "\%;\%",
  convergence_threshold = 1e-06,
  max_iterations = 1000L
)
}
\arguments{
\item{target}{Integer vector of binary target values (0 or 1).}

\item{feature}{Character vector of categorical feature values.}

\item{min_bins}{Minimum number of bins (default: 3).}

\item{max_bins}{Maximum number of bins (default: 5).}

\item{bin_cutoff}{Minimum frequency for a separate bin (default: 0.05).}

\item{max_n_prebins}{Maximum number of pre-bins before merging (default: 20).}

\item{bin_separator}{Separator used for merging category names (default: "\%;\%").}

\item{convergence_threshold}{Threshold for convergence (default: 1e-6).}

\item{max_iterations}{Maximum number of iterations (default: 1000).}
}
\value{
A list with the following elements:
\itemize{
\item id: Numeric vector of bin identifiers.
\item bin: Character vector of bin names (merged categories).
\item woe: Numeric vector of Weight of Evidence values for each bin.
\item iv: Numeric vector of Information Value for each bin.
\item count: Integer vector of total count for each bin.
\item count_pos: Integer vector of positive class count for each bin.
\item count_neg: Integer vector of negative class count for each bin.
\item total_iv: Total Information Value of the binning.
\item converged: Logical indicating whether the algorithm converged.
\item iterations: Integer indicating the number of iterations performed.
}
}
\description{
Implements optimal binning for categorical variables using a Greedy Merge approach,
calculating Weight of Evidence (WoE) and Information Value (IV).
}
\details{
The Greedy Merge Binning (GMB) algorithm finds an optimal binning solution by iteratively
merging adjacent bins to maximize Information Value (IV) while respecting constraints
on the number of bins.

The Weight of Evidence (WoE) measures the predictive power of a bin and is defined as:

\deqn{WoE_i = \ln\left(\frac{n^+_i/N^+}{n^-_i/N^-}\right)}

where:
\itemize{
\item \eqn{n^+_i} is the number of positive cases in bin i
\item \eqn{n^-_i} is the number of negative cases in bin i
\item \eqn{N^+} is the total number of positive cases
\item \eqn{N^-} is the total number of negative cases
}

The Information Value (IV) quantifies the predictive power of the entire binning and is:

\deqn{IV = \sum_{i=1}^{n} (p_i - q_i) \times WoE_i}

where:
\itemize{
\item \eqn{p_i = n^+_i/N^+} is the proportion of positive cases in bin i
\item \eqn{q_i = n^-_i/N^-} is the proportion of negative cases in bin i
}

This algorithm applies Bayesian smoothing to WoE calculations to improve stability, particularly
with small sample sizes or rare categories. The smoothing applies pseudo-counts based on the
overall population prevalence.

The algorithm includes the following main steps:
\enumerate{
\item Initialize bins with each unique category.
\item Merge rare categories based on the bin_cutoff.
\item Iteratively merge adjacent bins that result in the highest IV.
\item Stop merging when the number of bins reaches min_bins or max_bins.
\item Ensure monotonicity of WoE values across bins.
\item Calculate final WoE and IV for each bin.
}

Edge cases are handled as follows:
\itemize{
\item Empty strings in feature are rejected during input validation
\item Extremely imbalanced datasets (< 5 samples in either class) produce a warning
\item When merging bins, ties in IV improvement are resolved by preferring more balanced bins
\item Monotonicity violations are addressed with an adaptive threshold based on average WoE gaps
}
}
\examples{
\dontrun{
# Example data
target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")

# Run optimal binning
result <- optimal_binning_categorical_gmb(target, feature, min_bins = 2, max_bins = 4)

# View results
print(result)
}

}
\references{
\itemize{
\item Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. Risks, 9(3), 58.
\item Siddiqi, N. (2006). Credit risk scorecards: developing and implementing intelligent credit scoring (Vol. 3). John Wiley & Sons.
\item García-Magariño, I., Medrano, C., Lombas, A. S., & Barrasa, A. (2019). A hybrid approach with agent-based simulation and clustering for sociograms. Information Sciences, 499, 47-61.
\item Navas-Palencia, G. (2020). Optimal binning: mathematical programming formulations for binary classification. arXiv preprint arXiv:2001.08025.
\item Lin, X., Wang, G., & Zhang, T. (2022). Efficient monotonic binning for predictive modeling in high-dimensional spaces. Knowledge-Based Systems, 235, 107629.
\item Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y. S. (2008). A weakly informative default prior distribution for logistic and other regression models. The annals of applied statistics, 2(4), 1360-1383.
}
}
\author{
Lopes, J. E.
}
