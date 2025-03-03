% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{optimal_binning_categorical_sblp}
\alias{optimal_binning_categorical_sblp}
\title{Optimal Binning for Categorical Variables using Similarity-Based Logistic Partitioning (SBLP)}
\usage{
optimal_binning_categorical_sblp(
  target,
  feature,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L,
  convergence_threshold = 1e-06,
  max_iterations = 1000L,
  bin_separator = ";"
)
}
\arguments{
\item{target}{Integer binary vector (0 or 1) representing the response variable.}

\item{feature}{Character vector with the categories of the explanatory variable.}

\item{min_bins}{Minimum number of bins (default: 3).}

\item{max_bins}{Maximum number of bins (default: 5).}

\item{bin_cutoff}{Minimum frequency proportion for a category to be considered as a separate bin (default: 0.05).}

\item{max_n_prebins}{Maximum number of pre-bins before the partitioning process (default: 20).}

\item{convergence_threshold}{Threshold for algorithm convergence (default: 1e-6).}

\item{max_iterations}{Maximum number of iterations of the algorithm (default: 1000).}

\item{bin_separator}{Separator used to concatenate category names within bins (default: ";").}
}
\value{
A list containing:
\itemize{
\item bin: String vector with the names of the bins (concatenated categories).
\item woe: Numeric vector with the Weight of Evidence (WoE) values for each bin.
\item iv: Numeric vector with the Information Value (IV) values for each bin.
\item count: Integer vector with the total count of observations in each bin.
\item count_pos: Integer vector with the count of positive cases (target=1) in each bin.
\item count_neg: Integer vector with the count of negative cases (target=0) in each bin.
\item converged: Logical value indicating whether the algorithm converged.
\item iterations: Integer value indicating the number of iterations executed.
}
}
\description{
This function performs optimal binning for categorical variables using a Similarity-Based Logistic Partitioning (SBLP) approach.
The goal is to produce bins that maximize the Information Value (IV) and provide consistent Weight of Evidence (WoE), considering target rates
and ensuring quality through similarity-based merges.
The implementation has been revised to improve readability, efficiency, robustness, and to maintain compatibility
with the names and types of input/output parameters.
}
\details{
Steps of the SBLP algorithm:
\enumerate{
\item Validate input and calculate initial counts by category.
\item Handle rare categories by merging them with other similar ones in terms of target rate.
\item Ensure the maximum number of pre-bins by merging uninformative bins.
\item Sort categories by target rate.
\item Apply dynamic programming to determine the optimal partition, considering min_bins and max_bins.
\item Adjust WoE monotonicity, if necessary, provided the number of bins is greater than min_bins.
\item Perform final calculation of WoE and IV for each bin and return the result.
}

Key formulas:
\deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
\deqn{IV = \sum_{bins} (P(X|Y=1) - P(X|Y=0)) \times WoE}
}
\examples{
\dontrun{
set.seed(123)
target <- sample(0:1, 1000, replace = TRUE)
feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
result <- optimal_binning_categorical_sblp(target, feature)
print(result)
}

}
