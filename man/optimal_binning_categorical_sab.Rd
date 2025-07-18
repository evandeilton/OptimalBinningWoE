% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{optimal_binning_categorical_sab}
\alias{optimal_binning_categorical_sab}
\title{Optimal Binning for Categorical Variables using Simulated Annealing}
\usage{
optimal_binning_categorical_sab(
  target,
  feature,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L,
  bin_separator = "\%;\%",
  initial_temperature = 1,
  cooling_rate = 0.995,
  max_iterations = 1000L,
  convergence_threshold = 1e-06,
  adaptive_cooling = TRUE
)
}
\arguments{
\item{target}{An integer vector of binary target values (0 or 1).}

\item{feature}{A character vector of categorical feature values.}

\item{min_bins}{Minimum number of bins (default: 3).}

\item{max_bins}{Maximum number of bins (default: 5).}

\item{bin_cutoff}{Minimum proportion of observations in a bin (default: 0.05).}

\item{max_n_prebins}{Maximum number of pre-bins (default: 20).}

\item{bin_separator}{Separator string for merging categories (default: "\%;\%").}

\item{initial_temperature}{Initial temperature for Simulated Annealing (default: 1.0).}

\item{cooling_rate}{Cooling rate for Simulated Annealing (default: 0.995).}

\item{max_iterations}{Maximum number of iterations for Simulated Annealing (default: 1000).}

\item{convergence_threshold}{Threshold for convergence (default: 1e-6).}

\item{adaptive_cooling}{Whether to use adaptive cooling schedule (default: TRUE).}
}
\value{
A list containing the following elements:
\itemize{
\item id: Numeric vector of bin identifiers.
\item bin: Character vector of bin names.
\item woe: Numeric vector of Weight of Evidence (WoE) values for each bin.
\item iv: Numeric vector of Information Value (IV) for each bin.
\item count: Integer vector of total counts for each bin.
\item count_pos: Integer vector of positive counts for each bin.
\item count_neg: Integer vector of negative counts for each bin.
\item total_iv: Total Information Value of the binning.
\item converged: Logical value indicating whether the algorithm converged.
\item iterations: Integer value indicating the number of iterations run.
}
}
\description{
Performs optimal binning for categorical variables using an enhanced Simulated Annealing
approach. This implementation maximizes Information Value (IV) while maintaining monotonicity
in the bins, using Bayesian smoothing for robust estimation and adaptive temperature scheduling
for better convergence.
}
\details{
This enhanced version of the Simulated Annealing Binning (SAB) algorithm implements several
key improvements over traditional approaches:

\strong{Mathematical Framework:}

The Weight of Evidence (WoE) with Bayesian smoothing is calculated as:

\deqn{WoE_i = \ln\left(\frac{p_i^*}{q_i^*}\right)}

where:
\itemize{
\item \eqn{p_i^* = \frac{n_i^+ + \alpha \cdot \pi}{N^+ + \alpha}} is the smoothed proportion of
events in bin i
\item \eqn{q_i^* = \frac{n_i^- + \alpha \cdot (1-\pi)}{N^- + \alpha}} is the smoothed proportion of
non-events in bin i
\item \eqn{\pi = \frac{N^+}{N^+ + N^-}} is the overall event rate
\item \eqn{\alpha} is the prior strength parameter (default: 0.5)
\item \eqn{n_i^+} is the count of events in bin i
\item \eqn{n_i^-} is the count of non-events in bin i
\item \eqn{N^+} is the total number of events
\item \eqn{N^-} is the total number of non-events
}

The Information Value (IV) for each bin is calculated as:

\deqn{IV_i = (p_i^* - q_i^*) \times WoE_i}

\strong{Simulated Annealing:}

The algorithm uses an enhanced version of Simulated Annealing with these key features:
\itemize{
\item Multiple neighborhood generation strategies for better exploration
\item Adaptive temperature scheduling to escape local optima
\item Periodic restarting from the best known solution
\item Smart initialization using event rates for better starting points
}

The probability of accepting a worse solution is calculated as:

\deqn{P(accept) = \exp\left(\frac{\Delta IV}{T}\right)}

where \eqn{\Delta IV} is the change in Information Value and \eqn{T} is the current temperature.

\strong{Algorithm Phases:}
\enumerate{
\item \strong{Initialization:} Create initial bin assignments using a kmeans-like strategy based on event rates
\item \strong{Optimization:} Apply Simulated Annealing to find the optimal assignment of categories to bins
\item \strong{Monotonicity Enforcement:} Ensure the final solution has monotonic bin event rates
}

\strong{Key Features:}
\itemize{
\item Bayesian smoothing for robust estimation with small samples
\item Multiple neighbor generation strategies for better search space exploration
\item Adaptive temperature scheduling to escape local optima
\item Smart initialization for better starting points
\item Strong monotonicity enforcement
\item Comprehensive handling of edge cases
}
}
\examples{
\dontrun{
# Basic usage
set.seed(123)
target <- sample(0:1, 1000, replace = TRUE)
feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
result <- optimal_binning_categorical_sab(target, feature)
print(result)

# Adjust simulated annealing parameters
result2 <- optimal_binning_categorical_sab(
  target, feature,
  min_bins = 2,
  max_bins = 4,
  initial_temperature = 2.0,
  cooling_rate = 0.99,
  max_iterations = 2000
)
}

}
\references{
\itemize{
\item Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. science, 220(4598), 671-680.
\item Belotti, T., Crook, J. (2009). Credit Scoring with Macroeconomic Variables Using Survival Analysis. Journal of the Operational Research Society, 60(12), 1699-1707.
\item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. arXiv preprint arXiv:1711.05095.
\item Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y. S. (2008). A weakly informative default prior distribution for logistic and other regression models. The annals of applied statistics, 2(4), 1360-1383.
\item Navas-Palencia, G. (2020). Optimal binning: mathematical programming formulations for binary classification. arXiv preprint arXiv:2001.08025.
}
}
