% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{optimal_binning_numerical_bb}
\alias{optimal_binning_numerical_bb}
\title{Optimal Binning for Numerical Variables using Branch and Bound Algorithm}
\usage{
optimal_binning_numerical_bb(
  target,
  feature,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L,
  is_monotonic = TRUE,
  convergence_threshold = 1e-06,
  max_iterations = 1000L
)
}
\arguments{
\item{target}{An integer binary vector (0 or 1) representing the target variable.}

\item{feature}{A numeric vector of feature values to be binned.}

\item{min_bins}{Minimum number of bins to generate (default: 3).}

\item{max_bins}{Maximum number of bins to generate (default: 5).}

\item{bin_cutoff}{Minimum frequency fraction for each bin (default: 0.05).}

\item{max_n_prebins}{Maximum number of pre-bins generated before optimization (default: 20).}

\item{is_monotonic}{Logical value indicating whether to enforce monotonicity in WoE (default: TRUE).}

\item{convergence_threshold}{Convergence threshold for total Information Value (IV) change (default: 1e-6).}

\item{max_iterations}{Maximum number of iterations allowed for the optimization process (default: 1000).}
}
\value{
A list containing:
\item{id}{Numeric identifiers for each bin (1-based).}
\item{bin}{Character vector with the intervals of each bin (e.g., \verb{(-Inf; 0]}, \verb{(0; +Inf)}).}
\item{woe}{Numeric vector with the Weight of Evidence values for each bin.}
\item{iv}{Numeric vector with the Information Value contribution for each bin.}
\item{count}{Integer vector with the total number of observations in each bin.}
\item{count_pos}{Integer vector with the number of positive observations in each bin.}
\item{count_neg}{Integer vector with the number of negative observations in each bin.}
\item{cutpoints}{Numeric vector of cut points between bins (excluding infinity).}
\item{converged}{Logical value indicating whether the algorithm converged.}
\item{iterations}{Number of iterations executed by the optimization algorithm.}
\item{total_iv}{The total Information Value of the binning solution.}
}
\description{
Performs optimal binning for numerical variables using a Branch and Bound approach.
This method transforms continuous features into discrete bins by maximizing the statistical
relationship with a binary target variable while maintaining interpretability constraints.
The algorithm optimizes Weight of Evidence (WoE) and Information Value (IV) metrics
commonly used in risk modeling, credit scoring, and statistical analysis.
}
\details{
\subsection{Algorithm Overview}{

The implementation follows a five-phase approach:
\enumerate{
\item \strong{Input Validation}: Ensures data integrity and parameter validity.
\item \strong{Pre-Binning}:
\itemize{
\item Creates initial bins using quantile-based division
\item Handles special cases for limited unique values
\item Uses binary search for efficient observation assignment
}
\item \strong{Statistical Stabilization}:
\itemize{
\item Merges bins with frequencies below the specified threshold
\item Ensures each bin has sufficient observations for reliable statistics
}
\item \strong{Monotonicity Enforcement} (optional):
\itemize{
\item Ensures WoE values follow a consistent trend (increasing or decreasing)
\item Improves interpretability and aligns with business expectations
\item Selects optimal monotonicity direction based on IV preservation
}
\item \strong{Branch and Bound Optimization}:
\itemize{
\item Iteratively merges bins with minimal IV contribution
\item Continues until reaching the target number of bins or convergence
\item Preserves predictive power while reducing complexity
}
}
}

\subsection{Mathematical Foundation}{

The algorithm optimizes two key metrics:
\enumerate{
\item \strong{Weight of Evidence (WoE)} for bin \eqn{i}:
\deqn{WoE_i = \ln\left(\frac{p_i/P}{n_i/N}\right)}

Where:
\itemize{
\item \eqn{p_i}: Number of positive cases in bin \eqn{i}
\item \eqn{P}: Total number of positive cases
\item \eqn{n_i}: Number of negative cases in bin \eqn{i}
\item \eqn{N}: Total number of negative cases
}
\item \strong{Information Value (IV)} for bin \eqn{i}:
\deqn{IV_i = \left(\frac{p_i}{P} - \frac{n_i}{N}\right) \times WoE_i}

The total Information Value is the sum across all bins:
\deqn{IV_{total} = \sum_{i=1}^{k} IV_i}
\item \strong{Smoothing}:
The implementation uses Laplace smoothing to handle zero counts:
\deqn{\frac{p_i + \alpha}{P + k\alpha}, \frac{n_i + \alpha}{N + k\alpha}}

Where:
\itemize{
\item \eqn{\alpha}: Small constant (0.5 in this implementation)
\item \eqn{k}: Number of bins
}
}
}

\subsection{Branch and Bound Strategy}{

The core optimization uses a greedy iterative approach:
\enumerate{
\item Start with more bins than needed (from pre-binning)
\item Identify the bin with the smallest IV contribution
\item Merge this bin with an adjacent bin
\item Recompute WoE and IV values
\item If monotonicity is required, enforce it
\item Repeat until target number of bins is reached or convergence
}

This approach minimizes information loss while reducing model complexity.
}
}
\examples{
\dontrun{
# Generate synthetic data
set.seed(123)
n <- 10000
feature <- rnorm(n)
# Create target with logistic relationship
target <- rbinom(n, 1, plogis(0.5 * feature))

# Apply optimal binning
result <- optimal_binning_numerical_bb(target, feature, min_bins = 3, max_bins = 5)
print(result)

# Access specific components
bins <- result$bin
woe_values <- result$woe
total_iv <- result$total_iv

# Example with custom parameters
result2 <- optimal_binning_numerical_bb(
  target = target,
  feature = feature,
  min_bins = 2,
  max_bins = 8,
  bin_cutoff = 0.02,
  is_monotonic = TRUE
)
}

}
\references{
Belson, W. A. (1959). Matching and prediction on the principle of biological classification.
\emph{Journal of the Royal Statistical Society: Series C (Applied Statistics)}, 8(2), 65-75.

Siddiqi, N. (2006). \emph{Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring}.
John Wiley & Sons.

Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). \emph{Credit Scoring and Its Applications}.
Society for Industrial and Applied Mathematics.

Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization Techniques: A Recent Survey.
\emph{GESTS International Transactions on Computer Science and Engineering}, 32(1), 47-58.

Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and Unsupervised Discretization of
Continuous Features. \emph{Proceedings of the Twelfth International Conference on Machine Learning}, 194-202.

Bertsimas, D., & Dunn, J. (2017). Optimal classification trees. \emph{Machine Learning}, 106(7), 1039-1082.
}
