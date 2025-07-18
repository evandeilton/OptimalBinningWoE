% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{optimal_binning_numerical_mdlp}
\alias{optimal_binning_numerical_mdlp}
\title{Optimal Binning for Numerical Features using the Minimum Description Length Principle (MDLP)}
\usage{
optimal_binning_numerical_mdlp(
  target,
  feature,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L,
  convergence_threshold = 1e-06,
  max_iterations = 1000L,
  laplace_smoothing = 0.5
)
}
\arguments{
\item{target}{An integer binary vector (0 or 1) representing the target variable.}

\item{feature}{A numeric vector representing the feature to bin.}

\item{min_bins}{Minimum number of bins (default: 3).}

\item{max_bins}{Maximum number of bins (default: 5).}

\item{bin_cutoff}{Minimum proportion of records per bin (default: 0.05).}

\item{max_n_prebins}{Maximum number of pre-bins before merging (default: 20).}

\item{convergence_threshold}{Convergence threshold for IV optimization (default: 1e-6).}

\item{max_iterations}{Maximum number of iterations allowed (default: 1000).}

\item{laplace_smoothing}{Smoothing parameter for WoE calculation (default: 0.5).}
}
\value{
A list with the following components:
\itemize{
\item \code{id}: A numeric vector with bin identifiers (1-based).
\item \code{bin}: A vector of bin names representing the intervals.
\item \code{woe}: A numeric vector with the WoE values for each bin.
\item \code{iv}: A numeric vector with the IV values for each bin.
\item \code{count}: An integer vector with the total number of observations in each bin.
\item \code{count_pos}: An integer vector with the count of positive cases in each bin.
\item \code{count_neg}: An integer vector with the count of negative cases in each bin.
\item \code{cutpoints}: A numeric vector of cut points defining the bins.
\item \code{total_iv}: A numeric value representing the total information value of the binning.
\item \code{converged}: A boolean indicating whether the algorithm converged.
\item \code{iterations}: An integer with the number of iterations performed.
}
}
\description{
This function performs optimal binning for numerical features using the Minimum Description Length Principle (MDLP).
It minimizes information loss by merging adjacent bins that reduce the MDL cost, while ensuring monotonicity in the Weight of Evidence (WoE).
The algorithm adjusts the number of bins between \code{min_bins} and \code{max_bins} and handles rare bins by merging them iteratively.
Designed for robust and numerically stable calculations, it incorporates protections for extreme cases and convergence controls.
}
\details{
\subsection{Core Steps:}{
\enumerate{
\item \strong{Input Validation}: Ensures feature and target are valid, numeric, and binary respectively. Validates consistency between \code{min_bins} and \code{max_bins}.
\item \strong{Pre-Binning}: Creates pre-bins based on equal frequencies or unique values if there are few observations.
\item \strong{MDL-Based Merging}: Iteratively merges bins to minimize the MDL cost, which combines model complexity and data fit quality.
\item \strong{Rare Bin Handling}: Merges bins with frequencies below the \code{bin_cutoff} threshold to ensure statistical stability.
\item \strong{Monotonicity Enforcement}: Adjusts bins to ensure that the WoE values are monotonically increasing or decreasing.
\item \strong{Validation}: Validates the final bin structure for consistency and correctness.
}
}

\subsection{Mathematical Framework:}{

\strong{Entropy Calculation}: For a bin \( i \) with positive (\( p \)) and negative (\( n \)) counts:
\deqn{Entropy = -p \log_2(p) - n \log_2(n)}

\strong{MDL Cost}: Combines the cost of the model and data description:
\deqn{MDL\_Cost = Model\_Cost + Data\_Cost}
Where:
\deqn{Model\_Cost = \log_2(Number\_of\_bins - 1)}
\deqn{Data\_Cost = Total\_Entropy - \sum_{i} Count_i \times Entropy_i}

\strong{Weight of Evidence (WoE)}: For a bin \( i \) with Laplace smoothing parameter:
\deqn{WoE_i = \ln\left(\frac{n_{1i} + a}{n_{1} + ma} \cdot \frac{n_{0} + ma}{n_{0i} + a}\right)}
Where:
\itemize{
\item \eqn{n_{1i}} is the count of positive cases in bin \(i\)
\item \eqn{n_{0i}} is the count of negative cases in bin \(i\)
\item \eqn{n_{1}} is the total count of positive cases
\item \eqn{n_{0}} is the total count of negative cases
\item \eqn{m} is the number of bins
\item a is the Laplace smoothing parameter
}

\strong{Information Value (IV)}: Summarizes predictive power across all bins:
\deqn{IV = \sum_{i} (P(X|Y=1) - P(X|Y=0)) \times WoE_i}
}

\subsection{Features:}{
\itemize{
\item Merges bins iteratively to minimize the MDL cost.
\item Ensures monotonicity of WoE to improve model interpretability.
\item Handles rare bins by merging categories with low frequencies.
\item Stable against edge cases like all identical values or insufficient observations.
\item Efficiently processes large datasets with iterative binning and convergence checks.
\item Applies Laplace smoothing for robust WoE calculation in sparse bins.
}
}
}
\examples{
\dontrun{
# Example usage
set.seed(123)
target <- sample(0:1, 100, replace = TRUE)
feature <- runif(100)
result <- optimal_binning_numerical_mdlp(target, feature, min_bins = 3, max_bins = 5)
print(result)

# With different parameters
result2 <- optimal_binning_numerical_mdlp(
  target, 
  feature, 
  min_bins = 2, 
  max_bins = 10,
  bin_cutoff = 0.03,
  laplace_smoothing = 0.1
)

# Print summary statistics
print(paste("Total Information Value:", round(result2$total_iv, 4)))
print(paste("Number of bins created:", length(result2$bin)))
}

}
\references{
\itemize{
\item Fayyad, U. & Irani, K. (1993). "Multi-interval discretization of continuous-valued
attributes for classification learning." Proceedings of the International Joint
Conference on Artificial Intelligence, 1022-1027.
\item Rissanen, J. (1978). "Modeling by shortest data description." Automatica, 14(5), 465-471.
\item Good, I.J. (1952). "Rational Decisions." Journal of the Royal Statistical Society,
Series B, 14, 107-114. (Origin of Laplace smoothing/additive smoothing)
}
}
