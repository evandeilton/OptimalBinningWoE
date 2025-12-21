#' Optimal Binning for Numerical Variables using Branch and Bound Algorithm
#'
#' Performs supervised discretization of continuous numerical variables using a
#' Branch and Bound-style approach. This algorithm optimally creates bins based
#' on the relationship with a binary target variable, maximizing Information
#' Value (IV) while optionally enforcing monotonicity in Weight of Evidence (WoE).
#'
#' @param feature A numeric vector representing the continuous predictor variable
#'   to be binned. NA values are handled by exclusion during the pre-binning phase.
#' @param target An integer vector of binary outcomes (0/1) corresponding to
#'   each observation in \code{feature}. Must have the same length as \code{feature}.
#' @param min_bins Integer. The minimum number of bins to produce. Must be \eqn{\ge} 2.
#'   Defaults to 3.
#' @param max_bins Integer. The maximum number of bins to produce. Must be \eqn{\ge}
#'   \code{min_bins}. Defaults to 5.
#' @param bin_cutoff Numeric. The minimum fraction of total observations required
#'   for a bin to be considered valid. Bins with frequency < \code{bin_cutoff}
#'   will be merged with neighbors. Value must be in (0, 1). Defaults to 0.05.
#' @param max_n_prebins Integer. The number of initial quantiles to generate
#'   during the pre-binning phase. Higher values provide more granular starting
#'   points but increase computation time. Must be \eqn{\ge} \code{min_bins}.
#'   Defaults to 20.
#' @param is_monotonic Logical. If \code{TRUE}, the algorithm enforces a strict
#'   monotonic relationship (increasing or decreasing) between the bin indices
#'   and their WoE values. This makes the variable more interpretable for linear
#'   models. Defaults to \code{TRUE}.
#' @param convergence_threshold Numeric. The threshold for the change in total
#'   IV to determine convergence during the iterative merging process.
#'   Defaults to 1e-6.
#' @param max_iterations Integer. Safety limit for the maximum number of merging
#'   iterations. Defaults to 1000.
#'
#' @return A list containing the binning results:
#'   \itemize{
#'     \item \code{id}: Integer vector of bin identifiers (1 to k).
#'     \item \code{bin}: Character vector of bin labels in interval notation
#'           (e.g., \code{"(0.5;1.2]"}).
#'     \item \code{woe}: Numeric vector of Weight of Evidence for each bin.
#'     \item \code{iv}: Numeric vector of Information Value contribution per bin.
#'     \item \code{count}: Integer vector of total observations per bin.
#'     \item \code{count_pos}: Integer vector of positive cases (target=1) per bin.
#'     \item \code{count_neg}: Integer vector of negative cases (target=0) per bin.
#'     \item \code{cutpoints}: Numeric vector of upper boundaries for the bins
#'           (excluding Inf).
#'     \item \code{converged}: Logical indicating if the algorithm converged properly.
#'     \item \code{iterations}: Integer count of iterations performed.
#'     \item \code{total_iv}: The total Information Value of the binned variable.
#'   }
#'
#' @details
#' The algorithm proceeds in several distinct phases to ensure stability and
#' optimality:
#'
#' \enumerate{
#'   \item \strong{Pre-binning:} The numerical feature is initially discretized
#'   into \code{max_n_prebins} using quantiles. This handles outliers and
#'   provides a granular starting point.
#'
#'   \item \strong{Rare Bin Management:} Bins containing fewer observations
#'   than the threshold defined by \code{bin_cutoff} are iteratively merged
#'   with their nearest neighbors to ensure statistical robustness.
#'
#'   \item \strong{Monotonicity Enforcement (Optional):} If \code{is_monotonic = TRUE},
#'   the algorithm checks if the WoE trend is strictly increasing or decreasing.
#'   If not, it simulates merges in both directions to find the path that
#'   preserves the maximum possible Information Value while satisfying the
#'   monotonicity constraint.
#'
#'   \item \strong{Optimization Phase:} The algorithm iteratively merges adjacent
#'   bins that have the lowest contribution to the total Information Value (IV).
#'   This process continues until the number of bins is reduced to \code{max_bins}
#'   or the change in IV falls below \code{convergence_threshold}.
#' }
#'
#' \strong{Information Value (IV) Interpretation:}
#' \itemize{
#'   \item \eqn{< 0.02}: Not predictive
#'   \item \eqn{0.02 \text{ to } 0.1}: Weak predictive power
#'   \item \eqn{0.1 \text{ to } 0.3}: Medium predictive power
#'   \item \eqn{0.3 \text{ to } 0.5}: Strong predictive power
#'   \item \eqn{> 0.5}: Suspiciously high (check for leakage)
#' }
#'
#' @examples
#' # Example: Binning a variable with a sigmoid relationship to target
#' set.seed(123)
#' n <- 1000
#' # Generate feature
#' feature <- rnorm(n)
#'
#' # Generate target based on logistic probability
#' prob <- 1 / (1 + exp(-2 * feature))
#' target <- rbinom(n, 1, prob)
#'
#' # Perform Optimal Binning
#' result <- ob_numerical_bb(feature, target,
#'   min_bins = 3,
#'   max_bins = 5,
#'   is_monotonic = TRUE
#' )
#'
#' # Check results
#' print(data.frame(
#'   Bin = result$bin,
#'   Count = result$count,
#'   WoE = round(result$woe, 4),
#'   IV = round(result$iv, 4)
#' ))
#'
#' cat("Total IV:", result$total_iv, "\n")
#'
#' @export
ob_numerical_bb <- function(feature, target, min_bins = 3, max_bins = 5,
                            bin_cutoff = 0.05, max_n_prebins = 20,
                            is_monotonic = TRUE, convergence_threshold = 1e-6,
                            max_iterations = 1000) {
  # Type Coercion & Validation (Critical for preventing C++ Segfaults)
  # Ensure feature is numeric (double)
  if (!is.numeric(feature)) {
    warning("Feature converted to numeric for processing.")
    feature <- as.numeric(feature)
  }

  # Ensure target is integer
  if (!is.integer(target)) {
    target <- as.integer(target)
  }

  # Basic dimension check before C++ call
  if (length(feature) != length(target)) {
    stop("Length of 'feature' and 'target' must match.")
  }

  # .Call Interface
  # Note: The C++ function signature is (target, feature, ...).
  # We must respect this order in the .Call arguments.
  .Call("_OptimalBinningWoE_optimal_binning_numerical_bb",
    target,
    feature,
    as.integer(min_bins),
    as.integer(max_bins),
    as.numeric(bin_cutoff),
    as.integer(max_n_prebins),
    as.logical(is_monotonic),
    as.numeric(convergence_threshold),
    as.integer(max_iterations),
    PACKAGE = "OptimalBinningWoE"
  )
}
