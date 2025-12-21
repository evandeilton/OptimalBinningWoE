#' @title Optimal Binning for Numerical Variables using Dynamic Programming
#'
#' @description
#' Performs supervised discretization of continuous numerical variables using a
#' greedy heuristic approach that resembles Dynamic Programming. This method is
#' particularly effective at strictly enforcing monotonic trends (ascending or
#' descending) in the Weight of Evidence (WoE), which is critical for the
#' interpretability of logistic regression models in credit scoring.
#'
#' @param feature A numeric vector representing the continuous predictor variable.
#'   Missing values (NA) should be handled prior to binning, as they are not
#'   supported by this algorithm.
#' @param target An integer vector of binary outcomes (0/1) corresponding to
#'   each observation in \code{feature}. Must have the same length as \code{feature}.
#' @param min_bins Integer. The minimum number of bins to produce. Must be \eqn{\ge} 2.
#'   Defaults to 3.
#' @param max_bins Integer. The maximum number of bins to produce. Must be \eqn{\ge}
#'   \code{min_bins}. Defaults to 5.
#' @param bin_cutoff Numeric. The minimum fraction of total observations required
#'   for a bin to be considered valid. Bins with frequency < \code{bin_cutoff}
#'   will be merged. Value must be in (0, 1). Defaults to 0.05.
#' @param max_n_prebins Integer. The number of initial quantiles to generate
#'   during the pre-binning phase. Defaults to 20.
#' @param convergence_threshold Numeric. The threshold for the change in metrics
#'   to determine convergence during the iterative merging process.
#'   Defaults to 1e-6.
#' @param max_iterations Integer. Safety limit for the maximum number of merging
#'   iterations. Defaults to 1000.
#' @param monotonic_trend Character string specifying the desired direction of the
#'   Weight of Evidence (WoE) trend.
#'   \itemize{
#'     \item \code{"auto"}: Automatically determines the most likely trend (ascending or descending)
#'           based on the correlation between the feature and the target.
#'     \item \code{"ascending"}: Forces the WoE to increase as the feature value increases.
#'     \item \code{"descending"}: Forces the WoE to decrease as the feature value increases.
#'     \item \code{"none"}: Does not enforce any monotonic constraint (allows peaks and valleys).
#'   }
#'   Defaults to \code{"auto"}.
#'
#' @return A list containing the binning results:
#'   \itemize{
#'     \item \code{id}: Integer vector of bin identifiers.
#'     \item \code{bin}: Character vector of bin labels in interval notation.
#'     \item \code{woe}: Numeric vector of Weight of Evidence for each bin.
#'     \item \code{iv}: Numeric vector of Information Value contribution per bin.
#'     \item \code{count}: Integer vector of total observations per bin.
#'     \item \code{count_pos}: Integer vector of positive cases.
#'     \item \code{count_neg}: Integer vector of negative cases.
#'     \item \code{event_rate}: Numeric vector of the target event rate in each bin.
#'     \item \code{cutpoints}: Numeric vector of upper boundaries (excluding Inf).
#'     \item \code{total_iv}: The total Information Value of the binned variable.
#'     \item \code{monotonic_trend}: The actual trend enforced ("ascending", "descending", or "none").
#'     \item \code{execution_time_ms}: Execution time in milliseconds.
#'   }
#'
#' @details
#' Although named "DP" (Dynamic Programming) in some contexts, this implementation
#' primarily uses a \strong{greedy heuristic} to optimize the Information Value (IV)
#' while satisfying constraints.
#'
#' \strong{Algorithm Steps:}
#' \enumerate{
#'   \item \strong{Pre-binning:} Generates initial granular bins based on quantiles.
#'   \item \strong{Trend Determination:} If \code{monotonic_trend = "auto"}, calculates
#'         the Pearson correlation between the feature and target to decide if
#'         the WoE should increase or decrease.
#'   \item \strong{Monotonicity Enforcement:} Iteratively merges adjacent bins that
#'         violate the determined or requested trend.
#'   \item \strong{Constraint Satisfaction:} Merges rare bins (below \code{bin_cutoff})
#'         and ensures the number of bins is within \code{[min_bins, max_bins]}.
#'   \item \strong{Optimization:} Greedily merges similar bins (based on WoE difference)
#'         to reduce complexity while attempting to preserve information.
#' }
#'
#' This method is often preferred when strict business logic dictates a specific
#' relationship direction (e.g., "higher income must imply lower risk").
#'
#' @seealso \code{\link{ob_numerical_cm}}, \code{\link{ob_numerical_bb}}
#'
#' @examples
#' # Example: forcing a descending trend
#' set.seed(123)
#' feature <- runif(1000, 0, 100)
#' # Target has a complex relationship, but we want to force a linear view
#' target <- rbinom(1000, 1, 0.5 + 0.003 * feature) # slightly positive trend
#'
#' # Force "descending" (even if data suggests ascending) to see enforcement
#' result <- ob_numerical_dp(feature, target,
#'   min_bins = 3,
#'   max_bins = 5,
#'   monotonic_trend = "descending"
#' )
#'
#' print(result$bin)
#' print(result$woe) # Should be strictly decreasing
#'
#' @export
ob_numerical_dp <- function(feature, target, min_bins = 3, max_bins = 5,
                            bin_cutoff = 0.05, max_n_prebins = 20,
                            convergence_threshold = 1e-6, max_iterations = 1000,
                            monotonic_trend = c("auto", "ascending", "descending", "none")) {
  # Type Validation
  if (!is.numeric(feature)) {
    warning("Feature converted to numeric for processing.")
    feature <- as.numeric(feature)
  }

  if (!is.integer(target)) {
    target <- as.integer(target)
  }

  # Dimension Check
  if (length(feature) != length(target)) {
    stop("Length of 'feature' and 'target' must match.")
  }

  # NA check (C++ explicitly throws error for feature NAs here)
  if (any(is.na(feature))) {
    stop("Feature contains NA values. Impute missing values before using this algorithm.")
  }

  # Argument Match
  monotonic_trend <- match.arg(monotonic_trend)

  # .Call Interface
  # C++ expects: (target, feature, min, max, cutoff, prebins, thresh, iter, trend)
  .Call("_OptimalBinningWoE_optimal_binning_numerical_dp",
    target,
    feature,
    as.integer(min_bins),
    as.integer(max_bins),
    as.numeric(bin_cutoff),
    as.integer(max_n_prebins),
    as.numeric(convergence_threshold),
    as.integer(max_iterations),
    as.character(monotonic_trend),
    PACKAGE = "OptimalBinningWoE"
  )
}
