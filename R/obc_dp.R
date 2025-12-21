#' Optimal Binning for Categorical Variables using Dynamic Programming
#'
#' Performs supervised discretization of categorical variables using a dynamic
#' programming algorithm with optional monotonicity constraints. This method
#' maximizes the total Information Value (IV) while ensuring optimal bin
#' formation that respects user-defined constraints on bin count and frequency.
#' The algorithm guarantees global optimality through dynamic programming.
#'
#' @param feature A character vector or factor representing the categorical
#'   predictor variable to be binned. Missing values are automatically
#'   converted to the category \code{"NA"}.
#' @param target An integer vector of binary outcomes (0/1) corresponding
#'   to each observation in \code{feature}. Missing values are not permitted.
#' @param min_bins Integer. Minimum number of bins to produce. Must be >= 2.
#'   The algorithm searches for solutions within [\code{min_bins}, \code{max_bins}]
#'   that maximize total IV. Defaults to 3.
#' @param max_bins Integer. Maximum number of bins to produce. Must be >=
#'   \code{min_bins}. Defines the upper bound of the search space for the
#'   optimal solution. Defaults to 5.
#' @param bin_cutoff Numeric. Minimum proportion of total observations required
#'   for a category to remain separate. Categories below this threshold are
#'   merged with similar categories. Must be in (0, 1). Defaults to 0.05.
#' @param max_n_prebins Integer. Maximum number of initial bins before dynamic
#'   programming optimization. Controls computational complexity. Must be >= 2.
#'   Defaults to 20.
#' @param convergence_threshold Numeric. Convergence tolerance for the iterative
#'   dynamic programming updates. Smaller values require stricter convergence.
#'   Must be > 0. Defaults to 1e-6.
#' @param max_iterations Integer. Maximum number of dynamic programming iterations.
#'   Prevents excessive computation in edge cases. Must be > 0. Defaults to 1000.
#' @param bin_separator Character string used to concatenate category names
#'   when multiple categories are merged into a single bin. Defaults to "\%;\%".
#' @param monotonic_trend Character string specifying monotonicity constraint
#'   for Weight of Evidence. Must be one of:
#'   \describe{
#'     \item{\code{"auto"}}{Automatically determine trend direction (default)}
#'     \item{\code{"ascending"}}{Enforce increasing WoE across bins}
#'     \item{\code{"descending"}}{Enforce decreasing WoE across bins}
#'     \item{\code{"none"}}{No monotonicity constraint}
#'   }
#'   Monotonicity constraints are enforced during the DP optimization phase.
#'   Defaults to \code{"auto"}.
#'
#' @return A list containing the binning results with the following components:
#'   \describe{
#'     \item{\code{id}}{Integer vector of bin identifiers (1-indexed)}
#'     \item{\code{bin}}{Character vector of bin labels (merged category names)}
#'     \item{\code{woe}}{Numeric vector of Weight of Evidence values per bin}
#'     \item{\code{iv}}{Numeric vector of Information Value contribution per bin}
#'     \item{\code{count}}{Integer vector of total observations per bin}
#'     \item{\code{count_pos}}{Integer vector of positive cases (target=1) per bin}
#'     \item{\code{count_neg}}{Integer vector of negative cases (target=0) per bin}
#'     \item{\code{event_rate}}{Numeric vector of event rates per bin}
#'     \item{\code{total_iv}}{Numeric total Information Value of the binning solution}
#'     \item{\code{converged}}{Logical indicating if the DP algorithm converged}
#'     \item{\code{iterations}}{Integer count of DP iterations performed}
#'     \item{\code{execution_time_ms}}{Numeric execution time in milliseconds}
#'   }
#'
#' @details
#' This implementation uses dynamic programming to find the globally optimal
#' binning solution that maximizes total Information Value subject to constraints.
#'
#' \strong{Algorithm Workflow:}
#' \enumerate{
#'   \item Input validation and data preprocessing
#'   \item Rare category merging (frequencies below \code{bin_cutoff})
#'   \item Pre-binning limitation (if categories exceed \code{max_n_prebins})
#'   \item Category sorting by event rate
#'   \item Dynamic programming table initialization
#'   \item Iterative DP optimization with optional monotonicity constraints
#'   \item Backtracking to construct optimal bins
#'   \item Final metric computation
#' }
#'
#' \strong{Dynamic Programming Formulation:}
#'
#' Let \eqn{DP[i][k]} represent the maximum total IV achievable using the first
#' \eqn{i} categories partitioned into \eqn{k} bins. The recurrence relation is:
#'
#' \deqn{DP[i][k] = \max_{j<i} \{DP[j][k-1] + IV(j+1, i)\}}
#'
#' where \eqn{IV(j+1, i)} is the Information Value of a bin containing categories
#' from \eqn{j+1} to \eqn{i}. Monotonicity constraints are enforced by restricting
#' transitions that violate WoE ordering.
#'
#' \strong{Computational Complexity:}
#' \itemize{
#'   \item Time: \eqn{O(n^2 \cdot k \cdot m)} where \eqn{n} = categories,
#'     \eqn{k} = max_bins, \eqn{m} = iterations
#'   \item Space: \eqn{O(n \cdot k)} for DP tables
#' }
#'
#' \strong{Advantages over Heuristic Methods:}
#' \itemize{
#'   \item Guarantees global optimality (within constraint space)
#'   \item Explicit monotonicity enforcement
#'   \item Deterministic and reproducible results
#'   \item Efficient caching mechanism for bin statistics
#' }
#'
#' @references
#' Navas-Palencia, G. (2022). Optimal Binning: Mathematical Programming
#' Formulation. \emph{arXiv preprint arXiv:2001.08025}.
#'
#' Bellman, R. (1954). The theory of dynamic programming.
#' \emph{Bulletin of the American Mathematical Society}, 60(6), 503-515.
#'
#' Siddiqi, N. (2017). \emph{Intelligent Credit Scoring: Building and
#' Implementing Better Credit Risk Scorecards} (2nd ed.). Wiley.
#'
#' Thomas, L. C., Edelman, D. B., & Crook, J. N. (2017).
#' \emph{Credit Scoring and Its Applications} (2nd ed.). SIAM.
#'
#' @seealso
#' \code{\link{ob_categorical_cm}} for ChiMerge-based binning,
#' \code{\link{ob_categorical_dmiv}} for divergence measure-based binning
#'
#' @examples
#' \donttest{
#' # Example 1: Basic usage with monotonic WoE enforcement
#' set.seed(123)
#' n_obs <- 1000
#'
#' # Simulate education levels with increasing default risk
#' education <- c("High School", "Associate", "Bachelor", "Master", "PhD")
#' default_probs <- c(0.20, 0.15, 0.10, 0.06, 0.03)
#'
#' cat_feature <- sample(education, n_obs,
#'   replace = TRUE,
#'   prob = c(0.30, 0.25, 0.25, 0.15, 0.05)
#' )
#' bin_target <- sapply(cat_feature, function(x) {
#'   rbinom(1, 1, default_probs[which(education == x)])
#' })
#'
#' # Apply DP binning with ascending monotonicity
#' result_dp <- ob_categorical_dp(
#'   cat_feature,
#'   bin_target,
#'   min_bins = 2,
#'   max_bins = 4,
#'   monotonic_trend = "ascending"
#' )
#'
#' # Display results
#' print(data.frame(
#'   Bin = result_dp$bin,
#'   WoE = round(result_dp$woe, 3),
#'   IV = round(result_dp$iv, 4),
#'   Count = result_dp$count,
#'   EventRate = round(result_dp$event_rate, 3)
#' ))
#'
#' cat("Total IV:", round(result_dp$total_iv, 4), "\n")
#' cat("Converged:", result_dp$converged, "\n")
#'
#' # Example 2: Comparing monotonicity constraints
#' result_dp_asc <- ob_categorical_dp(
#'   cat_feature, bin_target,
#'   max_bins = 3,
#'   monotonic_trend = "ascending"
#' )
#'
#' result_dp_none <- ob_categorical_dp(
#'   cat_feature, bin_target,
#'   max_bins = 3,
#'   monotonic_trend = "none"
#' )
#'
#' cat("\nWith monotonicity:\n")
#' cat("  Bins:", length(result_dp_asc$bin), "\n")
#' cat("  Total IV:", round(result_dp_asc$total_iv, 4), "\n")
#'
#' cat("\nWithout monotonicity:\n")
#' cat("  Bins:", length(result_dp_none$bin), "\n")
#' cat("  Total IV:", round(result_dp_none$total_iv, 4), "\n")
#'
#' # Example 3: High cardinality with pre-binning
#' set.seed(456)
#' n_obs_large <- 5000
#'
#' # Simulate customer segments (high cardinality)
#' segments <- paste0("Segment_", LETTERS[1:20])
#' segment_probs <- runif(20, 0.01, 0.20)
#'
#' cat_feature_hc <- sample(segments, n_obs_large, replace = TRUE)
#' bin_target_hc <- rbinom(
#'   n_obs_large, 1,
#'   segment_probs[match(cat_feature_hc, segments)]
#' )
#'
#' result_dp_hc <- ob_categorical_dp(
#'   cat_feature_hc,
#'   bin_target_hc,
#'   min_bins = 3,
#'   max_bins = 5,
#'   bin_cutoff = 0.03,
#'   max_n_prebins = 10
#' )
#'
#' cat("\nHigh cardinality example:\n")
#' cat("  Original categories:", length(unique(cat_feature_hc)), "\n")
#' cat("  Final bins:", length(result_dp_hc$bin), "\n")
#' cat("  Execution time:", result_dp_hc$execution_time_ms, "ms\n")
#'
#' # Example 4: Handling missing values
#' set.seed(789)
#' cat_feature_na <- cat_feature
#' cat_feature_na[sample(n_obs, 50)] <- NA # Introduce 5% missing
#'
#' result_dp_na <- ob_categorical_dp(
#'   cat_feature_na,
#'   bin_target,
#'   min_bins = 2,
#'   max_bins = 4
#' )
#'
#' # Check if NA was treated as a category
#' na_bin <- grep("NA", result_dp_na$bin, value = TRUE)
#' if (length(na_bin) > 0) {
#'   cat("\nNA handling:\n")
#'   cat("  Bin containing NA:", na_bin, "\n")
#' }
#' }
#'
#' @export
ob_categorical_dp <- function(feature, target,
                              min_bins = 3,
                              max_bins = 5,
                              bin_cutoff = 0.05,
                              max_n_prebins = 20,
                              convergence_threshold = 1e-6,
                              max_iterations = 1000,
                              bin_separator = "%;%",
                              monotonic_trend = "auto") {
  # Input preprocessing
  if (!is.character(feature)) {
    feature <- as.character(feature)
  }
  feature[is.na(feature)] <- "NA"
  target <- as.integer(target)

  # Invoke C++ implementation
  .Call("_OptimalBinningWoE_optimal_binning_categorical_dp",
    target = target,
    feature = feature,
    min_bins = as.integer(min_bins),
    max_bins = as.integer(max_bins),
    bin_cutoff = bin_cutoff,
    max_n_prebins = as.integer(max_n_prebins),
    convergence_threshold = convergence_threshold,
    max_iterations = as.integer(max_iterations),
    bin_separator = bin_separator,
    monotonic_trend = monotonic_trend,
    PACKAGE = "OptimalBinningWoE"
  )
}
