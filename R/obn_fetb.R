#' Optimal Binning using Fisher's Exact Test
#'
#' Performs supervised discretization of continuous numerical variables using
#' Fisher's Exact Test. This method iteratively merges adjacent bins that are
#' statistically similar (highest p-value) while strictly enforcing a monotonic
#' Weight of Evidence (WoE) trend.
#'
#' @param feature A numeric vector representing the continuous predictor variable.
#'   Missing values (NA) should be handled prior to binning.
#' @param target An integer vector of binary outcomes (0/1) corresponding to
#'   each observation in \code{feature}. Must have the same length as \code{feature}.
#' @param min_bins Integer. The minimum number of bins to produce. Must be \eqn{\ge} 2.
#'   Defaults to 3.
#' @param max_bins Integer. The maximum number of bins to produce. Must be \eqn{\ge}
#'   \code{min_bins}. Defaults to 5.
#' @param max_n_prebins Integer. The number of initial quantiles to generate
#'   during the pre-binning phase. Defaults to 20.
#' @param convergence_threshold Numeric. The threshold for the change in Information
#'   Value (IV) to determine convergence during the iterative merging process.
#'   Defaults to 1e-6.
#' @param max_iterations Integer. Safety limit for the maximum number of merging
#'   iterations. Defaults to 1000.
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
#'     \item \code{cutpoints}: Numeric vector of upper boundaries (excluding Inf).
#'     \item \code{converged}: Logical indicating if the algorithm converged.
#'     \item \code{iterations}: Integer count of iterations performed.
#'   }
#'
#' @details
#' The \strong{Fisher's Exact Test Binning (FETB)} algorithm provides a robust statistical
#' alternative to ChiMerge.
#'
#' \strong{Key Differences from ChiMerge:}
#' \itemize{
#'   \item \strong{Exact Probability:} Instead of relying on the Chi-Square asymptotic
#'         approximation (which can be unreliable for small bin counts), FETB calculates
#'         the exact hypergeometric probability of independence between the bin index
#'         and the target.
#'   \item \strong{Merge Criterion:} In each step, the algorithm identifies the pair of
#'         adjacent bins with the \emph{highest} p-value (indicating they are the most
#'         statistically indistinguishable) and merges them.
#'   \item \strong{Monotonicity:} The algorithm incorporates a check after every merge
#'         to ensure the WoE trend remains monotonic, merging strictly violating bins
#'         immediately.
#' }
#'
#' This method is particularly recommended when working with smaller datasets or
#' highly imbalanced target classes, where the assumptions of the Chi-Square test
#' might be violated.
#'
#' @seealso \code{\link{ob_numerical_cm}}
#'
#' @examples
#' # Example: Binning a small dataset where Fisher's Exact Test excels
#' set.seed(123)
#' feature <- rnorm(100)
#' target <- rbinom(100, 1, 0.2)
#'
#' result <- ob_numerical_fetb(feature, target,
#'   min_bins = 2,
#'   max_bins = 4,
#'   max_n_prebins = 10
#' )
#'
#' print(result$bin)
#' print(result$woe)
#'
#' @export
ob_numerical_fetb <- function(feature, target, min_bins = 3, max_bins = 5,
                              max_n_prebins = 20, convergence_threshold = 1e-6,
                              max_iterations = 1000) {
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

  # .Call Interface
  .Call("_OptimalBinningWoE_optimal_binning_numerical_fetb",
    target,
    feature,
    as.integer(min_bins),
    as.integer(max_bins),
    as.integer(max_n_prebins),
    as.numeric(convergence_threshold),
    as.integer(max_iterations),
    PACKAGE = "OptimalBinningWoE"
  )
}
