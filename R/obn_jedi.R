#' Optimal Binning using Joint Entropy-Driven Interval Discretization (JEDI)
#'
#' Performs supervised discretization of continuous numerical variables using a
#' holistic approach that balances entropy reduction (information gain) with
#' statistical stability. The JEDI algorithm combines quantile-based initialization
#' with an iterative optimization process that enforces monotonicity and minimizes
#' Information Value (IV) loss.
#'
#' @param feature A numeric vector representing the continuous predictor variable.
#'   Missing values (NA) should be handled prior to binning.
#' @param target An integer vector of binary outcomes (0/1) corresponding to
#'   each observation in \code{feature}. Must have the same length as \code{feature}.
#' @param min_bins Integer. The minimum number of bins to produce. Must be \eqn{\ge} 2.
#'   Defaults to 3.
#' @param max_bins Integer. The maximum number of bins to produce. Must be \eqn{\ge}
#'   \code{min_bins}. Defaults to 5.
#' @param bin_cutoff Numeric. The minimum fraction of total observations required
#'   for a bin to be considered valid. Bins smaller than this threshold are merged.
#'   Value must be in (0, 1). Defaults to 0.05.
#' @param max_n_prebins Integer. The number of initial quantiles to generate
#'   during the initialization phase. Defaults to 20.
#' @param convergence_threshold Numeric. The threshold for the change in total IV
#'   to determine convergence during the iterative optimization. Defaults to 1e-6.
#' @param max_iterations Integer. Safety limit for the maximum number of iterations.
#'   Defaults to 1000.
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
#' The JEDI algorithm is designed to be a robust "all-rounder" for credit scoring
#' and risk modeling. Its methodology proceeds in four distinct stages:
#'
#' \enumerate{
#'   \item \strong{Initialization (Quantile Pre-binning):} The feature space is
#'   divided into \code{max_n_prebins} segments containing approximately equal
#'   numbers of observations. This ensures the algorithm starts with a statistically
#'   balanced view of the data.
#'
#'   \item \strong{Stabilization (Rare Bin Merging):} Adjacent bins with frequencies
#'   below \code{bin_cutoff} are merged. The merge direction is chosen to minimize
#'   the distortion of the event rate (similar to ChiMerge).
#'
#'   \item \strong{Monotonicity Enforcement:} The algorithm heuristically determines
#'   the dominant trend (increasing or decreasing) of the Weight of Evidence (WoE)
#'   and iteratively merges adjacent bins that violate this trend. This step effectively
#'   reduces the conditional entropy of the binning sequence with respect to the target.
#'
#'   \item \strong{IV Optimization:} If the number of bins exceeds \code{max_bins},
#'   the algorithm merges the pair of adjacent bins that results in the smallest
#'   decrease in total Information Value. This greedy approach ensures that the
#'   final discretization retains the maximum possible predictive power given the
#'   constraints.
#' }
#'
#' This joint approach (Entropy/IV + Stability constraints) makes JEDI particularly
#' effective for datasets with noise or non-monotonic initial distributions that
#' require smoothing.
#'
#' @seealso \code{\link{ob_numerical_cm}}, \code{\link{ob_numerical_ir}}
#'
#' @examples
#' # Example: Binning a variable with a complex relationship
#' set.seed(123)
#' feature <- rnorm(1000)
#' # Target probability has a quadratic component (non-monotonic)
#' # JEDI will try to force a monotonic approximation that maximizes IV
#' target <- rbinom(1000, 1, plogis(0.5 * feature + 0.1 * feature^2))
#'
#' result <- ob_numerical_jedi(feature, target,
#'   min_bins = 3,
#'   max_bins = 6,
#'   max_n_prebins = 20
#' )
#'
#' print(result$bin)
#'
#' @export
ob_numerical_jedi <- function(feature, target, min_bins = 3, max_bins = 5,
                              bin_cutoff = 0.05, max_n_prebins = 20,
                              convergence_threshold = 1e-6, max_iterations = 1000) {
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

  # NA Check
  if (any(is.na(feature))) {
    warning("Feature contains NA values. These will be excluded during pre-binning.")
  }

  # .Call Interface
  .Call("_OptimalBinningWoE_optimal_binning_numerical_jedi",
    target,
    feature,
    as.integer(min_bins),
    as.integer(max_bins),
    as.numeric(bin_cutoff),
    as.integer(max_n_prebins),
    as.numeric(convergence_threshold),
    as.integer(max_iterations),
    PACKAGE = "OptimalBinningWoE"
  )
}
