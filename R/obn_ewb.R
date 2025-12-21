#' Hybrid Optimal Binning using Equal-Width Initialization and IV Optimization
#'
#' Performs supervised discretization of continuous numerical variables using a
#' hybrid approach. The algorithm initializes with an Equal-Width Binning (EWB)
#' strategy to capture the scale of the variable, followed by an iterative,
#' supervised optimization phase that merges bins to maximize Information Value (IV)
#' and enforce monotonicity.
#'
#' @param feature A numeric vector representing the continuous predictor variable.
#'   Missing values (NA) are excluded during the pre-binning phase but should
#'   ideally be handled prior to binning.
#' @param target An integer vector of binary outcomes (0/1) corresponding to
#'   each observation in \code{feature}. Must have the same length as \code{feature}.
#' @param min_bins Integer. The minimum number of bins to produce. Must be \eqn{\ge} 2.
#'   Defaults to 3.
#' @param max_bins Integer. The maximum number of bins to produce. Must be \eqn{\ge}
#'   \code{min_bins}. Defaults to 5.
#' @param bin_cutoff Numeric. The minimum fraction of total observations required
#'   for a bin to be considered valid. Bins with frequency < \code{bin_cutoff}
#'   are merged with their most similar neighbor (based on event rate).
#'   Value must be in (0, 1). Defaults to 0.05.
#' @param max_n_prebins Integer. The number of initial equal-width intervals to
#'   generate during the pre-binning phase. This parameter defines the initial
#'   granularity/search space. Defaults to 20.
#' @param is_monotonic Logical. If \code{TRUE}, the algorithm enforces a strict
#'   monotonic relationship (increasing or decreasing) between the bin indices
#'   and their Weight of Evidence (WoE). Defaults to \code{TRUE}.
#' @param convergence_threshold Numeric. The threshold for determining convergence
#'   during the iterative merging process. Defaults to 1e-6.
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
#'     \item \code{total_iv}: The total Information Value of the binned variable.
#'     \item \code{converged}: Logical indicating if the algorithm converged.
#'   }
#'
#' @details
#' Unlike standard Equal-Width binning which is purely unsupervised, this function
#' implements a \strong{Hybrid Discretization Pipeline}:
#'
#' \enumerate{
#'   \item \strong{Phase 1: Unsupervised Initialization (Scale Preservation)}
#'   The range of the feature \eqn{[min(x), max(x)]} is divided into \code{max_n_prebins}
#'   intervals of equal width \eqn{w = (max(x) - min(x)) / N}. This step preserves
#'   the cardinal magnitude of the data but is sensitive to outliers.
#'
#'   \item \strong{Phase 2: Statistical Stabilization}
#'   Bins falling below the \code{bin_cutoff} threshold are merged. Unlike naive
#'   approaches, this implementation merges rare bins with the neighbor that has
#'   the most similar class distribution (event rate), minimizing the distortion
#'   of the predictive relationship.
#'
#'   \item \strong{Phase 3: Monotonicity Enforcement}
#'   If \code{is_monotonic = TRUE}, the algorithm checks for non-monotonic trends
#'   in the Weight of Evidence (WoE). Violating adjacent bins are iteratively merged
#'   to ensure a strictly increasing or decreasing relationship, which is a key
#'   requirement for interpretable Logistic Regression scorecards.
#'
#'   \item \strong{Phase 4: IV-Based Optimization}
#'   If the number of bins exceeds \code{max_bins}, the algorithm applies a
#'   hierarchical bottom-up merging strategy. It calculates the \emph{Information Value Loss}
#'   for every possible pair of adjacent bins:
#'   \deqn{\Delta IV = (IV_i + IV_{i+1}) - IV_{merged}}
#'   The pair minimizing this loss is merged, ensuring that the final coarse classes
#'   retain the maximum possible predictive power of the original variable.
#' }
#'
#' \strong{Technical Note on Outliers:}
#' Because the initialization is based on the range, extreme outliers can compress
#' the majority of the data into a single initial bin. If your data is highly
#' skewed or contains outliers, consider using \code{\link{ob_numerical_cm}} (Quantile/ChiMerge)
#' or winsorizing the data before using this function.
#'
#' @references
#' Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised
#' discretization of continuous features. \emph{Machine Learning Proceedings}, 194-202.
#'
#' Siddiqi, N. (2012). \emph{Credit Risk Scorecards: Developing and Implementing
#' Intelligent Credit Scoring}. John Wiley & Sons.
#'
#' Catlett, J. (1991). On changing continuous attributes into ordered discrete
#' attributes. \emph{Proceedings of the European Working Session on Learning on
#' Machine Learning}, 164-178.
#'
#' @seealso \code{\link{ob_numerical_cm}} for Quantile/Chi-Square binning,
#' \code{\link{ob_numerical_dp}} for Dynamic Programming approaches.
#'
#' @examples
#' # Example 1: Uniform distribution (Ideal for Equal-Width)
#' set.seed(123)
#' feature <- runif(1000, 0, 100)
#' target <- rbinom(1000, 1, plogis(0.05 * feature - 2))
#'
#' res_ewb <- ob_numerical_ewb(feature, target, max_bins = 5)
#' print(res_ewb$bin)
#' print(paste("Total IV:", round(res_ewb$total_iv, 4)))
#'
#' # Example 2: Effect of Outliers (The weakness of Equal-Width)
#' feature_outlier <- c(feature, 10000) # One extreme outlier
#' target_outlier <- c(target, 0)
#'
#' # Note: The algorithm tries to recover, but the initial split is distorted
#' res_outlier <- ob_numerical_ewb(feature_outlier, target_outlier, max_bins = 5)
#' print(res_outlier$bin)
#'
#' @export
ob_numerical_ewb <- function(feature, target, min_bins = 3, max_bins = 5,
                             bin_cutoff = 0.05, max_n_prebins = 20,
                             is_monotonic = TRUE, convergence_threshold = 1e-6,
                             max_iterations = 1000) {
  # Type Validation & Coercion
  if (!is.numeric(feature)) {
    warning("Feature converted to numeric for processing.")
    feature <- as.numeric(feature)
  }

  if (!is.integer(target)) {
    target <- as.integer(target)
  }

  # Dimension Validation
  if (length(feature) != length(target)) {
    stop("Length of 'feature' and 'target' must match.")
  }

  # NA Warning (The C++ logic filters them out, but good to warn R user)
  if (any(is.na(feature))) {
    warning("Feature contains NA values. These will be excluded during the pre-binning phase.")
  }

  # .Call Interface
  # Matches C++ signature:
  # (target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins,
  #  is_monotonic, convergence_threshold, max_iterations)
  .Call("_OptimalBinningWoE_optimal_binning_numerical_ewb",
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
