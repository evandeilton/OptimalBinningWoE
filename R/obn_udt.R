#' @title Optimal Binning for Numerical Variables using Entropy-Based Partitioning
#'
#' @description
#' Implements a supervised binning algorithm that uses Information Gain (Entropy)
#' to identify the most informative initial split points, followed by a bottom-up
#' merging process to satisfy constraints (minimum frequency, monotonicity, max bins).
#'
#' Although historically referred to as "Unsupervised Decision Trees" in some contexts,
#' this method is strictly **supervised** (uses target variable) and operates
#' **bottom-up** after an initial entropy-based selection of cutpoints. It is
#' particularly effective when the relationship between feature and target is
#' non-linear but highly informative in specific regions.
#'
#' @param feature Numeric vector of feature values. Missing values (NA) are handled
#'   by placing them in a separate bin. Infinite values are treated as valid numeric
#'   extremes or placed in the missing bin if they represent errors.
#' @param target Integer vector of binary target values (must contain only 0 and 1).
#'   Must have the same length as \code{feature}.
#' @param min_bins Minimum number of bins (default: 3). Must be at least 2.
#' @param max_bins Maximum number of bins (default: 5). Must be greater than or
#'   equal to \code{min_bins}.
#' @param bin_cutoff Minimum fraction of total observations per bin (default: 0.05).
#'   Bins below this threshold are merged based on Event Rate similarity.
#' @param max_n_prebins Maximum number of pre-bins (default: 20). The algorithm
#'   will select the top \code{max_n_prebins} cutpoints with highest Information Gain.
#'   \strong{Performance Note}: High values (>50) may significantly slow down
#'   processing for large datasets due to the O(N^2) nature of cutpoint selection.
#' @param laplace_smoothing Laplace smoothing parameter (default: 0.5) for WoE calculation.
#' @param monotonicity_direction String specifying monotonicity constraint:
#'   \itemize{
#'     \item \code{"none"} (default): No monotonicity enforcement.
#'     \item \code{"increasing"}: WoE must be non-decreasing.
#'     \item \code{"decreasing"}: WoE must be non-increasing.
#'     \item \code{"auto"}: Automatically determined by Pearson correlation.
#'   }
#' @param convergence_threshold Convergence threshold for IV optimization (default: 1e-6).
#' @param max_iterations Maximum iterations for optimization loop (default: 1000).
#'
#' @return A list containing:
#' \describe{
#'   \item{id}{Integer bin identifiers (1-based).}
#'   \item{bin}{Character bin intervals \code{"(lower;upper]"}.}
#'   \item{woe}{Numeric WoE values.}
#'   \item{iv}{Numeric IV contributions.}
#'   \item{event_rate}{Numeric event rates.}
#'   \item{count}{Integer total observations.}
#'   \item{count_pos}{Integer positive class counts.}
#'   \item{count_neg}{Integer negative class counts.}
#'   \item{cutpoints}{Numeric bin boundaries.}
#'   \item{total_iv}{Total Information Value.}
#'   \item{gini}{Gini index (2*AUC - 1) calculated on the binned data.}
#'   \item{ks}{Kolmogorov-Smirnov statistic calculated on the binned data.}
#'   \item{converged}{Logical convergence flag.}
#'   \item{iterations}{Integer iteration count.}
#' }
#'
#' @details
#' \strong{Algorithm Overview}
#'
#' The UDT algorithm executes in four phases:
#'
#' \strong{Phase 1: Entropy-Based Pre-binning}
#'
#' The algorithm evaluates every possible cutpoint \eqn{c} (midpoints between sorted
#' unique values) using Information Gain (IG):
#' \deqn{IG(S, c) = H(S) - \left( \frac{|S_L|}{|S|} H(S_L) + \frac{|S_R|}{|S|} H(S_R) \right)}
#'
#' The top \code{max_n_prebins} cutpoints with the highest IG are selected to form
#' the initial bins. This ensures that the starting bins capture the most discriminative
#' regions of the feature space.
#'
#' \strong{Phase 2: Rare Bin Merging}
#'
#' Bins with frequency \eqn{< \text{bin\_cutoff}} are merged. The merge partner is
#' chosen to minimize the difference in Event Rates:
#' \deqn{\text{merge\_idx} = \arg\min_{j \in \{i-1, i+1\}} |ER_i - ER_j|}
#' This differs from IV-based methods and aims to preserve local risk probability
#' smoothness.
#'
#' \strong{Phase 3: Monotonicity Enforcement}
#'
#' If requested, monotonicity is enforced by iteratively merging bins that violate
#' the specified direction (\code{"increasing"}, \code{"decreasing"}, or \code{"auto"}).
#' Auto-direction is determined by the sign of the Pearson correlation between
#' feature and target.
#'
#' \strong{Phase 4: Constraint Satisfaction}
#'
#' If \eqn{k > \text{max\_bins}}, bins are merged minimizing IV loss until the
#' constraint is met.
#'
#' \strong{Warning on Complexity}
#'
#' The pre-binning phase evaluates Information Gain for \emph{all} unique values.
#' For continuous features with many unique values (e.g., \eqn{N > 10,000}), this
#' step can be computationally intensive (\eqn{O(N^2)}). Consider rounding or using
#' \code{\link{ob_numerical_sketch}} for very large datasets.
#'
#' @references
#' \itemize{
#'   \item Quinlan, J. R. (1986). "Induction of Decision Trees". \emph{Machine Learning}, 1(1), 81-106.
#'   \item Fayyad, U. M., & Irani, K. B. (1992). "On the Handling of Continuous-Valued Attributes in Decision Tree Generation". \emph{Machine Learning}, 8, 87-102.
#'   \item Liu, H., et al. (2002). "Discretization: An Enabling Technique". \emph{Data Mining and Knowledge Discovery}, 6(4), 393-423.
#' }
#'
#' @author
#' Lopes, J. E.
#'
#' @seealso
#' \code{\link{ob_numerical_mdlp}} for a pure MDL-based approach,
#' \code{\link{ob_numerical_sketch}} for fast approximation on large data.
#'
#' @export
ob_numerical_udt <- function(feature,
                             target,
                             min_bins = 3,
                             max_bins = 5,
                             bin_cutoff = 0.05,
                             max_n_prebins = 20,
                             laplace_smoothing = 0.5,
                             monotonicity_direction = "none",
                             convergence_threshold = 1e-6,
                             max_iterations = 1000) {
  if (!is.numeric(feature)) {
    stop("Feature must be a numeric vector.")
  }

  if (!is.vector(target) || !(is.integer(target) || is.numeric(target))) {
    stop("Target must be an integer or numeric vector.")
  }

  if (length(feature) != length(target)) {
    stop("Feature and target must have the same length.")
  }

  feature <- as.numeric(feature)
  target <- as.integer(target)

  unique_target <- unique(target[!is.na(target)])
  if (!all(unique_target %in% c(0L, 1L)) || length(unique_target) != 2L) {
    stop("Target must contain exactly two classes: 0 and 1.")
  }

  if (min_bins < 2L) {
    stop("min_bins must be at least 2.")
  }

  if (max_bins < min_bins) {
    stop("max_bins must be greater than or equal to min_bins.")
  }

  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    stop("bin_cutoff must be in the range (0, 1).")
  }

  if (max_n_prebins < min_bins) {
    stop("max_n_prebins must be at least equal to min_bins.")
  }

  if (laplace_smoothing < 0) {
    stop("laplace_smoothing must be non-negative.")
  }

  valid_directions <- c("none", "increasing", "decreasing", "auto")
  if (!monotonicity_direction %in% valid_directions) {
    stop(paste("monotonicity_direction must be one of:", paste(valid_directions, collapse = ", ")))
  }

  result <- .Call(
    "_OptimalBinningWoE_optimal_binning_numerical_udt",
    target,
    feature,
    as.integer(min_bins),
    as.integer(max_bins),
    as.numeric(bin_cutoff),
    as.integer(max_n_prebins),
    as.numeric(laplace_smoothing),
    as.character(monotonicity_direction),
    as.numeric(convergence_threshold),
    as.integer(max_iterations),
    PACKAGE = "OptimalBinningWoE"
  )

  class(result) <- c("OptimalBinningUDT", "OptimalBinning", "list")

  return(result)
}
