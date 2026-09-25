#' Optimal Binning using MDLP with Monotonicity Constraints
#'
#' Performs supervised discretization of continuous numerical variables using the
#' Minimum Description Length Principle (MDLP) algorithm, enhanced with optional
#' monotonicity constraints on the Weight of Evidence (WoE). This method is
#' particularly suitable for creating interpretable bins for logistic regression
#' models in domains like credit scoring.
#'
#' @param feature A numeric vector representing the continuous predictor variable.
#'   Rows whose value is missing (\code{NA}/\code{NaN}) are excluded from the
#'   fit silently, so the bin counts sum to the number of non-missing rows;
#'   \code{-Inf} and \code{+Inf} are kept as extreme values of the first and
#'   last bin and never become a cutpoint.
#' @param target An integer vector of binary outcomes (0/1) corresponding to
#'   each observation in \code{feature}. Must have the same length as \code{feature}.
#'   A missing value in \code{target} is an error.
#' @param min_bins Integer. The minimum number of bins to produce. Must be \eqn{\ge} 2.
#'   Defaults to 2.
#' @param max_bins Integer. The maximum number of bins to produce. Must be \eqn{\ge}
#'   \code{min_bins}. Defaults to 5.
#' @param bin_cutoff Numeric. Currently unused in this implementation (reserved for
#'   future versions). Defaults to 0.05.
#' @param max_n_prebins Integer. Currently unused in this implementation (reserved for
#'   future versions). Defaults to 100.
#' @param convergence_threshold Numeric. The threshold for determining convergence
#'   during the iterative monotonicity enforcement process. Defaults to 1e-6.
#' @param max_iterations Integer. Safety limit for the maximum number of iterations
#'   in the monotonicity enforcement phase. Defaults to 1000.
#' @param force_monotonicity Logical. If \code{TRUE}, the algorithm enforces a strict
#'   monotonic relationship (increasing or decreasing) between the bin indices
#'   and their Weight of Evidence (WoE) values. Defaults to \code{TRUE}.
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
#'     \item \code{converged}: Logical indicating if the monotonicity enforcement converged.
#'     \item \code{iterations}: Integer count of iterations in monotonicity phase.
#'   }
#'
#' @details
#' This function implements a sophisticated hybrid approach combining the classic
#' MDLP algorithm with modern monotonicity constraints.
#'
#' \strong{Algorithm Pipeline:}
#' \enumerate{
#'   \item \strong{Data Preparation:} Removes NA values and sorts the data by feature value.
#'   \item \strong{MDLP Discretization (Fayyad & Irani, 1993):}
#'   \itemize{
#'     \item Recursively evaluates the binary splits of the sorted data. Only
#'           boundary points (cuts that do not fall between two values holding
#'           observations of a single, common class) are candidates: the
#'           entropy-minimising cut is always one of them (Fayyad & Irani, 1992).
#'     \item For each candidate split, calculates the Information Gain (IG).
#'     \item Applies the MDLP stopping criterion to the best split \eqn{T} of a
#'           set \eqn{S} into \eqn{S_1} and \eqn{S_2}:
#'           \deqn{IG > \frac{\log_2(N-1) + \Delta}{N}}
#'           where \eqn{N} is the number of samples in \eqn{S} and
#'           \eqn{\Delta = \log_2(3^k - 2) - [k E(S) - k_1 E(S_1) - k_2 E(S_2)]},
#'           with \eqn{k}, \eqn{k_1}, \eqn{k_2} the number of classes present
#'           in \eqn{S}, \eqn{S_1} and \eqn{S_2}.
#'     \item Only accepts splits that significantly reduce entropy beyond what would
#'           be expected by chance, balancing model fit with complexity.
#'   }
#'   \item \strong{Constraint Enforcement:}
#'   \itemize{
#'     \item \strong{Max Bins:} Accepted splits are applied best-first, in
#'           decreasing order of the entropy reduction they achieve, and at most
#'           \code{max_bins - 1} of them are kept. When \code{max_bins} is not
#'           binding this yields exactly the MDLP partition.
#'     \item \strong{Min Bins:} When MDLP accepts fewer splits, additional ones
#'           are placed between distinct values, spread evenly over the distinct
#'           values and then over the largest runs of observations.
#'     \item \strong{Monotonicity (if enabled):} Iteratively merges adjacent bins with
#'           the most similar WoE values until a strictly increasing or decreasing
#'           trend is achieved across all bins, or only \code{min_bins} bins remain.
#'   }
#' }
#'
#' \strong{Technical Notes:}
#' \itemize{
#'   \item The algorithm uses Laplace smoothing (\eqn{\alpha = 0.5}) when calculating
#'         WoE to prevent \eqn{\log(0)} errors for bins with pure class distributions.
#'   \item When all feature values are identical a single bin is returned, with
#'         a warning. When the feature has fewer distinct values than
#'         \code{min_bins}, each distinct value becomes its own bin and a warning
#'         reports that \code{min_bins} could not be met: a bin boundary can only
#'         fall between two different values.
#'   \item The monotonicity enforcement phase is iterative and uses the
#'         \code{convergence_threshold} to determine when changes in WoE become negligible.
#' }
#'
#' @references
#' Fayyad, U. M., & Irani, K. B. (1993). Multi-interval discretization of continuous-valued
#' attributes for classification learning. \emph{Proceedings of the 13th International
#' Joint Conference on Artificial Intelligence}, 1022-1029.
#'
#' Fayyad, U. M., & Irani, K. B. (1992). On the handling of continuous-valued
#' attributes in decision tree generation. \emph{Machine Learning}, 8, 87-102.
#'
#' Kurgan, L. A., & Musilek, P. (2006). A survey of techniques. \emph{IEEE Transactions
#' on Knowledge and Data Engineering}, 18(5), 673-689.
#'
#' Garcia, S., Luengo, J., & Herrera, F. (2013). Data preprocessing in data mining.
#' \emph{Springer Science & Business Media}.
#'
#' @seealso \code{\link{ob_numerical_cm}} for ChiMerge-based approaches,
#' \code{\link{ob_numerical_dp}} for dynamic programming methods.
#'
#' @examples
#' # Example: Standard usage with monotonicity
#' set.seed(123)
#' feature <- rnorm(1000)
#' target <- rbinom(1000, 1, plogis(2 * feature)) # Positive relationship
#'
#' result <- ob_numerical_fast_mdlp(feature, target,
#'   min_bins = 3,
#'   max_bins = 6,
#'   force_monotonicity = TRUE
#' )
#'
#' print(result$bin)
#' print(result$woe) # Should show a monotonic trend
#'
#' # Example: Disabling monotonicity for exploratory analysis
#' result_no_mono <- ob_numerical_fast_mdlp(feature, target,
#'   min_bins = 3,
#'   max_bins = 6,
#'   force_monotonicity = FALSE
#' )
#'
#' print(result_no_mono$woe) # May show non-monotonic patterns
#'
#' @export
ob_numerical_fast_mdlp <- function(feature, target,
                                   min_bins = 2L, max_bins = 5L,
                                   bin_cutoff = 0.05, max_n_prebins = 100L,
                                   convergence_threshold = 1e-6,
                                   max_iterations = 1000L,
                                   force_monotonicity = TRUE) {
  # Type Validation and Coercion
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

  # Parameter Validation
  if (min_bins < 2) {
    stop("min_bins must be >= 2.")
  }
  if (max_bins < min_bins) {
    stop("max_bins must be >= min_bins.")
  }

  # .Call Interface
  # Matches C++ signature:
  # (target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins,
  #  convergence_threshold, max_iterations, force_monotonicity)
  .Call("_OptimalBinningWoE_optimal_binning_numerical_fast_mdlpm",
    target,
    feature,
    as.integer(min_bins),
    as.integer(max_bins),
    as.numeric(bin_cutoff),
    as.integer(max_n_prebins),
    as.numeric(convergence_threshold),
    as.integer(max_iterations),
    as.logical(force_monotonicity),
    PACKAGE = "OptimalBinningWoE"
  )
}
