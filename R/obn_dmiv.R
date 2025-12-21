#' Optimal Binning using Metric Divergence Measures (Zeng, 2013)
#'
#' Performs supervised discretization of continuous numerical variables using the
#' theoretical framework proposed by Zeng (2013). This method creates bins that
#' maximize a specified divergence measure (e.g., Kullback-Leibler, Hellinger)
#' between the distributions of positive and negative cases, effectively maximizing
#' the Information Value (IV) or other discriminatory statistics.
#'
#' @param feature A numeric vector representing the continuous predictor variable.
#'   Missing values (NA) are excluded during the pre-binning phase.
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
#' @param is_monotonic Logical. If \code{TRUE}, the algorithm enforces a strict
#'   monotonic relationship (increasing or decreasing) between the bin indices
#'   and their WoE values. Defaults to \code{TRUE}.
#' @param convergence_threshold Numeric. The threshold for the change in total
#'   divergence to determine convergence during the iterative merging process.
#'   Defaults to 1e-6.
#' @param max_iterations Integer. Safety limit for the maximum number of merging
#'   iterations. Defaults to 1000.
#' @param bin_method Character string specifying the formula for Weight of Evidence calculation:
#'   \itemize{
#'     \item \code{"woe"}: Standard definition \eqn{\ln((p_i/P) / (n_i/N))}.
#'     \item \code{"woe1"}: Zeng's definition \eqn{\ln(p_i / n_i)} (direct log odds).
#'   }
#'   Defaults to \code{"woe1"}.
#' @param divergence_method Character string specifying the divergence measure to maximize.
#'   Available options:
#'   \itemize{
#'     \item \code{"iv"}: Information Value (conceptually similar to KL).
#'     \item \code{"he"}: Hellinger Distance.
#'     \item \code{"kl"}: Kullback-Leibler Divergence.
#'     \item \code{"tr"}: Triangular Discrimination.
#'     \item \code{"klj"}: Jeffrey's Divergence (Symmetric KL).
#'     \item \code{"sc"}: Symmetric Chi-Square Divergence.
#'     \item \code{"js"}: Jensen-Shannon Divergence.
#'     \item \code{"l1"}: Manhattan Distance (L1 Norm).
#'     \item \code{"l2"}: Euclidean Distance (L2 Norm).
#'     \item \code{"ln"}: Chebyshev Distance (L-infinity Norm).
#'   }
#'   Defaults to \code{"l2"}.
#'
#' @return A list containing the binning results:
#'   \itemize{
#'     \item \code{id}: Integer vector of bin identifiers.
#'     \item \code{bin}: Character vector of bin labels in interval notation.
#'     \item \code{woe}: Numeric vector of Weight of Evidence for each bin.
#'     \item \code{divergence}: Numeric vector of the chosen divergence contribution per bin.
#'     \item \code{count}: Integer vector of total observations per bin.
#'     \item \code{count_pos}: Integer vector of positive cases.
#'     \item \code{count_neg}: Integer vector of negative cases.
#'     \item \code{cutpoints}: Numeric vector of upper boundaries (excluding Inf).
#'     \item \code{total_divergence}: The sum of the divergence measure across all bins.
#'     \item \code{bin_method}: The WoE calculation method used.
#'     \item \code{divergence_method}: The divergence measure used.
#'   }
#'
#' @details
#' This algorithm implements the "Metric Divergence Measures" framework. Unlike
#' standard ChiMerge which uses statistical significance, this method uses a
#' branch-and-bound approach to minimize the loss of a specific divergence
#' metric when merging bins.
#'
#' \strong{The Process:}
#' \enumerate{
#'   \item \strong{Pre-binning:} Generates granular bins based on quantiles.
#'   \item \strong{Rare Merging:} Merges bins smaller than \code{bin_cutoff}.
#'   \item \strong{Monotonicity:} If \code{is_monotonic = TRUE}, forces the WoE trend
#'         to be monotonic by merging "violating" bins in the direction that
#'         maximizes the total divergence.
#'   \item \strong{Optimization:} Iteratively merges the pair of adjacent bins that
#'         results in the smallest loss of total divergence, until \code{max_bins}
#'         is reached.
#' }
#'
#' @references
#' Zeng, G. (2013). Metric Divergence Measures and Information Value in Credit Scoring.
#' \emph{Journal of the Operational Research Society}, 64(5), 712-731.
#'
#' @examples
#' # Example using the "he" (Hellinger) distance
#' set.seed(123)
#' feature <- rnorm(1000)
#' target <- rbinom(1000, 1, plogis(feature))
#'
#' result <- ob_numerical_dmiv(feature, target,
#'   min_bins = 3,
#'   max_bins = 5,
#'   divergence_method = "he",
#'   bin_method = "woe"
#' )
#'
#' print(result$bin)
#' print(result$divergence)
#' print(paste("Total Hellinger Distance:", round(result$total_divergence, 4)))
#'
#' @export
ob_numerical_dmiv <- function(feature, target, min_bins = 3, max_bins = 5,
                              bin_cutoff = 0.05, max_n_prebins = 20,
                              is_monotonic = TRUE, convergence_threshold = 1e-6,
                              max_iterations = 1000,
                              bin_method = c("woe1", "woe"),
                              divergence_method = c(
                                "l2", "he", "kl", "tr", "klj",
                                "sc", "js", "l1", "ln"
                              )) {
  # Type Validation and Coercion
  if (!is.numeric(feature)) {
    warning("Feature converted to numeric for processing.")
    feature <- as.numeric(feature)
  }

  if (!is.integer(target)) {
    target <- as.integer(target)
  }

  # Length Check
  if (length(feature) != length(target)) {
    stop("Length of 'feature' and 'target' must match.")
  }

  # Argument Validation for Strings (Safe CRAN Practice)
  # match.arg ensures the user passes a valid string from the allowed list
  bin_method <- match.arg(bin_method)
  divergence_method <- match.arg(divergence_method)

  # .Call Interface
  # C++ signature:
  # (target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins,
  #  is_monotonic, convergence_threshold, max_iterations, bin_method, divergence_method)
  .Call("_OptimalBinningWoE_optimal_binning_numerical_dmiv",
    target,
    feature,
    as.integer(min_bins),
    as.integer(max_bins),
    as.numeric(bin_cutoff),
    as.integer(max_n_prebins),
    as.logical(is_monotonic),
    as.numeric(convergence_threshold),
    as.integer(max_iterations),
    as.character(bin_method),
    as.character(divergence_method),
    PACKAGE = "OptimalBinningWoE"
  )
}
