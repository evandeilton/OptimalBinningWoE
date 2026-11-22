#' Optimal Binning using Metric Divergence Measures (Zeng, 2013)
#'
#' Performs supervised discretization of continuous numerical variables using the
#' theoretical framework proposed by Zeng (2013). This method creates bins that
#' maximize a specified divergence measure (e.g., Kullback-Leibler, Hellinger)
#' between the distributions of positive and negative cases, effectively maximizing
#' the Information Value (IV) or other discriminatory statistics.
#'
#' @param feature A numeric vector representing the continuous predictor variable.
#'   Missing values (\code{NA}/\code{NaN}) are excluded from the fit;
#'   \code{-Inf}/\code{Inf} fall in the first/last bin.
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
#' @param convergence_threshold Numeric. Accepted for backward compatibility and not
#'   used: merging always continues until the number of bins reaches
#'   \code{max_bins} (or \code{max_iterations} is exhausted). Defaults to 1e-6.
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
#'     \item \code{iv}: Numeric vector of standard Information Value per bin.
#'     \item \code{divergence}: Numeric vector of the chosen divergence contribution per bin.
#'     \item \code{count}: Integer vector of total observations per bin.
#'     \item \code{count_pos}: Integer vector of positive cases.
#'     \item \code{count_neg}: Integer vector of negative cases.
#'     \item \code{cutpoints}: Numeric vector of upper boundaries (excluding Inf).
#'     \item \code{converged}: Logical indicating if the algorithm converged.
#'     \item \code{iterations}: Integer count of merging iterations performed.
#'     \item \code{total_divergence}: The sum of the divergence measure across all bins.
#'     \item \code{total_iv}: The total standard Information Value.
#'     \item \code{bin_method}: The WoE calculation method used.
#'     \item \code{divergence_method}: The divergence measure used.
#'   }
#'
#' @details
#' This algorithm implements the "Metric Divergence Measures" framework. Unlike
#' standard ChiMerge which uses statistical significance, this method merges
#' adjacent bins greedily to minimize the loss of a specific divergence metric
#' when merging bins.
#'
#' \strong{The Process:}
#' \enumerate{
#'   \item \strong{Pre-binning:} Generates granular bins based on quantiles.
#'   \item \strong{Rare Merging:} Merges bins smaller than \code{bin_cutoff}.
#'   \item \strong{Pure bins:} A bin with no events (or no non-events) stays
#'         isolated only if its purity is significant at the 5\% level under the
#'         overall event rate \eqn{r_0}: \eqn{(1 - r_0)^n \le 0.05} for a bin of
#'         \eqn{n} observations without events (\eqn{r_0^n \le 0.05} without
#'         non-events). Otherwise it is merged with the neighbour whose merge keeps
#'         the larger total divergence, while more than \code{min_bins} bins remain.
#'   \item \strong{Monotonicity:} If \code{is_monotonic = TRUE}, forces the WoE trend
#'         to be monotonic by merging "violating" bins in the direction that
#'         maximizes the total divergence.
#'   \item \strong{Optimization:} Iteratively merges a pair of adjacent bins until
#'         \code{max_bins} is reached. For the divergences \code{"he"},
#'         \code{"kl"}, \code{"tr"}, \code{"klj"}, \code{"sc"} and \code{"js"}
#'         it is the pair whose merge keeps the largest total of the selected
#'         divergence. The distances \code{"l1"}, \code{"l2"} and \code{"ln"}
#'         are norms of the per-bin differences \eqn{p_i - q_i}, and merging two
#'         bins on the same side of \eqn{p = q} never lowers them (L1 and
#'         L-infinity are unchanged, L2 grows), so the distance cannot rank those
#'         merges: among such pairs the engine merges the one whose event rates
#'         are least distinguishable (smallest Pearson chi-square of the
#'         \eqn{2 \times 2}{2 x 2} table), and the selected distance decides only when
#'         every adjacent pair lies on opposite sides of \eqn{p = q}.
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
    target <- .ob_integer_target(target)
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
