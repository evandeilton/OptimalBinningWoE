#' Optimal Binning for Numerical Variables using Enhanced ChiMerge Algorithm
#'
#' Performs supervised discretization of continuous numerical variables using the
#' ChiMerge algorithm (Kerber, 1992) or the Chi2 algorithm (Liu & Setiono, 1995).
#' This function merges adjacent bins based on Chi-square statistics to maximize
#' the discrimination of the binary target variable while ensuring monotonicity
#' and statistical robustness.
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
#' @param max_n_prebins Integer. The number of initial bins created during the
#'   pre-binning phase before the merging process begins. Higher values provide
#'   more granular starting points. Must be \eqn{\ge} \code{max_bins}.
#'   Defaults to 20.
#' @param convergence_threshold Numeric. The threshold for the change in total
#'   IV to determine convergence during the iterative merging process.
#'   Defaults to 1e-6.
#' @param max_iterations Integer. Safety limit for the maximum number of merging
#'   iterations. Defaults to 1000.
#' @param init_method Character string specifying the initialization method.
#'   Options are \code{"equal_frequency"} (quantile-based) or \code{"equal_width"}.
#'   Defaults to \code{"equal_frequency"}.
#' @param chi_merge_threshold Numeric in (0, 1). The significance level
#'   (\eqn{\alpha}) of the Chi-square test; the critical value is
#'   \code{qchisq(1 - chi_merge_threshold, 1)}. \strong{It does not change the
#'   result of this function}: the initial bins are merged, smallest chi-square
#'   first, until at most \code{max_bins} remain, before the significance test
#'   is ever reached, so the test never has a merge left to decide (see
#'   Details). Control the number of bins with \code{max_bins}. The categorical
#'   \code{\link{ob_categorical_cm}} does use the level as its stopping rule.
#'   Defaults to 0.05.
#' @param use_chi2_algorithm Logical. If \code{TRUE}, runs the Chi2 variant, a
#'   schedule of decreasing significance levels (0.5, 0.1, 0.05, 0.01, 0.005,
#'   0.001). For the reason given for \code{chi_merge_threshold} the schedule
#'   merges nothing beyond \code{max_bins}; the bins match the default path
#'   except that the \code{bin_cutoff} rare-bin pass is skipped.
#'   Defaults to \code{FALSE}.
#'
#' @return A list containing the binning results:
#'   \itemize{
#'     \item \code{id}: Integer vector of bin identifiers (1 to k).
#'     \item \code{bin}: Character vector of bin labels in interval notation.
#'     \item \code{woe}: Numeric vector of Weight of Evidence for each bin.
#'     \item \code{iv}: Numeric vector of Information Value contribution per bin.
#'     \item \code{count}: Integer vector of total observations per bin.
#'     \item \code{count_pos}: Integer vector of positive cases (target=1).
#'     \item \code{count_neg}: Integer vector of negative cases (target=0).
#'     \item \code{cutpoints}: Numeric vector of upper boundaries (excluding Inf).
#'     \item \code{converged}: Logical indicating if the algorithm converged.
#'     \item \code{iterations}: Integer count of iterations performed.
#'     \item \code{total_iv}: The total Information Value of the binned variable.
#'     \item \code{algorithm}: String identifying the algorithm used ("ChiMerge" or "Chi2").
#'     \item \code{monotonic}: Logical indicating if the final WoE trend is monotonic.
#'   }
#'
#' @details
#' The function implements two major discretization strategies:
#'
#' \enumerate{
#'   \item \strong{Standard ChiMerge:}
#'   \itemize{
#'     \item Initializes bins using \code{init_method}.
#'     \item Iteratively merges the adjacent pair with the lowest \eqn{\chi^2}
#'           statistic (2x2 table, Yates continuity correction clamped at zero).
#'     \item Merging continues until the number of bins reaches \code{max_bins}.
#'           The significance test of Kerber (1992) is never reached: merging
#'           stops at \code{max_bins} first (see below).
#'   }
#'   \item \strong{Chi2 Algorithm:}
#'   \itemize{
#'     \item Activated when \code{use_chi2_algorithm = TRUE}.
#'     \item Loops over a schedule of decreasing significance levels
#'           (0.5 \eqn{\to}{->} 0.001) and checks the inconsistency rate of the data.
#'   }
#' }
#'
#' \strong{The significance level does not decide the bin count.} Before either
#' method starts, the initial bins are merged, smallest \eqn{\chi^2} first,
#' until at most \code{max_bins} remain. The significance tests never run,
#' because no merge is left to decide once \code{max_bins} is met, so the number
#' of bins is set by \code{max_bins} (and by \code{min_bins}, \code{bin_cutoff}
#' and monotonicity), not by \code{chi_merge_threshold}. The categorical
#' \code{\link{ob_categorical_cm}} differs: it merges down to \code{max_bins} and
#' then keeps merging while pairs are not significantly different at the chosen
#' level.
#'
#' Post-processing steps enforce:
#' \itemize{
#'   \item \strong{Minimum Bin Size:} Merging rare bins smaller than
#'         \code{bin_cutoff} (standard ChiMerge path only).
#'   \item \strong{Monotonicity:} Ensuring WoE trend is strictly increasing or decreasing
#'         to improve model interpretability.
#' }
#' Monotonicity merging stops at \code{min_bins}; the rare-bin pass does not
#' check it again between merges, so a large \code{bin_cutoff} (for example 0.3)
#' can leave fewer than \code{min_bins} bins.
#'
#' @references
#' Kerber, R. (1992). ChiMerge: Discretization of numeric attributes.
#' \emph{Proceedings of the Tenth National Conference on Artificial Intelligence}, 123-128.
#'
#' Liu, H., & Setiono, R. (1995). Chi2: Feature selection and discretization of
#' numeric attributes. \emph{Tools with Artificial Intelligence}, 388-391.
#'
#' @examples
#' # Example 1: Standard ChiMerge
#' set.seed(123)
#' feature <- rnorm(1000)
#' # Create a target with a relationship to the feature
#' target <- rbinom(1000, 1, plogis(2 * feature))
#'
#' res_cm <- ob_numerical_cm(feature, target,
#'   min_bins = 3,
#'   max_bins = 6,
#'   init_method = "equal_frequency"
#' )
#'
#' print(res_cm$bin)
#' print(res_cm$iv)
#'
#' # Example 2: Using the Chi2 Algorithm variant
#' res_chi2 <- ob_numerical_cm(feature, target,
#'   min_bins = 3,
#'   max_bins = 6,
#'   use_chi2_algorithm = TRUE
#' )
#'
#' cat("Total IV (ChiMerge):", res_cm$total_iv, "\n")
#' cat("Total IV (Chi2):", res_chi2$total_iv, "\n")
#'
#' @export
ob_numerical_cm <- function(feature, target, min_bins = 3, max_bins = 5,
                            bin_cutoff = 0.05, max_n_prebins = 20,
                            convergence_threshold = 1e-6, max_iterations = 1000,
                            init_method = "equal_frequency",
                            chi_merge_threshold = 0.05,
                            use_chi2_algorithm = FALSE) {
  # Type Coercion & Validation
  if (!is.numeric(feature)) {
    warning("Feature converted to numeric for processing.")
    feature <- as.numeric(feature)
  }

  if (!is.integer(target)) {
    target <- .ob_integer_target(target)
  }

  # Input Dimension Check
  if (length(feature) != length(target)) {
    stop("Length of 'feature' and 'target' must match.")
  }

  # Ensure method is valid before C++ to prevent cryptic errors
  if (!init_method %in% c("equal_frequency", "equal_width")) {
    warning("Invalid 'init_method'. Defaulting to 'equal_frequency'.")
    init_method <- "equal_frequency"
  }

  # Call C++ implementation
  # NOTE: The C++ signature expects 'target' first, then 'feature'.
  .Call("_OptimalBinningWoE_optimal_binning_numerical_cm",
    target,
    feature,
    as.integer(min_bins),
    as.integer(max_bins),
    as.numeric(bin_cutoff),
    as.integer(max_n_prebins),
    as.numeric(convergence_threshold),
    as.integer(max_iterations),
    as.character(init_method),
    as.numeric(chi_merge_threshold),
    as.logical(use_chi2_algorithm),
    PACKAGE = "OptimalBinningWoE"
  )
}
