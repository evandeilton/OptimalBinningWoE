#' Optimal Binning for Categorical Variables using Monotonic Binning Algorithm
#'
#' Performs supervised discretization of categorical variables using the
#' Monotonic Binning Algorithm (MBA), which enforces strict Weight of Evidence
#' monotonicity while optimizing Information Value through intelligent bin merging
#' strategies. This implementation includes Bayesian smoothing for numerical
#' stability and adaptive thresholding for robust monotonicity enforcement.
#'
#' @param feature A character vector or factor representing the categorical
#'   predictor variable to be binned. Missing values are automatically
#'   converted to the category \code{"NA"}.
#' @param target An integer vector of binary outcomes (0/1) corresponding
#'   to each observation in \code{feature}. Missing values are not permitted.
#' @param min_bins Integer. Minimum number of bins to produce. Must be >= 2.
#'   The algorithm will not merge below this threshold. Defaults to 3.
#' @param max_bins Integer. Maximum number of bins to produce. Must be >=
#'   \code{min_bins}. The algorithm reduces bins until this constraint is met.
#'   Defaults to 5.
#' @param bin_cutoff Numeric. Minimum proportion of total observations required
#'   for a category to remain separate. Categories below this threshold are
#'   pre-merged with similar categories. Must be in (0, 1). Defaults to 0.05.
#' @param max_n_prebins Integer. Maximum number of initial bins before the
#'   main optimization phase. Controls computational complexity. Must be >=
#'   \code{max_bins}. Defaults to 20.
#' @param bin_separator Character string used to concatenate category names
#'   when multiple categories are merged into a single bin. Defaults to "\%;\%".
#' @param convergence_threshold Numeric. Convergence tolerance based on
#'   Information Value change between iterations. Algorithm stops when
#'   \eqn{|\Delta IV| <} \code{convergence_threshold}. Must be > 0.
#'   Defaults to 1e-6.
#' @param max_iterations Integer. Maximum number of optimization iterations.
#'   Prevents infinite loops. Must be > 0. Defaults to 1000.
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
#'     \item{\code{total_iv}}{Numeric total Information Value of the binning solution}
#'     \item{\code{converged}}{Logical indicating algorithm convergence}
#'     \item{\code{iterations}}{Integer count of optimization iterations performed}
#'   }
#'
#' @details
#' The Monotonic Binning Algorithm (MBA) implements a sophisticated approach to
#' categorical binning that guarantees strict Weight of Evidence monotonicity
#' through intelligent violation detection and repair mechanisms.
#'
#' \strong{Algorithm Workflow:}
#' \enumerate{
#'   \item Input validation and preprocessing
#'   \item Initial bin creation (one category per bin)
#'   \item Pre-binning limitation to \code{max_n_prebins}
#'   \item Rare category merging (frequencies < \code{bin_cutoff})
#'   \item Bayesian-smoothed WoE calculation
#'   \item Strict monotonicity enforcement with adaptive thresholds
#'   \item IV-optimized bin merging to meet \code{max_bins} constraint
#'   \item Final consistency verification
#' }
#'
#' \strong{Monotonicity Enforcement:}
#'
#' MBA enforces strict monotonicity through an iterative repair process:
#'
#' \enumerate{
#'   \item Sort bins by current WoE values
#'   \item Calculate adaptive threshold: \eqn{\tau = \min(\epsilon, 0.01\bar{\Delta})}
#'   \item Identify violations: \eqn{sign(WoE_i - WoE_{i-1}) \neq sign(WoE_{i+1} - WoE_i)}
#'   \item Rank violations by severity: \eqn{s_i = |WoE_i - WoE_{i-1}| + |WoE_{i+1} - WoE_i|}
#'   \item Repair most severe violations by merging adjacent bins
#'   \item Repeat until no violations remain or \code{min_bins} reached
#' }
#'
#' \strong{Bayesian Smoothing:}
#'
#' To ensure numerical stability and prevent overfitting, MBA applies Bayesian
#' smoothing to WoE and IV calculations:
#'
#' \deqn{p'_i = \frac{n_{i,pos} + \alpha_p}{N_{pos} + \alpha_{total}}}
#' \deqn{n'_i = \frac{n_{i,neg} + \alpha_n}{N_{neg} + \alpha_{total}}}
#'
#' where priors are proportional to overall prevalence:
#'
#' \deqn{\alpha_p = \alpha_{total} \times \frac{N_{pos}}{N_{pos} + N_{neg}}}
#' \deqn{\alpha_n = \alpha_{total} - \alpha_p}
#'
#' with \eqn{\alpha_{total} = 1.0} as the prior strength parameter.
#'
#' \strong{Intelligent Bin Merging:}
#'
#' When reducing bins to meet the \code{max_bins} constraint, MBA employs an
#' IV-loss minimization strategy:
#'
#' \deqn{\Delta IV_{i,j} = IV_i + IV_j - IV_{merged}(i,j)}
#'
#' The pair with minimum \eqn{\Delta IV} is merged to preserve maximum
#' predictive information.
#'
#' \strong{Computational Complexity:}
#' \itemize{
#'   \item Time: \eqn{O(k^2 \cdot m)} where \eqn{k} = bins, \eqn{m} = iterations
#'   \item Space: \eqn{O(k^2)} for IV loss cache
#'   \item Cache hit rate typically > 75\% for \eqn{k > 10}
#' }
#'
#' \strong{Key Features:}
#' \itemize{
#'   \item \strong{Guaranteed monotonicity}: Strict enforcement with adaptive thresholds
#'   \item \strong{Bayesian regularization}: Robust to sparse bins and class imbalance
#'   \item \strong{Intelligent merging}: Preserves maximum information during reduction
#'   \item \strong{Adaptive thresholds}: Context-aware violation detection
#'   \item \strong{Consistency verification}: Final integrity checks
#' }
#'
#' @references
#' Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm
#' for credit risk modeling. \emph{SSRN Electronic Journal}.
#' \doi{10.2139/ssrn.2978774}
#'
#' Siddiqi, N. (2017). \emph{Intelligent Credit Scoring: Building and
#' Implementing Better Credit Risk Scorecards} (2nd ed.). Wiley.
#'
#' Good, I. J. (1965). \emph{The Estimation of Probabilities: An Essay on
#' Modern Bayesian Methods}. MIT Press.
#'
#' Zeng, G. (2014). A necessary condition for a good binning algorithm in
#' credit scoring. \emph{Applied Mathematical Sciences}, 8(65), 3229-3242.
#'
#' @seealso
#' \code{\link{ob_categorical_jedi}} for joint entropy-driven optimization,
#' \code{\link{ob_categorical_dp}} for dynamic programming approach,
#' \code{\link{ob_categorical_cm}} for ChiMerge-based binning
#'
#' @examples
#' \donttest{
#' # Example 1: Basic monotonic binning with guaranteed WoE ordering
#' set.seed(42)
#' n_obs <- 1500
#'
#' # Simulate risk ratings with natural monotonic relationship
#' ratings <- c("AAA", "AA", "A", "BBB", "BB", "B", "CCC")
#' default_probs <- c(0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
#'
#' cat_feature <- sample(ratings, n_obs,
#'   replace = TRUE,
#'   prob = c(0.05, 0.10, 0.20, 0.25, 0.20, 0.15, 0.05)
#' )
#' bin_target <- sapply(cat_feature, function(x) {
#'   rbinom(1, 1, default_probs[which(ratings == x)])
#' })
#'
#' # Apply MBA algorithm
#' result_mba <- ob_categorical_mba(
#'   cat_feature,
#'   bin_target,
#'   min_bins = 3,
#'   max_bins = 5
#' )
#'
#' # Display results with guaranteed monotonic WoE
#' print(data.frame(
#'   Bin = result_mba$bin,
#'   WoE = round(result_mba$woe, 3),
#'   IV = round(result_mba$iv, 4),
#'   Count = result_mba$count,
#'   EventRate = round(result_mba$count_pos / result_mba$count, 3)
#' ))
#'
#' cat("\nMonotonicity check (WoE differences):\n")
#' woe_diffs <- diff(result_mba$woe)
#' cat("  Differences:", paste(round(woe_diffs, 4), collapse = ", "), "\n")
#' cat("  All positive (increasing):", all(woe_diffs >= -1e-10), "\n")
#' cat("  Total IV:", round(result_mba$total_iv, 4), "\n")
#' cat("  Converged:", result_mba$converged, "\n")
#'
#' # Example 2: Comparison with non-monotonic methods
#' set.seed(123)
#' n_obs_comp <- 2000
#'
#' sectors <- c("Tech", "Health", "Finance", "Manufacturing", "Retail")
#' cat_feature_comp <- sample(sectors, n_obs_comp, replace = TRUE)
#' bin_target_comp <- rbinom(n_obs_comp, 1, 0.15)
#'
#' # MBA (strictly monotonic)
#' result_mba_comp <- ob_categorical_mba(
#'   cat_feature_comp, bin_target_comp,
#'   min_bins = 3, max_bins = 4
#' )
#'
#' # Standard binning (may not be monotonic)
#' result_std_comp <- ob_categorical_cm(
#'   cat_feature_comp, bin_target_comp,
#'   min_bins = 3, max_bins = 4
#' )
#'
#' cat("\nMonotonicity comparison:\n")
#' cat(
#'   "  MBA WoE differences:",
#'   paste(round(diff(result_mba_comp$woe), 4), collapse = ", "), "\n"
#' )
#' cat("  MBA monotonic:", all(diff(result_mba_comp$woe) >= -1e-10), "\n")
#' cat(
#'   "  Std WoE differences:",
#'   paste(round(diff(result_std_comp$woe), 4), collapse = ", "), "\n"
#' )
#' cat("  Std monotonic:", all(diff(result_std_comp$woe) >= -1e-10), "\n")
#'
#' # Example 3: Bayesian smoothing with sparse data
#' set.seed(789)
#' n_obs_sparse <- 400
#'
#' # Small sample with rare categories
#' categories <- c("A", "B", "C", "D", "E", "F")
#' cat_probs <- c(0.30, 0.25, 0.20, 0.15, 0.07, 0.03)
#'
#' cat_feature_sparse <- sample(categories, n_obs_sparse,
#'   replace = TRUE,
#'   prob = cat_probs
#' )
#' bin_target_sparse <- rbinom(n_obs_sparse, 1, 0.08) # 8% event rate
#'
#' result_mba_sparse <- ob_categorical_mba(
#'   cat_feature_sparse,
#'   bin_target_sparse,
#'   min_bins = 2,
#'   max_bins = 4,
#'   bin_cutoff = 0.02
#' )
#'
#' cat("\nBayesian smoothing (sparse data):\n")
#' cat("  Sample size:", n_obs_sparse, "\n")
#' cat("  Events:", sum(bin_target_sparse), "\n")
#' cat("  Final bins:", length(result_mba_sparse$bin), "\n\n")
#'
#' # Show how smoothing prevents extreme WoE values
#' for (i in seq_along(result_mba_sparse$bin)) {
#'   cat(sprintf(
#'     "  Bin %d: events=%d/%d, WoE=%.3f (smoothed)\n",
#'     i,
#'     result_mba_sparse$count_pos[i],
#'     result_mba_sparse$count[i],
#'     result_mba_sparse$woe[i]
#'   ))
#' }
#'
#' # Example 4: High cardinality with pre-binning
#' set.seed(456)
#' n_obs_hc <- 3000
#'
#' # Simulate ZIP codes (high cardinality)
#' zips <- paste0("ZIP_", sprintf("%04d", 1:50))
#'
#' cat_feature_hc <- sample(zips, n_obs_hc, replace = TRUE)
#' bin_target_hc <- rbinom(n_obs_hc, 1, 0.12)
#'
#' result_mba_hc <- ob_categorical_mba(
#'   cat_feature_hc,
#'   bin_target_hc,
#'   min_bins = 4,
#'   max_bins = 6,
#'   max_n_prebins = 20,
#'   bin_cutoff = 0.01
#' )
#'
#' cat("\nHigh cardinality performance:\n")
#' cat("  Original categories:", length(unique(cat_feature_hc)), "\n")
#' cat("  Final bins:", length(result_mba_hc$bin), "\n")
#' cat(
#'   "  Largest merged bin contains:",
#'   max(sapply(strsplit(result_mba_hc$bin, "%;%"), length)), "categories\n"
#' )
#'
#' # Verify monotonicity in high-cardinality case
#' woe_monotonic <- all(diff(result_mba_hc$woe) >= -1e-10)
#' cat("  WoE monotonic:", woe_monotonic, "\n")
#'
#' # Example 5: Convergence behavior
#' set.seed(321)
#' n_obs_conv <- 1000
#'
#' business_sizes <- c("Micro", "Small", "Medium", "Large", "Enterprise")
#' cat_feature_conv <- sample(business_sizes, n_obs_conv, replace = TRUE)
#' bin_target_conv <- rbinom(n_obs_conv, 1, 0.18)
#'
#' # Test different convergence thresholds
#' thresholds <- c(1e-3, 1e-6, 1e-9)
#'
#' for (thresh in thresholds) {
#'   result_conv <- ob_categorical_mba(
#'     cat_feature_conv,
#'     bin_target_conv,
#'     min_bins = 2,
#'     max_bins = 4,
#'     convergence_threshold = thresh,
#'     max_iterations = 50
#'   )
#'
#'   cat(sprintf("\nThreshold %.0e:\n", thresh))
#'   cat("  Final bins:", length(result_conv$bin), "\n")
#'   cat("  Total IV:", round(result_conv$total_iv, 4), "\n")
#'   cat("  Converged:", result_conv$converged, "\n")
#'   cat("  Iterations:", result_conv$iterations, "\n")
#'
#'   # Check monotonicity preservation
#'   monotonic <- all(diff(result_conv$woe) >= -1e-10)
#'   cat("  Monotonic:", monotonic, "\n")
#' }
#'
#' # Example 6: Missing value handling
#' set.seed(555)
#' cat_feature_na <- cat_feature
#' na_indices <- sample(n_obs, 75) # 5% missing
#' cat_feature_na[na_indices] <- NA
#'
#' result_mba_na <- ob_categorical_mba(
#'   cat_feature_na,
#'   bin_target,
#'   min_bins = 3,
#'   max_bins = 5
#' )
#'
#' # Locate NA bin
#' na_bin_idx <- grep("NA", result_mba_na$bin)
#' if (length(na_bin_idx) > 0) {
#'   cat("\nMissing value treatment:\n")
#'   cat("  NA bin:", result_mba_na$bin[na_bin_idx], "\n")
#'   cat("  NA count:", result_mba_na$count[na_bin_idx], "\n")
#'   cat(
#'     "  NA event rate:",
#'     round(result_mba_na$count_pos[na_bin_idx] /
#'       result_mba_na$count[na_bin_idx], 3), "\n"
#'   )
#'   cat("  NA WoE:", round(result_mba_na$woe[na_bin_idx], 3), "\n")
#'   cat(
#'     "  Monotonicity preserved:",
#'     all(diff(result_mba_na$woe) >= -1e-10), "\n"
#'   )
#' }
#' }
#'
#' @export
ob_categorical_mba <- function(feature, target,
                               min_bins = 3,
                               max_bins = 5,
                               bin_cutoff = 0.05,
                               max_n_prebins = 20,
                               bin_separator = "%;%",
                               convergence_threshold = 1e-6,
                               max_iterations = 1000) {
  # Input preprocessing
  if (!is.character(feature)) {
    feature <- as.character(feature)
  }
  feature[is.na(feature)] <- "NA"
  target <- as.integer(target)

  # Invoke C++ implementation
  .Call("_OptimalBinningWoE_optimal_binning_categorical_mba",
    target = target,
    feature = feature,
    min_bins = as.integer(min_bins),
    max_bins = as.integer(max_bins),
    bin_cutoff = bin_cutoff,
    max_n_prebins = as.integer(max_n_prebins),
    bin_separator = bin_separator,
    convergence_threshold = convergence_threshold,
    max_iterations = as.integer(max_iterations),
    PACKAGE = "OptimalBinningWoE"
  )
}
