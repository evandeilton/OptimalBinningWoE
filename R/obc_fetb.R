#' Optimal Binning for Categorical Variables using Fisher's Exact Test
#'
#' Performs supervised discretization of categorical variables using Fisher's
#' Exact Test as the similarity criterion for hierarchical bin merging. This
#' method iteratively merges the most statistically similar bins (highest
#' p-value) while enforcing Weight of Evidence monotonicity, providing a
#' statistically rigorous approach to optimal binning.
#'
#' @param feature A character vector or factor representing the categorical
#'   predictor variable to be binned. Missing values are automatically
#'   converted to the category \code{"NA"}.
#' @param target An integer vector of binary outcomes (0/1) corresponding
#'   to each observation in \code{feature}. Missing values are not permitted.
#' @param min_bins Integer. Minimum number of bins to produce. Must be >= 2.
#'   The algorithm will not merge below this threshold. Defaults to 3.
#' @param max_bins Integer. Maximum number of bins to produce. Must be >=
#'   \code{min_bins}. The algorithm merges bins until this constraint is
#'   satisfied. Defaults to 5.
#' @param bin_cutoff Numeric. Minimum proportion of total observations required
#'   for a category to avoid being classified as rare. Rare categories are
#'   pre-merged before the main algorithm. Must be in (0, 1). Defaults to 0.05.
#' @param max_n_prebins Integer. Maximum number of initial bins before the
#'   merging phase. Controls computational complexity for high-cardinality
#'   features. Must be >= 2. Defaults to 20.
#' @param convergence_threshold Numeric. Convergence tolerance based on
#'   Information Value change between iterations. Algorithm stops when
#'   \eqn{|\Delta IV| <} \code{convergence_threshold}. Must be > 0.
#'   Defaults to 1e-6.
#' @param max_iterations Integer. Maximum number of merge operations allowed.
#'   Prevents excessive computation. Must be > 0. Defaults to 1000.
#' @param bin_separator Character string used to concatenate category names
#'   when multiple categories are merged into a single bin. Defaults to "\%;\%".
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
#'     \item{\code{converged}}{Logical indicating algorithm convergence}
#'     \item{\code{iterations}}{Integer count of merge operations performed}
#'   }
#'
#' @details
#' This algorithm employs Fisher's Exact Test to quantify statistical similarity
#' between bins based on their 2×2 contingency tables. Unlike chi-square based
#' methods, Fisher's test provides exact p-values without relying on asymptotic
#' approximations, making it particularly suitable for small sample sizes.
#'
#' \strong{Algorithm Workflow:}
#' \enumerate{
#'   \item Data preprocessing and frequency computation
#'   \item Rare category identification and pre-merging (frequencies < \code{bin_cutoff})
#'   \item Initial bin creation (one category per bin)
#'   \item Iterative merging phase:
#'     \itemize{
#'       \item Compute Fisher's Exact Test p-values for all adjacent bin pairs
#'       \item Merge the pair with the \strong{highest} p-value (most similar)
#'       \item Enforce WoE monotonicity after each merge
#'       \item Check convergence based on IV change
#'     }
#'   \item Final monotonicity enforcement
#' }
#'
#' \strong{Fisher's Exact Test:}
#'
#' For two bins with contingency table:
#'
#' \tabular{lcc}{
#'   \tab Bin 1 \tab Bin 2 \cr
#'   Positives \tab \eqn{a} \tab \eqn{c} \cr
#'   Negatives \tab \eqn{b} \tab \eqn{d}
#' }
#'
#' The exact probability under the null hypothesis of independence is:
#'
#' \deqn{p = \frac{(a+b)!(c+d)!(a+c)!(b+d)!}{n! \cdot a! \cdot b! \cdot c! \cdot d!}}
#'
#' where \eqn{n = a + b + c + d}. Higher p-values indicate greater similarity
#' (less evidence against the null hypothesis of identical distributions).
#'
#' \strong{Key Features:}
#' \itemize{
#'   \item \strong{Exact inference}: No asymptotic approximations required
#'   \item \strong{Small sample robustness}: Valid for any sample size
#'   \item \strong{Automatic monotonicity}: WoE ordering enforced after each merge
#'   \item \strong{Efficient caching}: Log-factorial and p-value caching for speed
#'   \item \strong{Rare category handling}: Pre-merging prevents sparse bins
#' }
#'
#' \strong{Computational Complexity:}
#' \itemize{
#'   \item Time: \eqn{O(k^2 \cdot m)} where \eqn{k} = initial bins, \eqn{m} = iterations
#'   \item Space: \eqn{O(k + n_{max})} for bins and factorial cache
#' }
#'
#' @references
#' Fisher, R. A. (1922). On the interpretation of chi-squared from contingency tables,
#' and the calculation of P. \emph{Journal of the Royal Statistical Society},
#' 85(1), 87-94. \doi{10.2307/2340521}
#'
#' Agresti, A. (2013). \emph{Categorical Data Analysis} (3rd ed.). Wiley.
#'
#' Mehta, C. R., & Patel, N. R. (1983). A network algorithm for performing
#' Fisher's exact test in r×c contingency tables. \emph{Journal of the
#' American Statistical Association}, 78(382), 427-434.
#'
#' Zeng, G. (2014). A necessary condition for a good binning algorithm in
#' credit scoring. \emph{Applied Mathematical Sciences}, 8(65), 3229-3242.
#'
#' @seealso
#' \code{\link{ob_categorical_cm}} for ChiMerge-based binning,
#' \code{\link{ob_categorical_dp}} for dynamic programming approach,
#' \code{\link{ob_categorical_dmiv}} for divergence measure-based binning
#'
#' @examples
#' \donttest{
#' # Example 1: Basic usage with Fisher's Exact Test
#' set.seed(42)
#' n_obs <- 800
#'
#' # Simulate customer segments with different risk profiles
#' segments <- c("Premium", "Standard", "Basic", "Budget", "Economy")
#' risk_rates <- c(0.05, 0.10, 0.15, 0.22, 0.30)
#'
#' cat_feature <- sample(segments, n_obs,
#'   replace = TRUE,
#'   prob = c(0.15, 0.25, 0.30, 0.20, 0.10)
#' )
#' bin_target <- sapply(cat_feature, function(x) {
#'   rbinom(1, 1, risk_rates[which(segments == x)])
#' })
#'
#' # Apply Fisher's Exact Test binning
#' result_fetb <- ob_categorical_fetb(
#'   cat_feature,
#'   bin_target,
#'   min_bins = 2,
#'   max_bins = 4
#' )
#'
#' # Display results
#' print(data.frame(
#'   Bin = result_fetb$bin,
#'   WoE = round(result_fetb$woe, 3),
#'   IV = round(result_fetb$iv, 4),
#'   Count = result_fetb$count,
#'   EventRate = round(result_fetb$count_pos / result_fetb$count, 3)
#' ))
#'
#' cat("\nAlgorithm converged:", result_fetb$converged, "\n")
#' cat("Iterations performed:", result_fetb$iterations, "\n")
#'
#' # Example 2: Comparing with ChiMerge method
#' result_cm <- ob_categorical_cm(
#'   cat_feature,
#'   bin_target,
#'   min_bins = 2,
#'   max_bins = 4
#' )
#'
#' cat("\nFisher's Exact Test:\n")
#' cat("  Final bins:", length(result_fetb$bin), "\n")
#' cat("  Total IV:", round(sum(result_fetb$iv), 4), "\n")
#'
#' cat("\nChiMerge:\n")
#' cat("  Final bins:", length(result_cm$bin), "\n")
#' cat("  Total IV:", round(sum(result_cm$iv), 4), "\n")
#'
#' # Example 3: Small sample size (Fisher's advantage)
#' set.seed(123)
#' n_obs_small <- 150
#'
#' # Small sample with sparse categories
#' occupation <- c(
#'   "Doctor", "Lawyer", "Teacher", "Engineer",
#'   "Sales", "Manager"
#' )
#'
#' cat_feature_small <- sample(occupation, n_obs_small,
#'   replace = TRUE,
#'   prob = c(0.10, 0.10, 0.20, 0.25, 0.20, 0.15)
#' )
#' bin_target_small <- rbinom(n_obs_small, 1, 0.12)
#'
#' result_fetb_small <- ob_categorical_fetb(
#'   cat_feature_small,
#'   bin_target_small,
#'   min_bins = 2,
#'   max_bins = 3,
#'   bin_cutoff = 0.03 # Allow smaller bins for small sample
#' )
#'
#' cat("\nSmall sample binning:\n")
#' cat("  Observations:", n_obs_small, "\n")
#' cat("  Original categories:", length(unique(cat_feature_small)), "\n")
#' cat("  Final bins:", length(result_fetb_small$bin), "\n")
#' cat("  Converged:", result_fetb_small$converged, "\n")
#'
#' # Example 4: High cardinality with rare categories
#' set.seed(789)
#' n_obs_hc <- 2000
#'
#' # Simulate product codes (high cardinality)
#' product_codes <- paste0("PROD_", sprintf("%03d", 1:30))
#'
#' cat_feature_hc <- sample(product_codes, n_obs_hc,
#'   replace = TRUE,
#'   prob = c(
#'     rep(0.05, 10), rep(0.02, 10),
#'     rep(0.01, 10)
#'   )
#' )
#' bin_target_hc <- rbinom(n_obs_hc, 1, 0.08)
#'
#' result_fetb_hc <- ob_categorical_fetb(
#'   cat_feature_hc,
#'   bin_target_hc,
#'   min_bins = 3,
#'   max_bins = 6,
#'   bin_cutoff = 0.02,
#'   max_n_prebins = 15
#' )
#'
#' cat("\nHigh cardinality example:\n")
#' cat("  Original categories:", length(unique(cat_feature_hc)), "\n")
#' cat("  Final bins:", length(result_fetb_hc$bin), "\n")
#' cat("  Iterations:", result_fetb_hc$iterations, "\n")
#'
#' # Check for rare category merging
#' for (i in seq_along(result_fetb_hc$bin)) {
#'   n_merged <- length(strsplit(result_fetb_hc$bin[i], "%;%")[[1]])
#'   if (n_merged > 1) {
#'     cat("  Bin", i, "contains", n_merged, "merged categories\n")
#'   }
#' }
#'
#' # Example 5: Missing value handling
#' set.seed(456)
#' cat_feature_na <- cat_feature
#' na_indices <- sample(n_obs, 40) # 5% missing
#' cat_feature_na[na_indices] <- NA
#'
#' result_fetb_na <- ob_categorical_fetb(
#'   cat_feature_na,
#'   bin_target,
#'   min_bins = 2,
#'   max_bins = 4
#' )
#'
#' # Check NA treatment
#' na_bin_idx <- grep("NA", result_fetb_na$bin)
#' if (length(na_bin_idx) > 0) {
#'   cat("\nMissing value handling:\n")
#'   cat("  NA bin:", result_fetb_na$bin[na_bin_idx], "\n")
#'   cat("  NA count:", result_fetb_na$count[na_bin_idx], "\n")
#'   cat("  NA WoE:", round(result_fetb_na$woe[na_bin_idx], 3), "\n")
#' }
#' }
#'
#' @export
ob_categorical_fetb <- function(feature, target,
                                min_bins = 3,
                                max_bins = 5,
                                bin_cutoff = 0.05,
                                max_n_prebins = 20,
                                convergence_threshold = 1e-6,
                                max_iterations = 1000,
                                bin_separator = "%;%") {
  # Input preprocessing
  if (!is.character(feature)) {
    feature <- as.character(feature)
  }
  feature[is.na(feature)] <- "NA"
  target <- as.integer(target)

  # Invoke C++ implementation
  .Call("_OptimalBinningWoE_optimal_binning_categorical_fetb",
    target,
    feature,
    as.integer(min_bins),
    as.integer(max_bins),
    bin_cutoff,
    as.integer(max_n_prebins),
    convergence_threshold,
    as.integer(max_iterations),
    bin_separator,
    PACKAGE = "OptimalBinningWoE"
  )
}
