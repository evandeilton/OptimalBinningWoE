#' Optimal Binning for Categorical Variables using Monotonic Optimal Binning (MOB)
#'
#' This function performs optimal binning for categorical variables using the
#' Monotonic Optimal Binning (MOB) algorithm. It creates bins that maintain
#' monotonic Weight of Evidence (WoE) trends while maximizing Information Value.
#'
#' The MOB algorithm follows these steps:
#' \enumerate{
#'   \item Initial sorting: Categories are ordered by their individual WoE values
#'   \item Rare category handling: Categories below \code{bin_cutoff} frequency
#'         are merged with similar ones
#'   \item Pre-binning limitation: Reduces initial bins to \code{max_n_prebins}
#'         using similarity-based merging
#'   \item Monotonicity enforcement: Ensures WoE is either consistently
#'         increasing or decreasing across bins
#'   \item Bin count optimization: Adjusts to meet \code{min_bins}/\code{max_bins}
#'         constraints
#' }
#'
#' Key features include:
#' \itemize{
#'   \item Automatic sorting of categories by WoE for initial structure
#'   \item Bayesian smoothing to stabilize WoE estimates for sparse categories
#'   \item Guaranteed monotonic WoE trend across final bins
#'   \item Configurable minimum and maximum bin counts
#'   \item Similarity-based merging for optimal bin combinations
#' }
#'
#' Mathematical definitions:
#' \deqn{WoE_i = \ln\left(\frac{p_i^{(1)}}{p_i^{(0)}}\right)}{
#' WoE_i = ln((p_i^(1))/(p_i^(0)))}
#' where \eqn{p_i^{(1)}}{p_i^(1)} and \eqn{p_i^{(0)}}{p_i^(0)} are the
#' proportions of positive and negative cases in bin \eqn{i}, respectively,
#' adjusted using Bayesian smoothing.
#'
#' \deqn{IV = \sum_{i=1}^{n} (p_i^{(1)} - p_i^{(0)}) \times WoE_i}{
#' IV = sum((p_i^(1) - p_i^(0)) * WoE_i)}
#'
#' @param feature A character vector or factor representing the categorical
#'   predictor variable. Missing values (NA) will be converted to the string
#'   "NA" and treated as a separate category.
#' @param target An integer vector containing binary outcome values (0 or 1).
#'   Must be the same length as \code{feature}. Cannot contain missing values.
#' @param min_bins Integer. Minimum number of bins to create. Must be at least
#'   1. Default is 3.
#' @param max_bins Integer. Maximum number of bins to create. Must be greater
#'   than or equal to \code{min_bins}. Default is 5.
#' @param bin_cutoff Numeric. Minimum relative frequency threshold for
#'   individual categories. Categories with frequency below this proportion
#'   will be merged with others. Value must be between 0 and 1. Default is
#'   0.05 (5\%).
#' @param max_n_prebins Integer. Maximum number of initial bins before
#'   optimization. Used to control computational complexity when dealing with
#'   high-cardinality categorical variables. Default is 20.
#' @param bin_separator Character string used to separate category names when
#'   multiple categories are merged into a single bin. Default is "\%;\%".
#' @param convergence_threshold Numeric. Threshold for determining algorithm
#'   convergence based on changes in total Information Value. Must be positive.
#'   Default is 1e-6.
#' @param max_iterations Integer. Maximum number of iterations for the
#'   optimization process. Must be positive. Default is 1000.
#'
#' @return A list containing the results of the optimal binning procedure:
#' \describe{
#'   \item{\code{id}}{Numeric vector of bin identifiers (1 to n_bins)}
#'   \item{\code{bin}}{Character vector of bin labels, which are combinations
#'         of original categories separated by \code{bin_separator}}
#'   \item{\code{woe}}{Numeric vector of Weight of Evidence values for each bin}
#'   \item{\code{iv}}{Numeric vector of Information Values for each bin}
#'   \item{\code{count}}{Integer vector of total observations in each bin}
#'   \item{\code{count_pos}}{Integer vector of positive outcomes in each bin}
#'   \item{\code{count_neg}}{Integer vector of negative outcomes in each bin}
#'   \item{\code{total_iv}}{Numeric scalar. Total Information Value across all
#'         bins}
#'   \item{\code{converged}}{Logical. Whether the algorithm converged within
#'         the specified tolerance}
#'   \item{\code{iterations}}{Integer. Number of iterations performed}
#' }
#'
#' @note
#' \itemize{
#'   \item Target variable must contain both 0 and 1 values.
#'   \item Empty strings in the feature vector are not allowed and will cause
#'         an error.
#'   \item For datasets with very few observations in either class (<5),
#'         warnings will be issued as results may be unstable.
#'   \item The algorithm guarantees monotonic WoE across bins.
#'   \item When the number of unique categories is less than \code{max_bins},
#'         each category will form its own bin.
#' }
#'
#' @examples
#' # Generate sample data
#' set.seed(123)
#' n <- 1000
#' feature <- sample(letters[1:8], n, replace = TRUE)
#' target <- rbinom(n, 1, prob = ifelse(feature %in% c("a", "b"), 0.7, 0.3))
#'
#' # Perform optimal binning
#' result <- ob_categorical_mob(feature, target)
#' print(result[c("bin", "woe", "iv", "count")])
#'
#' # With custom parameters
#' result2 <- ob_categorical_mob(
#'   feature = feature,
#'   target = target,
#'   min_bins = 2,
#'   max_bins = 4,
#'   bin_cutoff = 0.03
#' )
#'
#' # Handling missing values
#' feature_with_na <- feature
#' feature_with_na[sample(length(feature_with_na), 50)] <- NA
#' result3 <- ob_categorical_mob(feature_with_na, target)
#'
#' @export
ob_categorical_mob <- function(feature,
                               target,
                               min_bins = 3L,
                               max_bins = 5L,
                               bin_cutoff = 0.05,
                               max_n_prebins = 20L,
                               bin_separator = "%;%",
                               convergence_threshold = 1e-6,
                               max_iterations = 1000L) {
  # Input validation and conversion
  if (!is.character(feature)) {
    feature <- as.character(feature)
  }

  # Convert NA values to "NA" string
  feature[is.na(feature)] <- "NA"
  target <- as.integer(target)

  # Call the C++ implementation
  .Call("_OptimalBinningWoE_optimal_binning_categorical_mob",
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
