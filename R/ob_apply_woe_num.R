#' @title Apply Optimal Weight of Evidence (WoE) to a Numerical Feature
#'
#' @description
#' Transforms a numerical feature into its corresponding Weight of Evidence (WoE)
#' values using pre-computed binning results from an optimal binning algorithm
#' (e.g., \code{\link{ob_numerical_mdlp}}, \code{\link{ob_numerical_mob}}).
#'
#' @param obresults List output from an optimal binning function for numerical
#'   variables. Must contain elements \code{cutpoints} (numeric vector of bin
#'   boundaries) and \code{woe} (numeric vector of WoE values). The number of
#'   WoE values should equal \code{length(cutpoints) + 1}.
#' @param feature Numeric vector of values to be transformed. Automatically
#'   coerced to numeric if provided in another type.
#' @param include_upper_bound Logical flag controlling interval boundary behavior
#'   (default: \code{TRUE}):
#'   \itemize{
#'     \item \code{TRUE}: Intervals are \code{(lower, upper]} (right-closed).
#'     \item \code{FALSE}: Intervals are \code{[lower, upper)} (left-closed).
#'   }
#'   This must match the convention used during binning.
#' @param missing_values Numeric vector of values to be treated as missing
#'   (default: \code{c(-999)}). These values are assigned the WoE of the special
#'   missing bin if it exists in \code{obresults}, or \code{NA} otherwise.
#'
#' @return Numeric vector of WoE values with the same length as \code{feature}.
#'   Values outside the range of \code{cutpoints} are assigned to the first or
#'   last bin. \code{NA} values in \code{feature} are propagated to the output
#'   unless explicitly listed in \code{missing_values}.
#'
#' @details
#' This function is typically used in a two-step workflow:
#' \enumerate{
#'   \item Train binning on training data: \code{bins <- ob_numerical_mdlp(feature_train, target_train)}
#'   \item Apply WoE to new data: \code{woe_test <- ob_apply_woe_num(bins, feature_test)}
#' }
#'
#' \strong{Bin Assignment Logic}:
#' For \code{k} cutpoints \eqn{c_1 < c_2 < \cdots < c_k}, values are assigned as:
#' \itemize{
#'   \item Bin 1: \eqn{x \le c_1} (if \code{include_upper_bound = TRUE})
#'   \item Bin i: \eqn{c_{i-1} < x \le c_i} for \eqn{i = 2, \ldots, k}
#'   \item Bin k+1: \eqn{x > c_k}
#' }
#'
#' \strong{Handling of Edge Cases}:
#' \itemize{
#'   \item Values in \code{missing_values} are matched against a bin labeled
#'     \code{"NA"} or \code{"Missing"} in \code{obresults$bin} (if available).
#'   \item \code{Inf} and \code{-Inf} are assigned to the last and first bins,
#'     respectively.
#'   \item Values exactly equal to cutpoints follow the \code{include_upper_bound}
#'     convention.
#' }
#'
#' @examples
#' \donttest{
#' # Mock data
#' train_data <- data.frame(
#'   income = c(50000, 75000, 30000, 45000, 80000, 60000),
#'   default = c(0, 0, 1, 1, 0, 0)
#' )
#' test_data <- data.frame(
#'   income = c(55000, 35000, 90000)
#' )
#'
#' # Train binning on training set
#' train_bins <- ob_numerical_mdlp(
#'   feature = train_data$income,
#'   target = train_data$default
#' )
#'
#' # Apply to test set
#' test_woe <- ob_apply_woe_num(
#'   obresults = train_bins,
#'   feature = test_data$income
#' )
#'
#' # Handle custom missing indicators (e.g., -999, -1)
#' test_woe <- ob_apply_woe_num(
#'   obresults = train_bins,
#'   feature = test_data$income,
#'   missing_values = c(-999, -1, -9999)
#' )
#'
#' # Use left-closed intervals (match scikit-learn convention)
#' test_woe <- ob_apply_woe_num(
#'   obresults = train_bins,
#'   feature = test_data$income,
#'   include_upper_bound = FALSE
#' )
#' }
#'
#' @seealso
#' \code{\link{ob_numerical_mdlp}} for MDLP binning,
#' \code{\link{ob_numerical_mob}} for monotonic binning,
#' \code{\link{ob_apply_woe_cat}} for applying WoE to categorical features.
#'
#' @export
ob_apply_woe_num <- function(obresults,
                             feature,
                             include_upper_bound = TRUE,
                             missing_values = c(-999)) {
  if (!is.list(obresults)) {
    stop("obresults must be a list (output from an optimal binning function).")
  }

  if (!all(c("cutpoints", "woe") %in% names(obresults))) {
    stop("obresults must contain 'cutpoints' and 'woe' elements.")
  }

  if (!is.numeric(obresults$cutpoints) || !is.numeric(obresults$woe)) {
    stop("obresults$cutpoints and obresults$woe must be numeric vectors.")
  }

  expected_woe_length <- length(obresults$cutpoints) + 1
  actual_woe_length <- length(obresults$woe)

  if (actual_woe_length != expected_woe_length &&
    actual_woe_length != expected_woe_length + 1) { # Allow +1 for missing bin
    stop(sprintf(
      "Inconsistent binning structure: %d cutpoints require %d or %d WoE values, but found %d.",
      length(obresults$cutpoints),
      expected_woe_length,
      expected_woe_length + 1,
      actual_woe_length
    ))
  }

  feature <- as.numeric(feature)

  if (!is.logical(include_upper_bound) || length(include_upper_bound) != 1) {
    stop("include_upper_bound must be a single logical value (TRUE or FALSE).")
  }

  if (!is.numeric(missing_values)) {
    missing_values <- as.numeric(missing_values)
  }

  .Call(
    "_OptimalBinningWoE_OBApplyWoENum",
    obresults,
    feature,
    include_upper_bound,
    missing_values,
    PACKAGE = "OptimalBinningWoE"
  )
}
