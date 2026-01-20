#' @title Apply Optimal Weight of Evidence (WoE) to a Categorical Feature
#'
#' @description
#' Transforms a categorical feature into its corresponding Weight of Evidence (WoE)
#' values using pre-computed binning results from an optimal binning algorithm
#' (e.g., \code{\link{ob_categorical_cm}}).
#'
#' @param obresults List output from an optimal binning function for categorical
#'   variables. Must contain elements \code{bin} (character vector of bin labels)
#'   and \code{woe} (numeric vector of WoE values). Bins may represent individual
#'   categories or merged groups separated by \code{bin_separator}.
#' @param feature Character or factor vector of categorical values to be transformed.
#'   Automatically coerced to character if provided as factor.
#' @param bin_separator Character string used to separate multiple categories
#'   within a single bin label (default: \code{"\%;\%"}). For example, a bin
#'   \code{"A\%;\%B\%;\%C"} contains categories A, B, and C.
#' @param missing_values Character vector specifying which values should be treated
#'   as missing (default: \code{c("NA", "Missing", "")}). These values are matched
#'   against a special bin labeled \code{"NA"} or \code{"Missing"} in \code{obresults}.
#'
#' @return Numeric vector of WoE values with the same length as \code{feature}.
#'   Categories not found in \code{obresults} will produce \code{NA} values with a warning.
#'
#' @details
#' This function is typically used in a two-step workflow:
#' \enumerate{
#'   \item Train binning on training data: \code{bins <- ob_categorical_cm(feature_train, target_train)}
#'   \item Apply WoE to new data: \code{woe_test <- ob_apply_woe_cat(bins, feature_test)}
#' }
#'
#' The function performs exact string matching between categories in \code{feature}
#' and the bin labels in \code{obresults$bin}. For merged bins (containing
#' \code{bin_separator}), the string is split and each component is matched
#' individually.
#'
#' @examples
#' \donttest{
#' # Mock data
#' train_data <- data.frame(
#'   category = c("A", "B", "A", "C", "B", "A"),
#'   default = c(0, 1, 0, 1, 0, 0)
#' )
#' test_data <- data.frame(
#'   category = c("A", "C", "B")
#' )
#'
#' # Train binning on training set
#' train_bins <- ob_categorical_cm(
#'   feature = train_data$category,
#'   target = train_data$default
#' )
#'
#' # Apply to test set
#' test_woe <- ob_apply_woe_cat(
#'   obresults = train_bins,
#'   feature = test_data$category
#' )
#'
#' # Handle custom missing indicators
#' test_woe <- ob_apply_woe_cat(
#'   obresults = train_bins,
#'   feature = test_data$category,
#'   missing_values = c("NA", "Unknown", "N/A", "")
#' )
#' }
#'
#' @export
ob_apply_woe_cat <- function(obresults,
                             feature,
                             bin_separator = "%;%",
                             missing_values = c("NA", "Missing", "")) {
  if (!is.list(obresults)) {
    stop("obresults must be a list (output from an optimal binning function).")
  }

  if (!all(c("bin", "woe") %in% names(obresults))) {
    stop("obresults must contain 'bin' and 'woe' elements.")
  }

  if (!is.character(feature)) {
    feature <- as.character(feature)
  }

  if (!is.character(bin_separator) || length(bin_separator) != 1) {
    stop("bin_separator must be a single character string.")
  }

  if (!is.character(missing_values)) {
    stop("missing_values must be a character vector.")
  }

  .Call(
    "_OptimalBinningWoE_OBApplyWoECat",
    obresults,
    feature,
    bin_separator,
    missing_values,
    PACKAGE = "OptimalBinningWoE"
  )
}
