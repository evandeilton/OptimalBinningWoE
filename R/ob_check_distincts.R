#' @title Check Distinct Length
#'
#' @description
#' Internal utility function that counts the number of distinct (unique) values
#' in a feature vector. Used for preprocessing validation before applying optimal
#' binning algorithms to determine if the feature has sufficient variability.
#'
#' @param x Vector (numeric, character, or factor) whose unique values are to be counted.
#'   Accepts any R vector type that can be compared for equality.
#' @param target Integer vector of binary target values (0/1). Must have the same
#'   length as \code{x}. While not used in the distinct count calculation, it is
#'   required for interface consistency and may be used for future extensions
#'   (e.g., counting distinct values per class).
#'
#' @return Integer scalar representing the number of unique values in \code{x},
#'   excluding \code{NA} values. Returns 0 if \code{x} is empty or contains only
#'   \code{NA} values.
#'
#' @details
#' This function is typically used internally by optimal binning algorithms to:
#' \itemize{
#'   \item Validate that the feature has at least 2 distinct values (required for binning).
#'   \item Determine if special handling is needed for low-cardinality features
#'     (e.g., \eqn{\le 2} unique values).
#'   \item Decide between binning strategies (continuous vs categorical).
#' }
#'
#' \strong{Handling of Missing Values}:
#' \code{NA}, \code{NaN}, and \code{Inf} values are excluded from the count.
#' To include missing values as a distinct category, preprocess \code{x} by
#' converting missings to a placeholder (e.g., \code{"-999"} for numeric,
#' \code{"Missing"} for character).
#'
#' @examples
#' \donttest{
#' # Continuous feature with many unique values
#' x_continuous <- rnorm(1000)
#' target <- sample(0:1, 1000, replace = TRUE)
#' ob_check_distincts(x_continuous, target)
#' # Returns: ~1000 (approximately all unique due to floating point)
#'
#' # Low-cardinality feature
#' x_binary <- sample(c("Yes", "No"), 1000, replace = TRUE)
#' ob_check_distincts(x_binary, target)
#' # Returns: 2
#'
#' # Feature with missing values
#' x_with_na <- c(1, 2, NA, 2, 3, NA, 1)
#' target_short <- c(1, 0, 1, 0, 1, 0, 1)
#' ob_check_distincts(x_with_na, target_short)
#' # Returns: 3 (counts: 1, 2, 3; NAs excluded)
#'
#' # Empty or all-NA feature
#' x_empty <- rep(NA, 100)
#' ob_check_distincts(x_empty, sample(0:1, 100, replace = TRUE))
#' # Returns: 0
#' }
#'
#' @keywords internal
#' @export
ob_check_distincts <- function(x, target) {
  if (length(x) == 0) {
    stop("x cannot be empty.")
  }

  if (!is.vector(target) || !(is.integer(target) || is.numeric(target))) {
    stop("target must be an integer or numeric vector.")
  }

  if (length(x) != length(target)) {
    stop("x and target must have the same length.")
  }

  target <- as.integer(target)

  unique_target <- unique(target[!is.na(target)])
  if (!all(unique_target %in% c(0L, 1L))) {
    stop("target must contain only binary values 0 and 1 (excluding NAs).")
  }

  .Call(
    "_OptimalBinningWoE_OBCheckDistinctsLength",
    x,
    target,
    PACKAGE = "OptimalBinningWoE"
  )
}
