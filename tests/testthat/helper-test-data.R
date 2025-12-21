# Helper functions for OptimalBinningWoE tests

#' Generate synthetic binary classification data for testing
#'
#' @param n Number of observations
#' @param seed Random seed for reproducibility
#' @return A list with target and feature vectors
generate_test_data <- function(n = 1000, seed = 42) {
  set.seed(seed)
  # Create feature with some relationship to target
  feature <- rnorm(n)
  # Target probability depends on feature value
  prob <- plogis(0.5 + 0.5 * feature)
  target <- rbinom(n, 1, prob)
  list(target = target, feature = feature)
}

#' Generate synthetic categorical data for testing
#'
#' @param n Number of observations
#' @param n_categories Number of categories
#' @param seed Random seed for reproducibility
#' @return A list with target and feature vectors
generate_categorical_data <- function(n = 1000, n_categories = 5, seed = 42) {
  set.seed(seed)
  categories <- LETTERS[1:n_categories]
  # Each category has different event probability
  probs <- seq(0.1, 0.9, length.out = n_categories)
  feature <- sample(categories, n, replace = TRUE)
  # Target depends on category
  target <- sapply(feature, function(cat) {
    idx <- which(categories == cat)
    rbinom(1, 1, probs[idx])
  })
  list(target = as.integer(target), feature = feature)
}

#' Safely call a function, skipping test if C++ export is not available
#'
#' @param fn The function to call
#' @param ... Arguments to pass to the function
#' @return Result of function call, or skips test if function not available
safe_call <- function(fn, ...) {
  tryCatch(
    fn(...),
    error = function(e) {
      if (grepl("not available for .Call()", e$message, fixed = TRUE)) {
        testthat::skip(paste("C++ function not available:", deparse(substitute(fn))))
      } else {
        stop(e)
      }
    }
  )
}

#' Validate binning result structure
#'
#' @param result A binning result object
#' @param type Either "numerical" or "categorical"
#' @return TRUE if valid, error otherwise
validate_binning_result <- function(result, type = "numerical") {
  # Check that result is a list
  expect_type(result, "list")

  # Common required fields
  expect_true("id" %in% names(result), info = "Missing 'id' field")
  expect_true("bin" %in% names(result), info = "Missing 'bin' field")
  expect_true("woe" %in% names(result), info = "Missing 'woe' field")
  expect_true("count" %in% names(result), info = "Missing 'count' field")
  expect_true("count_pos" %in% names(result), info = "Missing 'count_pos' field")
  expect_true("count_neg" %in% names(result), info = "Missing 'count_neg' field")

  # Check lengths match
  n_bins <- length(result$bin)
  expect_equal(length(result$id), n_bins, info = "id length mismatch")
  expect_equal(length(result$woe), n_bins, info = "woe length mismatch")
  expect_equal(length(result$count), n_bins, info = "count length mismatch")
  expect_equal(length(result$count_pos), n_bins, info = "count_pos length mismatch")
  expect_equal(length(result$count_neg), n_bins, info = "count_neg length mismatch")

  # Check counts sum correctly
  expect_equal(result$count, result$count_pos + result$count_neg,
    info = "count != count_pos + count_neg"
  )

  # Check numerical-specific fields

  if (type == "numerical") {
    expect_true("cutpoints" %in% names(result), info = "Missing 'cutpoints' field")
  }

  TRUE
}

#' Check WoE values are reasonable
#'
#' @param woe Vector of WoE values
#' @return TRUE if valid
validate_woe_values <- function(woe) {
  # WoE should be finite
  expect_true(all(is.finite(woe)), info = "WoE contains non-finite values")
  # WoE should typically be in reasonable range
  expect_true(all(abs(woe) < 10), info = "WoE values out of reasonable range")
  TRUE
}

#' Check IV values are reasonable
#'
#' @param iv Vector of IV values
#' @return TRUE if valid
validate_iv_values <- function(iv) {
  # IV should be non-negative
  expect_true(all(iv >= -0.001), info = "IV contains negative values")
  # IV should be finite
  expect_true(all(is.finite(iv)), info = "IV contains non-finite values")
  TRUE
}
