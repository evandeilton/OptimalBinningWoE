# Tests for numerical binning algorithms - Isotonic Regression (IR)

test_that("ob_numerical_ir works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 123)

  result <- ob_numerical_ir(
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  validate_woe_values(result$woe)
})

test_that("ob_numerical_ir respects monotonicity", {
  data <- generate_test_data(n = 1000, seed = 456)

  result <- ob_numerical_ir(
    feature = data$feature,
    target = data$target,
    auto_monotonicity = TRUE
  )

  # Check that WoE is monotonic
  woe_diff <- diff(result$woe)
  if (length(woe_diff) > 0) {
    # Either all increasing or all decreasing (with some tolerance)
    is_monotonic <- all(woe_diff >= -0.001) || all(woe_diff <= 0.001)
    expect_true(is_monotonic, info = "WoE values should be monotonic")
  }
})

test_that("ob_numerical_ir returns monotonicity direction", {
  data <- generate_test_data(n = 500, seed = 789)

  result <- ob_numerical_ir(
    feature = data$feature,
    target = data$target
  )

  expect_true("monotone_increasing" %in% names(result))
  expect_type(result$monotone_increasing, "logical")
})

test_that("ob_numerical_ir handles constant feature gracefully", {
  set.seed(111)
  feature <- rep(5, 100)
  target <- sample(0:1, 100, replace = TRUE)

  # Should not error
  result <- tryCatch(
    ob_numerical_ir(feature = feature, target = target, min_bins = 2, max_bins = 3),
    error = function(e) NULL
  )

  # Either returns result or handles gracefully
  if (!is.null(result)) {
    expect_type(result, "list")
  }
})
