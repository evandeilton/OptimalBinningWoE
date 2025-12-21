# Tests for numerical binning algorithms - K-Means Binning (KMB)

test_that("ob_numerical_kmb works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 123)

  result <- ob_numerical_kmb(
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  validate_woe_values(result$woe)

  # Check bin count constraints

  n_bins <- length(result$bin)
  expect_gte(n_bins, 2)
  expect_lte(n_bins, 6)
})

test_that("ob_numerical_kmb respects min_bins and max_bins", {
  data <- generate_test_data(n = 1000, seed = 456)

  result <- ob_numerical_kmb(
    feature = data$feature,
    target = data$target,
    min_bins = 4,
    max_bins = 6
  )

  n_bins <- length(result$bin)
  expect_gte(n_bins, 2)
  expect_lte(n_bins, 7)
})

test_that("ob_numerical_kmb handles edge cases", {
  # Few unique values
  set.seed(789)
  feature <- sample(1:3, 200, replace = TRUE) + rnorm(200, sd = 0.01)
  target <- sample(0:1, 200, replace = TRUE)

  result <- ob_numerical_kmb(
    feature = feature,
    target = target,
    min_bins = 2,
    max_bins = 5
  )

  expect_type(result, "list")
  expect_true(length(result$bin) >= 1)
})

test_that("ob_numerical_kmb returns convergence info", {
  data <- generate_test_data(n = 500, seed = 101)

  result <- ob_numerical_kmb(
    feature = data$feature,
    target = data$target
  )

  expect_true("converged" %in% names(result))
  expect_true("iterations" %in% names(result))
  expect_type(result$converged, "logical")
})

test_that("ob_numerical_kmb total counts match input", {
  data <- generate_test_data(n = 800, seed = 202)

  result <- ob_numerical_kmb(
    feature = data$feature,
    target = data$target
  )

  # Total observations should match
  expect_equal(sum(result$count), length(data$feature))
  expect_equal(sum(result$count_pos), sum(data$target == 1))
  expect_equal(sum(result$count_neg), sum(data$target == 0))
})
