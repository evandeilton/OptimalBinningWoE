# Tests for utility functions
# This file tests: obwoe_gains_score, obwoe_gains_variable, ob_check_distincts,
#                   ob_preprocess, ob_apply_woe_num, ob_apply_woe_cat,
#                   ob_cutpoints_num, ob_cutpoints_cat

# ==============================================================================
# obwoe_gains_score
# ==============================================================================
test_that("obwoe_gains_score works with valid binning result", {
  data <- generate_test_data(n = 500, seed = 300)

  # First create a binning result
  binning_result <- ob_numerical_kmb(
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  # Then compute gains table
  gains <- obwoe_gains_score(binning_result)

  expect_type(gains, "list")
  expect_true("bin" %in% names(gains))
  expect_true("woe" %in% names(gains))
  expect_true("iv" %in% names(gains))
})

test_that("obwoe_gains_score computes correct totals", {
  data <- generate_test_data(n = 500, seed = 301)

  binning_result <- ob_numerical_ir(
    feature = data$feature,
    target = data$target
  )

  gains <- obwoe_gains_score(binning_result)

  # Check counts match
  expect_equal(sum(gains$count), length(data$feature))
})

# ==============================================================================
# obwoe_gains_variable
# ==============================================================================
test_that("obwoe_gains_variable works with binned data", {
  data <- generate_test_data(n = 500, seed = 302)

  # Create binning and apply WoE
  binning_result <- ob_numerical_kmb(
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  woe_applied <- ob_apply_woe_num(binning_result, data$feature)

  # Compute gains table from feature
  gains <- obwoe_gains_variable(woe_applied, data$target, group_var = "bin")

  expect_type(gains, "list")
  expect_true("bin" %in% names(gains))
})

# ==============================================================================
# ob_check_distincts
# ==============================================================================
test_that("ob_check_distincts returns correct counts", {
  feature <- c(1, 2, 3, 4, 5, 1, 2, 3)
  target <- c(0, 1, 0, 1, 0, 1, 0, 1)

  result <- ob_check_distincts(feature, target)

  expect_type(result, "integer")
  expect_equal(length(result), 2)
})

test_that("ob_check_distincts works with categorical feature", {
  feature <- c("A", "B", "C", "A", "B", "C")
  target <- c(0, 1, 0, 1, 0, 1)

  result <- ob_check_distincts(feature, target)

  expect_type(result, "integer")
})

# ==============================================================================
# ob_preprocess
# ==============================================================================
test_that("ob_preprocess handles numeric features", {
  set.seed(303)
  feature <- c(rnorm(50), NA, NA, 100) # With missing and outlier
  target <- sample(0:1, 53, replace = TRUE)

  result <- ob_preprocess(
    feature = feature,
    target = target,
    outlier_process = FALSE,
    preprocess = "both"
  )

  expect_type(result, "list")
})

test_that("ob_preprocess handles categorical features", {
  feature <- c("A", "B", NA, "C", "A", "B")
  target <- c(0, 1, 0, 1, 0, 1)

  result <- ob_preprocess(
    feature = feature,
    target = target,
    preprocess = "both"
  )

  expect_type(result, "list")
})

# ==============================================================================
# ob_apply_woe_num
# ==============================================================================
test_that("ob_apply_woe_num applies WoE correctly", {
  data <- generate_test_data(n = 500, seed = 304)

  # Create binning
  binning_result <- ob_numerical_kmb(
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  # Apply WoE
  result <- ob_apply_woe_num(binning_result, data$feature)

  expect_type(result, "list")
  expect_true("feature" %in% names(result))
  expect_true("bin" %in% names(result))
  expect_true("woe" %in% names(result))
  expect_true("idbin" %in% names(result))
  expect_equal(length(result$feature), length(data$feature))
})

test_that("ob_apply_woe_num handles missing values", {
  data <- generate_test_data(n = 100, seed = 305)

  binning_result <- ob_numerical_kmb(
    feature = data$feature,
    target = data$target
  )

  # Add some missing values
  feature_with_na <- c(data$feature[1:90], rep(NA, 10))

  result <- ob_apply_woe_num(binning_result, feature_with_na)

  expect_equal(length(result$feature), length(feature_with_na))
  expect_true(any(result$ismissing == 1))
})

# ==============================================================================
# ob_apply_woe_cat
# ==============================================================================
test_that("ob_apply_woe_cat applies WoE correctly", {
  data <- generate_categorical_data(n = 500, n_categories = 5, seed = 306)

  # Create binning
  binning_result <- ob_categorical_jedi(
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  # Apply WoE
  result <- ob_apply_woe_cat(binning_result, data$feature)

  expect_type(result, "list")
  expect_true("feature" %in% names(result))
  expect_true("bin" %in% names(result))
  expect_true("woe" %in% names(result))
  expect_equal(length(result$feature), length(data$feature))
})

test_that("ob_apply_woe_cat handles missing values", {
  data <- generate_categorical_data(n = 100, n_categories = 4, seed = 307)

  binning_result <- ob_categorical_cm(
    feature = data$feature,
    target = data$target
  )

  # Add some missing values
  feature_with_na <- c(data$feature[1:90], rep(NA, 10))

  result <- ob_apply_woe_cat(binning_result, feature_with_na)

  expect_equal(length(result$feature), length(feature_with_na))
})

# ==============================================================================
# ob_cutpoints_num
# ==============================================================================
test_that("ob_cutpoints_num works with custom cutpoints", {
  set.seed(308)
  feature <- rnorm(200)
  target <- sample(0:1, 200, replace = TRUE)
  cutpoints <- c(-1, 0, 1)

  result <- ob_cutpoints_num(feature, target, cutpoints)

  expect_type(result, "list")
  expect_true("woefeature" %in% names(result))
  expect_true("woebin" %in% names(result))
  expect_equal(length(result$woefeature), length(feature))
})

# ---------------------------------------------------------------------------
# [C6/A-04,A-05] ob_cutpoints_num() -> ob_apply_woe_num() must round-trip
#
# ob_cutpoints_num() used to emit left-closed [a, b) bins, the opposite of
# ob_apply_woe_num()'s include_upper_bound = TRUE default (right-closed
# (a, b]), so a value exactly on a cutpoint got a different (often
# sign-flipped) WoE depending on which side of the fit/apply boundary it
# went through.
# ---------------------------------------------------------------------------
test_that("[C6/A-04,A-05] ob_cutpoints_num() -> ob_apply_woe_num() agree for a value on a cutpoint", {
  feature <- c(5, 15, 25, 30, 45, 55, 60, 75)
  target <- c(0, 0, 1, 1, 1, 1, 0, 0)
  cutpoints <- c(30, 60)

  result <- ob_cutpoints_num(feature, target, cutpoints)

  # The result must actually carry what ob_apply_woe_num() needs.
  expect_true(all(c("cutpoints", "id") %in% names(result)))
  expect_true("id" %in% names(result$woebin))

  apply_obj <- list(
    cutpoints = result$cutpoints, woe = result$woebin$woe,
    id = result$id, bin = result$woebin$bin
  )
  woe_new <- ob_apply_woe_num(apply_obj, feature)

  # Every observation, including the two that sit exactly on a cutpoint
  # (30 and 60), must get the identical WoE on both sides.
  expect_equal(woe_new$woe, result$woefeature)

  on_cutpoint <- feature %in% cutpoints
  expect_true(any(on_cutpoint))
  expect_equal(woe_new$woe[on_cutpoint], result$woefeature[on_cutpoint])
})

# ==============================================================================
# ob_cutpoints_cat
# ==============================================================================
test_that("ob_cutpoints_cat works with custom cutpoints", {
  feature <- c("A", "B", "C", "D", "A", "B", "C", "D", "A", "B")
  target <- c(1, 0, 1, 0, 0, 1, 0, 1, 1, 0)
  cutpoints <- c("A+B", "C+D")

  result <- ob_cutpoints_cat(feature, target, cutpoints)

  expect_type(result, "list")
  expect_true("woefeature" %in% names(result))
  expect_true("woebin" %in% names(result))
  expect_equal(length(result$woefeature), length(feature))
})

# ---------------------------------------------------------------------------
# [C6/A-04,A-05] ob_cutpoints_cat() -> ob_apply_woe_cat() must round-trip
#
# ob_cutpoints_cat() used to echo the "+"-joined input cutpoints straight
# into the emitted bin labels, and carried no 'id' element. Since
# ob_apply_woe_cat() defaults to bin_separator = "%;%" (the separator every
# ob_categorical_*() algorithm in the main pipeline uses) and requires an
# 'id' element, a round trip with defaults matched no category at all and
# every observation silently fell back to a "Special"/NA bin.
# ---------------------------------------------------------------------------
test_that("[C6/A-04,A-05] ob_cutpoints_cat() -> ob_apply_woe_cat() round-trips with defaults", {
  feature <- c("A", "B", "C", "D", "A", "B", "C", "D")
  target <- c(1, 0, 1, 0, 1, 1, 0, 0)
  cutpoints <- c("A+B", "C+D")

  result <- ob_cutpoints_cat(feature, target, cutpoints)

  # The result must actually carry what ob_apply_woe_cat() needs.
  expect_true("id" %in% names(result$woebin))

  woe_new <- ob_apply_woe_cat(result$woebin, feature)

  expect_false(any(woe_new$bin == "Special"))
  expect_false(anyNA(woe_new$woe))
  expect_equal(woe_new$woe, result$woefeature)
})
