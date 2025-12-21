# Tests for ALL categorical binning algorithms
# This file tests: cm, dmiv, dp, fetb, gmb, ivb, jedi, jedi_mwoe, mba,
#                   milp, mob, sab, sblp, sketch, swb, udt

# Helper to safely test categorical functions (skips if C++ not available)
test_categorical <- function(func_name, ...) {
  fn <- get(func_name, envir = asNamespace("OptimalBinningWoE"))
  tryCatch(
    fn(...),
    error = function(e) {
      if (grepl("not available for .Call()", e$message, fixed = TRUE)) {
        skip(paste("C++ function not available:", func_name))
      } else {
        stop(e)
      }
    }
  )
}

# ==============================================================================
# ob_categorical_cm - ChiMerge
# ==============================================================================
test_that("ob_categorical_cm works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 200)

  result <- test_categorical("ob_categorical_cm",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_dmiv - Divergence Measures IV
# ==============================================================================
test_that("ob_categorical_dmiv works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 201)

  result <- test_categorical("ob_categorical_dmiv",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_dp - Dynamic Programming
# ==============================================================================
test_that("ob_categorical_dp works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 202)

  result <- test_categorical("ob_categorical_dp",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_fetb - FETB
# ==============================================================================
test_that("ob_categorical_fetb works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 203)

  result <- test_categorical("ob_categorical_fetb",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_gmb - GMB
# ==============================================================================
test_that("ob_categorical_gmb works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 204)

  result <- test_categorical("ob_categorical_gmb",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_ivb - IV-based Binning
# ==============================================================================
test_that("ob_categorical_ivb works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 205)

  result <- test_categorical("ob_categorical_ivb",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_jedi - JEDI
# ==============================================================================
test_that("ob_categorical_jedi works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 206)

  result <- test_categorical("ob_categorical_jedi",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_jedi_mwoe - JEDI with Modified WoE (Multiclass)
# ==============================================================================
test_that("ob_categorical_jedi_mwoe works with basic inputs", {
  # Note: jedi_mwoe returns multiclass format (woe/iv as matrices)
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 207)

  result <- test_categorical("ob_categorical_jedi_mwoe",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  # Multiclass validation: woe/iv are matrices, count is vector
  expect_true(is.list(result))
  expect_true("bin" %in% names(result) || "id" %in% names(result))
  expect_true("count" %in% names(result))
  expect_equal(sum(result$count), length(data$feature))
})


# ==============================================================================
# ob_categorical_mba - MBA
# ==============================================================================
test_that("ob_categorical_mba works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 208)

  result <- test_categorical("ob_categorical_mba",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_milp - MILP
# ==============================================================================
test_that("ob_categorical_milp works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 209)

  result <- test_categorical("ob_categorical_milp",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_mob - Monotone Optimal Binning
# ==============================================================================
test_that("ob_categorical_mob works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 210)

  result <- test_categorical("ob_categorical_mob",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_sab - SAB
# ==============================================================================
test_that("ob_categorical_sab works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 211)

  result <- test_categorical("ob_categorical_sab",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_sblp - SBLP
# ==============================================================================
test_that("ob_categorical_sblp works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 212)

  result <- test_categorical("ob_categorical_sblp",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_sketch - Sketch-based Binning
# ==============================================================================
test_that("ob_categorical_sketch works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 213)

  result <- test_categorical("ob_categorical_sketch",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_swb - SWB
# ==============================================================================
test_that("ob_categorical_swb works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 214)

  result <- test_categorical("ob_categorical_swb",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_categorical_udt - UDT
# ==============================================================================
test_that("ob_categorical_udt works with basic inputs", {
  data <- generate_categorical_data(n = 500, n_categories = 6, seed = 215)

  result <- test_categorical("ob_categorical_udt",
    feature = data$feature,
    target = data$target,
    min_bins = 2,
    max_bins = 4
  )

  validate_binning_result(result, type = "categorical")
  expect_equal(sum(result$count), length(data$feature))
})
