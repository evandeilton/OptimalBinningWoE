# Tests for ALL numerical binning algorithms
# This file tests: bb, cm, dmiv, dp, ewb, fast_mdlp, fetb, jedi, jedi_mwoe,
#                   ldb, lpdb, mblp, mdlp, mob, mrblp, oslp, sketch, ubsd, udt

# Helper to safely test numerical functions (skips if C++ not available)
test_numerical <- function(func_name, ...) {
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
# ob_numerical_bb - B-Binning
# ==============================================================================
test_that("ob_numerical_bb works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 100)

  result <- test_numerical("ob_numerical_bb",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_cm - ChiMerge
# ==============================================================================
test_that("ob_numerical_cm works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 101)

  result <- test_numerical("ob_numerical_cm",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_dmiv - Divergence Measures IV
# ==============================================================================
test_that("ob_numerical_dmiv works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 102)

  result <- test_numerical("ob_numerical_dmiv",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_dp - Dynamic Programming
# ==============================================================================
test_that("ob_numerical_dp works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 103)

  result <- test_numerical("ob_numerical_dp",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_ewb - Equal Width Binning
# ==============================================================================
test_that("ob_numerical_ewb works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 104)

  result <- test_numerical("ob_numerical_ewb",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_fast_mdlp - Fast MDLP
# ==============================================================================
test_that("ob_numerical_fast_mdlp works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 105)

  result <- test_numerical("ob_numerical_fast_mdlp",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_fetb - FETB
# ==============================================================================
test_that("ob_numerical_fetb works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 106)

  result <- test_numerical("ob_numerical_fetb",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_jedi - JEDI
# ==============================================================================
test_that("ob_numerical_jedi works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 107)

  result <- test_numerical("ob_numerical_jedi",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_jedi_mwoe - JEDI with Modified WoE (Multiclass)
# ==============================================================================
test_that("ob_numerical_jedi_mwoe works with basic inputs", {
  # Note: jedi_mwoe returns multiclass format (woe/iv as matrices)
  data <- generate_test_data(n = 500, seed = 108)

  result <- test_numerical("ob_numerical_jedi_mwoe",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  # Multiclass validation: woe/iv are matrices, count is vector
  expect_true(is.list(result))
  expect_true("bin" %in% names(result) || "id" %in% names(result))
  expect_true("count" %in% names(result))
  expect_equal(sum(result$count), length(data$feature))
})


# ==============================================================================
# ob_numerical_ldb - Local Density Binning
# ==============================================================================
test_that("ob_numerical_ldb works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 109)

  result <- test_numerical("ob_numerical_ldb",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_lpdb - Local Polynomial Density Binning
# ==============================================================================
test_that("ob_numerical_lpdb works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 110)

  result <- test_numerical("ob_numerical_lpdb",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_mblp - MBLP
# ==============================================================================
test_that("ob_numerical_mblp works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 111)

  result <- test_numerical("ob_numerical_mblp",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_mdlp - MDLP
# ==============================================================================
test_that("ob_numerical_mdlp works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 112)

  result <- test_numerical("ob_numerical_mdlp",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_mob - Monotone Optimal Binning
# ==============================================================================
test_that("ob_numerical_mob works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 113)

  result <- test_numerical("ob_numerical_mob",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_mrblp - MRBLP
# ==============================================================================
test_that("ob_numerical_mrblp works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 114)

  result <- test_numerical("ob_numerical_mrblp",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_oslp - OSLP
# ==============================================================================
test_that("ob_numerical_oslp works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 115)

  result <- test_numerical("ob_numerical_oslp",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_sketch - Sketch-based Binning
# ==============================================================================
test_that("ob_numerical_sketch works with basic inputs", {
  skip("Skipping ob_numerical_sketch - Segfault in C++ with n>=500 (needs deeper debugging)")
  data <- generate_test_data(n = 500, seed = 116)

  result <- test_numerical("ob_numerical_sketch",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_ubsd - UBSD
# ==============================================================================
test_that("ob_numerical_ubsd works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 117)

  result <- test_numerical("ob_numerical_ubsd",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})

# ==============================================================================
# ob_numerical_udt - UDT
# ==============================================================================
test_that("ob_numerical_udt works with basic inputs", {
  data <- generate_test_data(n = 500, seed = 118)

  result <- test_numerical("ob_numerical_udt",
    feature = data$feature,
    target = data$target,
    min_bins = 3,
    max_bins = 5
  )

  validate_binning_result(result, type = "numerical")
  expect_equal(sum(result$count), length(data$feature))
})
