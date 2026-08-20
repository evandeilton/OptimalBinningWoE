# ============================================================================ #
# Tests for step_obwoe
# ============================================================================ #

# Load required packages for testing
library(recipes)

# Setup: Create mock data and functions for testing
# These tests assume OptimalBinningWoE package is installed with obwoe() and
# control.obwoe() functions available.

# ---------------------------------------------------------------------------- #
# Test Fixtures
# ---------------------------------------------------------------------------- #

# Helper function to create test data
create_test_data <- function(n = 500, seed = 42) {
  set.seed(seed)

  data.frame(
    # Numerical features
    age = rnorm(n, 45, 12),
    income = exp(rnorm(n, 10, 0.6)),
    score = runif(n, 300, 850),

    # Categorical features
    employment = sample(
      c("Employed", "Self-Employed", "Unemployed", "Retired"),
      n,
      replace = TRUE, prob = c(0.6, 0.2, 0.1, 0.1)
    ),
    education = factor(
      sample(c("HighSchool", "Bachelor", "Master", "PhD"),
        n,
        replace = TRUE, prob = c(0.3, 0.4, 0.2, 0.1)
      )
    ),

    # Binary target (factor - tidymodels standard)
    target = factor(
      rbinom(n, 1, 0.15),
      levels = c(0, 1),
      labels = c("No", "Yes")
    ),
    stringsAsFactors = FALSE
  )
}

# Create small test data for faster tests
create_small_data <- function(n = 100, seed = 123) {
  create_test_data(n = n, seed = seed)
}

# ---------------------------------------------------------------------------- #
# Test Group: Internal Helper Functions
# ---------------------------------------------------------------------------- #

test_that(".valid_algorithms returns correct algorithms", {
  algos <- .valid_algorithms()

  expect_type(algos, "character")
  expect_true(length(algos) == 29L) # 28 algorithms + "auto"
  expect_true("auto" %in% algos)
  expect_true("jedi" %in% algos)
  expect_true("mdlp" %in% algos)
  expect_true("mob" %in% algos)
})

test_that(".numerical_only_algorithms returns correct list", {
  num_algos <- .numerical_only_algorithms()

  expect_type(num_algos, "character")
  expect_true("mdlp" %in% num_algos)
  expect_true("fast_mdlp" %in% num_algos)
  expect_true("ewb" %in% num_algos)
  expect_false("jedi" %in% num_algos)
  expect_false("ivb" %in% num_algos)
})

test_that(".categorical_only_algorithms returns correct list", {
  cat_algos <- .categorical_only_algorithms()

  expect_type(cat_algos, "character")
  expect_true("ivb" %in% cat_algos)
  expect_true("gmb" %in% cat_algos)
  expect_false("jedi" %in% cat_algos)
  expect_false("mdlp" %in% cat_algos)
})

test_that(".universal_algorithms returns correct list", {
  uni_algos <- .universal_algorithms()

  expect_type(uni_algos, "character")
  expect_true("jedi" %in% uni_algos)
  expect_true("jedi_mwoe" %in% uni_algos)
  expect_true("mob" %in% uni_algos)
  expect_false("mdlp" %in% uni_algos)
  expect_false("ivb" %in% uni_algos)
})


# ---------------------------------------------------------------------------- #
# Test Group: Constructor Validation
# ---------------------------------------------------------------------------- #

test_that("step_obwoe requires outcome argument", {
  df <- create_small_data()

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(all_predictors()),
    "outcome.*required"
  )
})

test_that("step_obwoe validates outcome is single character", {
  df <- create_small_data()

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(all_predictors(), outcome = c("a", "b")),
    "single character"
  )

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(all_predictors(), outcome = 123),
    "single character"
  )
})

test_that("step_obwoe validates algorithm parameter", {
  df <- create_small_data()

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(all_predictors(), outcome = "target", algorithm = "invalid_algo"),
    "not recognized"
  )

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(all_predictors(), outcome = "target", algorithm = c("jedi", "mdlp")),
    "single character"
  )
})

test_that("step_obwoe validates min_bins parameter", {
  df <- create_small_data()

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(age, outcome = "target", min_bins = 1),
    "at least 2"
  )

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(age, outcome = "target", min_bins = c(2, 3)),
    "single integer"
  )
})

test_that("step_obwoe validates max_bins parameter", {
  df <- create_small_data()

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(age, outcome = "target", min_bins = 5, max_bins = 3),
    "greater than or equal"
  )
})

test_that("step_obwoe validates bin_cutoff parameter", {
  df <- create_small_data()

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(age, outcome = "target", bin_cutoff = 0),
    "between 0 and 1"
  )

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(age, outcome = "target", bin_cutoff = 1),
    "between 0 and 1"
  )

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(age, outcome = "target", bin_cutoff = -0.1),
    "between 0 and 1"
  )
})

test_that("step_obwoe validates na_woe parameter", {
  df <- create_small_data()

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(age, outcome = "target", na_woe = "zero"),
    "single numeric"
  )
})

test_that("step_obwoe validates suffix parameters", {
  df <- create_small_data()

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(age, outcome = "target", suffix_woe = 123),
    "single character"
  )

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(age, outcome = "target", suffix_bin = c("_a", "_b")),
    "single character"
  )
})

test_that("step_obwoe validates control parameter", {
  df <- create_small_data()

  expect_error(
    recipe(target ~ ., data = df) %>%
      step_obwoe(age, outcome = "target", control = "invalid"),
    "named list"
  )
})

test_that("step_obwoe accepts valid algorithm names", {
  df <- create_small_data()

  for (algo in c("auto", "jedi", "mob", "mdlp")) {
    rec <- recipe(target ~ age, data = df) %>%
      step_obwoe(age, outcome = "target", algorithm = algo)

    expect_s3_class(rec, "recipe")
  }
})


test_that("[B3/C-05] print.step_obwoe works with algorithm = tune::tune()", {
  # x$algorithm can be an unevaluated language object before the step is
  # prepped -- tune::tune() returns a call, not a string. print.step_obwoe()
  # used to do sprintf("%s", x$algorithm), which errors on a language
  # object instead of formatting it, so print(rec) broke for any recipe
  # tagging step_obwoe(algorithm = tune()) for hyperparameter tuning.
  skip_if_not_installed("tune")

  df <- create_small_data()
  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target", algorithm = tune::tune())

  expect_no_error(print(rec))
})


# ---------------------------------------------------------------------------- #
# Test Group: Recipe Preparation (prep)
# ---------------------------------------------------------------------------- #

test_that("prep.step_obwoe works with default parameters", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age + income, data = df) %>%
    step_obwoe(all_numeric_predictors(), outcome = "target")

  prepped <- prep(rec, training = df)

  expect_true(prepped$steps[[1]]$trained)
  expect_true(length(prepped$steps[[1]]$binning_results) > 0L)
})

test_that("prep.step_obwoe resolves 'auto' algorithm for binary target", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target", algorithm = "auto")

  prepped <- prep(rec, training = df)

  # Algorithm should be resolved to "jedi" for binary target

  expect_equal(prepped$steps[[1]]$algorithm, "jedi")
})

test_that("[B2/C-02] prep.step_obwoe resolves 'auto' using observed levels only", {
  # 'auto' used to switch to the multiclass algorithm ("jedi_mwoe") whenever
  # the outcome factor *declared* more than 2 levels, even if only 2 were
  # actually observed in the training rows (e.g. after a filter or a
  # resample). That produced a matrix-valued 'woe' and corrupted bake()
  # with silent linear-index recycling. It must resolve like a genuine
  # binary target when only two levels are observed.
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()
  # Declare a third level that is never observed in the data.
  df$target <- factor(as.character(df$target), levels = c("No", "Yes", "Unused"))
  expect_equal(nlevels(df$target), 3L)
  expect_equal(length(unique(df$target)), 2L)

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target", algorithm = "auto")

  prepped <- prep(rec, training = df)

  expect_equal(prepped$steps[[1]]$algorithm, "jedi")

  baked <- bake(prepped, new_data = df)
  # output = "woe" (the default) replaces the column in place; it must be
  # a plain numeric vector, not a matrix (which is what jedi_mwoe would
  # have produced here).
  expect_true(is.numeric(baked$age))
  expect_null(dim(baked$age))
})


test_that("prep.step_obwoe handles missing outcome column", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()
  df_no_target <- df[, setdiff(names(df), "target")]

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target")

  # The recipes package validates formula variables before our prep method
  # So we expect the recipes error message, not our custom one
  expect_error(
    prep(rec, training = df_no_target),
    "Not all variables|not found"
  )
})

test_that("prep.step_obwoe warns for incompatible algorithm-feature combinations", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  # Numerical-only algorithm on categorical feature
  rec <- recipe(target ~ employment, data = df) %>%
    step_obwoe(employment, outcome = "target", algorithm = "mdlp")

  expect_warning(
    prepped <- prep(rec, training = df),
    "does not support categorical"
  )

  # The feature should be skipped
  expect_length(prepped$steps[[1]]$binning_results, 0L)
})

test_that("prep.step_obwoe warns for categorical-only algorithm on numerical", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target", algorithm = "ivb")

  expect_warning(
    prepped <- prep(rec, training = df),
    "does not support numerical"
  )

  expect_length(prepped$steps[[1]]$binning_results, 0L)
})

test_that("prep.step_obwoe handles empty selection gracefully", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  # Select columns that don't exist
  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(starts_with("nonexistent"), outcome = "target")

  prepped <- prep(rec, training = df)

  expect_true(prepped$steps[[1]]$trained)
  expect_length(prepped$steps[[1]]$binning_results, 0L)
})

test_that("prep.step_obwoe removes outcome from selection", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  # all_predictors() should not include target
  rec <- recipe(target ~ ., data = df) %>%
    step_obwoe(everything(), outcome = "target")

  prepped <- prep(rec, training = df)

  # Target should not be in binning_results
  expect_false("target" %in% names(prepped$steps[[1]]$binning_results))
})

test_that("prep.step_obwoe stores efficient data structure", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target")

  prepped <- prep(rec, training = df)
  res <- prepped$steps[[1]]$binning_results$age

  # Check required fields exist
  expect_true("type" %in% names(res))
  expect_true("bin" %in% names(res))
  expect_true("woe" %in% names(res))
  expect_true("iv" %in% names(res))
  expect_true("cutpoints" %in% names(res))
  expect_true("total_iv" %in% names(res))
})

test_that("prep.step_obwoe creates category mapping for categorical features", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ employment, data = df) %>%
    step_obwoe(employment, outcome = "target", algorithm = "jedi")

  prepped <- prep(rec, training = df)

  # Skip if feature was not successfully binned
  skip_if(length(prepped$steps[[1]]$binning_results) == 0L)

  res <- prepped$steps[[1]]$binning_results$employment

  expect_equal(res$type, "categorical")
  expect_true("cat_map_keys" %in% names(res))
  expect_true("cat_map_bins" %in% names(res))
  expect_true("cat_map_woes" %in% names(res))
})


# ---------------------------------------------------------------------------- #
# Test Group: Baking (bake)
# ---------------------------------------------------------------------------- #

test_that("bake.step_obwoe works with output='woe'", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target", output = "woe")

  prepped <- prep(rec, training = df)
  baked <- bake(prepped, new_data = df)

  expect_s3_class(baked, "tbl_df")
  expect_true("age" %in% names(baked))
  expect_type(baked$age, "double") # WoE values are numeric
})

test_that("[C5/A-03] bake.step_obwoe handles a numerical column arriving as a factor", {
  # vals_num <- as.numeric(vals) returns the integer LEVEL CODES when vals
  # is a factor, not the numeric values the levels represent, so bin
  # assignment was silently wrong for a numerical predictor delivered as a
  # factor at bake() time (e.g. a data source that stores everything as
  # factors). This compares the WoE assigned to a factor-typed copy of the
  # column against the WoE assigned to the original numeric column: they
  # must be identical, and bake() must warn about the type mismatch.
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()
  # Round first: factor() -> as.character() -> as.numeric() is a lossless
  # round-trip for these values, so any remaining difference below can only
  # come from bin assignment, not from floating-point noise introduced by
  # the factor conversion itself.
  df$age <- round(df$age, 4)

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target", output = "woe")
  prepped <- prep(rec, training = df)

  baked_numeric <- bake(prepped, new_data = df)

  df_factor <- df
  df_factor$age <- factor(df_factor$age)
  expect_warning(
    baked_factor <- bake(prepped, new_data = df_factor),
    "arrived in new_data as a factor"
  )

  expect_equal(baked_factor$age, baked_numeric$age)
})


test_that("bake.step_obwoe works with output='bin'", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target", output = "bin")

  prepped <- prep(rec, training = df)
  baked <- bake(prepped, new_data = df)

  expect_s3_class(baked, "tbl_df")
  expect_true("age" %in% names(baked))
  expect_type(baked$age, "character") # Bin labels are character
})

test_that("bake.step_obwoe works with output='both'", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target", output = "both")

  prepped <- prep(rec, training = df)
  baked <- bake(prepped, new_data = df)

  expect_s3_class(baked, "tbl_df")
  expect_true("age" %in% names(baked))
  expect_true("age_woe" %in% names(baked))
  expect_true("age_bin" %in% names(baked))

  # Original should be unchanged
  expect_equal(baked$age, df$age)
})

test_that("bake.step_obwoe maintains column order with output='both'", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age + income + score, data = df) %>%
    step_obwoe(age, income, outcome = "target", output = "both")

  prepped <- prep(rec, training = df)
  baked <- bake(prepped, new_data = df)

  col_names <- names(baked)

  # Check that _woe and _bin columns follow their respective originals
  age_idx <- which(col_names == "age")
  age_woe_idx <- which(col_names == "age_woe")
  age_bin_idx <- which(col_names == "age_bin")

  expect_equal(age_woe_idx, age_idx + 1L)
  expect_equal(age_bin_idx, age_idx + 2L)
})

test_that("bake.step_obwoe uses custom suffixes", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age,
      outcome = "target", output = "both",
      suffix_woe = "_weight", suffix_bin = "_category"
    )

  prepped <- prep(rec, training = df)
  baked <- bake(prepped, new_data = df)

  expect_true("age_weight" %in% names(baked))
  expect_true("age_category" %in% names(baked))
  expect_false("age_woe" %in% names(baked))
  expect_false("age_bin" %in% names(baked))
})

test_that("bake.step_obwoe handles NA values with na_woe", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()
  df_with_na <- df
  df_with_na$age[1:10] <- NA

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target", output = "woe", na_woe = -999)

  prepped <- prep(rec, training = df)
  baked <- bake(prepped, new_data = df_with_na)

  # NAs should get na_woe value
  expect_equal(baked$age[1:10], rep(-999, 10))
})

test_that("bake.step_obwoe handles unseen categories", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()
  df_new <- df
  df_new$employment[1:5] <- "NewCategory"

  rec <- recipe(target ~ employment, data = df) %>%
    step_obwoe(employment, outcome = "target", output = "woe", na_woe = 0)

  prepped <- prep(rec, training = df)

  # Skip if employment was not successfully binned
  skip_if(length(prepped$steps[[1]]$binning_results) == 0L)

  baked <- bake(prepped, new_data = df_new)

  # Unseen category should get na_woe value
  expect_equal(baked$employment[1:5], rep(0, 5))
})

test_that("bake.step_obwoe returns tibble", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target")

  prepped <- prep(rec, training = df)
  baked <- bake(prepped, new_data = df)

  expect_s3_class(baked, "tbl_df")
})

test_that("bake.step_obwoe works with new_data = NULL", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target")

  prepped <- prep(rec, training = df)
  baked <- bake(prepped, new_data = NULL)

  expect_s3_class(baked, "tbl_df")
  expect_equal(nrow(baked), nrow(df))
})


# ---------------------------------------------------------------------------- #
# Test Group: Print Method
# ---------------------------------------------------------------------------- #

test_that("print.step_obwoe works for untrained step", {
  df <- create_small_data()

  rec <- recipe(target ~ age + income, data = df) %>%
    step_obwoe(all_numeric_predictors(), outcome = "target", algorithm = "jedi")

  output <- capture.output(print(rec$steps[[1]]))

  expect_true(any(grepl("Optimal Binning WoE", output)))
  expect_true(any(grepl("jedi", output)))
})

test_that("print.step_obwoe works for trained step", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target")

  prepped <- prep(rec, training = df)
  output <- capture.output(print(prepped$steps[[1]]))

  expect_true(any(grepl("trained", output)))
  expect_true(any(grepl("IV=", output)))
})


# ---------------------------------------------------------------------------- #
# Test Group: Tidy Method
# ---------------------------------------------------------------------------- #

test_that("tidy.step_obwoe returns correct structure for untrained step", {
  df <- create_small_data()

  rec <- recipe(target ~ age + income, data = df) %>%
    step_obwoe(all_numeric_predictors(), outcome = "target")

  tid <- tidy(rec, number = 1)

  expect_s3_class(tid, "tbl_df")
  expect_named(tid, c("terms", "bin", "woe", "iv", "id"))
  expect_true(all(is.na(tid$woe)))
})

test_that("tidy.step_obwoe returns one row per bin for trained step", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target")

  prepped <- prep(rec, training = df)
  tid <- tidy(prepped, number = 1)

  expect_s3_class(tid, "tbl_df")
  expect_named(tid, c("terms", "bin", "woe", "iv", "id"))

  # Should have multiple rows (one per bin)
  expect_gt(nrow(tid), 1L)

  # All terms should be "age"
  expect_true(all(tid$terms == "age"))

  # WoE should be numeric
  expect_type(tid$woe, "double")
})

test_that("tidy.step_obwoe handles empty results", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  # Force empty results with incompatible algorithm
  rec <- recipe(target ~ employment, data = df) %>%
    step_obwoe(employment, outcome = "target", algorithm = "mdlp")

  suppressWarnings(prepped <- prep(rec, training = df))
  tid <- tidy(prepped, number = 1)

  expect_s3_class(tid, "tbl_df")
  expect_equal(nrow(tid), 0L)
})


# ---------------------------------------------------------------------------- #
# Test Group: Tunable Method
# ---------------------------------------------------------------------------- #

test_that("tunable.step_obwoe returns correct parameters", {
  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target")

  tun <- tunable(rec$steps[[1]])

  expect_s3_class(tun, "tbl_df")
  expect_true("algorithm" %in% tun$name)
  expect_true("min_bins" %in% tun$name)
  expect_true("max_bins" %in% tun$name)
  expect_true("bin_cutoff" %in% tun$name)

  # Check call_info structure
  expect_equal(tun$call_info[[1]]$pkg, "OptimalBinningWoE")
  expect_equal(tun$call_info[[1]]$fun, "obwoe_algorithm")
})


# ---------------------------------------------------------------------------- #
# Test Group: Required Packages
# ---------------------------------------------------------------------------- #

test_that("required_pkgs.step_obwoe returns correct packages", {
  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target")

  pkgs <- required_pkgs(rec$steps[[1]])

  expect_type(pkgs, "character")
  expect_true("OptimalBinningWoE" %in% pkgs)
  expect_true("recipes" %in% pkgs)
})


# ---------------------------------------------------------------------------- #
# Test Group: Dials Parameters
# ---------------------------------------------------------------------------- #

test_that("obwoe_algorithm returns valid dials parameter", {
  skip_if_not_installed("dials")

  param <- obwoe_algorithm()

  expect_s3_class(param, "qual_param")
  expect_true(length(param$values) == 29L)
  expect_true("auto" %in% param$values)
  expect_true("jedi" %in% param$values)
})

test_that("obwoe_algorithm accepts custom values", {
  skip_if_not_installed("dials")

  param <- obwoe_algorithm(values = c("jedi", "mob", "dp"))

  expect_equal(param$values, c("jedi", "mob", "dp"))
})

test_that("obwoe_min_bins returns valid dials parameter", {
  skip_if_not_installed("dials")

  param <- obwoe_min_bins()

  expect_s3_class(param, "quant_param")
  expect_equal(param$type, "integer")
  expect_equal(param$range$lower, 2L)
  expect_equal(param$range$upper, 5L)
})

test_that("obwoe_min_bins accepts custom range", {
  skip_if_not_installed("dials")

  param <- obwoe_min_bins(range = c(3L, 8L))

  expect_equal(param$range$lower, 3L)
  expect_equal(param$range$upper, 8L)
})

test_that("obwoe_max_bins returns valid dials parameter", {
  skip_if_not_installed("dials")

  param <- obwoe_max_bins()

  expect_s3_class(param, "quant_param")
  expect_equal(param$type, "integer")
  expect_equal(param$range$lower, 5L)
  expect_equal(param$range$upper, 20L)
})

test_that("obwoe_bin_cutoff returns valid dials parameter", {
  skip_if_not_installed("dials")

  param <- obwoe_bin_cutoff()

  expect_s3_class(param, "quant_param")
  expect_equal(param$type, "double")
  expect_equal(param$range$lower, 0.01)
  expect_equal(param$range$upper, 0.10)
})


# ---------------------------------------------------------------------------- #
# Test Group: Integration with Workflows
# ---------------------------------------------------------------------------- #

test_that("step_obwoe works in a workflow", {
  skip_if_not_installed("OptimalBinningWoE")
  skip_if_not_installed("workflows")
  skip_if_not_installed("parsnip")

  # n = 200 rather than the 100 default: the target runs at 15% prevalence, so
  # 100 rows leave ~12 events, and a WoE transform fitted on those same rows
  # separates them perfectly in-sample -- glm then warns about fitted
  # probabilities of 0 or 1. 200 rows (~27 events) fit without separation, so
  # this smoke test proves the workflow wires up without leaving a warning
  # behind that would mask a real one.
  df <- create_small_data(n = 200)

  rec <- recipe(target ~ age + income, data = df) %>%
    step_obwoe(all_numeric_predictors(), outcome = "target")

  # Create a simple model spec
  mod <- parsnip::logistic_reg() %>%
    parsnip::set_engine("glm")

  wf <- workflows::workflow() %>%
    workflows::add_recipe(rec) %>%
    workflows::add_model(mod)

  # Fit the workflow
  fitted_wf <- generics::fit(wf, data = df)

  expect_s3_class(fitted_wf, "workflow")
})


# ---------------------------------------------------------------------------- #
# Test Group: Edge Cases
# ---------------------------------------------------------------------------- #

test_that("step_obwoe handles single observation per level", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data(n = 50)
  # Create rare category
  df$rare_cat <- "common"
  df$rare_cat[1] <- "rare"
  df$rare_cat <- as.factor(df$rare_cat)

  rec <- recipe(target ~ rare_cat, data = df) %>%
    step_obwoe(rare_cat, outcome = "target", algorithm = "jedi")

  # Should either work or warn, but not error
  result <- tryCatch(
    {
      prepped <- prep(rec, training = df)
      TRUE
    },
    warning = function(w) TRUE,
    error = function(e) FALSE
  )

  expect_true(result)
})

test_that("step_obwoe handles constant column", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()
  df$constant <- 5

  rec <- recipe(target ~ constant, data = df) %>%
    step_obwoe(constant, outcome = "target")

  # Should warn or handle gracefully
  result <- tryCatch(
    {
      suppressWarnings(prepped <- prep(rec, training = df))
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(result)
})

test_that("step_obwoe handles all NA column", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()
  df$all_na <- NA_real_

  rec <- recipe(target ~ all_na, data = df) %>%
    step_obwoe(all_na, outcome = "target")

  # Should warn or handle gracefully
  result <- tryCatch(
    {
      suppressWarnings(prepped <- prep(rec, training = df))
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(result)
})


# ---------------------------------------------------------------------------- #
# Test Group: Multiple Features
# ---------------------------------------------------------------------------- #

test_that("step_obwoe processes multiple numerical features", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age + income + score, data = df) %>%
    step_obwoe(all_numeric_predictors(), outcome = "target")

  prepped <- prep(rec, training = df)
  baked <- bake(prepped, new_data = df)

  # All columns should be transformed
  expect_true(all(c("age", "income", "score") %in% names(baked)))
})

test_that("step_obwoe processes mixed feature types with universal algorithm", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age + employment, data = df) %>%
    step_obwoe(all_predictors(), outcome = "target", algorithm = "jedi")

  prepped <- prep(rec, training = df)
  baked <- bake(prepped, new_data = df)

  expect_s3_class(baked, "tbl_df")
})


# ---------------------------------------------------------------------------- #
# Test Group: Reproducibility
# ---------------------------------------------------------------------------- #

test_that("step_obwoe produces reproducible results", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target")

  prepped1 <- prep(rec, training = df)
  prepped2 <- prep(rec, training = df)

  baked1 <- bake(prepped1, new_data = df)
  baked2 <- bake(prepped2, new_data = df)

  expect_equal(baked1$age, baked2$age)
})


# ---------------------------------------------------------------------------- #
# Test Group: Control Parameters
# ---------------------------------------------------------------------------- #

test_that("step_obwoe passes control parameters correctly", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age,
      outcome = "target",
      bin_cutoff = 0.08,
      control = list(max_n_prebins = 15)
    )

  prepped <- prep(rec, training = df)

  # Just verify it runs without error
  expect_true(prepped$steps[[1]]$trained)
})


# ---------------------------------------------------------------------------- #
# Test Group: Skip Parameter
# ---------------------------------------------------------------------------- #

test_that("step_obwoe respects skip = TRUE", {
  skip_if_not_installed("OptimalBinningWoE")

  df <- create_small_data()

  rec <- recipe(target ~ age, data = df) %>%
    step_obwoe(age, outcome = "target", skip = TRUE)

  prepped <- prep(rec, training = df)

  # When skipped, bake should return original data
  baked <- bake(prepped, new_data = df)

  # With skip = TRUE, the step is skipped during bake on new data
  # but the step is still trained
  expect_true(prepped$steps[[1]]$trained)
})
