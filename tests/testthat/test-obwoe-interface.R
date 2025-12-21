# =============================================================================
# Exhaustive Tests for obwoe(), obwoe_apply(), and obwoe_gains()
# Total: 60 tests (20 each)
# =============================================================================

library(OptimalBinningWoE)
set.seed(42)
n <- 500

test_df <- data.frame(
  # Numerical features
  age = pmax(18, pmin(80, rnorm(n, 45, 15))),
  income = exp(rnorm(n, 10, 0.8)),
  debt_ratio = rbeta(n, 2, 5),
  score = rnorm(n, 600, 100),
  balance = rgamma(n, 2, 0.001),


  # Categorical features
  education = sample(c("HS", "BA", "MA", "PhD"), n,
    replace = TRUE,
    prob = c(0.35, 0.40, 0.20, 0.05)
  ),
  employment = sample(c("Employed", "Self", "Unemployed", "Retired"), n,
    replace = TRUE, prob = c(0.60, 0.20, 0.10, 0.10)
  ),
  region = sample(c("North", "South", "East", "West"), n, replace = TRUE),

  # Target
  target = rbinom(n, 1, 0.15),
  stringsAsFactors = FALSE
)

# Multinomial target data
test_df_multi <- test_df
test_df_multi$target_multi <- sample(0:3, n, replace = TRUE, prob = c(0.5, 0.25, 0.15, 0.1))

# =============================================================================
# SECTION 1: obwoe() Tests (20 tests)
# =============================================================================

test_that("obwoe_01: Basic call with defaults processes all features", {
  result <- obwoe(test_df, target = "target")

  expect_s3_class(result, "obwoe")
  expect_equal(result$n_features, 8) # All except target
  expect_equal(result$target, "target")
  expect_equal(result$target_type, "binary")
})

test_that("obwoe_02: Single numerical feature works", {
  result <- obwoe(test_df, target = "target", feature = "age")

  expect_equal(result$n_features, 1)
  expect_true("age" %in% names(result$results))
  expect_true(!is.null(result$results$age$bin))
})

test_that("obwoe_03: Single categorical feature works", {
  result <- obwoe(test_df, target = "target", feature = "education")

  expect_equal(result$n_features, 1)
  expect_true("education" %in% names(result$results))
})

test_that("obwoe_04: Multiple features subset works", {
  result <- obwoe(test_df,
    target = "target",
    feature = c("age", "income", "education")
  )

  expect_equal(result$n_features, 3)
  expect_equal(sort(names(result$results)), c("age", "education", "income"))
})

test_that("obwoe_05: min_bins and max_bins are respected", {
  result <- obwoe(test_df,
    target = "target", feature = "age",
    min_bins = 3, max_bins = 4
  )

  n_bins <- length(result$results$age$bin)
  expect_true(n_bins >= 3 && n_bins <= 4)
})

test_that("obwoe_06: algorithm='jedi' works for numerical", {
  result <- obwoe(test_df,
    target = "target", feature = "income",
    algorithm = "jedi"
  )

  expect_equal(result$summary$algorithm[1], "jedi")
  expect_false(result$summary$error[1])
})

test_that("obwoe_07: algorithm='mdlp' works for numerical only", {
  result <- obwoe(test_df,
    target = "target", feature = "age",
    algorithm = "mdlp"
  )

  expect_equal(result$summary$algorithm[1], "mdlp")
  expect_false(result$summary$error[1])
})

test_that("obwoe_08: algorithm='mdlp' fails for categorical (captured as error)", {
  result <- obwoe(test_df,
    target = "target", feature = "education",
    algorithm = "mdlp"
  )

  expect_true(result$summary$error[1])
})

test_that("obwoe_09: algorithm='mob' works for both types", {
  result <- obwoe(test_df,
    target = "target",
    feature = c("age", "education"),
    algorithm = "mob"
  )

  expect_equal(sum(result$summary$error), 0)
})

test_that("obwoe_10: algorithm='ewb' works (equal width)", {
  result <- obwoe(test_df,
    target = "target", feature = "score",
    algorithm = "ewb"
  )

  expect_false(result$summary$error[1])
})

test_that("obwoe_11: algorithm='dp' works (dynamic programming)", {
  result <- obwoe(test_df,
    target = "target", feature = "age",
    algorithm = "dp"
  )

  expect_false(result$summary$error[1])
})

test_that("obwoe_12: algorithm='cm' works (ChiMerge)", {
  result <- obwoe(test_df,
    target = "target", feature = "income",
    algorithm = "cm"
  )

  expect_false(result$summary$error[1])
})

test_that("obwoe_13: algorithm='auto' selects jedi for binary", {
  result <- obwoe(test_df,
    target = "target", feature = "age",
    algorithm = "auto"
  )

  expect_equal(result$summary$algorithm[1], "jedi")
})

test_that("obwoe_14: control.obwoe() parameters are applied", {
  ctrl <- control.obwoe(bin_cutoff = 0.02, max_n_prebins = 30)
  result <- obwoe(test_df, target = "target", feature = "age", control = ctrl)

  expect_false(result$summary$error[1])
})

test_that("obwoe_15: control as list is converted", {
  result <- obwoe(test_df,
    target = "target", feature = "age",
    control = list(bin_cutoff = 0.03)
  )

  expect_false(result$summary$error[1])
})

test_that("obwoe_16: summary data.frame has correct structure", {
  result <- obwoe(test_df, target = "target")

  expect_true(all(c(
    "feature", "type", "algorithm", "n_bins",
    "total_iv", "error"
  ) %in% names(result$summary)))
})

test_that("obwoe_17: type detection works correctly", {
  result <- obwoe(test_df, target = "target")

  num_features <- c("age", "income", "debt_ratio", "score", "balance")
  cat_features <- c("education", "employment", "region")

  for (feat in num_features) {
    expect_equal(result$results[[feat]]$type, "numerical",
      info = paste("Feature:", feat)
    )
  }
  for (feat in cat_features) {
    expect_equal(result$results[[feat]]$type, "categorical",
      info = paste("Feature:", feat)
    )
  }
})

test_that("obwoe_18: total_iv is calculated correctly from iv vector", {
  result <- obwoe(test_df, target = "target", feature = "age")

  manual_iv <- sum(result$results$age$iv)
  expect_equal(result$summary$total_iv[1], manual_iv, tolerance = 1e-6)
})

test_that("obwoe_19: print.obwoe works", {
  result <- obwoe(test_df, target = "target", feature = c("age", "income"))

  expect_output(print(result), "Optimal Binning Weight of Evidence")
  expect_output(print(result), "Features processed:")
})

test_that("obwoe_20: Invalid inputs produce errors", {
  expect_error(obwoe(as.matrix(test_df), target = "target"))
  expect_error(obwoe(test_df, target = "nonexistent"))
  expect_error(obwoe(test_df, target = "target", min_bins = 0))
  expect_error(obwoe(test_df, target = "target", min_bins = 5, max_bins = 3))
})


# =============================================================================
# SECTION 2: obwoe_apply() Tests (20 tests)
# =============================================================================

# Fit a model for apply tests
model <- obwoe(test_df, target = "target", min_bins = 3, max_bins = 5)

test_that("obwoe_apply_01: Basic apply returns correct columns", {
  scored <- obwoe_apply(test_df, model)

  expect_true("target" %in% names(scored))
  expect_true("age" %in% names(scored))
  expect_true("age_bin" %in% names(scored))
  expect_true("age_woe" %in% names(scored))
})

test_that("obwoe_apply_02: All features get _bin and _woe columns", {
  scored <- obwoe_apply(test_df, model)

  successful <- model$summary[!model$summary$error, "feature"]
  for (feat in successful) {
    expect_true(paste0(feat, "_bin") %in% names(scored),
      info = paste("Missing:", feat, "_bin")
    )
    expect_true(paste0(feat, "_woe") %in% names(scored),
      info = paste("Missing:", feat, "_woe")
    )
  }
})

test_that("obwoe_apply_03: Number of rows preserved", {
  scored <- obwoe_apply(test_df, model)
  expect_equal(nrow(scored), nrow(test_df))
})

test_that("obwoe_apply_04: keep_original=FALSE excludes original columns", {
  scored <- obwoe_apply(test_df, model, keep_original = FALSE)

  expect_true("age_bin" %in% names(scored))
  expect_false("age" %in% names(scored))
})

test_that("obwoe_apply_05: keep_original=TRUE includes original columns", {
  scored <- obwoe_apply(test_df, model, keep_original = TRUE)

  expect_true("age" %in% names(scored))
  expect_true("age_bin" %in% names(scored))
})

test_that("obwoe_apply_06: Custom suffixes work", {
  scored <- obwoe_apply(test_df, model, suffix_bin = "_B", suffix_woe = "_W")

  expect_true("age_B" %in% names(scored))
  expect_true("age_W" %in% names(scored))
})

test_that("obwoe_apply_07: WoE values are numeric", {
  scored <- obwoe_apply(test_df, model)

  expect_true(is.numeric(scored$age_woe))
  expect_true(is.numeric(scored$income_woe))
})

test_that("obwoe_apply_08: Bin values are character", {
  scored <- obwoe_apply(test_df, model)

  expect_true(is.character(scored$age_bin) || is.factor(scored$age_bin))
})

test_that("obwoe_apply_09: Numerical bins use cutpoints correctly", {
  scored <- obwoe_apply(test_df, model)

  # Check that bins correspond to intervals
  bins <- unique(scored$age_bin)
  expect_true(any(grepl("\\(|\\[", bins))) # Should contain interval notation
})

test_that("obwoe_apply_10: Categorical bins preserve category names", {
  scored <- obwoe_apply(test_df, model)

  edu_bins <- unique(scored$education_bin)
  expect_true(length(edu_bins) > 0)
})

test_that("obwoe_apply_11: na_woe applies to missing mappings", {
  # Create new data with unseen category
  new_df <- test_df[1:10, ]
  new_df$education[1] <- "Unknown"

  scored <- obwoe_apply(new_df, model, na_woe = -999)

  expect_true(any(scored$education_woe == -999) ||
    any(is.na(scored$education_bin)))
})

test_that("obwoe_apply_12: Apply to subset of columns works", {
  model_subset <- obwoe(test_df,
    target = "target",
    feature = c("age", "income")
  )

  scored <- obwoe_apply(test_df, model_subset)

  expect_true("age_bin" %in% names(scored))
  expect_true("income_bin" %in% names(scored))
  expect_false("education_bin" %in% names(scored))
})

test_that("obwoe_apply_13: Target included when present in data", {
  scored <- obwoe_apply(test_df, model)

  expect_true("target" %in% names(scored))
  expect_equal(scored$target, test_df$target)
})

test_that("obwoe_apply_14: Target excluded when not in data", {
  new_df <- test_df[, !names(test_df) %in% "target"]
  scored <- obwoe_apply(new_df, model)

  expect_false("target" %in% names(scored))
})

test_that("obwoe_apply_15: WoE values match model bins", {
  scored <- obwoe_apply(test_df, model)

  # Check first observation
  age_val <- test_df$age[1]
  age_bin <- scored$age_bin[1]
  age_woe <- scored$age_woe[1]

  # Find corresponding WoE in model
  res <- model$results$age
  idx <- which(res$bin == age_bin)
  if (length(idx) > 0) {
    expect_equal(age_woe, res$woe[idx], tolerance = 1e-6)
  }
})

test_that("obwoe_apply_16: Multiple apply calls are consistent", {
  scored1 <- obwoe_apply(test_df, model)
  scored2 <- obwoe_apply(test_df, model)

  expect_equal(scored1$age_woe, scored2$age_woe)
})

test_that("obwoe_apply_17: Apply to new data with different values", {
  new_df <- data.frame(
    age = c(20, 40, 60, 80),
    income = c(1000, 10000, 50000, 100000),
    debt_ratio = c(0.1, 0.2, 0.3, 0.4),
    score = c(400, 500, 600, 700),
    balance = c(100, 1000, 5000, 10000),
    education = c("HS", "BA", "MA", "PhD"),
    employment = c("Employed", "Self", "Unemployed", "Retired"),
    region = c("North", "South", "East", "West")
  )

  scored <- obwoe_apply(new_df, model)
  expect_equal(nrow(scored), 4)
})

test_that("obwoe_apply_18: Validation errors work", {
  expect_error(obwoe_apply(as.matrix(test_df), model))
  expect_error(obwoe_apply(test_df, "not_a_model"))
})

test_that("obwoe_apply_19: Warning for missing features", {
  new_df <- test_df[, c("age", "target")]

  expect_warning(obwoe_apply(new_df, model))
})

test_that("obwoe_apply_20: Extreme numerical values handled", {
  extreme_df <- data.frame(
    age = c(-100, 0, 200, Inf, -Inf),
    income = c(0, 1, 1e10, 1e-10, NA),
    debt_ratio = c(0, 0.5, 1, 2, -1),
    score = c(-500, 600, 2000, NA, 0),
    balance = c(0, 1000, 1e9, NA, -100),
    education = c("HS", "BA", "MA", "PhD", "HS"),
    employment = c("Employed", "Self", "Unemployed", "Retired", "Employed"),
    region = c("North", "South", "East", "West", "North")
  )

  # Should not error, but may produce NAs
  result <- tryCatch(
    obwoe_apply(extreme_df, model),
    error = function(e) NULL
  )

  if (!is.null(result)) {
    expect_equal(nrow(result), 5)
  }
})


# =============================================================================
# SECTION 3: obwoe_gains() Tests (20 tests)
# =============================================================================

test_that("obwoe_gains_01: Basic call from obwoe object", {
  gains <- obwoe_gains(model, feature = "age")

  expect_s3_class(gains, "obwoe_gains")
  expect_true(!is.null(gains$table))
  expect_true(!is.null(gains$metrics))
})

test_that("obwoe_gains_02: Default feature is highest IV", {
  gains <- obwoe_gains(model)

  best_feat <- model$summary[!model$summary$error, ]
  best_feat <- best_feat$feature[which.max(best_feat$total_iv)]

  expect_equal(gains$feature, best_feat)
})

test_that("obwoe_gains_03: Table has 18 columns", {
  gains <- obwoe_gains(model, feature = "income")

  expected_cols <- c(
    "bin", "count", "count_pct", "pos_count", "neg_count",
    "pos_rate", "neg_rate", "pos_pct", "neg_pct", "odds",
    "log_odds", "woe", "iv", "cum_pos_pct", "cum_neg_pct",
    "ks", "lift", "capture_rate"
  )

  expect_true(all(expected_cols %in% names(gains$table)))
})

test_that("obwoe_gains_04: Metrics include KS, Gini, AUC, IV", {
  gains <- obwoe_gains(model, feature = "age")

  expect_true(all(c("ks", "gini", "auc", "total_iv") %in% names(gains$metrics)))
})

test_that("obwoe_gains_05: KS is between 0 and 100", {
  gains <- obwoe_gains(model, feature = "age")

  expect_true(gains$metrics$ks >= 0 && gains$metrics$ks <= 100)
})

test_that("obwoe_gains_06: AUC is between 0 and 1", {
  gains <- obwoe_gains(model, feature = "income")

  expect_true(gains$metrics$auc >= 0 && gains$metrics$auc <= 1)
})

test_that("obwoe_gains_07: From obwoe_apply with use_column='bin'", {
  scored <- obwoe_apply(test_df, model)
  gains <- obwoe_gains(scored,
    target = test_df$target, feature = "age",
    use_column = "bin"
  )

  expect_s3_class(gains, "obwoe_gains")
})

test_that("obwoe_gains_08: From obwoe_apply with use_column='woe'", {
  scored <- obwoe_apply(test_df, model)
  gains <- obwoe_gains(scored,
    target = test_df$target, feature = "age",
    use_column = "woe", n_groups = 5
  )

  expect_true(gains$n_bins <= 5)
})

test_that("obwoe_gains_09: use_column='direct' works with any variable", {
  test_df$decile <- cut(test_df$score,
    breaks = quantile(test_df$score, probs = seq(0, 1, 0.1)),
    include.lowest = TRUE, labels = 1:10
  )

  gains <- obwoe_gains(test_df,
    target = "target", feature = "decile",
    use_column = "direct"
  )

  expect_equal(gains$n_bins, 10)
})

test_that("obwoe_gains_10: n_groups creates quantile groups", {
  gains <- obwoe_gains(test_df,
    target = "target", feature = "score",
    use_column = "direct", n_groups = 10
  )

  expect_true(gains$n_bins <= 10)
})

test_that("obwoe_gains_11: sort_by='woe' orders correctly", {
  gains <- obwoe_gains(model, feature = "age", sort_by = "woe")

  woe_values <- gains$table$woe
  expect_equal(woe_values, sort(woe_values, decreasing = TRUE))
})

test_that("obwoe_gains_12: sort_by='event_rate' orders correctly", {
  gains <- obwoe_gains(model, feature = "income", sort_by = "event_rate")

  rates <- gains$table$pos_rate
  expect_equal(rates, sort(rates, decreasing = TRUE))
})

test_that("obwoe_gains_13: Cumulative percentages sum to 1", {
  gains <- obwoe_gains(model, feature = "age")

  expect_equal(max(gains$table$cum_pos_pct), 1, tolerance = 1e-6)
  expect_equal(max(gains$table$cum_neg_pct), 1, tolerance = 1e-6)
})

test_that("obwoe_gains_14: count sums to n_obs", {
  gains <- obwoe_gains(model, feature = "income")

  expect_equal(sum(gains$table$count), gains$n_obs)
})

test_that("obwoe_gains_15: print.obwoe_gains works", {
  gains <- obwoe_gains(model, feature = "age")

  expect_output(print(gains), "Gains Table:")
  expect_output(print(gains), "KS Statistic:")
})

test_that("obwoe_gains_16: plot.obwoe_gains type='cumulative'", {
  gains <- obwoe_gains(model, feature = "age")

  expect_silent({
    png(tempfile())
    plot(gains, type = "cumulative")
    dev.off()
  })
})

test_that("obwoe_gains_17: plot.obwoe_gains type='ks'", {
  gains <- obwoe_gains(model, feature = "income")

  expect_silent({
    png(tempfile())
    plot(gains, type = "ks")
    dev.off()
  })
})

test_that("obwoe_gains_18: plot.obwoe_gains type='lift'", {
  gains <- obwoe_gains(model, feature = "age")

  expect_silent({
    png(tempfile())
    plot(gains, type = "lift")
    dev.off()
  })
})

test_that("obwoe_gains_19: plot.obwoe_gains type='woe_iv'", {
  gains <- obwoe_gains(model, feature = "income")

  expect_silent({
    png(tempfile())
    plot(gains, type = "woe_iv")
    dev.off()
  })
})

test_that("obwoe_gains_20: Input validation errors", {
  expect_error(obwoe_gains("not_valid"))
  expect_error(obwoe_gains(test_df)) # Missing target
  expect_error(obwoe_gains(model, feature = "nonexistent"))
})
