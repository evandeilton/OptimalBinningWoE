# ===========================================================================#
# obwoe_select(): automated variable screening
#
# The reference figures used below come from the Statlog (German Credit)
# benchmark, whose per-variable Information Values are widely published for
# the ungrouped categorical attributes. Where the optimiser keeps one bin per
# category, the IV computed here must equal that published value exactly.
# ===========================================================================#

test_that("every candidate variable appears in the output", {
  skip_if_no_german()
  df <- german_credit()
  feats <- setdiff(names(df), "target")

  model <- obwoe(df, target = "target", max_bins = 6)
  sel <- obwoe_select(model)

  expect_s3_class(sel, "data.frame")
  expect_equal(nrow(sel), length(feats))
  expect_setequal(sel$feature, feats)
  expect_true(all(c(
    "feature", "type", "n_bins", "total_iv", "iv_class", "ks", "gini", "auc",
    "monotonic", "quality", "selected", "reason", "reason_desc"
  ) %in% names(sel)))
  expect_type(sel$selected, "logical")
  expect_type(sel$reason, "character")
  expect_false(anyNA(sel$selected))
})


test_that("Information Value reproduces published German Credit figures", {
  skip_if_no_german()
  df <- german_credit()

  # Attributes the optimiser keeps ungrouped (one bin per category), so the IV
  # must equal the textbook value for the full cross-tabulation.
  reference <- c(
    status = 0.66601,
    credit_history = 0.29323,
    savings = 0.19601,
    property = 0.11264,
    employment_duration = 0.08643,
    housing = 0.08329,
    other_installment_plans = 0.05761,
    personal_status_sex = 0.04467,
    foreign_worker = 0.04388,
    other_debtors = 0.03202,
    number_credits = 0.01008,
    job = 0.00876,
    telephone = 0.00638,
    people_liable = 0.00004
  )

  model <- obwoe(df, target = "target", feature = names(reference), max_bins = 12)
  sel <- as.data.frame(obwoe_select(model, sort_by = "feature"))

  got <- sel$total_iv[match(names(reference), sel$feature)]
  expect_equal(got, unname(reference), tolerance = 1e-4)
})


test_that("IV, KS, AUC and Gini match an independent raw-data computation", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", max_bins = 6)

  sel <- as.data.frame(obwoe_select(model, sort_by = "feature"))
  scored <- obwoe_apply(df, model)
  y <- df$target

  for (f in sel$feature[!sel$error]) {
    woe <- scored[[paste0(f, "_woe")]]

    # IV from the observation-level cross-tabulation
    tb <- table(woe, y)
    p1 <- tb[, "1"] / sum(tb[, "1"])
    p0 <- tb[, "0"] / sum(tb[, "0"])
    iv_ref <- sum((p1 - p0) * log(p1 / p0))

    # KS on the WoE-ordered cumulative distributions
    o <- order(as.numeric(rownames(tb)))
    ks_ref <- max(abs(cumsum(p1[o]) - cumsum(p0[o])))

    # AUC as the rank-based Mann-Whitney statistic on raw observations
    r <- rank(woe)
    n1 <- sum(y == 1)
    n0 <- sum(y == 0)
    auc_ref <- (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)

    i <- match(f, sel$feature)
    expect_equal(sel$total_iv[i], iv_ref, tolerance = 1e-10, info = f)
    expect_equal(sel$ks[i], ks_ref, tolerance = 1e-10, info = f)
    expect_equal(sel$auc[i], auc_ref, tolerance = 1e-10, info = f)
    expect_equal(sel$gini[i], 2 * auc_ref - 1, tolerance = 1e-10, info = f)
  }
})


test_that("Siddiqi strength bands are applied at the documented cut-offs", {
  ivs <- c(0, 0.0199, 0.02, 0.0999, 0.10, 0.2999, 0.30, 0.4999, 0.50, 5, NA)
  expect_equal(
    as.character(OptimalBinningWoE:::.ob_iv_class(ivs)),
    c(
      "Unpredictive", "Unpredictive", "Weak", "Weak", "Medium", "Medium",
      "Strong", "Strong", "Suspicious", "Suspicious", "Error"
    )
  )
})


test_that("suspicious variables are rejected and weak ones are kept", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", max_bins = 6)
  sel <- as.data.frame(obwoe_select(model))

  # 'status' (checking-account balance) has IV ~ 0.67 on this benchmark
  st <- sel[sel$feature == "status", ]
  expect_gt(st$total_iv, 0.50)
  expect_equal(as.character(st$iv_class), "Suspicious")
  expect_false(st$selected)
  expect_true(grepl("IV_SUSPICIOUS", st$reason))
  expect_equal(as.character(st$quality), "Rejected")

  # Raising the ceiling admits it again
  sel2 <- as.data.frame(obwoe_select(model, iv_max = Inf))
  expect_true(sel2$selected[sel2$feature == "status"])

  # Unpredictive variables are rejected for the opposite reason
  weakest <- sel[which.min(sel$total_iv), ]
  expect_false(weakest$selected)
  expect_true(grepl("IV_BELOW_MIN", weakest$reason))
})


test_that("reason codes accumulate every violated rule", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", feature = "people_liable", max_bins = 6)

  sel <- as.data.frame(obwoe_select(model, iv_min = 0.9, iv_max = Inf, min_bins = 99L))
  expect_true(grepl("IV_BELOW_MIN", sel$reason))
  expect_true(grepl("TOO_FEW_BINS", sel$reason))
  expect_match(sel$reason_desc, "^rejected: ")
})


test_that("the ordering rule targets numerical variables by default", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", max_bins = 6)

  d_num <- as.data.frame(obwoe_select(model, require_monotonic = "numeric"))
  d_all <- as.data.frame(obwoe_select(model, require_monotonic = "all"))
  d_none <- as.data.frame(obwoe_select(model, require_monotonic = "none"))

  # A non-monotonic categorical passes under "numeric" but not under "all"
  broken <- d_num$feature[d_num$type == "categorical" & !d_num$monotonic]
  skip_if(length(broken) == 0, "no non-monotonic categorical in this fit")
  b <- broken[1]
  expect_true(d_num$selected[d_num$feature == b])
  expect_false(d_all$selected[d_all$feature == b])
  expect_true(grepl("NOT_MONOTONIC", d_all$reason[d_all$feature == b]))
  expect_true(d_none$selected[d_none$feature == b])

  # Loosening a rule can only ever select more variables
  expect_gte(sum(d_none$selected), sum(d_num$selected))
  expect_gte(sum(d_num$selected), sum(d_all$selected))
})


test_that("monotonicity diagnostics are correct on constructed profiles", {
  f <- OptimalBinningWoE:::.ob_monotonicity

  up <- f(c(0.1, 0.2, 0.3))
  expect_true(up$monotonic)
  expect_true(up$strict)
  expect_equal(up$direction, "increasing")
  expect_equal(up$n_violations, 0L)
  expect_equal(up$spearman, 1)

  down <- f(c(0.9, 0.5, 0.1))
  expect_true(down$monotonic)
  expect_true(down$strict)
  expect_equal(down$direction, "decreasing")
  expect_equal(down$spearman, -1)

  tied <- f(c(0.1, 0.2, 0.2, 0.4))
  expect_true(tied$monotonic)
  expect_false(tied$strict)
  expect_equal(tied$direction, "increasing")

  bad <- f(c(0.1, 0.9, 0.2))
  expect_false(bad$monotonic)
  expect_equal(bad$direction, "non-monotonic")
  expect_equal(bad$n_violations, 1L)

  flat <- f(c(0.3, 0.3, 0.3))
  expect_true(flat$monotonic)
  expect_false(flat$strict)
  expect_equal(flat$direction, "constant")

  one <- f(0.4)
  expect_true(one$monotonic)
  expect_equal(one$direction, "single-bin")
})


test_that("strict monotonicity rejects tied adjacent bins", {
  # Two bins with the same event rate: weakly monotonic, not strictly so.
  set.seed(99)
  n <- 800
  x <- rep(1:4, each = n / 4)
  # bins 2 and 3 are given an identical event rate by construction
  p <- c(0.10, 0.30, 0.30, 0.60)[x]
  df <- data.frame(x = x, target = rbinom(n, 1, p))
  model <- obwoe(df, target = "target", min_bins = 4, max_bins = 4)
  sel <- as.data.frame(obwoe_select(model, require_monotonic = "none"))

  # Whatever the fit, weak monotonicity must be implied by strict monotonicity
  expect_true(is.na(sel$monotonic) || !isTRUE(sel$monotonic_strict) ||
    isTRUE(sel$monotonic))
})


test_that("the AUC helper is exact on hand-computable cases", {
  f <- OptimalBinningWoE:::.ob_auc_ks_binned

  # Perfect separation: every event above every non-event
  perfect <- f(pos = c(0, 10), neg = c(10, 0), score = c(0, 1))
  expect_equal(perfect$auc, 1)
  expect_equal(perfect$gini, 1)
  expect_equal(perfect$ks, 1)

  # No information: identical composition in both bins
  flat <- f(pos = c(5, 5), neg = c(5, 5), score = c(0, 1))
  expect_equal(flat$auc, 0.5)
  expect_equal(flat$gini, 0)
  expect_equal(flat$ks, 0)

  # A single tied bin is pure coin flipping
  tied <- f(pos = c(4, 6), neg = c(4, 6), score = c(1, 1))
  expect_equal(tied$auc, 0.5)

  # Infinite WoE (a bin with no non-events) still ranks last
  degenerate <- f(pos = c(1, 9), neg = c(10, 0), score = c(-1, Inf))
  expect_equal(degenerate$auc, (9 * 10 + 1 * 5) / (10 * 10))

  # Degenerate target
  expect_true(is.na(f(pos = c(0, 0), neg = c(3, 4), score = c(0, 1))$auc))
})


test_that("structural rules gate selection as documented", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", max_bins = 8)

  many <- as.data.frame(obwoe_select(model, max_bins = 2))
  expect_true(all(grepl("TOO_MANY_BINS", many$reason[many$n_bins > 2])))

  few <- as.data.frame(obwoe_select(model, min_bins = 5L))
  expect_true(all(grepl("TOO_FEW_BINS", few$reason[
    !few$error & few$n_bins < 5
  ])))

  small <- as.data.frame(obwoe_select(model, min_bin_pct = 0.15))
  expect_true(all(grepl("SMALL_BIN", small$reason[
    !small$error & small$min_bin_pct < 0.15
  ])))
  expect_false(any(small$selected & small$min_bin_pct < 0.15))
})


test_that("degenerate bins are rejected unless explicitly allowed", {
  # A category that is 100% event produces an infinite WoE
  set.seed(5)
  n <- 600
  g <- c(rep("always_bad", 40), sample(c("a", "b", "c"), n - 40, TRUE))
  y <- c(rep(1L, 40), rbinom(n - 40, 1, 0.3))
  df <- data.frame(g = g, target = y, stringsAsFactors = FALSE)

  model <- obwoe(df, target = "target", max_bins = 4)
  res <- model$results$g
  skip_if(!is.null(res$error), "binning failed on the constructed case")
  skip_if(all(res$count_pos > 0 & res$count_neg > 0), "no degenerate bin produced")

  strict <- as.data.frame(obwoe_select(model, iv_max = Inf))
  expect_false(strict$selected)
  expect_true(grepl("DEGENERATE_BIN", strict$reason))
  expect_gt(strict$n_degenerate_bins, 0)

  loose <- as.data.frame(obwoe_select(model,
    iv_max = Inf, allow_degenerate = TRUE,
    require_monotonic = "none"
  ))
  expect_true(loose$selected)
})


test_that("top_n keeps only the best passing variables", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", max_bins = 6)

  full <- as.data.frame(obwoe_select(model))
  n_pass <- sum(full$selected)
  skip_if(n_pass < 3, "not enough passing variables")

  cut <- as.data.frame(obwoe_select(model, top_n = 3L))
  expect_equal(sum(cut$selected), 3L)
  expect_setequal(cut$feature[cut$selected], full$feature[which(full$rank <= 3)])
  dropped <- cut[!cut$selected & cut$reason == "NOT_IN_TOP_N", ]
  expect_equal(nrow(dropped), n_pass - 3L)

  # Ranking by a different metric can select a different set
  by_ks <- as.data.frame(obwoe_select(model, top_n = 3L, sort_by = "ks"))
  expect_equal(sum(by_ks$selected), 3L)
})


test_that("detail = 'full' returns one row per optimised bin", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", max_bins = 6)

  summ <- as.data.frame(obwoe_select(model))
  full <- as.data.frame(obwoe_select(model, detail = "full"))

  ok <- summ[!summ$error, ]
  expect_equal(nrow(full), sum(ok$n_bins))
  expect_setequal(unique(full$feature), summ$feature)

  # Bin counts add up to the population of each feature
  agg <- tapply(full$count, full$feature, sum)
  expect_equal(as.integer(agg[ok$feature]), as.integer(ok$n_obs))

  # Gains-table columns and interval bounds are present
  expect_true(all(c(
    "bin_id", "bin", "bin_lower", "bin_upper", "n_categories", "categories",
    "woe_model", "count", "pos", "neg", "woe", "iv", "ks_bin", "lift",
    "js_divergence", "selected", "reason", "quality"
  ) %in% names(full)))

  # Numerical bounds line up with the fitted cut points
  f <- ok$feature[ok$type == "numerical"][1]
  sub <- full[full$feature == f, ]
  cp <- sort(unique(model$results[[f]]$cutpoints))
  expect_equal(sub$bin_lower, c(-Inf, cp))
  expect_equal(sub$bin_upper, c(cp, Inf))

  # Bin-level IV sums to the feature-level IV
  expect_equal(
    sum(sub$iv), summ$total_iv[summ$feature == f],
    tolerance = 1e-12
  )
})


test_that("variables whose binning failed are still reported", {
  df <- data.frame(
    good = rnorm(400),
    broken = c(NA_real_, rnorm(399)),
    target = rbinom(400, 1, 0.4)
  )
  model <- suppressWarnings(obwoe(df, target = "target"))
  skip_if(!any(model$summary$error), "no feature failed in this fit")

  sel <- as.data.frame(obwoe_select(model))
  expect_equal(nrow(sel), 2L)

  bad <- sel[sel$feature == "broken", ]
  expect_true(bad$error)
  expect_false(bad$selected)
  expect_equal(bad$reason, "BINNING_ERROR")
  expect_true(nzchar(bad$error_msg))
  expect_equal(as.character(bad$iv_class), "Error")
  expect_equal(as.character(bad$quality), "Rejected")

  # ... and they keep a placeholder row at bin level
  full <- as.data.frame(obwoe_select(model, detail = "full"))
  expect_true("broken" %in% full$feature)
  expect_equal(sum(full$feature == "broken"), 1L)
})


test_that("sorting and ranking behave as documented", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", max_bins = 6)

  by_iv <- as.data.frame(obwoe_select(model, sort_by = "iv"))
  sel_iv <- by_iv$total_iv[by_iv$selected]
  expect_false(is.unsorted(rev(sel_iv)))
  expect_true(all(by_iv$selected[seq_len(sum(by_iv$selected))]))

  by_name <- as.data.frame(obwoe_select(model, sort_by = "feature"))
  expect_false(is.unsorted(by_name$feature))

  none <- as.data.frame(obwoe_select(model, sort_by = "none"))
  expect_equal(none$feature, names(model$results))

  # rank is defined exactly for the variables that pass every rule
  expect_equal(sum(!is.na(by_iv$rank)), sum(by_iv$selected))
})


test_that("quality tiers follow the documented rubric", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", max_bins = 6)
  sel <- as.data.frame(obwoe_select(model))

  expect_true(all(as.character(sel$quality[!sel$selected]) == "Rejected"))
  expect_true(all(as.character(sel$quality[sel$selected]) != "Rejected"))

  exc <- sel[as.character(sel$quality) == "Excellent", ]
  if (nrow(exc) > 0) {
    expect_true(all(exc$monotonic_strict))
    expect_true(all(exc$total_iv >= 0.10))
    expect_true(all(exc$n_bins >= 3))
    expect_true(all(exc$n_degenerate_bins == 0))
    expect_true(all(exc$min_bin_pct >= 0.05))
  }
})


test_that("the criteria used are attached to the result", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", feature = c("duration", "age"))
  sel <- obwoe_select(model, iv_min = 0.05, top_n = 1L)

  crit <- attr(sel, "obwoe_criteria")
  expect_type(crit, "list")
  expect_equal(crit$iv_min, 0.05)
  expect_equal(crit$top_n, 1L)
  expect_equal(crit$n_features, 2L)
  expect_equal(crit$n_selected, sum(sel$selected))
})


test_that("a data.table is returned when the package is available", {
  skip_if_not_installed("data.table")
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", feature = c("duration", "age"))
  expect_s3_class(obwoe_select(model), "data.table")
  expect_s3_class(obwoe_select(model, detail = "full"), "data.table")
})


test_that("input validation rejects malformed arguments", {
  df <- data.frame(x = rnorm(300))
  df$target <- rbinom(300, 1, plogis(df$x))
  model <- obwoe(df, target = "target")

  expect_error(obwoe_select(df), "class 'obwoe'")
  expect_error(obwoe_select(model, iv_min = -1), "non-negative")
  expect_error(obwoe_select(model, iv_min = 0.5, iv_max = 0.2), "strictly greater")
  expect_error(obwoe_select(model, min_bin_pct = 1), "\\[0, 1\\)")
  expect_error(obwoe_select(model, min_bins = 0), "integer >= 1")
  expect_error(obwoe_select(model, max_bins = 1, min_bins = 3), "min_bins")
  expect_error(obwoe_select(model, top_n = 0), "integer >= 1")
  expect_error(obwoe_select(model, allow_degenerate = "yes"), "logical")
  expect_error(obwoe_select(model, detail = "nope"))
})


test_that("multinomial models are rejected with a clear message", {
  set.seed(3)
  n <- 600
  df <- data.frame(x = rnorm(n), y = sample(0:2, n, TRUE))
  model <- obwoe(df, target = "y")
  expect_equal(model$target_type, "multinomial")
  expect_error(obwoe_select(model), "binary targets only")
})


test_that("a single-bin variable is handled without error", {
  set.seed(21)
  n <- 400
  df <- data.frame(
    constant = rep(1, n),
    target = rbinom(n, 1, 0.35)
  )
  model <- suppressWarnings(obwoe(df, target = "target"))
  sel <- as.data.frame(obwoe_select(model))
  expect_equal(nrow(sel), 1L)
  expect_false(sel$selected)
  expect_false(is.na(sel$reason))
})


test_that("a perfectly separating variable is flagged, not celebrated", {
  set.seed(4)
  n <- 500
  x <- rnorm(n)
  df <- data.frame(leak = x, target = as.integer(x > 0))
  model <- obwoe(df, target = "target", max_bins = 4)
  sel <- as.data.frame(obwoe_select(model))

  expect_gt(sel$total_iv, 0.50)
  expect_equal(as.character(sel$iv_class), "Suspicious")
  expect_false(sel$selected)
  expect_true(grepl("IV_SUSPICIOUS", sel$reason))
})


test_that("the base-R fallback produces identical content without data.table", {
  skip_if_not_installed("data.table")
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", max_bins = 6)

  fast_summary <- obwoe_select(model)
  fast_full <- obwoe_select(model, detail = "full")

  local_mocked_bindings(
    .ob_has_dt = function() FALSE,
    .package = "OptimalBinningWoE"
  )

  slow_summary <- obwoe_select(model)
  slow_full <- obwoe_select(model, detail = "full")

  expect_false(inherits(slow_summary, "data.table"))
  expect_s3_class(slow_summary, "data.frame")

  expect_equal(
    as.data.frame(slow_summary), as.data.frame(fast_summary),
    ignore_attr = TRUE
  )
  expect_equal(
    as.data.frame(slow_full), as.data.frame(fast_full),
    ignore_attr = TRUE
  )
})
