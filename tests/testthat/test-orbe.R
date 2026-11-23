# Tests for the Order-Restricted Bayesian Evidence engines:
# ob_numerical_orbe() and ob_categorical_orbe().

# ------------------------------------------------------------------------------
# Reference implementation in R: exhaustive search over contiguous partitions
# ------------------------------------------------------------------------------

# Beta prior with mean pi0 and `strength` pseudo-events of the minority class
orbe_prior <- function(pi0, strength = 1) {
  strength * c(pi0, 1 - pi0) / min(pi0, 1 - pi0)
}

# Log evidence of a bin with n rows and e events
orbe_f <- function(n, e, ab) {
  lbeta(ab[1] + e, ab[2] + n - e) - lbeta(ab[1], ab[2])
}

# Leave-one-out log predictive score of a bin
orbe_f_loo <- function(n, e, ab) {
  den <- sum(ab) + n - 1
  ifelse(e > 0, e * log((ab[1] + e - 1) / den), 0) +
    ifelse(n - e > 0, (n - e) * log((ab[2] + n - e - 1) / den), 0)
}

# Best additive score for every bin count, by enumeration of the cuts.
# `ok` receives the posterior logits of the bins and returns TRUE or FALSE.
orbe_brute <- function(n, e, ab, n_min, kmax, ok, f = orbe_f) {
  u <- length(n)
  best <- rep(-Inf, kmax)
  part <- vector("list", kmax)
  for (k in seq_len(min(kmax, u))) {
    cuts <- if (k == 1) list(integer(0)) else utils::combn(u - 1, k - 1, simplify = FALSE)
    for (cc in cuts) {
      g <- findInterval(seq_len(u), c(cc, u) + 1, left.open = FALSE) + 1
      bn <- as.numeric(tapply(n, g, sum))
      be <- as.numeric(tapply(e, g, sum))
      if (any(bn < n_min)) next
      if (!ok(digamma(ab[1] + be) - digamma(ab[2] + bn - be))) next
      val <- sum(f(bn, be, ab))
      if (val > best[k]) {
        best[k] <- val
        part[[k]] <- bn
      }
    }
  }
  list(best = best, part = part)
}

# Small numerical data set with few distinct values
orbe_small <- function(seed, u = 9, n = 400) {
  set.seed(seed)
  x <- sample(seq_len(u), n, replace = TRUE)
  y <- rbinom(n, 1, plogis(-1.5 + 0.35 * x + rnorm(u, 0, 0.4)[x]))
  list(x = x, y = y)
}

strictly_increasing <- function(r) all(diff(r) > 0)
strictly_decreasing <- function(r) all(diff(r) < 0)
is_peak <- function(r) {
  d <- sign(diff(r))
  all(d != 0) && !is.unsorted(-d)
}

# ------------------------------------------------------------------------------
# Numerical: exactness
# ------------------------------------------------------------------------------

test_that("numerical orbe matches exhaustive search without a shape", {
  for (seed in 1:6) {
    d <- orbe_small(seed)
    n <- as.numeric(table(d$x))
    e <- as.numeric(tapply(d$y, d$x, sum))
    ab <- orbe_prior(mean(d$y))
    n_min <- ceiling(0.05 * length(d$x))
    ref <- orbe_brute(n, e, ab, n_min, 6, function(r) TRUE)
    ref_score <- ref$best

    fit <- ob_numerical_orbe(d$x, d$y,
      min_bins = 1, max_bins = 6,
      monotonic_trend = "none", criterion = "evidence"
    )
    got <- fit$log_score_by_bins
    expect_equal(got[is.finite(ref_score)], ref_score[is.finite(ref_score)],
      tolerance = 1e-9
    )
    k <- which.max(ref_score)
    expect_equal(length(fit$bin), k)
    expect_equal(as.numeric(fit$count), ref$part[[k]])
    expect_equal(fit$log_score, max(ref_score), tolerance = 1e-9)
  }
})

test_that("numerical orbe matches exhaustive search under each shape", {
  shapes <- list(
    ascending = strictly_increasing,
    descending = strictly_decreasing,
    peak = is_peak,
    valley = function(r) is_peak(-r)
  )
  for (seed in 11:14) {
    d <- orbe_small(seed)
    n <- as.numeric(table(d$x))
    e <- as.numeric(tapply(d$y, d$x, sum))
    ab <- orbe_prior(mean(d$y))
    n_min <- ceiling(0.05 * length(d$x))
    for (shape in names(shapes)) {
      ref <- orbe_brute(n, e, ab, n_min, 5, shapes[[shape]])
      fit <- ob_numerical_orbe(d$x, d$y,
        min_bins = 1, max_bins = 5,
        monotonic_trend = shape, criterion = "evidence"
      )
      k <- length(fit$bin)
      expect_true(is.finite(ref$best[k]))
      expect_equal(as.numeric(fit$count), ref$part[[k]], info = paste(seed, shape))

      # Score = additive part + log Bayes factor of the order
      log_mass <- -lfactorial(k)
      if (shape %in% c("peak", "valley")) log_mass <- log_mass + (k - 1) * log(2)
      bf <- if (k > 1) log(fit$order_probability) - log_mass else 0
      expect_equal(fit$log_score,
        ref$best[k] + bf,
        tolerance = 1e-8, info = paste(seed, shape)
      )
      # Feasibility of every bin count agrees with the enumeration
      expect_equal(
        is.finite(fit$log_score_by_bins),
        is.finite(ref$best[seq_along(fit$log_score_by_bins)])
      )
    }
  }
})

test_that("order_probability agrees with simulation", {
  set.seed(42)
  x <- rnorm(1500)
  y <- rbinom(1500, 1, plogis(-1 + 0.25 * x))
  fit <- ob_numerical_orbe(x, y,
    min_bins = 4, max_bins = 4, bin_cutoff = 0.1,
    monotonic_trend = "ascending"
  )
  ab <- orbe_prior(mean(y))
  draws <- sapply(seq_along(fit$bin), function(i) {
    rbeta(2e5, ab[1] + fit$count_pos[i], ab[2] + fit$count_neg[i])
  })
  mc <- mean(apply(draws, 1, function(r) all(diff(r) > 0)))
  expect_equal(fit$order_probability, mc, tolerance = 0.01)

  # A peak includes both monotone orders, so its probability is larger
  fit_p <- ob_numerical_orbe(x, y,
    min_bins = 4, max_bins = 4, bin_cutoff = 0.1,
    monotonic_trend = "peak"
  )
  ab_draws <- sapply(seq_along(fit_p$bin), function(i) {
    rbeta(2e5, ab[1] + fit_p$count_pos[i], ab[2] + fit_p$count_neg[i])
  })
  mc_p <- mean(apply(ab_draws, 1, is_peak))
  expect_equal(fit_p$order_probability, mc_p, tolerance = 0.01)
})

# ------------------------------------------------------------------------------
# Numerical: contract and invariants
# ------------------------------------------------------------------------------

test_that("numerical orbe returns a valid, monotone fit", {
  data <- generate_test_data(n = 1000, seed = 501)
  fit <- ob_numerical_orbe(data$feature, data$target)

  validate_binning_result(fit, type = "numerical")
  validate_woe_values(fit$woe)
  expect_equal(sum(fit$count), length(data$feature))
  expect_equal(length(fit$cutpoints), length(fit$bin) - 1)
  expect_true(fit$monotonic_trend %in% c("ascending", "descending"))
  expect_true(strictly_increasing(fit$woe) || strictly_decreasing(fit$woe))
  expect_true(all(fit$count >= ceiling(0.05 * length(data$feature))))
  expect_true(all(fit$woe_lower < fit$woe & fit$woe < fit$woe_upper))
  expect_true(all(fit$woe_se > 0))
  expect_true(fit$total_iv_corrected <= fit$total_iv)
  expect_true(fit$order_probability > 0 && fit$order_probability <= 1)
  expect_equal(fit$total_iv, sum(fit$iv))
})

test_that("numerical orbe WoE is the digamma posterior mean", {
  data <- generate_test_data(n = 800, seed = 502)
  fit <- ob_numerical_orbe(data$feature, data$target, prior_strength = 2)
  pi0 <- mean(data$target)
  ab <- orbe_prior(pi0, 2)
  A <- ab[1] + fit$count_pos
  B <- ab[2] + fit$count_neg
  expect_equal(fit$woe, digamma(A) - digamma(B) - qlogis(pi0), tolerance = 1e-10)
  expect_equal(fit$woe_se, sqrt(trigamma(A) + trigamma(B)), tolerance = 1e-10)
  expect_equal(fit$woe_lower, qlogis(qbeta(0.025, A, B)) - qlogis(pi0), tolerance = 1e-8)
  expect_equal(fit$woe_upper, qlogis(qbeta(0.975, A, B)) - qlogis(pi0), tolerance = 1e-8)
})

test_that("numerical orbe selects the number of bins from the data", {
  set.seed(503)
  x <- rnorm(3000)
  # No relationship: one bin under the evidence, when allowed
  y0 <- rbinom(3000, 1, 0.2)
  fit0 <- ob_numerical_orbe(x, y0, min_bins = 1, max_bins = 10, criterion = "evidence")
  expect_equal(length(fit0$bin), 1)
  expect_equal(fit0$bin, "(-Inf;+Inf]")
  expect_equal(length(fit0$cutpoints), 0)

  # Two-level step: max_bins is not reached and a cut falls near the step
  y2 <- rbinom(3000, 1, ifelse(x > 0.3, 0.45, 0.10))
  fit2 <- ob_numerical_orbe(x, y2, min_bins = 1, max_bins = 10)
  expect_lt(length(fit2$bin), 10)
  expect_lt(min(abs(fit2$cutpoints - 0.3)), 0.15)

  # A penalty per bin recovers the two levels
  fit3 <- ob_numerical_orbe(x, y2,
    min_bins = 1, max_bins = 10,
    criterion = "evidence", bin_penalty = 3
  )
  expect_equal(length(fit3$bin), 2)
  expect_lt(abs(fit3$cutpoints - 0.3), 0.15)
})

test_that("numerical orbe recovers a peak", {
  set.seed(504)
  x <- runif(4000, -2, 2)
  y <- rbinom(4000, 1, plogis(0.5 - 1.5 * x^2))
  fit <- ob_numerical_orbe(x, y, monotonic_trend = "peak")
  expect_true(is_peak(fit$woe))
  expect_gt(length(fit$bin), 2)
  expect_gt(which.max(fit$woe), 1)
  expect_lt(which.max(fit$woe), length(fit$bin))

  fit_v <- ob_numerical_orbe(x, 1L - y, monotonic_trend = "valley")
  expect_equal(fit_v$count, fit$count)
  expect_true(is_peak(-fit_v$woe))
})

test_that("numerical orbe is invariant to row order and monotone transformations", {
  set.seed(505)
  x <- round(rexp(1500), 2)
  y <- rbinom(1500, 1, plogis(-2 + x))
  fit <- ob_numerical_orbe(x, y)

  idx <- sample(length(x))
  fit_perm <- ob_numerical_orbe(x[idx], y[idx])
  expect_identical(fit_perm$bin, fit$bin)
  expect_identical(fit_perm$woe, fit$woe)

  fit_log <- ob_numerical_orbe(log1p(x), y)
  expect_identical(fit_log$count, fit$count)
  expect_equal(fit_log$woe, fit$woe)
  expect_equal(expm1(fit_log$cutpoints), fit$cutpoints, tolerance = 1e-10)

  # A decreasing transformation mirrors the bins
  fit_neg <- ob_numerical_orbe(-x, y)
  expect_equal(fit_neg$monotonic_trend, "descending")
})

test_that("numerical orbe follows the NA and infinity rules", {
  set.seed(506)
  x <- rnorm(600)
  y <- rbinom(600, 1, plogis(x))
  fit <- ob_numerical_orbe(x, y)

  x_na <- c(x, NA, NaN, NA)
  y_na <- c(y, 1L, 0L, 1L)
  fit_na <- ob_numerical_orbe(x_na, y_na)
  expect_identical(fit_na$bin, fit$bin)
  expect_identical(fit_na$count, fit$count)

  x_inf <- c(x, -Inf, Inf)
  y_inf <- c(y, 0L, 1L)
  fit_inf <- ob_numerical_orbe(x_inf, y_inf)
  expect_equal(sum(fit_inf$count), 602)
  expect_true(all(is.finite(fit_inf$cutpoints)))

  expect_error(ob_numerical_orbe(c(1, 2, 3), c(1L, NA, 0L)), "missing")
  expect_error(ob_numerical_orbe(rep(NA_real_, 4), c(0L, 1L, 0L, 1L)), "non-missing")
})

test_that("numerical orbe validates its arguments", {
  x <- c(1, 2, 3, 4, 5, 6)
  y <- c(0L, 0L, 1L, 0L, 1L, 1L)
  expect_error(ob_numerical_orbe(x, y[-1]), "match")
  expect_error(ob_numerical_orbe(numeric(0), integer(0)), "empty")
  expect_error(ob_numerical_orbe(x, y, min_bins = 0), "min_bins")
  expect_error(ob_numerical_orbe(x, y, min_bins = 4, max_bins = 3), "max_bins")
  expect_error(ob_numerical_orbe(x, y, bin_cutoff = 1), "bin_cutoff")
  expect_error(ob_numerical_orbe(x, y, max_n_prebins = 1), "max_n_prebins")
  expect_error(ob_numerical_orbe(x, y, monotonic_trend = "up"), "monotonic_trend")
  expect_error(ob_numerical_orbe(x, y, prior_strength = 0), "prior_strength")
  expect_error(ob_numerical_orbe(x, y, credible_level = 1), "credible_level")
  expect_error(ob_numerical_orbe(x, rep(0L, 6)), "both classes")
  expect_error(ob_numerical_orbe(x, c(0L, 1L, 2L, 0L, 1L, 2L)), "binary")
})

test_that("numerical orbe handles degenerate features", {
  y <- rep(c(0L, 1L), 50)
  # Constant feature: one bin whatever min_bins
  fit <- ob_numerical_orbe(rep(3, 100), y, min_bins = 3)
  expect_equal(length(fit$bin), 1)
  expect_equal(fit$count, 100L)

  # Two distinct values and min_bins above what is feasible
  fit2 <- ob_numerical_orbe(rep(c(1, 2), each = 50), c(rep(0:1, c(40, 10)), rep(0:1, c(15, 35))),
    min_bins = 4
  )
  expect_equal(length(fit2$bin), 2)

  # Pure bins have finite WoE and intervals
  x <- rep(1:4, each = 50)
  yp <- c(rep(0L, 50), rbinom(100, 1, 0.5), rep(1L, 50))
  fit3 <- ob_numerical_orbe(x, yp, bin_cutoff = 0)
  expect_true(all(is.finite(c(fit3$woe, fit3$woe_lower, fit3$woe_upper, fit3$iv))))
})

test_that("numerical orbe groups distinct values above max_n_prebins", {
  set.seed(507)
  x <- rnorm(5000)
  y <- rbinom(5000, 1, plogis(-1 + x))
  fit <- ob_numerical_orbe(x, y, max_n_prebins = 50)
  validate_binning_result(fit, type = "numerical")
  expect_equal(sum(fit$count), 5000)
  expect_true(strictly_increasing(fit$woe))
  # Cutpoints are observed values
  expect_true(all(fit$cutpoints %in% x))
  # The bins reproduce the counts
  expect_equal(
    as.integer(table(cut(x, c(-Inf, fit$cutpoints, Inf)))),
    as.integer(fit$count)
  )
})

# ------------------------------------------------------------------------------
# Categorical
# ------------------------------------------------------------------------------

test_that("categorical orbe matches exhaustive search over the rate order", {
  for (seed in 21:25) {
    set.seed(seed)
    lev <- letters[1:8]
    f <- sample(lev, 500, replace = TRUE, prob = c(rep(0.14, 6), 0.08, 0.08))
    y <- rbinom(500, 1, plogis(-1.5 + rnorm(8)[match(f, lev)]))
    ab <- orbe_prior(mean(y))
    n <- as.numeric(table(f))
    e <- as.numeric(tapply(y, f, sum))
    ord <- order((ab[1] + e) / (sum(ab) + n), n, names(table(f)))
    n_min <- ceiling(0.05 * 500)
    ref <- orbe_brute(n[ord], e[ord], ab, n_min, 6, strictly_increasing)

    ref_score <- ref$best

    fit <- ob_categorical_orbe(f, y, min_bins = 1, max_bins = 6, criterion = "evidence")
    ok <- is.finite(ref_score)
    expect_equal(fit$log_score_by_bins[ok], ref_score[ok], tolerance = 1e-9)
    k <- which.max(ref_score)
    expect_equal(length(fit$bin), k)
    expect_equal(as.numeric(fit$count), ref$part[[k]])
    # Labels follow the rate order
    expect_identical(
      unlist(strsplit(fit$bin, "%;%", fixed = TRUE)),
      names(table(f))[ord]
    )
  }
})

test_that("categorical orbe returns a valid fit ordered by WoE", {
  set.seed(601)
  f <- sample(LETTERS[1:10], 2000, replace = TRUE)
  y <- rbinom(2000, 1, plogis(-2 + match(f, LETTERS) / 4))
  fit <- ob_categorical_orbe(f, y)

  validate_binning_result(fit, type = "categorical")
  validate_woe_values(fit$woe)
  expect_equal(sum(fit$count), 2000)
  expect_true(strictly_increasing(fit$woe))
  expect_true(all(fit$woe_lower < fit$woe & fit$woe < fit$woe_upper))
  expect_setequal(unlist(strsplit(fit$bin, "%;%", fixed = TRUE)), LETTERS[1:10])
  expect_true(length(fit$bin) < 10)
})

test_that("categorical orbe merges rare categories by the evidence", {
  set.seed(602)
  lev <- c("a", "b", "c", "d", "rare1", "rare2", "rare3")
  f <- sample(lev, 3000, replace = TRUE, prob = c(rep(0.2475, 4), rep(0.0033, 3)))
  rate <- c(0.05, 0.15, 0.30, 0.50, 0.20, 0.20, 0.20)
  y <- rbinom(3000, 1, rate[match(f, lev)])
  fit <- ob_categorical_orbe(f, y, bin_cutoff = 0, min_bins = 1)

  # The four frequent categories are separated; no rare one stands alone
  expect_equal(length(fit$bin), 4)
  members <- strsplit(fit$bin, "%;%", fixed = TRUE)
  expect_false(any(vapply(members, function(m) all(grepl("^rare", m)), logical(1))))
})

test_that("categorical orbe is invariant to row order and handles NA", {
  set.seed(603)
  f <- sample(c(letters[1:6], NA), 800, replace = TRUE)
  y <- rbinom(800, 1, 0.3)
  y[!is.na(f) & f %in% c("a", "b")] <- rbinom(sum(!is.na(f) & f %in% c("a", "b")), 1, 0.7)
  fit <- ob_categorical_orbe(f, y)
  idx <- sample(800)
  fit_perm <- ob_categorical_orbe(f[idx], y[idx])
  expect_identical(fit_perm$bin, fit$bin)
  expect_identical(fit_perm$woe, fit$woe)
  expect_true(any(grepl("NA", fit$bin, fixed = TRUE)))

  # Factors are accepted
  fit_fac <- ob_categorical_orbe(factor(f), y)
  expect_identical(fit_fac$bin, fit$bin)
})

test_that("categorical orbe validates its arguments and degenerate input", {
  f <- c("a", "b", "a", "b", "c", "c")
  y <- c(0L, 1L, 0L, 1L, 1L, 0L)
  expect_error(ob_categorical_orbe(f, y[-1]), "same size")
  expect_error(ob_categorical_orbe(character(0), integer(0)), "empty")
  expect_error(ob_categorical_orbe(f, y, min_bins = 0), "min_bins")
  expect_error(ob_categorical_orbe(f, y, min_bins = 3, max_bins = 2), "max_bins")
  expect_error(ob_categorical_orbe(f, y, bin_cutoff = -0.1), "bin_cutoff")
  expect_error(ob_categorical_orbe(f, y, prior_strength = -1), "prior_strength")
  expect_error(ob_categorical_orbe(f, rep(1L, 6)), "both values")
  expect_error(ob_categorical_orbe(f, c(y[-1], NA)), "missing")

  # One category: one bin
  fit <- ob_categorical_orbe(rep("only", 20), rep(0:1, 10), min_bins = 2)
  expect_equal(fit$bin, "only")
  expect_equal(fit$count, 20L)
})

# ------------------------------------------------------------------------------
# Integration with the package interface
# ------------------------------------------------------------------------------

test_that("orbe is registered for numerical and categorical features", {
  algos <- obwoe_algorithms()
  row <- algos[algos$algorithm == "orbe", ]
  expect_equal(nrow(row), 1)
  expect_true(row$numerical)
  expect_true(row$categorical)
  expect_false(row$multinomial)
})

test_that("obwoe() fits and applies orbe on mixed features", {
  set.seed(701)
  n <- 1200
  df <- data.frame(
    num = rnorm(n),
    cat = sample(c("x", "y", "z", "w"), n, replace = TRUE),
    stringsAsFactors = FALSE
  )
  df$target <- rbinom(n, 1, plogis(-1 + 0.9 * df$num + 0.8 * (df$cat == "x")))
  df$num[sample(n, 30)] <- NA

  fit <- obwoe(df, target = "target", algorithm = "orbe")
  expect_s3_class(fit, "obwoe")
  expect_false(any(vapply(fit$results, function(r) !is.null(r$error), logical(1))))
  expect_true(all(is.finite(fit$summary$total_iv)))

  scored <- obwoe_apply(df, fit)
  expect_equal(nrow(scored), n)
  expect_true(all(c("num_woe", "cat_woe") %in% names(scored)))
  expect_true(all(is.finite(scored$cat_woe)))
})

test_that("orbe works through ob_apply_woe_num and ob_apply_woe_cat", {
  set.seed(702)
  x <- rnorm(500)
  y <- rbinom(500, 1, plogis(x))
  fit <- ob_numerical_orbe(x, y)
  applied <- ob_apply_woe_num(fit, x)
  expect_equal(nrow(applied), 500)
  expect_setequal(unique(applied$woe), fit$woe)

  f <- sample(letters[1:5], 500, replace = TRUE)
  yc <- rbinom(500, 1, plogis(-1 + match(f, letters) / 2))
  fit_c <- ob_categorical_orbe(f, yc)
  applied_c <- ob_apply_woe_cat(fit_c, f)
  expect_equal(nrow(applied_c), 500)
  expect_true(all(applied_c$woe %in% fit_c$woe))
})

test_that("step_obwoe accepts orbe", {
  skip_if_not_installed("recipes")
  set.seed(703)
  df <- data.frame(x = rnorm(400), g = sample(c("a", "b", "c"), 400, replace = TRUE))
  df$y <- factor(rbinom(400, 1, plogis(df$x + (df$g == "a"))))
  rec <- recipes::recipe(y ~ ., data = df)
  rec <- step_obwoe(rec, recipes::all_predictors(), outcome = "y", algorithm = "orbe")
  prepped <- recipes::prep(rec, training = df)
  baked <- recipes::bake(prepped, new_data = df)
  expect_equal(nrow(baked), 400)
  expect_true(all(vapply(baked[c("x", "g")], is.numeric, logical(1))))
})

test_that("orbe keeps the atoms when max_bins is very large", {
  set.seed(801)
  x <- rnorm(500)
  y <- rbinom(500, 1, plogis(x))
  fit <- ob_numerical_orbe(x, y, max_bins = 1e6, bin_cutoff = 0.01)
  ref <- ob_numerical_orbe(x, y, max_bins = 100, bin_cutoff = 0.01)
  expect_identical(fit$bin, ref$bin)
  expect_equal(fit$log_score, ref$log_score)
})

test_that("orbe accepts a bin with exactly bin_cutoff of the rows", {
  # 0.07 * 100 is slightly above 7 in floating point
  x <- c(rep(1, 7), rep(2, 93))
  y <- c(rep(1L, 7), rep(0:1, c(90, 3)))
  fit <- ob_numerical_orbe(x, y, min_bins = 2, bin_cutoff = 0.07, monotonic_trend = "none")
  expect_equal(as.integer(fit$count), c(7L, 93L))
})

test_that("order_probability is accurate for many similar bins", {
  # Twelve nearly equal bins, where the quadrature error is largest
  x <- rep(1:12, each = 100)
  y2 <- unlist(lapply(1:12, function(i) rep(0:1, c(70 - i, 30 + i))))
  fit2 <- ob_numerical_orbe(x, y2,
    min_bins = 12, max_bins = 12, bin_cutoff = 0,
    monotonic_trend = "ascending"
  )
  ab <- orbe_prior(mean(y2))
  set.seed(802)
  draws <- sapply(1:12, function(i) rbeta(4e5, ab[1] + 30 + i, ab[2] + 70 - i))
  mc <- mean(apply(draws, 1, function(r) all(diff(r) > 0)))
  expect_equal(fit2$order_probability, mc, tolerance = 0.05)
})

test_that("categorical orbe warns when the separator is inside a category", {
  f <- rep(c("a%;%b", "c", "d"), each = 30)
  y <- rep(c(0L, 1L, 1L, 0L, 0L, 1L), 15)
  expect_warning(ob_categorical_orbe(f, y), "bin_separator")
})

test_that("bin_penalty subtracts a cost per additional bin", {
  d <- orbe_small(31)
  free <- ob_numerical_orbe(d$x, d$y, min_bins = 1, max_bins = 6, monotonic_trend = "none")
  pen <- ob_numerical_orbe(d$x, d$y,
    min_bins = 1, max_bins = 6, monotonic_trend = "none",
    bin_penalty = 1.5
  )
  k <- seq_along(free$log_score_by_bins)
  expect_equal(pen$log_score_by_bins, free$log_score_by_bins - 1.5 * (k - 1))
  expect_lte(length(pen$bin), length(free$bin))

  f <- sample(letters[1:8], 600, replace = TRUE)
  y <- rbinom(600, 1, plogis(-1 + match(f, letters) / 5))
  free_c <- ob_categorical_orbe(f, y, min_bins = 1)
  pen_c <- ob_categorical_orbe(f, y, min_bins = 1, bin_penalty = 2)
  kc <- seq_along(free_c$log_score_by_bins)
  expect_equal(pen_c$log_score_by_bins, free_c$log_score_by_bins - 2 * (kc - 1))

  expect_error(ob_numerical_orbe(d$x, d$y, bin_penalty = -1), "bin_penalty")
  expect_error(ob_categorical_orbe(f, y, bin_penalty = NA), "bin_penalty")
})

test_that("orbe matches exhaustive search with the predictive criterion", {
  for (seed in 41:46) {
    d <- orbe_small(seed)
    n <- as.numeric(table(d$x))
    e <- as.numeric(tapply(d$y, d$x, sum))
    ab <- orbe_prior(mean(d$y))
    n_min <- ceiling(0.05 * length(d$x))

    ref <- orbe_brute(n, e, ab, n_min, 6, function(r) TRUE, orbe_f_loo)
    fit <- ob_numerical_orbe(d$x, d$y, min_bins = 1, max_bins = 6, monotonic_trend = "none")
    expect_equal(fit$criterion, "predictive")
    ok <- is.finite(ref$best)
    expect_equal(fit$log_score_by_bins[ok], ref$best[ok], tolerance = 1e-9)
    expect_equal(as.numeric(fit$count), ref$part[[which.max(ref$best)]])

    ref_a <- orbe_brute(n, e, ab, n_min, 6, strictly_increasing, orbe_f_loo)
    fit_a <- ob_numerical_orbe(d$x, d$y, min_bins = 1, max_bins = 6, monotonic_trend = "ascending")
    k <- length(fit_a$bin)
    expect_equal(as.numeric(fit_a$count), ref_a$part[[k]])
    bf <- if (k > 1) log(fit_a$order_probability) + lfactorial(k) else 0
    expect_equal(fit_a$log_score, ref_a$best[k] + bf, tolerance = 1e-8)
  }

  # Categorical: same engine over the rate order
  set.seed(47)
  lev <- letters[1:7]
  f <- sample(lev, 400, replace = TRUE)
  y <- rbinom(400, 1, plogis(-1 + rnorm(7)[match(f, lev)]))
  ab <- orbe_prior(mean(y))
  n <- as.numeric(table(f))
  e <- as.numeric(tapply(y, f, sum))
  ord <- order((ab[1] + e) / (sum(ab) + n), n, names(table(f)))
  ref <- orbe_brute(n[ord], e[ord], ab, 20, 6, strictly_increasing, orbe_f_loo)
  fit <- ob_categorical_orbe(f, y, min_bins = 1, max_bins = 6)
  ok <- is.finite(ref$best)
  expect_equal(fit$log_score_by_bins[ok], ref$best[ok], tolerance = 1e-9)
  expect_equal(as.numeric(fit$count), ref$part[[which.max(ref$best)]])
})

test_that("orbe validates the criterion and the evidence is more parsimonious", {
  set.seed(48)
  x <- rnorm(3000)
  y <- rbinom(3000, 1, plogis(-2 + 0.8 * x))
  expect_error(ob_numerical_orbe(x, y, criterion = "aic"), "criterion")
  expect_error(ob_numerical_orbe(x, y, criterion = c("predictive", "evidence")), "criterion")
  expect_error(ob_numerical_orbe(x, y, bin_penalty = 1e308), "bin_penalty")
  expect_error(ob_categorical_orbe(letters[1 + (x > 0)], y, criterion = "aic"), "criterion")
  pred <- ob_numerical_orbe(x, y)
  evid <- ob_numerical_orbe(x, y, criterion = "evidence", bin_penalty = 2)
  expect_equal(evid$criterion, "evidence")
  expect_gt(length(pred$bin), length(evid$bin))
  expect_gte(length(pred$bin), 5)
})
