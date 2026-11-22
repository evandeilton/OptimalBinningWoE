# Regression tests for the forensic audit fixes of group G1 (statistical tests
# and information theory engines). See audit/AUDITORIA_FORENSE_BINNING_CPP.md,
# Section 5 (X-nn) and Annex A (A-nn).

ref_iv_counts <- function(pos, neg, alpha = 0.5) {
  K <- length(pos)
  dp <- (pos + alpha) / (sum(pos) + K * alpha)
  dn <- (neg + alpha) / (sum(neg) + K * alpha)
  sum((dp - dn) * log(dp / dn))
}

# ---- X-03 / A-06: OBN_CM merge_zero_bins ignored min_bins -------------------

test_that("X-03: numerical cm keeps a perfect separator (no collapse to 1 bin)", {
  set.seed(1)
  x <- rnorm(2000)
  y <- as.integer(x > 0)
  r <- ob_numerical_cm(x, y, min_bins = 3)
  # baseline 1.14.1: 1 bin, IV = 0
  expect_gte(length(r$bin), 3L)
  expect_gt(r$total_iv, 1)
  expect_true(all(is.finite(r$woe)))
  expect_equal(sum(r$count), 2000L)
})

test_that("X-03: numerical cm keeps a binary flag whose level 0 has no events", {
  x <- c(rep(0, 500), rep(1, 500))
  set.seed(2)
  y <- c(rep(0L, 500), rbinom(500, 1, 0.1))
  r <- ob_numerical_cm(x, y)
  # baseline 1.14.1: 1 bin, WoE ~ 0, IV ~ 0
  expect_length(r$bin, 2L)
  expect_identical(r$count_pos[1], 0L)
  expect_lt(r$woe[1], r$woe[2])
  expect_gt(ref_iv_counts(r$count_pos, r$count_neg), 0.5)
})

test_that("X-03: pure bins are never merged below min_bins", {
  set.seed(11)
  x <- rnorm(3000)
  y <- as.integer(x > 0.5)
  k <- sample.int(3000, 2)
  y[k] <- 1L - y[k]
  for (mb in 2:4) {
    r <- ob_numerical_cm(x, y, min_bins = mb, max_bins = 6)
    expect_gte(length(r$bin), mb)
  }
})

# ---- X-05 / A-07: OBC_CM stopped on |delta min chi2| < tol ------------------

test_that("X-05: categorical cm merges levels with identical rates", {
  set.seed(7)
  lv <- c(sprintf("s%02d", 1:10), "h1", "h2")
  n_ev <- setNames(c(rep(20L, 10), 80L, 80L), lv)
  # exactly 100 rows per level: 20 events (s01..s10) or 80 events (h1, h2)
  f <- rep(lv, each = 100)
  y <- unlist(lapply(lv, function(l) sample(rep(c(1L, 0L), c(n_ev[[l]], 100L - n_ev[[l]])))))
  # baseline 1.14.1: 11 bins (stopped after one merge because two consecutive
  # minimum chi-squares were both 0)
  r2 <- ob_categorical_cm(f, y, min_bins = 2, max_bins = 12, chi_merge_threshold = 0.05)
  expect_length(r2$bin, 2L)
  r3 <- ob_categorical_cm(f, y, min_bins = 3, max_bins = 12, chi_merge_threshold = 0.05)
  expect_length(r3$bin, 3L)
})

# ---- X-07 / A-04: FastMDLPM force_min_bins dropped the MDL cuts -------------

test_that("X-07: fast_mdlp keeps the MDL cut when min_bins adds cuts", {
  set.seed(3)
  x <- runif(3000)
  y <- rbinom(3000, 1, ifelse(x > 0.5, 0.6, 0.1))
  mdl <- ob_numerical_fast_mdlp(x, y, min_bins = 2, max_bins = 10, force_monotonicity = FALSE)
  step_cut <- mdl$cutpoints[which.min(abs(mdl$cutpoints - 0.5))]
  expect_lt(abs(step_cut - 0.5), 0.02)
  for (mb in 3:5) {
    r <- ob_numerical_fast_mdlp(x, y, min_bins = mb, max_bins = 6, force_monotonicity = FALSE)
    # baseline 1.14.1: the MDL cut was replaced by equally spaced cuts
    expect_true(all(mdl$cutpoints %in% r$cutpoints), info = mb)
    expect_equal(length(r$cutpoints), max(mb - 1L, length(mdl$cutpoints)))
  }
})

test_that("X-07: forced cuts bisect the most populated bin, not the noise", {
  # Only the step at 0.3 is MDL-significant. The forced cut is NOT placed by
  # entropy gain (that selects on noise and lost hold-out IV in the audit
  # gate): it bisects the most populated bin, here the bin above 0.3, at its
  # median.
  set.seed(5)
  x <- runif(4000)
  y <- rbinom(4000, 1, ifelse(x > 0.3, 0.5, 0.1))
  mdl <- ob_numerical_fast_mdlp(x, y, min_bins = 2, max_bins = 10, force_monotonicity = FALSE)
  expect_length(mdl$cutpoints, 1L)
  r <- ob_numerical_fast_mdlp(x, y, min_bins = 3, max_bins = 3, force_monotonicity = FALSE)
  expect_length(r$cutpoints, 2L)
  expect_true(mdl$cutpoints %in% r$cutpoints)
  upper <- sort(x[x > mdl$cutpoints])
  added <- setdiff(r$cutpoints, mdl$cutpoints)
  expect_equal(mean(upper <= added), 0.5, tolerance = 0.01)
})

test_that("X-07: an MDL cut is kept in the min_bins completion only if it pays its code length", {
  # n = 28 (corpus size/small_n40, seed 1008): the only MDL cut (0.363)
  # isolates a pure tail of 10 events. Kept and completed by bisection it
  # shortens the label code by less than log2(27) = 4.75 bits against the
  # target-blind completion of 1.14.1, so the latter is returned; keeping the
  # MDL cut there was significantly worse out of sample (6/20 seeds worse,
  # none better, in the 20-seed gate).
  set.seed(1008)
  x <- rnorm(40)
  y <- rbinom(40, 1, plogis(x))
  idx <- sample.int(40, 28)
  x <- x[idx]
  y <- y[idx]
  expect_equal(ob_numerical_fast_mdlp(x, y, min_bins = 2, max_bins = 8)$cutpoints,
               0.3628241, tolerance = 1e-6)
  r <- ob_numerical_fast_mdlp(x, y, min_bins = 3, max_bins = 8)
  expect_equal(r$cutpoints, c(0.03692722, 0.69069198), tolerance = 1e-6)
  expect_equal(r$count_pos, c(5L, 4L, 7L))
  expect_equal(r$count_neg, c(9L, 3L, 0L))
})
