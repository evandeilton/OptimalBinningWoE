# ---------------------------------------------------------------------------#
# Audit regressions and invariants for ldb, lpdb, mblp and mdlp.
#
# Every test in the "regression" blocks fails on the code before this audit
# and passes after it; the invariant blocks pin the output contract of the
# four engines over a battery of ordinary and adversarial inputs.
# ---------------------------------------------------------------------------#

numc_capture <- function(expr) {
  w <- character()
  r <- withCallingHandlers(expr, warning = function(e) {
    w <<- c(w, conditionMessage(e))
    invokeRestart("muffleWarning")
  })
  list(res = r, warnings = w)
}

# WoE as documented for ldb/lpdb/mblp: Laplace alpha = 0.5 and K = the final
# number of bins.
numc_doc_woe <- function(r, alpha = 0.5) {
  k <- length(r$bin)
  p <- sum(r$count_pos)
  n <- sum(r$count_neg)
  log(((r$count_pos + alpha) / (p + k * alpha)) /
    ((r$count_neg + alpha) / (n + k * alpha)))
}

numc_monotone <- function(w, tol = 1e-9) {
  length(w) < 2 || all(diff(w) >= -tol) || all(diff(w) <= tol)
}

# Output contract; returns the violated clauses (empty when all hold).
numc_problems <- function(r, n_expected, max_bins, min_bins = NULL) {
  nb <- length(r$bin)
  p <- c(
    if (nb < 1 || nb > max_bins) "bin count outside [1, max_bins]",
    if (sum(r$count) != n_expected) "counts do not sum to n",
    if (any(r$count != r$count_pos + r$count_neg)) "count != pos + neg",
    if (any(r$count <= 0)) "empty bin",
    if (!all(is.finite(r$woe)) || !all(is.finite(r$iv))) "non-finite WoE/IV",
    if (!isTRUE(all.equal(r$total_iv, sum(r$iv), tolerance = 1e-12))) "total_iv != sum(iv)",
    if (length(r$cutpoints) != nb - 1) "cutpoints/bins length mismatch",
    if (!all(is.finite(r$cutpoints))) "non-finite cutpoint",
    if (length(r$cutpoints) > 1 && any(diff(r$cutpoints) <= 0)) "cutpoints not increasing",
    # monotone WoE whenever the engine could still merge (above min_bins)
    if (!is.null(min_bins) && nb > min_bins && !numc_monotone(r$woe)) "WoE not monotone above min_bins"
  )
  p
}

numc_fns <- list(
  ldb = function(x, y, ...) ob_numerical_ldb(x, y, ...),
  lpdb = function(x, y, ...) ob_numerical_lpdb(x, y, ...),
  mblp = function(x, y, ...) ob_numerical_mblp(x, y, ...),
  mdlp = function(x, y, ...) ob_numerical_mdlp(x, y, ...)
)

# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------
test_that("ldb/lpdb/mblp/mdlp keep the output contract on a fuzz battery", {
  gens <- list(
    normal = function(n) rnorm(n),
    bimodal = function(n) c(rnorm(n %/% 2, -3), rnorm(n - n %/% 2, 3)),
    rounded = function(n) round(rnorm(n), 1),
    five_values = function(n) sample(1:5, n, TRUE),
    two_values = function(n) sample(c(0, 1), n, TRUE),
    constant = function(n) rep(3.5, n),
    lognormal = function(n) rlnorm(n, 0, 2),
    cauchy = function(n) rcauchy(n),
    extremes = function(n) c(rnorm(n - 2), 1e300, -1e300),
    tiny_scale = function(n) runif(n) * 1e-9,
    skewed_three = function(n) sample(1:3, n, TRUE, prob = c(.8, .15, .05)),
    zero_mass = function(n) c(rep(0, n %/% 2), rnorm(n - n %/% 2)),
    offset = function(n) rnorm(n) * 1e-3 + 1e6
  )
  set.seed(20260925)
  failures <- character()
  for (g in names(gens)) {
    for (n in c(7L, 25L, 400L)) {
      x <- gens[[g]](n)
      z <- rank(x) / n - 0.5
      y <- rbinom(n, 1, plogis(-0.5 + 3 * z))
      y[1:2] <- c(0L, 1L)
      for (a in names(numc_fns)) {
        for (pars in list(list(), list(min_bins = 2, max_bins = 7, bin_cutoff = 0.02))) {
          label <- sprintf("%s/%s/n=%d/%d", a, g, n, length(pars))
          out <- numc_capture(do.call(numc_fns[[a]], c(list(x, y), pars)))
          mx <- if (length(pars)) 7 else 5
          mn <- if (length(pars)) 2 else 3
          problems <- c(
            if (length(out$warnings)) paste("warning:", out$warnings),
            numc_problems(out$res, n, mx, mn)
          )
          if (length(problems)) failures <- c(failures, paste(label, problems))
        }
      }
    }
  }
  expect_identical(failures, character())
})

test_that("ldb, lpdb and mblp report WoE with the final number of bins", {
  set.seed(1)
  x <- rlnorm(300)
  y <- rbinom(300, 1, plogis(-1 + 0.5 * x))
  r <- ob_numerical_ldb(x, y)
  expect_equal(r$woe, numc_doc_woe(r), tolerance = 1e-12)

  set.seed(31)
  x <- c(rnorm(1500), rnorm(1500, 4))
  y <- rbinom(3000, 1, plogis(-1 + 0.5 * x))
  r <- ob_numerical_lpdb(x, y, max_bins = 4)
  expect_equal(r$woe, numc_doc_woe(r), tolerance = 1e-12)

  r <- ob_numerical_mblp(x, y, max_bins = 4)
  expect_equal(r$woe, numc_doc_woe(r), tolerance = 1e-12)
})

# ---------------------------------------------------------------------------
# LDB regressions
# ---------------------------------------------------------------------------
test_that("ldb returns total_iv on the few-distinct-values path", {
  set.seed(6)
  x <- sample(1:3, 300, TRUE)
  y <- rbinom(300, 1, c(.2, .4, .6)[x])
  r <- ob_numerical_ldb(x, y)
  expect_length(r$bin, 3L)
  expect_gt(r$total_iv, 0.5)
  expect_equal(r$total_iv, sum(r$iv))
})

test_that("ldb and lpdb honour max_bins even with a small max_iterations", {
  # The iteration cap used to stop the max_bins merge as well (7-11 bins
  # returned for max_bins = 3); now it is only reported via `converged`.
  set.seed(1)
  x <- rnorm(12)
  y <- rbinom(12, 1, .5)
  r <- ob_numerical_ldb(x, y, max_iterations = 2, min_bins = 2, max_bins = 3)
  expect_lte(length(r$bin), 3L)
  expect_false(r$converged)

  set.seed(13)
  x <- rnorm(2000)
  y <- rbinom(2000, 1, plogis(x))
  r <- ob_numerical_lpdb(x, y, max_iterations = 2, min_bins = 2, max_bins = 3)
  expect_lte(length(r$bin), 3L)
  expect_false(r$converged)
})

test_that("ldb keeps WoE monotone after its frequency merges", {
  set.seed(3)
  x <- rnorm(15)
  y <- rbinom(15, 1, .5)
  r <- ob_numerical_ldb(x, y)
  expect_gt(length(r$bin), 3L)
  expect_true(numc_monotone(r$woe))
})

test_that("ldb does not emit an empty top bin from a quantile cut at the maximum", {
  set.seed(22)
  x <- sample(1:4, 1000, TRUE, prob = c(.2, .2, .1, .5))
  y <- rbinom(1000, 1, .3)
  r <- ob_numerical_ldb(x, y)
  expect_true(all(r$count > 0))
  expect_equal(sum(r$count), 1000)
})

test_that("ldb finds density minima on features with tied values", {
  # Rounded data: every value repeats, so on the raw sorted vector no point
  # was ever a strict local minimum and ldb fell back to a median cut (0.7,
  # inside the large cluster). The valley between the clusters is near 3.
  set.seed(32)
  x <- round(c(rnorm(4000, 0, 1), rnorm(2000, 6, 1)), 1)
  y <- rbinom(6000, 1, plogis(-2 + 0.5 * x))
  r <- ob_numerical_ldb(x, y,
    min_bins = 2, max_bins = 10, bin_cutoff = 0.01,
    enforce_monotonic = FALSE
  )
  expect_true(any(r$cutpoints > 2 & r$cutpoints < 4.5))
})

# ---------------------------------------------------------------------------
# LPDB regressions
# ---------------------------------------------------------------------------
test_that("lpdb does not return empty bins on heavy-tailed data", {
  set.seed(10)
  x <- rlnorm(1000, 0, 2)
  y <- rbinom(1000, 1, .3)
  r <- ob_numerical_lpdb(x, y)
  expect_true(all(r$count > 0))
})

test_that("lpdb keeps +/-Inf out of the cutpoints and centroids", {
  x <- c(3, 3, 3, Inf, 5, 5, 5, 5)
  y <- c(0L, 1L, 0L, 1L, 1L, 0L, 1L, 1L)
  r <- ob_numerical_lpdb(x, y)
  expect_true(all(is.finite(r$cutpoints)))
  expect_true(all(r$count > 0))
  expect_equal(sum(r$count), 8)

  set.seed(11)
  x <- c(rnorm(200), Inf, Inf, -Inf)
  y <- rbinom(203, 1, .3)
  r <- ob_numerical_lpdb(x, y)
  expect_true(all(is.finite(r$centroids)))
  expect_true(all(is.finite(r$cutpoints)))
  expect_equal(sum(r$count), 203)
})

test_that("lpdb is scale invariant and silent on small-scale features", {
  set.seed(51)
  x <- rnorm(3000)
  y <- rbinom(3000, 1, plogis(-1 - 0.8 * x))
  r1 <- ob_numerical_lpdb(x, y)
  expect_no_warning(r2 <- ob_numerical_lpdb(x * 1e-9, y))
  expect_equal(r2$count, r1$count)
  expect_equal(r2$monotonicity, r1$monotonicity)
})

test_that("lpdb and ldb survive values near the double range limit", {
  set.seed(12)
  x <- c(rnorm(200), 1.5e308, -1.5e308)
  y <- rbinom(202, 1, .3)
  for (f in list(ob_numerical_ldb, ob_numerical_lpdb)) {
    expect_no_warning(r <- f(x, y))
    expect_equal(sum(r$count), 202)
    expect_true(all(is.finite(r$cutpoints)))
  }
})

test_that("lpdb excludes NA feature rows silently", {
  set.seed(14)
  x <- rnorm(300)
  x[c(3, 30)] <- NA
  y <- rbinom(300, 1, .4)
  expect_no_warning(r <- ob_numerical_lpdb(x, y))
  expect_equal(sum(r$count), 298)
})

# ---------------------------------------------------------------------------
# MBLP regressions
# ---------------------------------------------------------------------------
test_that("mblp delivers the monotone WoE it guarantees", {
  set.seed(41)
  x <- sample(1:5, 2000, TRUE)
  y <- rbinom(2000, 1, c(.10, .30, .15, .35, .40)[x])
  r <- ob_numerical_mblp(x, y, min_bins = 2, max_bins = 5)
  expect_true(numc_monotone(r$woe))

  set.seed(42)
  x <- rnorm(2000)
  y <- rbinom(2000, 1, plogis(-1 + x - 1.5 * (abs(x) < .3)))
  r <- ob_numerical_mblp(x, y, min_bins = 2, max_bins = 6, max_n_prebins = 6)
  expect_true(numc_monotone(r$woe))
  expect_true(r$converged)
})

test_that("mblp gives each distinct value a pre-bin instead of an empty top bin", {
  set.seed(2)
  x <- sample(1:3, 300, TRUE, prob = c(.8, .15, .05))
  y <- rbinom(300, 1, c(.2, .3, .5)[x])
  r <- ob_numerical_mblp(x, y)
  expect_true(all(r$count > 0))
  expect_equal(r$cutpoints, c(1, 2))

  # fewer distinct values than min_bins is valid input: no warning
  set.seed(5)
  x <- sample(1:3, 300, TRUE)
  y <- rbinom(300, 1, c(.2, .4, .6)[x])
  expect_no_warning(r <- ob_numerical_mblp(x, y, min_bins = 4, max_bins = 6))
  expect_length(r$bin, 3L)
  expect_true(all(r$count > 0))
})

test_that("mblp rejects a missing target instead of counting it negative", {
  # The engine read NA_integer_ as a negative; the wrapper now stops, as
  # obwoe() does, and the engine itself skips such rows if called directly.
  set.seed(3)
  x <- rnorm(500)
  y <- rbinom(500, 1, plogis(x))
  y[c(5, 50, 100)] <- NA
  expect_error(ob_numerical_mblp(x, y), "missing values")
  r <- OptimalBinningWoE:::optimal_binning_numerical_mblp(as.integer(y), x)
  expect_equal(sum(r$count), 497)
  expect_equal(sum(r$count_pos), sum(y, na.rm = TRUE))
})

test_that("mblp does not collapse features measured on a small scale", {
  set.seed(23)
  x <- runif(500)
  y <- rbinom(500, 1, plogis((x - 0.5) * 4))
  r1 <- ob_numerical_mblp(x, y)
  expect_no_warning(r2 <- ob_numerical_mblp(x * 1e-11, y))
  expect_equal(r2$count, r1$count)
})

test_that("mblp and mdlp never emit a -Inf cutpoint", {
  set.seed(15)
  x <- c(rep(-Inf, 30), rep(5, 70))
  y <- rbinom(100, 1, .4)
  r <- ob_numerical_mblp(x, y)
  expect_true(all(is.finite(r$cutpoints)))
  expect_equal(r$count, c(30L, 70L))
  expect_no_warning(r <- ob_numerical_mdlp(x, y))
  expect_true(all(is.finite(r$cutpoints)))
  expect_equal(r$count, c(30L, 70L))
  expect_false(any(grepl("-inf", r$bin, fixed = TRUE)))

  set.seed(16)
  x <- c(-Inf, -Inf, rnorm(98))
  y <- rbinom(100, 1, .4)
  expect_no_warning(r <- ob_numerical_mdlp(x, y))
  expect_true(all(is.finite(r$cutpoints)))
  expect_equal(sum(r$count), 100)
})

# ---------------------------------------------------------------------------
# MDLP regressions
# ---------------------------------------------------------------------------
test_that("mdlp enforces monotonicity on fresh WoE values", {
  set.seed(35)
  x <- rnorm(400)
  y <- rbinom(400, 1, plogis(sin(2 * x)))
  r <- ob_numerical_mdlp(x, y,
    min_bins = 2, max_bins = 8, bin_cutoff = 0.02,
    max_n_prebins = 40
  )
  expect_gt(length(r$bin), 2L)
  expect_true(numc_monotone(r$woe))
})

test_that("mdlp reports non-convergence only when the cap left work undone", {
  set.seed(3)
  x <- rnorm(1000)
  y <- rbinom(1000, 1, plogis(-1 + x))
  expect_no_warning(r <- ob_numerical_mdlp(x, y, max_iterations = 6))
  expect_true(r$converged)

  set.seed(1)
  x <- rnorm(1000)
  y <- rbinom(1000, 1, plogis(-1 + x))
  out <- numc_capture(ob_numerical_mdlp(x, y, max_iterations = 1))
  expect_true(any(grepl("MDL merging did not complete", out$warnings)))
  expect_false(any(grepl("Rare bin merging", out$warnings)))
  expect_lte(length(out$res$bin), 5L)
})

test_that("mdlp explains an all-missing feature", {
  expect_error(
    ob_numerical_mdlp(c(NaN, NaN, NA), c(0L, 1L, 0L)),
    "All feature values are NA"
  )
})

test_that("numerical NA contract: silent NA exclusion, Inf in the end bins, NA target is an error", {
  set.seed(17)
  x <- rnorm(300)
  x[c(1, 2)] <- NaN
  x[c(3, 4)] <- NA
  x[5] <- Inf
  x[6] <- -Inf
  y <- rbinom(300, 1, .4)
  y[1:2] <- c(0L, 1L)
  for (a in names(numc_fns)) {
    expect_no_warning(r <- numc_fns[[a]](x, y))
    expect_equal(sum(r$count), 296, info = a)
    expect_true(all(is.finite(r$cutpoints)), info = a)
    expect_true(all(is.finite(r$woe)), info = a)
    expect_true(all(r$count > 0), info = a)
    y_na <- y
    y_na[10] <- NA
    expect_error(numc_fns[[a]](x, y_na), "missing values", info = a)
  }
})

# ---------------------------------------------------------------------------
# Edge paths and input validation (R wrappers and the C++ entry points)
# ---------------------------------------------------------------------------
test_that("ldb and lpdb handle +/-Inf, overflowing midpoints and missing values", {
  set.seed(18)
  x <- c(-Inf, rnorm(200), Inf, NA)
  y <- rbinom(203, 1, .4)
  r <- ob_numerical_ldb(x, y)
  expect_equal(sum(r$count), 202)
  expect_true(all(is.finite(r$cutpoints)))

  # two distinct values whose sum overflows: the midpoint must stay finite
  x <- rep(c(1e308, 1.7e308), 20)
  y <- rep(c(0L, 1L, 1L, 0L), 10)
  for (f in list(ob_numerical_ldb, ob_numerical_lpdb)) {
    r <- f(x, y)
    expect_length(r$cutpoints, 1L)
    expect_true(is.finite(r$cutpoints))
    expect_equal(r$count, c(20L, 20L))
  }
  x <- rep(c(-1.7e308, 1e308, 1.7e308), 20)
  y <- rep(c(0L, 1L, 1L, 0L, 1L, 0L), 10)
  for (f in list(ob_numerical_ldb, ob_numerical_lpdb)) {
    r <- f(x, y)
    expect_true(all(is.finite(r$cutpoints)))
    expect_equal(sum(r$count), 60)
  }

  # every positive has a missing feature: one class left after NA removal
  x <- c(rnorm(10), rep(NA, 5))
  y <- c(rep(0L, 10), rep(1L, 5))
  expect_error(ob_numerical_ldb(x, y), "both positive and negative")
})

test_that("lpdb small-sample paths: trend direction and flat WoE", {
  x <- rep(1:3, each = 20)
  y <- c(rep(c(1L, 1L, 1L, 0L), 5), rep(c(1L, 0L), 10), rep(c(0L, 0L, 0L, 1L), 5))
  r <- ob_numerical_lpdb(x, y)
  expect_equal(r$monotonicity, "decreasing")
  expect_equal(r$count, c(20L, 20L, 20L))

  # identical event rates in every bin: no trend to detect, and no warning
  x <- rep(1:3, each = 10)
  y <- rep(c(0L, 1L), 15)
  expect_no_warning(r <- ob_numerical_lpdb(x, y))
  expect_true(all(abs(r$woe) < 1e-12))
})

test_that("lpdb wrapper coerces its inputs", {
  set.seed(19)
  x <- rnorm(60) > 0
  y <- as.numeric(rbinom(60, 1, .5))
  expect_warning(r <- ob_numerical_lpdb(x, y), "converted to numeric")
  expect_equal(sum(r$count), 60)
  expect_error(ob_numerical_lpdb(1:3, c(0, 1)), "must match")
})

test_that("mblp honours a forced direction and reports an exhausted cap", {
  set.seed(24)
  x <- rnorm(1000)
  y <- rbinom(1000, 1, plogis(-1 - x))
  r <- ob_numerical_mblp(x, y, force_monotonic_direction = -1)
  expect_equal(r$monotonicity, "decreasing")
  expect_true(all(diff(r$woe) <= 1e-10))
  # forcing the wrong direction merges down to min_bins, where it must stop
  r <- ob_numerical_mblp(x, y, force_monotonic_direction = 1)
  expect_equal(r$monotonicity, "increasing")
  expect_length(r$bin, 3L)

  set.seed(42)
  x <- rnorm(2000)
  y <- rbinom(2000, 1, plogis(-1 + x - 1.5 * (abs(x) < .3)))
  expect_warning(
    r <- ob_numerical_mblp(x, y, max_iterations = 1, min_bins = 2),
    "Convergence not reached"
  )
  expect_false(r$converged)
  expect_lte(length(r$bin), 5L)
  expect_error(ob_numerical_mblp(rep(NA_real_, 3), c(0, 1, 0)), "NA")
})

test_that("mdlp edge paths: capped WoE, capped phases, leftmost rare bin", {
  # laplace_smoothing = 0 with pure bins: WoE is capped at +/-20
  x <- 1:200
  y <- c(rep(0L, 60), rep(c(0L, 1L), 40), rep(1L, 60))
  r <- ob_numerical_mdlp(x, y, laplace_smoothing = 0)
  expect_true(all(is.finite(r$woe)))
  expect_true(any(abs(r$woe) == 20))

  # a pure, small first pre-bin survives the merges and is rare
  set.seed(25)
  x <- 1:1000
  y <- c(rep(0L, 25), rbinom(975, 1, .5))
  r <- ob_numerical_mdlp(x, y, max_n_prebins = 40)
  expect_true(all(r$count / 1000 >= 0.05))
  out <- numc_capture(ob_numerical_mdlp(x, y, max_n_prebins = 40, max_iterations = 1))
  expect_true(any(grepl("Rare bin merging did not complete", out$warnings)))

  set.seed(3)
  x <- rnorm(1000)
  y <- rbinom(1000, 1, plogis(-1 + x))
  expect_warning(
    r <- ob_numerical_mdlp(x, y, max_iterations = 5),
    "Monotonicity enforcement did not converge"
  )
  expect_false(r$converged)

  # all feature values of one class are missing: one class left
  expect_error(
    ob_numerical_mdlp(c(1, 2, 3, NaN, NaN), c(0L, 0L, 0L, 1L, 1L)),
    "same"
  )
})

test_that("R wrappers reject invalid arguments", {
  x <- rnorm(50)
  y <- rep(0:1, 25)
  for (f in list(ob_numerical_ldb, ob_numerical_mblp, ob_numerical_mdlp)) {
    expect_error(f(letters[1:50], y), "numeric")
    expect_error(f(x, as.list(y)), "Target")
    expect_error(f(x[-1], y), "same length")
    expect_error(f(x, rep(0L, 50)), "two classes")
    expect_error(f(x, y, min_bins = 1), "min_bins")
    expect_error(f(x, y, min_bins = 4, max_bins = 3), "max_bins")
    expect_error(f(x, y, bin_cutoff = 0), "bin_cutoff")
    expect_error(f(x, y, max_n_prebins = 1), "max_n_prebins")
    expect_error(f(x, y, max_iterations = 0), "max_iterations")
  }
  expect_error(ob_numerical_mblp(x, y, force_monotonic_direction = 2), "force_monotonic")
  expect_error(ob_numerical_mdlp(x, y, laplace_smoothing = -1), "laplace")
})

test_that("C++ entry points validate their arguments", {
  ldb <- OptimalBinningWoE:::optimal_binning_numerical_ldb
  lpdb <- OptimalBinningWoE:::optimal_binning_numerical_lpdb
  mblp <- OptimalBinningWoE:::optimal_binning_numerical_mblp
  mdlp <- OptimalBinningWoE:::optimal_binning_numerical_mdlp
  x <- as.numeric(1:40)
  y <- rep(0:1, 20)

  expect_error(ldb(integer(0), numeric(0)), "empty")
  expect_error(ldb(y[-1], x), "same length")
  expect_error(ldb(y, x, min_bins = 1L), "min_bins")
  expect_error(ldb(y, x, min_bins = 4L, max_bins = 3L), "max_bins")
  expect_error(ldb(y, x, bin_cutoff = 2), "bin_cutoff")
  expect_error(ldb(y, x, max_n_prebins = 2L), "max_n_prebins")
  expect_error(ldb(rep(2L, 40), x), "binary")
  expect_error(ldb(rep(1L, 40), x), "both classes")
  expect_error(ldb(y, c(1, 2, rep(NA, 38))), "Not enough valid")

  expect_error(lpdb(y, x, min_bins = 1L), "min_bins")
  expect_error(lpdb(y, x, min_bins = 4L, max_bins = 3L), "max_bins")
  expect_error(lpdb(y, x, bin_cutoff = 2), "bin_cutoff")
  expect_error(lpdb(y, x, max_n_prebins = 2L), "max_n_prebins")
  expect_error(lpdb(y, x, polynomial_degree = 0L), "polynomial_degree")
  expect_error(lpdb(y[-1], x), "same length")
  expect_error(lpdb(rep(2L, 40), x), "binary")
  expect_error(lpdb(y, rep(NA_real_, 40)), "No valid")
  expect_error(lpdb(y, ifelse(y == 1L, NA_real_, x)), "both positive")

  expect_error(mblp(y[-1], x), "same length")
  expect_error(mblp(rep(2L, 40), x), "binary")
  expect_error(mblp(y, x, min_bins = 1L), "min_bins")
  expect_error(mblp(y, x, min_bins = 4L, max_bins = 3L), "max_bins")
  expect_error(mblp(y, x, bin_cutoff = 0), "bin_cutoff")
  expect_error(mblp(y, x, max_n_prebins = 2L), "max_n_prebins")
  expect_error(mblp(y, x, force_monotonic_direction = 3L), "force_monotonic")
  expect_error(mblp(y, x, convergence_threshold = 0), "convergence_threshold")
  expect_error(mblp(y, x, max_iterations = 0L), "max_iterations")

  expect_error(mdlp(y[-1], x), "same size")
  expect_error(mdlp(integer(0), numeric(0)), "empty")
  expect_error(mdlp(y, x, min_bins = 0L), "min_bins")
  expect_error(mdlp(y, x, min_bins = 4L, max_bins = 3L), "max_bins")
  expect_error(mdlp(y, x, bin_cutoff = 0), "bin_cutoff")
  expect_error(mdlp(y, x, max_n_prebins = 1L), "max_n_prebins")
  expect_error(mdlp(y, x, laplace_smoothing = -1), "laplace")
  expect_error(mdlp(rep(2L, 40), x), "only 0 and 1")
  r <- mdlp(rep(1L, 40), x)
  expect_length(r$bin, 1L)
  expect_equal(r$count, 40L)
})

test_that("ldb and lpdb recover when quantile cuts fall in a tie run at the top", {
  # Most observations share the maximum: the fallback quantile cuts all
  # landed on it, which used to leave one populated bin plus an empty one.
  set.seed(62)
  x <- c(sample(1:3, 200, TRUE), rep(9, 800))
  y <- rbinom(1000, 1, .3)
  for (f in list(ob_numerical_ldb, ob_numerical_lpdb)) {
    r <- f(x, y)
    expect_length(r$bin, 3L)
    expect_true(all(r$count > 0))
    expect_equal(r$count[3], 800L)
  }

  x <- c(1, 2, 3, rep(10, 6))
  y <- c(0L, 1L, 0L, 1L, 1L, 0L, 1L, 0L, 1L)
  r <- ob_numerical_lpdb(x, y)
  expect_true(all(r$count > 0))
  expect_equal(sum(r$count), 9)
  expect_gte(length(r$bin), 2L)
})
