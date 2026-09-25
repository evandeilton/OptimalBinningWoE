# Audit regressions and invariants for the numerical binners
#   ob_numerical_mob, ob_numerical_mrblp, ob_numerical_oslp,
#   ob_numerical_sketch, ob_numerical_ubsd
#
# Every result must be a valid right-closed binning: counts add up, each bin is
# non-empty, cutpoints are finite and strictly increasing, there is one
# cutpoint fewer than bins, and the counts are reproduced exactly by assigning
# the data to (c[j-1], c[j]] intervals.

numd_fns <- list(
  mob = ob_numerical_mob,
  mrblp = ob_numerical_mrblp,
  oslp = ob_numerical_oslp,
  sketch = ob_numerical_sketch,
  ubsd = ob_numerical_ubsd
)

# Run and collect warnings, so a test can assert exactly which ones occur.
numd_run <- function(fn, ...) {
  ws <- character(0)
  res <- withCallingHandlers(fn(...), warning = function(w) {
    ws <<- c(ws, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(res = res, warnings = ws)
}

# Names of the violated invariants (character(0) when the binning is valid).
numd_problems <- function(res, x, y, max_bins) {
  keep <- !is.na(x)
  x <- x[keep]
  y <- y[keep]
  nb <- length(res$count)
  cp <- res$cutpoints
  p <- c(
    no_bins = nb < 1,
    bin_length = length(res$bin) != nb || length(res$woe) != nb,
    count_sum = sum(res$count) != length(x),
    pos_neg = any(res$count_pos + res$count_neg != res$count),
    pos_sum = sum(res$count_pos) != sum(y),
    empty_bin = any(res$count <= 0),
    too_many_bins = nb > max_bins,
    nonfinite_woe_iv = !all(is.finite(res$woe)) || !all(is.finite(res$iv)),
    cutpoint_length = length(cp) != nb - 1,
    nonfinite_cutpoint = !all(is.finite(cp)),
    cutpoints_not_increasing = length(cp) > 1 && !isTRUE(all(diff(cp) > 0)),
    label_not_right_closed = !all(startsWith(res$bin, "(") & endsWith(res$bin, "]")),
    first_label = !startsWith(res$bin[1], "(-Inf;"),
    last_label = !endsWith(res$bin[nb], ";+Inf]")
  )
  if (!p[["cutpoint_length"]] && !p[["nonfinite_cutpoint"]] && !p[["cutpoints_not_increasing"]]) {
    idx <- findInterval(x, cp, left.open = TRUE) + 1L
    p <- c(p,
      counts_not_reproducible = !identical(tabulate(idx, nbins = nb), as.integer(res$count)),
      pos_not_reproducible = !identical(
        tabulate(idx[y == 1], nbins = nb), as.integer(res$count_pos)
      )
    )
  }
  names(p)[p]
}

numd_expect_valid <- function(res, x, y, max_bins, info = "") {
  expect_identical(numd_problems(res, x, y, max_bins), character(0), info = info)
}

numd_is_monotone <- function(w) {
  d <- diff(w)
  all(d >= 0) || all(d <= 0)
}

# ---------------------------------------------------------------------------
# Invariants on a spread of shapes, for every algorithm
# ---------------------------------------------------------------------------
test_that("all five binners return valid right-closed binnings", {
  set.seed(20260925)
  n <- 400
  shapes <- list(
    normal = rnorm(n),
    ties = round(rnorm(n) * 2),
    lognormal = rlnorm(n, 0, 2),
    cauchy = rcauchy(n),
    discrete4 = sample(1:4, n, TRUE),
    small_scale = rnorm(n) * 1e-12,
    large_scale = rnorm(n) * 1e300,
    two_values = sample(c(3, 8), n, TRUE)
  )
  params <- list(
    list(),
    list(min_bins = 2, max_bins = 8, bin_cutoff = 0.02),
    list(min_bins = 4, max_bins = 4, bin_cutoff = 0.1)
  )
  problems <- character(0)
  for (sh in names(shapes)) {
    x <- shapes[[sh]]
    y <- rbinom(n, 1, plogis(rank(x) / n * 2 - 1))
    for (a in names(numd_fns)) {
      for (p in params) {
        max_bins <- if (is.null(p$max_bins)) 5 else p$max_bins
        out <- numd_run(numd_fns[[a]],
          feature = x, target = y,
          min_bins = if (is.null(p$min_bins)) 3 else p$min_bins,
          max_bins = max_bins,
          bin_cutoff = if (is.null(p$bin_cutoff)) 0.05 else p$bin_cutoff
        )
        pr <- numd_problems(out$res, x, y, max_bins)
        if (length(out$warnings)) pr <- c(pr, "warning")
        if (a != "sketch" && !isTRUE(out$res$converged)) pr <- c(pr, "not_converged")
        if (length(pr)) {
          problems <- c(problems, paste(a, sh, max_bins, paste(pr, collapse = ",")))
        }
      }
    }
  }
  expect_identical(problems, character(0))
})

test_that("German credit numeric features bin validly with every algorithm", {
  skip_if_no_german()
  gc <- german_credit()
  num <- names(gc)[vapply(gc, is.numeric, logical(1)) & names(gc) != "target"]
  for (v in num) {
    for (a in names(numd_fns)) {
      out <- numd_run(numd_fns[[a]], feature = gc[[v]], target = gc$target)
      numd_expect_valid(out$res, gc[[v]], gc$target, 5, paste(a, v))
      expect_length(out$warnings, 0)
    }
  }
})

test_that("binners are deterministic without set.seed", {
  set.seed(1)
  x <- rlnorm(3000)
  y <- rbinom(3000, 1, 0.3)
  for (a in names(numd_fns)) {
    r1 <- suppressWarnings(numd_fns[[a]](x, y))
    set.seed(999)
    r2 <- suppressWarnings(numd_fns[[a]](x, y))
    expect_identical(r1, r2, info = a)
  }
})

# ---------------------------------------------------------------------------
# ob_numerical_mob
# ---------------------------------------------------------------------------
test_that("mob: WoE stays monotonic after the max_bins reduction", {
  # The max_bins reduction merged bins after monotonicity had been enforced and,
  # with Laplace smoothing, re-created a violation (8 bins, "Final bins do not
  # have monotonic WoE values") although min_bins = 2 allowed further merging.
  x <- c(
    0.4332263, 0.004806331, 0.009775463, 0.1908803, 0.08791471,
    0.3102341, 0.1539959, 2.695633, 0.2494239, 0.5937712, 0.2064175,
    0.07092108, 2.273397, 1.898254, 0.01454574, 1.685211, 12.13717,
    0.2362929, 4.968069, 1.962615
  )
  y <- c(0L, 0L, 0L, 0L, 0L, 1L, 0L, 1L, 0L, 0L, 0L, 0L, 0L, 1L, 0L, 1L, 1L, 0L, 1L, 1L)
  expect_no_warning(
    r <- ob_numerical_mob(x, y, min_bins = 2, max_bins = 8, bin_cutoff = 0.02, max_n_prebins = 30)
  )
  expect_true(numd_is_monotone(r$woe))
  numd_expect_valid(r, x, y, 8)
})

test_that("mob: the monotonic direction is not decided by the first two bins", {
  # The first two pre-bins of this increasing relationship happened to have
  # decreasing WoE; enforcing "decreasing" merged everything above them and
  # returned counts 150/150/2700 with non-monotonic WoE.
  set.seed(7)
  x <- rexp(3000)
  y <- rbinom(3000, 1, plogis(x - 1))
  expect_no_warning(r <- ob_numerical_mob(x, y))
  expect_true(all(diff(r$woe) > 0))
  expect_gt(length(r$count), 3)
  numd_expect_valid(r, x, y, 5)
})

test_that("mob: a tie between the first two WoE values is not a violation", {
  # WoE (pos counts 3, 3, 1, 0 in bins of 5) is non-increasing. The old check
  # took the direction from the first pair only (equal -> "increasing") and
  # merged these already monotonic bins.
  x <- 1:20
  y <- c(1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  expect_no_warning(
    r <- ob_numerical_mob(x, y, min_bins = 2, max_bins = 4, bin_cutoff = 0.01, max_n_prebins = 4)
  )
  expect_equal(r$count, c(5L, 5L, 5L, 5L))
  expect_equal(r$count_pos, c(3L, 3L, 1L, 0L))
  expect_true(numd_is_monotone(r$woe))
})

test_that("mob: missing values are excluded from the totals, not only from bins", {
  set.seed(7)
  x <- rexp(3000)
  y <- rbinom(3000, 1, plogis(x - 1))
  xna <- x
  xna[sample(3000, 300)] <- NA
  out_na <- numd_run(ob_numerical_mob, xna, y)
  expect_length(out_na$warnings, 0)
  r_na <- out_na$res
  r_clean <- numd_run(ob_numerical_mob, xna[!is.na(xna)], y[!is.na(xna)])$res
  for (f in c("bin", "woe", "iv", "count", "count_pos", "cutpoints", "total_iv")) {
    expect_equal(r_na[[f]], r_clean[[f]], info = f)
  }
  numd_expect_valid(r_na, xna, y, 5)
  expect_error(ob_numerical_mob(rep(NA_real_, 10), rep(0:1, 5)), "All feature values are missing")
})

test_that("mob: infinite values never produce a non-finite cutpoint", {
  x <- c(-Inf, -Inf, 0.5, 1, 2, 3, 4, 5, 6, Inf)
  y <- c(1L, 0L, 1L, 0L, 1L, 0L, 0L, 1L, 0L, 1L)
  out <- numd_run(ob_numerical_mob, x, y, bin_cutoff = 0.01)
  expect_length(out$warnings, 0)
  numd_expect_valid(out$res, x, y, 5)
  # two distinct values, one of them -Inf
  expect_no_warning(r2 <- ob_numerical_mob(c(-Inf, -Inf, 2, 2, 2), c(1L, 0L, 1L, 0L, 0L)))
  numd_expect_valid(r2, c(-Inf, -Inf, 2, 2, 2), c(1L, 0L, 1L, 0L, 0L), 5)
})

test_that("mob: small-scale features keep every cutpoint", {
  # Cutpoints closer than an absolute 1e-10 used to be dropped as duplicates.
  set.seed(3)
  x <- rnorm(500) * 1e-12
  y <- rbinom(500, 1, plogis(x * 1e12))
  r <- ob_numerical_mob(x, y)
  expect_equal(length(r$cutpoints), length(r$count) - 1)
  expect_identical(r$count, ob_numerical_mob(x * 1e12, y)$count)
})

test_that("mob/mrblp/oslp/ubsd: two huge distinct values get a finite cut", {
  # (a + b) / 2 overflowed to Inf for values near the largest double.
  x <- rep(c(1.6e308, 1.7e308), 20)
  y <- rep(c(0L, 1L, 1L, 0L), 10)
  for (a in c("mob", "mrblp", "oslp", "ubsd")) {
    r <- numd_fns[[a]](x, y)
    numd_expect_valid(r, x, y, 5, a)
    expect_equal(r$count, c(20L, 20L), info = a)
  }
})

test_that("mob: reaching max_iterations sets converged = FALSE without a warning", {
  set.seed(11)
  x <- rnorm(2000)
  y <- rbinom(2000, 1, plogis(x))
  out <- numd_run(ob_numerical_mob, x, y, max_bins = 3, max_n_prebins = 50, max_iterations = 2)
  expect_false(out$res$converged)
  expect_length(out$warnings, 0)
})

test_that("mob: one or two distinct values are binned exactly", {
  r1 <- ob_numerical_mob(rep(4, 10), rep(0:1, 5))
  expect_equal(r1$count, 10L)
  expect_equal(r1$bin, "(-Inf;+Inf]")
  r2 <- ob_numerical_mob(rep(c(1, 2), 10), rep(c(0L, 0L, 1L, 1L), 5))
  expect_equal(r2$count, c(10L, 10L))
  expect_equal(r2$cutpoints, 1.5)
})

# ---------------------------------------------------------------------------
# ob_numerical_mrblp
# ---------------------------------------------------------------------------
test_that("mrblp: converged is TRUE for an ordinary monotonic binning", {
  # Pre-bins whose WoE is already monotonic skipped the only assignment of
  # `converged = TRUE`, so well-behaved features reported FALSE.
  x <- 1:2000
  y <- as.integer((x %% 10) < (x %/% 200))
  r <- ob_numerical_mrblp(x, y)
  expect_true(numd_is_monotone(r$woe))
  expect_true(r$converged)
  r_lim <- ob_numerical_mrblp(x, y, max_n_prebins = 50, max_iterations = 2)
  expect_false(r_lim$converged)
})

test_that("mrblp: equal first/last WoE no longer stops monotonicity enforcement", {
  # |WoE_last - WoE_first| < convergence_threshold ended the enforcement loop
  # with WoE (-0.41, 0.69, -0.41, -0.41) although min_bins = 2 allowed merging.
  x <- c(-4.186767, -0.4653132, -2.867764, 0.5519313, -0.2257682)
  y <- c(0L, 0L, 1L, 0L, 0L)
  r <- ob_numerical_mrblp(x, y, min_bins = 2, max_bins = 8, bin_cutoff = 0.02, max_n_prebins = 30)
  expect_true(numd_is_monotone(r$woe))
  numd_expect_valid(r, x, y, 8)
})

# ---------------------------------------------------------------------------
# ob_numerical_oslp
# ---------------------------------------------------------------------------
test_that("oslp: labels are right-closed like the assignment", {
  x <- rep(1:10, each = 10)
  y <- rep(c(0L, 1L), 50)
  y[x <= 3] <- 0L
  r <- ob_numerical_oslp(x, y)
  expect_true(all(startsWith(r$bin, "(") & endsWith(r$bin, "]")))
  numd_expect_valid(r, x, y, 5)
  # A value equal to a cutpoint is counted in the bin below it
  cp <- r$cutpoints[1]
  expect_equal(r$count[1], sum(x <= cp))
})

test_that("oslp: a small IV change no longer stops monotonicity enforcement", {
  x <- c(
    1.20568, 0.1217733, 0.07082062, 0.5768472, 5.064433, 22.11132,
    8.466307, 20.85885, 1.76452, 0.03360825, 5.640905, 2.633374
  )
  y <- c(0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L, 0L, 0L, 1L, 1L)
  r <- ob_numerical_oslp(x, y, min_bins = 2, max_bins = 8, bin_cutoff = 0.02, max_n_prebins = 30)
  expect_true(numd_is_monotone(r$woe))
  numd_expect_valid(r, x, y, 8)
})

test_that("oslp: converged is FALSE only when max_iterations is exhausted", {
  set.seed(8)
  x <- rnorm(3000)
  y <- rbinom(3000, 1, plogis(x))
  expect_true(ob_numerical_oslp(x, y)$converged)
  expect_false(ob_numerical_oslp(x, y, max_n_prebins = 50, bin_cutoff = 0.001, max_iterations = 2)$converged)
})

test_that("oslp: a negative zero is labelled 0", {
  x <- c(-0, 0, 0, -1, -1, 1, 1, 2, 2, 3)
  y <- c(0L, 1L, 0L, 0L, 0L, 1L, 1L, 1L, 0L, 1L)
  r <- ob_numerical_oslp(x, y, min_bins = 2, bin_cutoff = 0.01)
  expect_false(any(grepl("-0.000000", r$bin, fixed = TRUE)))
})

# ---------------------------------------------------------------------------
# ob_numerical_ubsd
# ---------------------------------------------------------------------------
test_that("ubsd: two distinct values return their cutpoint", {
  r <- ob_numerical_ubsd(c(4, 3, 3, 4, 3, 4), c(0L, 0L, 0L, 0L, 1L, 1L))
  expect_equal(r$cutpoints, 3.5)
  expect_equal(r$bin, c("(-Inf;3.500000]", "(3.500000;+Inf]"))
  expect_equal(r$count, c(3L, 3L))
})

test_that("ubsd: empty sd-based bins are merged instead of returned", {
  # With min_bins = 4 the empty bins created by mean +/- k sd edges outside
  # the data used to be returned (count 0, "NumericalBin # is empty" warning).
  x <- c(-5, -1, 1, -2, 1)
  y <- c(1L, 0L, 1L, 0L, 1L)
  out <- numd_run(ob_numerical_ubsd, x, y,
    min_bins = 4, max_bins = 4, bin_cutoff = 0.1, max_n_prebins = 5, laplace_smoothing = 0
  )
  expect_true(all(out$res$count > 0))
  expect_false(any(grepl("empty", out$warnings)))
  numd_expect_valid(out$res, x, y, 4)
})

test_that("ubsd: features near the largest double give finite edges", {
  set.seed(9)
  x <- c(rep(-1.7e308, 30), rep(1.7e308, 30), rnorm(40))
  y <- rbinom(100, 1, 0.4)
  r <- suppressWarnings(ob_numerical_ubsd(x, y))
  numd_expect_valid(r, x, y, 5)
})

# ---------------------------------------------------------------------------
# ob_numerical_sketch
# ---------------------------------------------------------------------------
test_that("sketch: result carries its documented class", {
  set.seed(2)
  x <- rnorm(300)
  r <- ob_numerical_sketch(x, rbinom(300, 1, plogis(x)))
  expect_s3_class(r, "OptimalBinningSketch")
  expect_s3_class(r, "OptimalBinning")
})

test_that("sketch: first and last labels cover the whole line", {
  set.seed(2)
  x <- rnorm(1000)
  y <- rbinom(1000, 1, plogis(x))
  r <- ob_numerical_sketch(x, y)
  numd_expect_valid(r, x, y, 5)
  expect_equal(r$bin_lower[1], min(x))
  expect_equal(r$bin_upper[length(r$bin_upper)], max(x))
})

test_that("sketch: no duplicate cutpoints or empty bins on tied data", {
  x <- c(0, 0, 1, 0, 1, 1, 0, 0, 0, 1)
  y <- c(0L, 0L, 1L, 0L, 0L, 0L, 0L, 0L, 0L, 1L)
  r <- ob_numerical_sketch(x, y, min_bins = 4, max_bins = 4, bin_cutoff = 0.1)
  numd_expect_valid(r, x, y, 4)
  expect_equal(r$count, c(6L, 4L))

  x2 <- c(-5, -1, 1, -2, 1)
  y2 <- c(1L, 0L, 1L, 0L, 1L)
  r2 <- ob_numerical_sketch(x2, y2, min_bins = 4, max_bins = 4, bin_cutoff = 0.1)
  numd_expect_valid(r2, x2, y2, 4)
})

test_that("sketch: dynamic programming splits only between distinct values", {
  set.seed(4)
  for (i in 1:20) {
    x <- sample(1:6, 40, TRUE)
    y <- rbinom(40, 1, x / 7)
    r <- ob_numerical_sketch(x, y, monotonic = FALSE)
    numd_expect_valid(r, x, y, 5, paste("rep", i))
  }
})

test_that("sketch: only an exactly constant feature collapses to one bin", {
  y <- rep(0:1, 50)
  expect_no_warning(r <- ob_numerical_sketch(rep(2, 100), y))
  expect_equal(r$bin, "(-Inf;+Inf]")
  expect_equal(r$woe, 0)
  # distinct values spanning less than 1e-10 are not constant
  set.seed(6)
  x <- rnorm(100) * 1e-12
  y2 <- rbinom(100, 1, plogis(x * 1e12))
  expect_no_warning(r2 <- ob_numerical_sketch(x, y2))
  expect_gt(length(r2$count), 1)
  numd_expect_valid(r2, x, y2, 5)
})

test_that("sketch: large samples take the greedy path and stay valid", {
  set.seed(10)
  x <- c(rlnorm(20000), rep(0, 5000))
  y <- rbinom(25000, 1, plogis(log1p(x)))
  for (k in c(10, 200)) {
    r <- ob_numerical_sketch(x, y, sketch_k = k, max_bins = 7)
    numd_expect_valid(r, x, y, 7, paste("k", k))
    expect_true(numd_is_monotone(r$woe) || length(r$woe) <= 3)
  }
})

# ---------------------------------------------------------------------------
# Argument validation inside the compiled routines (direct calls)
# ---------------------------------------------------------------------------
test_that("compiled routines validate their arguments", {
  ns <- asNamespace("OptimalBinningWoE")
  mob <- get("optimal_binning_numerical_mob", ns)
  mrblp <- get("optimal_binning_numerical_mrblp", ns)
  oslp <- get("optimal_binning_numerical_oslp", ns)
  ubsd <- get("optimal_binning_numerical_ubsd", ns)
  sk <- get("optimal_binning_numerical_sketch", ns)
  x <- as.numeric(1:10)
  y <- rep(0:1, 5)

  expect_error(mob(y, x, min_bins = 1), "min_bins")
  expect_error(mob(y, x, bin_cutoff = 0), "bin_cutoff")
  expect_error(mob(y, x, convergence_threshold = 0), "convergence_threshold")
  expect_error(mob(y, x, max_iterations = 0), "max_iterations")
  expect_error(mob(y, x, laplace_smoothing = -1), "laplace_smoothing")
  expect_error(mob(y, x[1:5]), "same length")
  expect_error(mob(integer(0), numeric(0)), "empty")
  expect_error(mob(rep(2L, 10), x), "only 0 and 1")
  expect_error(mob(rep(1L, 10), x), "both classes")

  expect_error(mrblp(y, x, min_bins = 0), "min_bins")
  expect_error(mrblp(y, x, bin_cutoff = 1), "bin_cutoff")
  expect_error(mrblp(y, x, convergence_threshold = 0), "convergence_threshold")
  expect_error(mrblp(y, x, max_iterations = 0), "max_iterations")
  expect_error(mrblp(y, x, laplace_smoothing = -1), "laplace_smoothing")
  expect_error(mrblp(y, x[1:5]), "same length")
  expect_error(mrblp(integer(0), numeric(0)), "empty")
  expect_error(mrblp(rep(2L, 10), x), "only 0 and 1")
  expect_error(mrblp(rep(1L, 10), x), "both classes")
  r1 <- mrblp(y, rep(5, 10), min_bins = 1, max_bins = 1)
  expect_equal(r1$count, 10L)

  expect_error(oslp(y, x, min_bins = 1), "min_bins")
  expect_error(oslp(y, x, max_bins = 2), "max_bins")
  expect_error(oslp(y, x, bin_cutoff = 0), "bin_cutoff")
  expect_error(oslp(y, x, max_n_prebins = 2), "max_n_prebins")
  expect_error(oslp(y, x, convergence_threshold = 0), "convergence_threshold")
  expect_error(oslp(y, x, max_iterations = 0), "max_iterations")
  expect_error(oslp(y, x, laplace_smoothing = -1), "laplace_smoothing")
  expect_error(oslp(y, x[1:5]), "same length")
  expect_error(oslp(numeric(0), numeric(0)), "empty")
  expect_error(oslp(rep(2, 10), x), "only 0 and 1")
  expect_error(oslp(rep(1, 10), x), "both classes")
  r2 <- oslp(y, rep(5, 10))
  expect_equal(r2$bin, "(-Inf;+Inf]")

  expect_error(ubsd(y, x, min_bins = 1), "min_bins")
  expect_error(ubsd(y, x, max_bins = 2), "max_bins")
  expect_error(ubsd(y, x, bin_cutoff = 0), "bin_cutoff")
  expect_error(ubsd(y, x, max_n_prebins = 2), "max_n_prebins")
  expect_error(ubsd(y, x, convergence_threshold = 0), "convergence_threshold")
  expect_error(ubsd(y, x, max_iterations = 0), "max_iterations")
  expect_error(ubsd(y, x, laplace_smoothing = -1), "laplace_smoothing")
  expect_error(ubsd(y, x[1:5]), "same length")
  expect_error(ubsd(numeric(0), numeric(0)), "empty")
  expect_error(ubsd(rep(2, 10), x), "only 0 and 1")
  expect_error(ubsd(rep(1, 10), x), "both classes")
  r3 <- ubsd(y, rep(5, 10))
  expect_equal(r3$bin, "(-Inf;+Inf]")

  expect_error(sk(integer(0), numeric(0)), "empty")
  expect_error(sk(y, x[1:5]), "same size")
  expect_error(sk(y, rep(NA_real_, 10)), "All feature values are missing")
  expect_error(sk(c(y[1:9], NA), x), "missing")
  expect_error(sk(rep(1L, 10), x), "both 0 and 1")
  expect_error(sk(c(y[1:9], 2L), x), "only 0 and 1")
  expect_error(sk(y, x, min_bins = 1), "min_bins")
  expect_error(sk(y, x, max_bins = 2), "max_bins")
  expect_error(sk(y, x, bin_cutoff = 0), "bin_cutoff")
  expect_error(sk(y, x, sketch_k = 5), "sketch_k")
  expect_error(sk(y, x, max_iterations = 0), "max_iterations")
})

test_that("R wrappers reject invalid arguments before calling C++", {
  x <- as.numeric(1:10)
  y <- rep(0:1, 5)
  for (a in c("mob", "mrblp", "oslp", "ubsd")) {
    fn <- numd_fns[[a]]
    expect_error(fn(letters[1:10], y), "numeric", info = a)
    expect_error(fn(x, as.character(y)), "Target must be", info = a)
    expect_error(fn(x, y[1:5]), "same length", info = a)
    expect_error(fn(x, rep(0:2, length.out = 10)), "two classes", info = a)
    expect_error(fn(x, c(y[1:9], NA)), "Target contains missing values", info = a)
    expect_error(fn(x, y, min_bins = 1), "min_bins", info = a)
    expect_error(fn(x, y, min_bins = 4, max_bins = 3), "max_bins", info = a)
    expect_error(fn(x, y, bin_cutoff = 1), "bin_cutoff", info = a)
    expect_error(fn(x, y, max_n_prebins = 2), "max_n_prebins", info = a)
    expect_error(fn(x, y, max_iterations = 0), "max_iterations", info = a)
    expect_error(fn(x, y, laplace_smoothing = -1), "laplace_smoothing", info = a)
    if (a != "mob") {
      expect_error(fn(x, y, convergence_threshold = 0), "convergence_threshold", info = a)
    }
  }

  sk <- ob_numerical_sketch
  expect_error(sk(letters[1:10], y), "feature")
  expect_error(sk(x, as.character(y)), "target")
  expect_error(sk(x, y[1:5]), "Length mismatch")
  expect_error(sk(numeric(0), integer(0)), "empty")
  expect_error(sk(x, c(y[1:9], NA)), "Missing values")
  expect_error(sk(x, rep(0:2, length.out = 10)), "only 0 and 1")
  expect_error(sk(x, rep(1L, 10)), "both classes")
  expect_error(sk(x, y, min_bins = "a"), "min_bins")
  expect_error(sk(x, y, min_bins = 1), "min_bins")
  expect_error(sk(x, y, max_bins = NA), "max_bins")
  expect_error(sk(x, y, max_bins = 2), "max_bins")
  expect_error(sk(x, y, bin_cutoff = c(0.1, 0.2)), "bin_cutoff")
  expect_error(sk(x, y, bin_cutoff = 1), "bin_cutoff")
  expect_error(sk(x, y, max_n_prebins = NA), "max_n_prebins")
  expect_error(sk(x, y, max_n_prebins = 1), "max_n_prebins")
  expect_error(sk(x, y, monotonic = NA), "monotonic")
  expect_error(sk(x, y, convergence_threshold = "a"), "convergence_threshold")
  expect_error(sk(x, y, convergence_threshold = 0), "convergence_threshold")
  expect_error(sk(x, y, max_iterations = NA), "max_iterations")
  expect_error(sk(x, y, max_iterations = 0), "max_iterations")
  expect_error(sk(x, y, sketch_k = NA), "sketch_k")
  expect_error(sk(x, y, sketch_k = 5), "sketch_k")
})

test_that("laplace_smoothing = 0 keeps WoE finite with pure bins (capped at +/-20)", {
  set.seed(12)
  x <- runif(400)
  y <- as.integer(x > 0.5)
  y[c(1, 2)] <- 1L - y[c(1, 2)]
  for (a in c("mob", "mrblp", "oslp", "ubsd")) {
    out <- numd_run(numd_fns[[a]], x, y, laplace_smoothing = 0, max_bins = 4, max_n_prebins = 30)
    numd_expect_valid(out$res, x, y, 4, a)
    expect_true(all(abs(out$res$woe) <= 20), info = a)
    expect_length(out$warnings, 0)
  }
  xl <- rlnorm(300, 0, 2)
  yl <- as.integer(xl > 1)
  out <- numd_run(ob_numerical_ubsd, xl, yl, laplace_smoothing = 0)
  numd_expect_valid(out$res, xl, yl, 5)
})

test_that("mob: all positives with a missing feature gives zero WoE, not NaN", {
  x <- c(1:10, rep(NA, 5))
  y <- c(rep(0L, 10), rep(1L, 5))
  out <- numd_run(ob_numerical_mob, x, y, laplace_smoothing = 0)
  expect_true(all(out$res$woe == 0) && all(out$res$iv == 0))
  expect_equal(sum(out$res$count), 10L)
})

test_that("adjacent doubles are split at the lower value", {
  # (a + b) / 2 rounds up to b for these two neighbours; the cut must stay
  # below b so that the two values land in different bins.
  x <- rep(c(1 + .Machine$double.eps, 1 + 2 * .Machine$double.eps), 10)
  y <- rep(c(0L, 1L, 1L, 0L), 5)
  for (a in c("mob", "mrblp", "oslp", "ubsd", "sketch")) {
    r <- numd_fns[[a]](x, y)
    expect_equal(r$count, c(10L, 10L), info = a)
    expect_equal(r$cutpoints, 1 + .Machine$double.eps, info = a)
  }
})

test_that("ubsd: nearly constant features and few pre-bins", {
  # sd < 1e-10 with a range >= 1e-10: documented equal-width fallback
  x <- c(rep(0, 2000), 1e-9, 2e-9, 3e-9)
  y <- c(rep(0:1, 1000), 1L, 0L, 1L)
  out0 <- numd_run(ob_numerical_ubsd, x, y, bin_cutoff = 0.0001)
  expect_length(out0$warnings, 0)
  numd_expect_valid(out0$res, x, y, 5)
  # more candidate edges than max_n_prebins: edges are sub-sampled
  set.seed(13)
  x2 <- rnorm(500)
  y2 <- rbinom(500, 1, plogis(x2))
  out <- numd_run(ob_numerical_ubsd, x2, y2, min_bins = 2, max_bins = 5, max_n_prebins = 2)
  numd_expect_valid(out$res, x2, y2, 5)
})

test_that("ubsd: noisy features stay valid across many seeds", {
  problems <- character(0)
  for (s in 1:40) {
    set.seed(100 + s)
    n <- sample(c(30, 80, 200), 1)
    x <- rcauchy(n)
    y <- rbinom(n, 1, 0.4)
    out <- numd_run(ob_numerical_ubsd, x, y, min_bins = 2, max_bins = 8, bin_cutoff = 0.01, max_n_prebins = 20)
    pr <- numd_problems(out$res, x, y, 8)
    if (length(out$warnings)) pr <- c(pr, "warning")
    if (length(pr)) problems <- c(problems, paste(s, paste(pr, collapse = ",")))
  }
  expect_identical(problems, character(0))
})

test_that("merge decisions with laplace_smoothing = 0 handle pure bins", {
  x <- 1:200
  y <- as.integer(x > 100)
  for (a in c("mob", "mrblp", "oslp", "ubsd")) {
    out <- numd_run(numd_fns[[a]], x, y,
      min_bins = 2, max_bins = 2, bin_cutoff = 0.01, max_n_prebins = 20, laplace_smoothing = 0
    )
    numd_expect_valid(out$res, x, y, 2, a)
    # (mrblp reduces by the smallest |IV_i - IV_i+1|, a documented heuristic
    # that does not target the IV-maximising split)
    if (a != "mrblp") expect_equal(out$res$count_pos, c(0L, 100L), info = a)
  }
  problems <- character(0)
  for (s in 1:30) {
    set.seed(200 + s)
    n <- sample(c(20, 60, 150), 1)
    x <- round(rnorm(n) * 3)
    y <- rbinom(n, 1, plogis(x / 2))
    for (a in c("mob", "oslp")) {
      out <- numd_run(numd_fns[[a]], x, y,
        min_bins = 2, max_bins = 3, bin_cutoff = 0.01, max_n_prebins = 30, laplace_smoothing = 0
      )
      pr <- numd_problems(out$res, x, y, 3)
      if (length(out$warnings)) pr <- c(pr, "warning")
      if (length(pr)) problems <- c(problems, paste(a, s, paste(pr, collapse = ",")))
    }
  }
  expect_identical(problems, character(0))
})

# ---------------------------------------------------------------------------
# Unified missing-value / infinite-value contract (all five algorithms)
# ---------------------------------------------------------------------------
test_that("NA/NaN rows are dropped silently and +/-Inf go to the end bins", {
  set.seed(21)
  n <- 2000
  x <- rnorm(n)
  y <- rbinom(n, 1, plogis(x))
  xs <- x
  xs[sample(n, 100)] <- NA
  xs[sample(which(!is.na(xs)), 5)] <- NaN
  xs[which(!is.na(xs))[1:4]] <- c(-Inf, -Inf, Inf, Inf)
  keep <- !is.na(xs)
  for (a in names(numd_fns)) {
    out <- numd_run(numd_fns[[a]], xs, y)
    expect_length(out$warnings, 0)
    r <- out$res
    numd_expect_valid(r, xs, y, 5, a)
    expect_equal(sum(r$count), sum(keep), info = a)
    expect_true(all(is.finite(r$cutpoints)), info = a)
    # identical to binning the complete cases
    rc <- numd_fns[[a]](xs[keep], y[keep])
    for (f in c("bin", "woe", "count", "count_pos", "cutpoints")) {
      expect_identical(r[[f]], rc[[f]], info = paste(a, f))
    }
  }
})

test_that("degenerate non-missing parts give a single valid bin", {
  y <- rep(0:1, 10)
  cases <- list(
    constant_with_na = c(rep(3, 15), rep(NA, 5)),
    one_value_and_infs = c(rep(3, 16), -Inf, Inf, NA, NaN),
    only_infs = c(rep(-Inf, 10), rep(Inf, 10))
  )
  for (nm in names(cases)) {
    x <- cases[[nm]]
    for (a in names(numd_fns)) {
      out <- numd_run(numd_fns[[a]], x, y)
      expect_length(out$warnings, 0)
      numd_expect_valid(out$res, x, y, 5, paste(a, nm))
    }
  }
  for (a in names(numd_fns)) {
    expect_error(numd_fns[[a]](rep(NA_real_, 20), y), "All feature values are missing", info = a)
  }
})

test_that("a class present only among missing-feature rows gives zero WoE", {
  x <- c(1:30, rep(NA, 5))
  y <- c(rep(0L, 30), rep(1L, 5))
  for (a in names(numd_fns)) {
    args <- list(x, y)
    if (a != "sketch") args$laplace_smoothing <- 0
    out <- do.call(numd_run, c(list(numd_fns[[a]]), args))
    expect_length(out$warnings, 0)
    expect_true(all(out$res$woe == 0) && all(out$res$iv == 0), info = a)
    expect_equal(sum(out$res$count), 30L, info = a)
  }
})

test_that("obwoe() and the ob_numerical_* wrappers agree on NA/Inf features", {
  set.seed(22)
  n <- 600
  x <- rnorm(n)
  y <- rbinom(n, 1, plogis(x))
  x[sample(n, 30)] <- NA
  x[which(!is.na(x))[1:2]] <- c(-Inf, Inf)
  df <- data.frame(target = y, f = x)
  for (a in names(numd_fns)) {
    out <- numd_run(obwoe, df, target = "target", feature = "f", algorithm = a)
    expect_length(out$warnings, 0)
    d <- numd_fns[[a]](x, y, min_bins = 2, max_bins = 7)
    expect_identical(out$res$results$f$cutpoints, d$cutpoints, info = a)
    expect_equal(sum(out$res$results$f$count), sum(!is.na(x)), info = a)
  }
})
