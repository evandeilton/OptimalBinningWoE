# Audit group numA: bb, cm, dmiv, dp, ewb, fetb (numerical engines).
#
# Every test named "regression: ..." fails on the code before this audit and
# passes after it. The "invariants" block checks, for each engine, the output
# contract on a range of shapes (ties, heavy tails, tiny/huge scales, tiny n,
# German credit): counts sum to the number of non-missing observations, no
# empty bin, at most max_bins bins, finite strictly increasing cutpoints (one
# fewer than bins), counts consistent with the cutpoints under the (a, b]
# rule, finite WoE/IV, and no warning of any kind.

ob_num <- function(algo) {
  get(paste0("ob_numerical_", algo), envir = asNamespace("OptimalBinningWoE"))
}

# Contract checks shared by the tests below.
binning_problems <- function(r, x, y, max_bins) {
  keep <- !is.na(x)
  x <- x[keep]
  y <- y[keep]
  nb <- length(r$bin)
  cnt <- as.numeric(r$count)
  cp <- as.numeric(r$cutpoints)
  p <- character(0)
  if (sum(cnt) != length(x)) p <- c(p, "counts do not sum to n")
  if (any(cnt <= 0)) p <- c(p, "empty bin")
  if (nb > max_bins) p <- c(p, "more than max_bins bins")
  if (length(cp) != nb - 1) p <- c(p, "length(cutpoints) != bins - 1")
  if (any(!is.finite(cp))) p <- c(p, "non-finite cutpoint")
  if (length(cp) > 1 && !isTRUE(all(diff(cp) > 0))) p <- c(p, "cutpoints not increasing")
  if (any(!is.finite(r$woe))) p <- c(p, "non-finite woe")
  if (any(!is.finite(r$iv))) p <- c(p, "non-finite iv")
  if (!isTRUE(all(as.numeric(r$count_pos) + as.numeric(r$count_neg) == cnt))) {
    p <- c(p, "count_pos + count_neg != count")
  }
  # counts must be what the cutpoints imply for right-closed bins (a, b]
  if (length(cp) == nb - 1 && all(is.finite(cp))) {
    idx <- findInterval(x, cp, left.open = TRUE) + 1L
    if (!identical(as.numeric(tabulate(idx, nb)), cnt)) p <- c(p, "counts vs cutpoints")
    if (!identical(as.numeric(tabulate(idx[y == 1], nb)), as.numeric(r$count_pos))) {
      p <- c(p, "positives vs cutpoints")
    }
  }
  p
}

# Contract checks shared by the tests below (one expectation per fit).
expect_valid_binning <- function(r, x, y, max_bins, info = "") {
  expect_identical(binning_problems(r, x, y, max_bins), character(0), info = info)
}

is_monotone <- function(w) {
  d <- diff(w)
  all(d >= -1e-9) || all(d <= 1e-9)
}

# ---------------------------------------------------------------------------
# Invariants for all six engines
# ---------------------------------------------------------------------------
test_that("invariants hold for bb, cm, dmiv, dp, ewb and fetb", {
  set.seed(20260925)
  shapes <- list(
    norm = function(n) rnorm(n),
    lnorm = function(n) rlnorm(n, 0, 2),
    cauchy = function(n) rcauchy(n),
    ties = function(n) round(rnorm(n), 1),
    int = function(n) sample(1:10, n, TRUE),
    dup = function(n) c(rep(0, floor(0.7 * n)), rnorm(n - floor(0.7 * n))),
    extreme = function(n) c(rnorm(n - 2), 1e300, -1e300),
    tiny = function(n) runif(n) * 1e-11,
    huge = function(n) rnorm(n) * 1e12 + 1e15
  )
  params <- list(
    bb = list(), cm = list(), dmiv = list(), dp = list(), ewb = list(), fetb = list(),
    cm_ew = list(init_method = "equal_width"), cm_chi2 = list(use_chi2_algorithm = TRUE),
    dmiv_ln = list(divergence_method = "ln"), dp_none = list(monotonic_trend = "none")
  )
  for (nm in names(shapes)) {
    for (n in c(12, 300)) {
      x <- shapes[[nm]](n)
      y <- rbinom(n, 1, plogis(as.numeric(scale(rank(x)))))
      y[1:2] <- c(0L, 1L)
      for (a in names(params)) {
        fn <- ob_num(sub("_.*", "", a))
        info <- paste(a, nm, n)
        r <- NULL
        expect_no_warning(r <- do.call(fn, c(list(feature = x, target = y), params[[a]])))
        # equal-width EWB cannot always produce bins on a range dominated by
        # outliers; everything else must satisfy the contract.
        expect_valid_binning(r, x, y, max_bins = 5, info = info)
      }
    }
  }
})

test_that("invariants hold on the German credit numerical features", {
  skip_if_no_german()
  gc <- german_credit()
  num <- names(gc)[vapply(gc, is.numeric, TRUE)]
  num <- setdiff(num, "target")
  for (v in num) {
    for (a in c("bb", "cm", "dmiv", "dp", "ewb", "fetb")) {
      r <- NULL
      expect_no_warning(r <- ob_num(a)(gc[[v]], gc$target))
      expect_valid_binning(r, gc[[v]], gc$target, max_bins = 5, info = paste(a, v))
    }
  }
})

test_that("monotonic engines return monotonic WoE whenever min_bins allows it", {
  set.seed(1)
  for (s in 1:15) {
    n <- 150
    x <- rnorm(n)
    y <- rbinom(n, 1, plogis(x + 0.5 * sin(3 * x)))
    for (a in c("bb", "dmiv", "dp")) {
      r <- ob_num(a)(x, y)
      if (length(r$bin) > 3) expect_true(is_monotone(r$woe), info = paste(a, s))
    }
  }
})

test_that("regression: unified NA / Inf contract for all six engines", {
  # NA/NaN rows are excluded silently (cm and dp used to stop, ewb warned,
  # fetb counted NA in a bin); -Inf/+Inf are extreme values in the first/last
  # bin and never cut points (cm used to stop); an NA target is an error.
  set.seed(12)
  n <- 2000
  x <- rnorm(n)
  y <- rbinom(n, 1, plogis(x))
  x[sample(n, 100)] <- NA
  x[sample(which(!is.na(x)), 3)] <- NaN
  x[sample(which(!is.na(x)), 5)] <- Inf
  x[sample(which(!is.na(x)), 5)] <- -Inf
  for (a in c("bb", "cm", "dmiv", "dp", "ewb", "fetb")) {
    r <- NULL
    expect_no_warning(r <- ob_num(a)(x, y))
    expect_equal(sum(r$count), sum(!is.na(x)), info = a)
    expect_valid_binning(r, x, y, max_bins = 5, info = a)
    y2 <- y
    y2[10] <- NA
    expect_error(ob_num(a)(x, y2), info = a)
  }
  # fewer than two distinct finite values: a valid result, never a crash
  xc <- rep(c(-Inf, 3, Inf), 20)
  yc <- rep(0:1, 30)
  for (a in c("bb", "cm", "dmiv", "dp", "ewb", "fetb")) {
    r <- NULL
    expect_no_warning(r <- ob_num(a)(xc, yc, min_bins = 2))
    expect_valid_binning(r, xc, yc, max_bins = 5, info = a)
  }
  # constant feature: one bin, or ewb's documented error
  for (a in c("bb", "cm", "dmiv", "dp", "fetb")) {
    r <- ob_num(a)(rep(3, 40), rep(0:1, 20))
    expect_identical(as.numeric(r$count), 40, info = a)
  }
  expect_error(ob_numerical_ewb(rep(3, 40), rep(0:1, 20)), "two unique")
})

# ---------------------------------------------------------------------------
# FETB
# ---------------------------------------------------------------------------
test_that("regression: fetb merges the pair with the highest two-sided p-value", {
  # Two large bins with IDENTICAL event rates and one small, different bin.
  # The old criterion (hypergeometric point probability of the observed
  # table) is small for large tables even under perfect independence, so it
  # merged the small bin instead of the two identical ones.
  x <- c(rep(1, 300), rep(2, 300), rep(3, 6))
  y <- c(rep(0:1, c(200, 100)), rep(0:1, c(200, 100)), rep(0:1, c(3, 3)))
  r <- ob_numerical_fetb(x, y, min_bins = 2, max_bins = 2, max_n_prebins = 3)

  p_ab <- fisher.test(matrix(c(100, 200, 100, 200), 2, byrow = TRUE))$p.value
  p_bc <- fisher.test(matrix(c(100, 200, 3, 3), 2, byrow = TRUE))$p.value
  expect_gt(p_ab, p_bc)
  expect_equal(as.numeric(r$count), c(600, 6))
  expect_equal(as.numeric(r$cutpoints), 2)
})

test_that("regression: fetb honours max_bins (no early IV-convergence exit)", {
  set.seed(99)
  n <- 20000
  x <- runif(n)
  y <- rbinom(n, 1, 0.3)
  r <- ob_numerical_fetb(x, y, max_bins = 3)
  expect_lte(length(r$bin), 3)
  expect_true(r$converged)
})

test_that("regression: fetb creates at most max_n_prebins pre-bins and no empty bin", {
  # step = floor(39 / 20) = 1 used to make 39 pre-bins, the last one empty.
  x <- 1:39
  y <- as.integer(x > 20)
  r <- ob_numerical_fetb(x, y, min_bins = 2, max_bins = 50, max_n_prebins = 20)
  expect_lte(length(r$bin), 20)
  expect_true(all(r$count > 0))
  expect_equal(sum(r$count), 39)
})

test_that("regression: fetb excludes missing feature values and rejects NA targets", {
  set.seed(3)
  x <- rnorm(200)
  y <- rbinom(200, 1, plogis(x))
  x[c(5, 50, 150)] <- NA
  r <- ob_numerical_fetb(x, y)
  expect_equal(sum(r$count), 197)
  expect_valid_binning(r, x, y, max_bins = 5)

  y2 <- y
  y2[7] <- NA
  expect_error(ob_numerical_fetb(rnorm(200), y2), "binary")
  expect_error(ob_numerical_fetb(c(NA_real_, NA_real_), c(0, 1)), "non-missing")
})

test_that("fetb handles infinite values and validates its parameters", {
  x <- c(-Inf, rnorm(50), Inf, Inf)
  y <- c(0, rep(0:1, 25), 1, 1)
  r <- ob_numerical_fetb(x, y, min_bins = 2)
  expect_valid_binning(r, x, y, max_bins = 5)
  expect_error(ob_numerical_fetb(rnorm(10), rep(0:1, 5), max_n_prebins = 1), "max_n_prebins")
  expect_error(ob_numerical_fetb(rnorm(10), rep(0:1, 5), max_iterations = -1), "max_iterations")
  r0 <- ob_numerical_fetb(rnorm(100), rep(0:1, 50), max_iterations = 0)
  expect_false(r0$converged)
  expect_identical(r0$iterations, 0L)
})

# ---------------------------------------------------------------------------
# ChiMerge
# ---------------------------------------------------------------------------

# Independent reference: equal-frequency chunks, then repeatedly merge the
# adjacent pair with the smallest chisq.test(correct = TRUE) statistic.
cm_reference <- function(x, y, n_init, n_final) {
  o <- order(x)
  xs <- x[o]
  ys <- y[o]
  n <- length(x)
  rpb <- n %/% n_init
  grp <- (seq_len(n) - 1L) %/% rpb
  up <- as.numeric(tapply(xs, grp, max))
  pos <- as.numeric(tapply(ys, grp, sum))
  cnt <- as.numeric(tapply(ys, grp, length))
  stat <- function(i) {
    m <- rbind(c(pos[i], cnt[i] - pos[i]), c(pos[i + 1], cnt[i + 1] - pos[i + 1]))
    suppressWarnings(unname(stats::chisq.test(m, correct = TRUE)$statistic))
  }
  while (length(cnt) > n_final) {
    i <- which.min(vapply(seq_len(length(cnt) - 1L), stat, 0))
    pos[i] <- pos[i] + pos[i + 1]
    cnt[i] <- cnt[i] + cnt[i + 1]
    up[i] <- up[i + 1]
    pos <- pos[-(i + 1)]
    cnt <- cnt[-(i + 1)]
    up <- up[-(i + 1)]
  }
  list(cut = up[-length(up)], count = cnt)
}

test_that("regression: cm merges by the current chi-square of each adjacent pair", {
  # The old triangular cache returned stale statistics after every merge and
  # used the unclamped Yates term; both made the greedy merge sequence differ
  # from ChiMerge. min_bins = max_bins isolates the chi-square merging.
  for (s in 1:5) {
    set.seed(s)
    x <- rnorm(400)
    y <- rbinom(400, 1, plogis(0.7 * x))
    ref <- cm_reference(x, y, 20, 5)
    r <- ob_numerical_cm(x, y, min_bins = 5, max_bins = 5)
    expect_equal(as.numeric(r$cutpoints), ref$cut, info = s)
    expect_equal(as.numeric(r$count), ref$count, info = s)
  }
})

test_that("regression: cm chi-square does not overflow on large bins", {
  # count * pair_pos was an int product: > 2^31 for bins of 50,000 rows.
  set.seed(1)
  n <- 250000
  x <- rnorm(n)
  y <- rbinom(n, 1, plogis(0.05 * x))
  ref <- cm_reference(x, y, 5, 4)
  r <- ob_numerical_cm(x, y, min_bins = 4, max_bins = 4, max_n_prebins = 5)
  expect_equal(as.numeric(r$cutpoints), ref$cut)
})

test_that("regression: cm never splits tied values across bins", {
  # A run of ties reaching into the final chunk was split: part of it went
  # to a new last bin although the cutpoint assigns all of it below.
  x <- c(1:85, rep(100, 15))
  y <- rep(0:1, 50)
  for (mb in c(5, 10)) {
    r <- ob_numerical_cm(x, y, min_bins = 2, max_bins = mb, max_n_prebins = 20)
    expect_valid_binning(r, x, y, max_bins = mb)
  }
})

test_that("regression: cm equal-width bins are right-closed and labels contiguous", {
  x <- rep(0:10, each = 20)
  y <- rep(c(0, 1, 1, 0, 1), length.out = length(x))
  r <- ob_numerical_cm(x, y, min_bins = 2, max_bins = 10, max_n_prebins = 20,
                       init_method = "equal_width")
  expect_valid_binning(r, x, y, max_bins = 10)
  # each label's lower bound is the previous bin's upper bound
  lo <- sub("^\\((.*);.*$", "\\1", r$bin)[-1]
  up <- sub("^.*;(.*)\\]$", "\\1", r$bin)[-length(r$bin)]
  expect_identical(lo, up)
})

test_that("regression: cm keeps distinct values on a tiny scale apart", {
  set.seed(4)
  x <- runif(300) * 1e-11
  y <- rbinom(300, 1, plogis(as.numeric(scale(x)) * 2))
  r <- ob_numerical_cm(x, y)
  expect_gte(length(r$bin), 3)
  expect_valid_binning(r, x, y, max_bins = 5)
})

test_that("regression: cm reports each warning once, for the final binning", {
  set.seed(5)
  x <- rnorm(1000)
  y <- rbinom(1000, 1, plogis(2 * x))
  r <- ob_numerical_cm(x, y)
  expect_equal(anyDuplicated(r$warnings), 0L)
  iv_notes <- grep("total IV", r$warnings, value = TRUE)
  expect_lte(length(iv_notes), 1L)
  if (length(iv_notes)) {
    expect_true(grepl(sprintf("%.6f", r$total_iv), iv_notes, fixed = TRUE))
  }
})

test_that("cm Chi2 variant and parameter checks", {
  set.seed(6)
  x <- rnorm(500)
  y <- rbinom(500, 1, plogis(x))
  r <- ob_numerical_cm(x, y, use_chi2_algorithm = TRUE)
  expect_identical(r$algorithm, "Chi2")
  expect_valid_binning(r, x, y, max_bins = 5)
  expect_warning(ob_numerical_cm(x, y, max_n_prebins = 2), "max_n_prebins")
  expect_error(ob_numerical_cm(rep(NA_real_, 4), c(0, 1, 0, 1)), "non-missing")
  r2 <- ob_numerical_cm(sample(c(1, 2), 100, TRUE), rep(0:1, 50))
  expect_lte(length(r2$bin), 2)
})

# ---------------------------------------------------------------------------
# BB / DMIV
# ---------------------------------------------------------------------------
test_that("regression: bb and dmiv return monotonic WoE", {
  # A single merge pass comparing stale WoE left 4 non-monotonic bins
  # (min_bins = 3) for this sample.
  set.seed(134)
  x <- rnorm(150)
  y <- rbinom(150, 1, plogis(x))
  for (a in c("bb", "dmiv")) {
    r <- ob_num(a)(x, y)
    expect_true(is_monotone(r$woe), info = a)
    expect_valid_binning(r, x, y, max_bins = 5, info = a)
  }
})

test_that("regression: bb and dmiv never return an empty bin", {
  set.seed(2)
  x <- c(rep(0, 5), rnorm(3))
  y <- c(0, 1, rbinom(6, 1, 0.5))
  for (a in c("bb", "dmiv")) {
    r <- ob_num(a)(x, y)
    expect_true(all(r$count > 0), info = a)
    expect_valid_binning(r, x, y, max_bins = 5, info = a)
  }
})

test_that("regression: bb and dmiv separate two values closer than 1e-10", {
  x <- rep(c(1e-11, 2e-11), each = 50)
  y <- rep(c(0, 1), c(50, 50))
  for (a in c("bb", "dmiv")) {
    r <- ob_num(a)(x, y)
    expect_equal(as.numeric(r$count), c(50, 50), info = a)
  }
})

test_that("regression: bb and dmiv never use an infinite boundary", {
  x <- c(rep(-Inf, 10), rnorm(80), rep(Inf, 10))
  y <- rep(0:1, 50)
  for (a in c("bb", "dmiv")) {
    r <- ob_num(a)(x, y, min_bins = 2)
    expect_valid_binning(r, x, y, max_bins = 5, info = a)
  }
  x2 <- rep(c(-Inf, 0, Inf), 10)
  for (a in c("bb", "dmiv")) {
    r <- ob_num(a)(x2, rep(0:1, 15))
    expect_valid_binning(r, x2, rep(0:1, 15), max_bins = 5, info = a)
  }
})

test_that("bb and dmiv exclude NA, report errors, and accept all divergences", {
  set.seed(8)
  x <- rnorm(300)
  y <- rbinom(300, 1, plogis(x))
  x[1:5] <- NA
  for (a in c("bb", "dmiv")) {
    r <- ob_num(a)(x, y)
    expect_equal(sum(r$count), 295)
  }
  for (dm in c("he", "kl", "tr", "klj", "sc", "js", "l1", "l2", "ln")) {
    r <- ob_numerical_dmiv(x, y, divergence_method = dm, bin_method = "woe")
    expect_valid_binning(r, x, y, max_bins = 5, info = dm)
    expect_equal(r$total_divergence, sum(r$divergence))
  }
  expect_error(ob_numerical_bb(rep(NA_real_, 4), c(0, 1, 0, 1)), "non-NA")
  expect_error(ob_numerical_dmiv(rep(NA_real_, 4), c(0, 1, 0, 1)), "non-NA")
  expect_error(ob_numerical_bb(rnorm(10), rep(0:1, 5), max_n_prebins = 2), "max_n_prebins")
  expect_error(ob_numerical_dmiv(rnorm(10), rep(0:1, 5), min_bins = 1), "min_bins")
})

# ---------------------------------------------------------------------------
# DP
# ---------------------------------------------------------------------------
test_that("regression: dp never returns an empty bin", {
  set.seed(1)
  x <- rnorm(5)
  y <- c(0, 1, rbinom(3, 1, 0.5))
  r <- ob_numerical_dp(x, y)
  expect_true(all(r$count > 0))
  expect_valid_binning(r, x, y, max_bins = 5)
})

test_that("regression: dp result respects the monotonic trend after rare-bin merges", {
  set.seed(6)
  x <- rlnorm(60, 0, 2)
  y <- rbinom(60, 1, plogis(0.8 * as.numeric(scale(rank(x)))))
  r <- ob_numerical_dp(x, y)
  expect_true(is_monotone(r$woe))
  expect_valid_binning(r, x, y, max_bins = 5)
})

test_that("regression: dp splits two values without a non-finite cutpoint", {
  for (v in list(c(1, Inf), c(-Inf, 5), c(-1.5e308, 1.6e308))) {
    x <- rep(v, 10)
    y <- rep(0:1, each = 10)
    r <- ob_numerical_dp(x, y)
    expect_equal(as.numeric(r$count), c(10, 10), info = format(v))
    expect_true(all(is.finite(r$cutpoints)))
  }
  x <- c(-Inf, rnorm(40), Inf)
  y <- rep(0:1, 21)
  r <- ob_numerical_dp(x, y, min_bins = 2)
  expect_valid_binning(r, x, y, max_bins = 5)
  expect_error(ob_numerical_dp(rep(NA_real_, 4), c(0, 1, 0, 1)), "non-missing")
})

# ---------------------------------------------------------------------------
# EWB
# ---------------------------------------------------------------------------
test_that("regression: ewb bins a tiny-scale feature and counts each row once", {
  set.seed(9)
  x <- runif(200) * 1e-11
  y <- rbinom(200, 1, plogis(as.numeric(scale(x)) * 2))
  r <- ob_numerical_ewb(x, y)
  expect_gte(length(r$bin), 3)
  expect_valid_binning(r, x, y, max_bins = 5)
})

test_that("regression: ewb assigns boundary values to the bin below", {
  x <- rep(0:10, each = 10)
  y <- rep(c(0, 1, 0, 1, 1), length.out = 110)
  r <- ob_numerical_ewb(x, y, min_bins = 2, max_bins = 10, max_n_prebins = 5,
                        is_monotonic = FALSE, bin_cutoff = 0.01)
  expect_valid_binning(r, x, y, max_bins = 10)
})

test_that("regression: ewb handles overflowing ranges and infinite values", {
  x <- c(-1e308, 1e308, rnorm(40))
  y <- rep(0:1, 21)
  r <- ob_numerical_ewb(x, y, min_bins = 2)
  expect_valid_binning(r, x, y, max_bins = 5)
  x2 <- c(-Inf, rnorm(40), Inf)
  r2 <- ob_numerical_ewb(x2, y, min_bins = 2)
  expect_valid_binning(r2, x2, y, max_bins = 5)
  x3 <- rep(c(-Inf, 0, Inf), 14)
  r3 <- ob_numerical_ewb(x3, y)
  expect_valid_binning(r3, x3, y, max_bins = 5)
})

test_that("regression: ewb WoE uses the class totals of the binned rows", {
  set.seed(10)
  x <- rnorm(300)
  y <- rbinom(300, 1, plogis(x))
  x[y == 1][1:20] <- NA
  r <- NULL
  expect_no_warning(r <- ob_numerical_ewb(x, y))
  nb <- length(r$bin)
  tp <- sum(y[!is.na(x)] == 1)
  tn <- sum(y[!is.na(x)] == 0)
  woe <- log(((r$count_pos + 0.5) / (tp + 0.5 * nb)) / ((r$count_neg + 0.5) / (tn + 0.5 * nb)))
  expect_equal(r$woe, woe)
})

test_that("regression: ewb drops empty equal-width intervals", {
  set.seed(11)
  x <- c(rlnorm(500, 0, 2), 1e4)
  y <- rbinom(501, 1, 0.3)
  r <- ob_numerical_ewb(x, y)
  expect_true(all(r$count > 0))
  expect_valid_binning(r, x, y, max_bins = 5)
})

test_that("ewb few-unique path labels the last bin open-ended", {
  x <- rep(c(1, 2), 20)
  y <- rep(0:1, each = 20)
  r <- ob_numerical_ewb(x, y)
  expect_match(r$bin[length(r$bin)], "\\+Inf\\]$")
  expect_error(ob_numerical_ewb(rep(1, 10), rep(0:1, 5)), "two unique")
})

# ---------------------------------------------------------------------------
# Inputs that used to crash R (kept last: on the old code they abort the
# process rather than fail).
# ---------------------------------------------------------------------------
test_that("regression: cm equal-width no longer crashes on an overflowing range", {
  # (1e308 - (-1e308)) / width was Inf / Inf = NaN, cast to int and used as
  # an index: segmentation fault.
  x <- c(0.26, -1e308, 0, -1e308, 0.54, 1, 0, 1e308, 0.54, -1, 0.26, 0.54)
  y <- c(0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0)
  r <- ob_numerical_cm(x, y, min_bins = 2, max_bins = 8, max_n_prebins = 40,
                       init_method = "equal_width")
  expect_equal(sum(r$count), length(x))
})

test_that("regression: fetb rejects max_n_prebins = 0 instead of dividing by zero", {
  expect_error(ob_numerical_fetb(rnorm(10), rep(0:1, 5), max_n_prebins = 0), "max_n_prebins")
})
