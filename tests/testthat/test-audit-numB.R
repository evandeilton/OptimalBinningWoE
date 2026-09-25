# Audit regression and invariant tests for fast_mdlp, ir, jedi, jedi_mwoe, kmb.
#
# Every "fix" test below fails on the pre-audit sources; the invariant tests
# pin properties every binning must have: counts add up, cutpoints strictly
# increase and reproduce the reported counts under the right-closed (a, b]
# convention, no bin is empty, WoE is finite and no unexpected warning is
# raised.

# Recount the non-missing observations per bin from the reported cutpoints.
numB_recount <- function(x, cutpoints) {
  x <- x[!is.na(x)]
  idx <- findInterval(x, cutpoints, left.open = TRUE) + 1L
  tabulate(idx, nbins = length(cutpoints) + 1L)
}

numB_check_binning <- function(res, x, max_bins = Inf, info = "") {
  cnt <- res$count
  cut <- res$cutpoints
  expect_equal(sum(cnt), sum(!is.na(x)), info = info)
  expect_length(cut, length(cnt) - 1L)
  if (length(cut) > 1) expect_true(all(diff(cut) > 0), info = info)
  expect_true(all(cnt > 0), info = info)
  expect_lte(length(cnt), max_bins)
  expect_true(all(is.finite(res$woe)), info = info)
  expect_equal(numB_recount(x, cut), as.integer(cnt), info = info)
  if (!is.null(res$count_pos)) {
    expect_equal(res$count_pos + res$count_neg, cnt, info = info)
  }
}

# Independent reference implementation of Fayyad & Irani (1993) MDLP:
# all cuts between distinct values are evaluated, the best one (largest
# information gain) is accepted iff
#   Gain > (log2(N - 1) + log2(3^k - 2) - [k E(S) - k1 E(S1) - k2 E(S2)]) / N.
numB_mdlp_reference <- function(x, y) {
  ent <- function(p, n) {
    q <- c(p, n) / (p + n)
    q <- q[q > 0]
    -sum(q * log2(q))
  }
  o <- order(x)
  x <- x[o]
  y <- y[o]
  cuts <- numeric()
  rec <- function(lo, hi) {
    xs <- x[lo:hi]
    ys <- y[lo:hi]
    N <- length(xs)
    if (N <= 1) return(invisible())
    P <- sum(ys)
    if (P == 0 || P == N) return(invisible())
    ES <- ent(P, N - P)
    cand <- which(diff(xs) != 0)
    if (!length(cand)) return(invisible())
    cp <- cumsum(ys)
    gains <- vapply(cand, function(i) {
      pl <- cp[i]
      pr <- P - pl
      ES - (i / N * ent(pl, i - pl) + (N - i) / N * ent(pr, N - i - pr))
    }, numeric(1))
    i <- cand[which(gains > max(gains) - 1e-12)[1]]
    pl <- cp[i]
    nl <- i - pl
    pr <- P - pl
    nr <- N - i - pr
    k1 <- (pl > 0) + (nl > 0)
    k2 <- (pr > 0) + (nr > 0)
    delta <- log2(3^2 - 2) - (2 * ES - k1 * ent(pl, nl) - k2 * ent(pr, nr))
    if (max(gains) > (log2(N - 1) + delta) / N) {
      cuts <<- c(cuts, xs[i])
      rec(lo, lo + i - 1)
      rec(lo + i, hi)
    }
    invisible()
  }
  rec(1, length(x))
  sort(cuts)
}

# ---------------------------------------------------------------------------
# fast_mdlp
# ---------------------------------------------------------------------------

test_that("fast_mdlp applies the Fayyad-Irani MDL acceptance criterion", {
  # The previous criterion used Delta = log2(7) - 2 E(S), dropping the
  # k1 E(S1) + k2 E(S2) term. That lowers the acceptance threshold, so it kept
  # cuts that the MDL principle rejects (about 1 dataset in 10 of this kind).
  n_multi <- 0
  for (s in 1:40) {
    set.seed(s)
    n <- c(60, 200, 800)[s %% 3 + 1]
    x <- switch(s %% 3 + 1, rnorm(n), round(rnorm(n), 1), sample(1:15, n, TRUE) + 0)
    y <- rbinom(n, 1, plogis(if (s %% 2) 2 * x else 2 * sin(2 * x)))
    ref <- numB_mdlp_reference(x, y)
    res <- ob_numerical_fast_mdlp(x, y, min_bins = 2, max_bins = 100,
                                  force_monotonicity = FALSE)
    if (length(ref) == 0) {
      # no MDL cut: min_bins = 2 forces one cut
      expect_length(res$cutpoints, 1)
    } else {
      expect_equal(res$cutpoints, ref, info = paste("seed", s))
      n_multi <- n_multi + (length(ref) > 1)
    }
  }
  expect_gt(n_multi, 5)
})

test_that("fast_mdlp monotonicity merging keeps counts consistent", {
  # Merging bins more than once used to index the already-merged count vectors
  # with the original bin numbers: counts no longer summed to n (e.g. 9737 for
  # n = 3000), and reads went past the end of the vectors.
  for (s in 1:5) {
    set.seed(s)
    n <- 3000
    x <- runif(n)
    y <- rbinom(n, 1, 0.2 + 0.6 * (sin(12 * x) > 0))
    res <- ob_numerical_fast_mdlp(x, y, min_bins = 2, max_bins = 20)
    numB_check_binning(res, x, max_bins = 20, info = paste("seed", s))
    expect_true(all(diff(res$woe) > 0) || all(diff(res$woe) < 0))
  }
})

test_that("fast_mdlp keeps the most informative cuts when max_bins binds", {
  # max_bins used to be enforced by dropping the rightmost cuts, whatever
  # their information: here the strongest cut (0.85) was the one discarded.
  set.seed(2024)
  n <- 4000
  x <- runif(n)
  y <- rbinom(n, 1, ifelse(x < 0.85, ifelse(x < 0.2, 0.05, ifelse(x < 0.45, 0.35, 0.15)), 0.97))
  full <- ob_numerical_fast_mdlp(x, y, min_bins = 2, max_bins = 50, force_monotonicity = FALSE)
  expect_equal(full$cutpoints, c(0.2, 0.45, 0.85), tolerance = 0.02)
  two <- ob_numerical_fast_mdlp(x, y, min_bins = 2, max_bins = 2, force_monotonicity = FALSE)
  expect_equal(two$cutpoints, full$cutpoints[3])
  expect_gt(sum(two$iv), 2)
})

test_that("fast_mdlp returns one bin for a constant feature", {
  x <- rep(7, 50)
  y <- rep(0:1, 25)
  res <- NULL
  expect_warning(res <- ob_numerical_fast_mdlp(x, y, min_bins = 3, max_bins = 5),
                 "identical")
  expect_equal(res$count, 50L)
  expect_length(res$cutpoints, 0)
  expect_equal(res$bin, "(-Inf;+Inf]")
})

test_that("fast_mdlp never splits a run of tied values", {
  # With fewer distinct values than min_bins, equal-frequency cuts used to
  # fall inside runs of ties: duplicated cutpoints and counts that could not
  # be reproduced from them.
  set.seed(3)
  x <- as.numeric(sample(1:3, 300, TRUE))
  y <- rbinom(300, 1, 0.4)
  res <- NULL
  expect_warning(res <- ob_numerical_fast_mdlp(x, y, min_bins = 5, max_bins = 5),
                 "failed to respect min_bins")
  numB_check_binning(res, x)
  expect_equal(res$cutpoints, c(1, 2))
})

test_that("fast_mdlp reaches min_bins whenever the distinct values allow it", {
  # The gap bisection gave up as soon as the widest gap held no admissible
  # cut, and never cut after a singleton first value: here 2 bins and a
  # warning, although min_bins = 3 bins exist.
  x <- c(1, 2, 2, 2, 3, 3, 3)
  y <- c(0L, 1L, 0L, 1L, 0L, 1L, 0L)
  res <- NULL
  expect_no_warning(res <- ob_numerical_fast_mdlp(x, y, min_bins = 3, max_bins = 3))
  expect_equal(res$cutpoints, c(1, 2))
  numB_check_binning(res, x, max_bins = 3)
  res2 <- NULL
  expect_no_warning(res2 <- ob_numerical_fast_mdlp(c(1, 2), c(0L, 1L), min_bins = 2, max_bins = 2))
  expect_equal(res2$count, c(1L, 1L))
})

test_that("fast_mdlp rejects a non-binary target", {
  expect_error(ob_numerical_fast_mdlp(rnorm(20), rep(1:2, 10)), "binary")
})

test_that("fast_mdlp excludes missing values", {
  set.seed(4)
  x <- rnorm(500)
  y <- rbinom(500, 1, plogis(x))
  x[1:25] <- NA
  x[26] <- NaN
  res <- NULL
  expect_no_warning(res <- ob_numerical_fast_mdlp(x, y))
  numB_check_binning(res, x, max_bins = 5)
})

# ---------------------------------------------------------------------------
# jedi (default algorithm of obwoe) and jedi_mwoe
# ---------------------------------------------------------------------------

test_that("jedi excludes missing feature values instead of failing", {
  set.seed(5)
  x <- rnorm(400)
  y <- rbinom(400, 1, plogis(x))
  x[sample(400, 30)] <- NA
  x[1] <- NaN
  res <- NULL
  expect_no_warning(res <- ob_numerical_jedi(x, y))
  numB_check_binning(res, x, max_bins = 5)
  expect_equal(sum(res$count_pos), sum(y[!is.na(x)]))
})

test_that("obwoe's default algorithm bins numeric features that contain NA", {
  set.seed(6)
  d <- data.frame(x = rnorm(1000), z = rnorm(1000))
  d$target <- rbinom(1000, 1, plogis(d$x))
  d$x[sample(1000, 40)] <- NA
  fit <- NULL
  expect_no_warning(fit <- obwoe(d, target = "target"))
  expect_false(any(fit$summary$error))
  expect_null(fit$results$x$error)
  expect_equal(sum(fit$results$x$count), 960)
})

test_that("jedi and jedi_mwoe never create empty bins", {
  # With fewer distinct values than min_bins, the pre-binning used to halve
  # intervals between distinct values, creating bins with no observation.
  set.seed(11)
  x <- as.numeric(sample(1:3, 300, TRUE))
  y <- rbinom(300, 1, 0.4)
  r1 <- ob_numerical_jedi(x, y, min_bins = 5, max_bins = 5)
  numB_check_binning(r1, x, max_bins = 5)
  expect_equal(r1$cutpoints, c(1, 2))
  y3 <- as.integer((seq_len(300) + sample(0:2, 300, TRUE)) %% 3)
  r2 <- ob_numerical_jedi_mwoe(x, y3, min_bins = 5, max_bins = 5)
  numB_check_binning(r2, x, max_bins = 5)
  expect_equal(r2$cutpoints, c(1, 2))
  expect_equal(rowSums(r2$class_counts), r2$count)
})

test_that("jedi_mwoe excludes missing values and validates class labels", {
  set.seed(12)
  x <- rnorm(600)
  y3 <- as.integer(cut(x + rnorm(600, 0, 0.5), c(-Inf, -0.5, 0.5, Inf))) - 1L
  x[1:20] <- NA
  res <- ob_numerical_jedi_mwoe(x, y3)
  numB_check_binning(res, x, max_bins = 5)
  expect_equal(colSums(res$class_counts), as.numeric(tabulate(y3[!is.na(x)] + 1L, 3)))
  expect_error(ob_numerical_jedi_mwoe(rnorm(30), rep(c(0L, 2L), 15)), "n_classes")
})

# ---------------------------------------------------------------------------
# Numerical NA contract, every algorithm:
#   NA / NaN feature rows are excluded silently; -Inf / +Inf are extreme values
#   of the first / last bin and never a cutpoint; a missing target is an error;
#   a feature with fewer than two finite distinct values yields a valid
#   single- (or two-) bin result.
# ---------------------------------------------------------------------------

numB_algos <- list(
  fast_mdlp = function(x, y, ...) ob_numerical_fast_mdlp(x, y, ...),
  ir = function(x, y, ...) ob_numerical_ir(x, y, ...),
  jedi = function(x, y, ...) ob_numerical_jedi(x, y, ...),
  jedi_mwoe = function(x, y, ...) ob_numerical_jedi_mwoe(x, y, ...),
  kmb = function(x, y, ...) ob_numerical_kmb(x, y, ...)
)

test_that("NA feature rows are excluded silently by every algorithm", {
  set.seed(41)
  x <- rnorm(2000)
  y <- rbinom(2000, 1, plogis(x))
  x[sample(2000, 100)] <- NA
  x[3] <- NaN
  for (a in names(numB_algos)) {
    res <- NULL
    expect_no_warning(res <- numB_algos[[a]](x, y))
    numB_check_binning(res, x, max_bins = 5, info = a)
    expect_equal(sum(res$count), 1899L, info = a)
  }
})

test_that("-Inf and +Inf are extreme values of the outer bins, never cutpoints", {
  set.seed(42)
  x <- c(rnorm(600), -Inf, -Inf, Inf, Inf, Inf)
  y <- c(rbinom(600, 1, 0.4), 0L, 1L, 1L, 0L, 1L)
  for (a in names(numB_algos)) {
    res <- NULL
    expect_no_warning(res <- numB_algos[[a]](x, y))
    expect_true(all(is.finite(res$cutpoints)), info = a)
    numB_check_binning(res, x, max_bins = 5, info = a)
    expect_equal(sum(res$count), 605L, info = a)
  }
  # an infinite value next to the only finite ones is still no cutpoint
  x2 <- c(-Inf, rep(c(1, 2, 3), 40), Inf)
  y2 <- c(1L, rep(c(0L, 1L, 1L), 40), 0L)
  for (a in names(numB_algos)) {
    res <- suppressWarnings(numB_algos[[a]](x2, y2, min_bins = 2, max_bins = 5))
    expect_true(all(is.finite(res$cutpoints)), info = a)
    expect_equal(sum(res$count), 122L, info = a)
    expect_equal(numB_recount(x2, res$cutpoints), as.integer(res$count), info = a)
  }
})

test_that("a missing target is an error for every algorithm", {
  x <- c(1, 2, 3, 4, 5, 6)
  y <- c(0L, 1L, NA, 1L, 0L, 1L)
  for (a in names(numB_algos)) {
    expect_error(numB_algos[[a]](x, y), "missing|NA", info = a)
  }
  # also when the feature of that row is missing
  x[3] <- NA
  for (a in names(numB_algos)) {
    expect_error(numB_algos[[a]](x, y), "missing|NA", info = a)
  }
})

test_that("fewer than two finite distinct values give a valid small binning", {
  y <- rep(c(0L, 1L), 10)
  x1 <- c(rep(5, 18), -Inf, Inf)
  x2 <- rep(c(-Inf, Inf), 10)
  x3 <- c(rep(5, 10), rep(7, 8), -Inf, Inf)
  for (a in names(numB_algos)) {
    for (xx in list(x1, x2)) {
      res <- NULL
      if (a == "fast_mdlp") {
        # fast_mdlp documents a warning for a feature without two distinct values
        expect_warning(res <- numB_algos[[a]](xx, y), "identical")
      } else {
        expect_no_warning(res <- numB_algos[[a]](xx, y))
      }
      expect_length(res$count, 1)
      expect_equal(sum(res$count), 20L, info = a)
      expect_length(res$cutpoints, 0)
    }
    res <- numB_algos[[a]](x3, y, min_bins = 2, max_bins = 5)
    expect_equal(res$cutpoints, 5, info = a)
    expect_equal(res$count, c(11L, 9L), info = a)
  }
  expect_error(ob_numerical_jedi(rep(NA_real_, 3), c(0L, 1L, 0L)), "non-missing")
  expect_error(ob_numerical_fast_mdlp(c(NA_real_, NA), c(0L, 1L)), "non-missing")
})

test_that("jedi and jedi_mwoe results do not depend on the sort path", {
  # Large inputs (>= 4096 values per class) are sorted with a radix sort,
  # small ones with std::sort; binning a sample and a replicated copy of it
  # must give the same cutpoints and proportional counts.
  set.seed(13)
  x <- c(round(rnorm(300), 2), -0, 0, 1e-300, -1e-300)
  y <- rbinom(length(x), 1, plogis(x))
  y3 <- as.integer((rank(x, ties.method = "first") + rbinom(length(x), 1, 0.3)) %% 3)
  k <- 40
  for (fn in c("ob_numerical_jedi", "ob_numerical_jedi_mwoe")) {
    yy <- if (fn == "ob_numerical_jedi") y else y3
    small <- do.call(fn, list(feature = x, target = yy))
    big <- do.call(fn, list(feature = rep(x, k), target = rep(yy, k)))
    expect_equal(big$cutpoints, small$cutpoints, info = fn)
    expect_equal(big$count, small$count * k, info = fn)
    numB_check_binning(big, rep(x, k), max_bins = 5, info = fn)
  }
})

# ---------------------------------------------------------------------------
# ir
# ---------------------------------------------------------------------------

test_that("ir rejects empty input instead of reading out of range", {
  expect_error(ob_numerical_ir(numeric(0), integer(0)), "empty")
})

test_that("ir never creates empty bins or duplicated cutpoints", {
  # splitLargestBin() used to cut at a value equal to a bin bound, producing
  # bins such as (2;2] or (3;+Inf] with no observation.
  set.seed(21)
  x <- as.numeric(sample(1:3, 1000, TRUE))
  y <- rbinom(1000, 1, c(0.7, 0.45, 0.25)[x])
  res <- ob_numerical_ir(x, y, min_bins = 4, max_bins = 4, bin_cutoff = 0.02,
                         max_n_prebins = 50)
  numB_check_binning(res, x, max_bins = 4)
  set.seed(22)
  x <- as.numeric(sample(1:4, 30, TRUE))
  y <- rbinom(30, 1, 0.5)
  res <- ob_numerical_ir(x, y, min_bins = 5, max_bins = 5)
  numB_check_binning(res, x, max_bins = 5)
})

test_that("ir event rates are monotone after pooling", {
  for (s in 1:15) {
    set.seed(100 + s)
    n <- sample(c(50, 400, 3000), 1)
    x <- rnorm(n)
    y <- rbinom(n, 1, plogis(sin(2 * x)))
    res <- ob_numerical_ir(x, y, min_bins = 3, max_bins = 8, max_n_prebins = 30)
    numB_check_binning(res, x, max_bins = 8, info = paste("seed", s))
    rate <- res$count_pos / res$count
    if (isTRUE(res$monotone_increasing)) {
      expect_true(all(diff(rate) >= 0), info = paste("seed", s))
    } else {
      expect_true(all(diff(rate) <= 0), info = paste("seed", s))
    }
  }
})

# ---------------------------------------------------------------------------
# kmb
# ---------------------------------------------------------------------------

test_that("kmb centroids are the mean feature value of each bin", {
  set.seed(2)
  x <- rlnorm(1000)
  y <- rbinom(1000, 1, 0.3)
  res <- ob_numerical_kmb(x, y)
  bin_of <- findInterval(x, res$cutpoints, left.open = TRUE) + 1L
  expect_equal(res$centroids, as.numeric(tapply(x, bin_of, mean)), tolerance = 1e-12)
})

test_that("kmb removes empty equal-width intervals and keeps min_bins", {
  set.seed(31)
  x <- c(runif(999), 100)
  y <- rbinom(1000, 1, 0.3)
  res <- ob_numerical_kmb(x, y, min_bins = 3, max_bins = 5)
  numB_check_binning(res, x, max_bins = 5)
  expect_gte(length(res$count), 3)
})

test_that("kmb does not depend on the scale of the feature", {
  # A range below 1e-10 used to collapse the feature into a single bin, with a
  # warning, whatever its number of distinct values.
  set.seed(32)
  x <- runif(500)
  y <- rbinom(500, 1, plogis(3 * x - 1.5))
  a <- ob_numerical_kmb(x, y)
  b <- ob_numerical_kmb(x * 1e-12, y)
  expect_equal(b$count, a$count)
  expect_equal(b$cutpoints, a$cutpoints * 1e-12)
})

test_that("kmb handles values at the limits of double precision", {
  set.seed(33)
  x <- c(-1.7e308, 1.7e308, runif(300))
  y <- rbinom(302, 1, 0.4)
  res <- ob_numerical_kmb(x, y, min_bins = 3, max_bins = 9)
  numB_check_binning(res, x, max_bins = 9)
  expect_true(all(is.finite(res$cutpoints)))
})

test_that("kmb rejects a non-binary target", {
  expect_error(ob_numerical_kmb(rnorm(20), rep(1:2, 10)), "binary")
})

# ---------------------------------------------------------------------------
# Invariants on a battery of shapes, for every algorithm
# ---------------------------------------------------------------------------

test_that("numB algorithms return valid binnings without warnings", {
  shapes <- list(
    normal = function(n) rnorm(n),
    rounded = function(n) round(rnorm(n), 1),
    lognormal = function(n) rlnorm(n, 0, 2),
    cauchy = function(n) rcauchy(n),
    extremes = function(n) c(1e300, -1e300, rnorm(n - 2)),
    small = function(n) runif(n) * 1e-9,
    integer = function(n) as.numeric(sample(0:9, n, TRUE)),
    zeros = function(n) sample(c(-0, 0, 1, 2, 3), n, TRUE)
  )
  algos <- list(
    fast_mdlp = function(x, y) ob_numerical_fast_mdlp(x, y, min_bins = 2, max_bins = 6),
    ir = function(x, y) ob_numerical_ir(x, y, min_bins = 3, max_bins = 6),
    jedi = function(x, y) ob_numerical_jedi(x, y, min_bins = 3, max_bins = 6),
    jedi_mwoe = function(x, y) ob_numerical_jedi_mwoe(x, as.integer((rank(x, ties.method = "first") + y) %% 3), min_bins = 3, max_bins = 6),
    kmb = function(x, y) ob_numerical_kmb(x, y, min_bins = 3, max_bins = 6)
  )
  for (sh in names(shapes)) {
    for (n in c(12, 300)) {
      set.seed(nchar(sh) * 100 + n)
      x <- shapes[[sh]](n)
      y <- rbinom(n, 1, plogis(scale(rank(x))[, 1]))
      if (length(unique(y)) < 2) y[1:2] <- 0:1
      for (a in names(algos)) {
        info <- paste(a, sh, n)
        res <- NULL
        expect_no_warning(res <- algos[[a]](x, y))
        numB_check_binning(res, x, max_bins = 6, info = info)
      }
    }
  }
})

test_that("numB algorithms validate their arguments with clear errors", {
  x <- rnorm(40)
  y <- rep(0:1, 20)
  y3 <- rep(0:2, length.out = 40)
  expect_error(ob_numerical_fast_mdlp(x, y[-1]), "match")
  expect_error(ob_numerical_fast_mdlp(x, y, min_bins = 1), "min_bins")
  expect_error(ob_numerical_fast_mdlp(x, y, min_bins = 4, max_bins = 3), "max_bins")
  expect_error(ob_numerical_ir(x, y[-1]), "match")
  expect_error(ob_numerical_ir(x, y, min_bins = 1), "min_bins")
  expect_error(ob_numerical_ir(x, y, min_bins = 4, max_bins = 3), "max_bins")
  expect_error(ob_numerical_ir(x, y, bin_cutoff = 1.5), "bin_cutoff")
  expect_error(ob_numerical_ir(x, y, max_n_prebins = 2), "max_n_prebins")
  expect_error(ob_numerical_ir(x, y, max_iterations = 0), "max_iterations")
  expect_error(ob_numerical_ir(x, y, convergence_threshold = 0), "convergence_threshold")
  expect_error(ob_numerical_ir(x, rep(1:2, 20)), "binary")
  expect_error(ob_numerical_ir(x, rep(1L, 40)), "both classes")
  expect_error(ob_numerical_ir(rep(NA_real_, 40), y), "non-missing")
  expect_error(ob_numerical_jedi(x, y[-1]), "match")
  expect_error(ob_numerical_jedi(x, y, bin_cutoff = 0), "bin_cutoff")
  expect_error(ob_numerical_jedi(x, y, convergence_threshold = -1), "convergence_threshold")
  expect_error(ob_numerical_jedi(x, y, max_iterations = 0), "max_iterations")
  expect_error(ob_numerical_jedi(x, rep(2L, 40)), "0 and 1")
  expect_error(ob_numerical_jedi(x, rep(1L, 40)), "both classes")
  expect_error(ob_numerical_jedi_mwoe(x, y3[-1]), "match")
  expect_error(ob_numerical_jedi_mwoe(x, rep(1L, 40)), "2 distinct")
  expect_error(ob_numerical_jedi_mwoe(x, y3, bin_cutoff = 2), "bin_cutoff")
  expect_error(ob_numerical_jedi_mwoe(x, y3, convergence_threshold = 0), "convergence_threshold")
  expect_error(ob_numerical_jedi_mwoe(x, y3, max_iterations = 0), "max_iterations")
  expect_error(ob_numerical_jedi_mwoe(x, c(-1L, y3[-1])), "n_classes")
  expect_error(ob_numerical_jedi_mwoe(rep(NA_real_, 40), y3), "non-missing")
  expect_error(ob_numerical_kmb(x, y[-1]), "match")
  expect_error(ob_numerical_kmb(numeric(0), integer(0)), "empty")
  expect_error(ob_numerical_kmb(rep(NA_real_, 40), y), "non-missing")
  expect_error(ob_numerical_kmb(x, rep(1L, 40)), "both")
  expect_error(ob_numerical_kmb(x, y, min_bins = 1), "min_bins")
  expect_error(ob_numerical_kmb(x, y, min_bins = 4, max_bins = 3), "max_bins")
  expect_error(ob_numerical_kmb(x, y, bin_cutoff = 0), "bin_cutoff")
  expect_error(ob_numerical_kmb(x, y, max_n_prebins = 0), "max_n_prebins")
  expect_error(ob_numerical_kmb(x, y, max_iterations = 0), "max_iterations")
  # a character feature is coerced, with the documented warning
  xc <- as.character(round(x, 1))
  expect_warning(ob_numerical_fast_mdlp(xc, y), "converted to numeric")
  expect_warning(ob_numerical_ir(xc, y), "converted to numeric")
  expect_warning(ob_numerical_jedi(xc, y), "converted to numeric")
  expect_warning(ob_numerical_jedi_mwoe(xc, y3), "converted to numeric")
  expect_warning(ob_numerical_kmb(xc, y), "converted to numeric")
})

test_that("kmb gives valid bins on the large max_n_prebins path", {
  set.seed(51)
  x <- round(rnorm(3000), 2)
  y <- rbinom(3000, 1, plogis(x))
  a <- ob_numerical_kmb(x, y, max_n_prebins = 20, max_bins = 20)
  b <- ob_numerical_kmb(x, y, max_n_prebins = 500, max_bins = 20)
  numB_check_binning(a, x, max_bins = 20)
  numB_check_binning(b, x, max_bins = 20)
})
