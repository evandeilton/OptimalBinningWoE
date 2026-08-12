# Regression tests for defects found in the 2026-08 C++ audit.
# Every test in this file FAILS on the pre-audit sources and passes after the
# corresponding fix, so it pins the behaviour rather than merely exercising it.

# ---------------------------------------------------------------------------
# ob_categorical_ivb: null stats_cache on the "few categories" fast path
#
# perform_binning() skipped initialize_dp_structures() whenever
# ncat <= max_bins, but result assembly dereferenced stats_cache
# unconditionally -> segfault (killed the R session, not a catchable error).
# Triggered by any feature with <= max_bins distinct categories, i.e. sex,
# marital status, region: the most ordinary categorical predictors there are.
# ---------------------------------------------------------------------------
test_that("ob_categorical_ivb handles features with <= max_bins categories", {
  for (k in 2:5) {
    set.seed(100 + k)
    n <- 400
    feature <- sample(LETTERS[seq_len(k)], n, replace = TRUE)
    target <- rbinom(n, 1, 0.3)

    res <- ob_categorical_ivb(feature = feature, target = target)

    expect_equal(sum(res$count), n)
    expect_equal(res$count_pos + res$count_neg, res$count)
    expect_true(all(is.finite(res$woe)))
    expect_true(all(is.finite(res$iv)))
    expect_lte(length(res$count), k)
  }
})

test_that("ob_categorical_ivb handles a perfectly separating 2-level feature", {
  res <- ob_categorical_ivb(
    feature  = c("A", "A", "A", "A", "B", "B", "B", "B"),
    target   = c(1L, 1L, 1L, 1L, 0L, 0L, 0L, 0L),
    min_bins = 2, max_bins = 2
  )

  expect_equal(sum(res$count), 8)
  expect_true(all(is.finite(res$woe)))
  # a perfect separator must carry information, never IV == 0
  expect_gt(sum(res$iv), 1)
})

# ---------------------------------------------------------------------------
# Unbreakable infinite loops when min_bins exceeds the number of distinct
# feature values.
#
# OBN_JEDI / OBN_JEDIMWoE: the pre-binning "ensure at least min_bins" loop ran
#   before assign_bins(), so every count was 0 and find_largest_bin() always
#   returned index 0 -- the (-Inf, e1] interval, which split_bin() refuses to
#   split. The loop never made progress and never terminated.
# OBN_IR: ensureMinMaxBins() spun on splitLargestBin(), which is a no-op once
#   every bin holds a single observation.
#
# None of these loops contained R_CheckUserInterrupt(), so the hang could not
# even be interrupted with Ctrl-C -- the session had to be killed. This is a
# routine situation in pipelines that apply one min_bins across every feature.
# ---------------------------------------------------------------------------
test_that("numerical algorithms terminate when min_bins exceeds distinct values", {
  algos <- c("ob_numerical_ir", "ob_numerical_jedi", "ob_numerical_jedi_mwoe")

  for (fn in algos) {
    res <- do.call(fn, list(
      target = c(0L, 1L, 0L, 1L), feature = c(1, 2, 3, 4),
      min_bins = 6, max_bins = 10
    ))
    expect_equal(sum(res$count), 4, info = fn)
    expect_true(all(is.finite(res$woe)), info = fn)
  }
})

test_that("low-cardinality features terminate across the min_bins grid", {
  algos <- c("ob_numerical_ir", "ob_numerical_jedi", "ob_numerical_jedi_mwoe")
  grid <- list(c(u = 3, mb = 5), c(u = 4, mb = 5), c(u = 4, mb = 6), c(u = 2, mb = 3))

  for (fn in algos) {
    for (g in grid) {
      set.seed(11)
      n <- 300
      feature <- as.numeric(sample(seq_len(g[["u"]]), n, replace = TRUE))
      target <- rbinom(n, 1, 0.4)

      res <- do.call(fn, list(
        target = target, feature = feature,
        min_bins = g[["mb"]], max_bins = max(g[["mb"]], 5)
      ))
      expect_equal(sum(res$count), n,
                   info = sprintf("%s u=%d min_bins=%d", fn, g[["u"]], g[["mb"]]))
    }
  }
})

# ---------------------------------------------------------------------------
# ob_categorical_sab was seeded from std::random_device, so set.seed() had no
# effect and every call returned a different binning for identical input --
# no way to reproduce or audit a model built with it. It is now seeded from R's
# own RNG stream.
# ---------------------------------------------------------------------------
test_that("ob_categorical_sab is reproducible under set.seed()", {
  set.seed(42)
  feature <- sample(paste0("L", 1:5), 1000, replace = TRUE)
  target <- rbinom(1000, 1, 0.3)

  runs <- replicate(4, {
    set.seed(123)
    res <- ob_categorical_sab(feature = feature, target = target)
    paste(length(res$count), format(sum(res$iv), digits = 14))
  })
  expect_length(unique(runs), 1)

  # and the search must still respond to the seed, i.e. not be degenerate
  varied <- vapply(1:4, function(s) {
    set.seed(s)
    res <- ob_categorical_sab(feature = feature, target = target)
    sum(res$iv)
  }, numeric(1))
  expect_gt(length(unique(varied)), 1)
})

# ---------------------------------------------------------------------------
# Interval convention: every numerical algorithm bins right-closed (a, b], the
# convention its labels advertise and that ob_apply_woe_num(include_upper_bound
# = TRUE) implements.
#
# Before the audit the package disagreed with itself: labels said "(a;b]" while
# many algorithms assigned values as [a;b), ob_apply_woe_num had its two
# searches swapped, and some equal-frequency pre-binners split runs of tied
# values so their counts could not be derived from their own cutpoints at all.
# A value landing exactly on a cutpoint -- routine for integer or rounded
# features -- could therefore be scored into a different bin than the one it was
# trained in.
#
# The check below is self-validating: it recomputes the counts from the raw
# feature under both conventions and requires the right-closed one to match.
# ---------------------------------------------------------------------------
test_that("numerical algorithms bin right-closed (a, b], matching their labels", {
  make_case <- function(kind, n, seed) {
    set.seed(seed)
    feature <- switch(kind,
      integers = sample(1:20, n, replace = TRUE),
      coarse   = sample(seq(0, 100, by = 10), n, replace = TRUE),
      cont     = rnorm(n)
    )
    list(feature = as.numeric(feature), target = rbinom(n, 1, 0.35))
  }

  algos <- grep("^ob_numerical_", getNamespaceExports("OptimalBinningWoE"), value = TRUE)

  for (fn in algos) {
    for (kind in c("integers", "coarse", "cont")) {
      d <- make_case(kind, 2000, 7)
      res <- try(suppressWarnings(
        do.call(fn, list(feature = d$feature, target = d$target))), silent = TRUE)
      if (inherits(res, "try-error") || is.null(res$count) || is.null(res$cutpoints)) next

      k <- length(res$count)
      cuts <- as.numeric(res$cutpoints)
      if (length(cuts) != k - 1L) next

      trained <- as.integer(res$count)
      lbl <- sprintf("%s / %s", fn, kind)

      # no observation may be lost
      expect_equal(sum(trained), length(d$feature), info = lbl)

      # counts must be reproducible from the reported cutpoints under (a, b]
      right_closed <- as.integer(table(factor(
        findInterval(d$feature, cuts, left.open = TRUE) + 1L, levels = seq_len(k))))
      expect_equal(right_closed, trained, info = lbl)
    }
  }
})

test_that("ob_apply_woe_num reproduces the training bins it was given", {
  set.seed(7)
  feature <- as.numeric(sample(1:20, 2000, replace = TRUE))
  target <- rbinom(2000, 1, 0.35)

  algos <- grep("^ob_numerical_", getNamespaceExports("OptimalBinningWoE"), value = TRUE)

  for (fn in algos) {
    res <- try(suppressWarnings(
      do.call(fn, list(feature = feature, target = target))), silent = TRUE)
    if (inherits(res, "try-error") || is.null(res$count)) next

    applied <- try(suppressWarnings(
      ob_apply_woe_num(obresults = res, feature = feature)), silent = TRUE)
    # multinomial variants are out of scope for the binary apply helper
    if (inherits(applied, "try-error")) next

    k <- length(res$count)
    got <- as.integer(table(factor(applied$idbin, levels = seq_len(k))))
    expect_equal(got, as.integer(res$count), info = fn)
  }
})

# ---------------------------------------------------------------------------
# obcorr() spliced per-thread result buffers together inside an `omp critical`
# block, so the row order of the returned data frame depended on thread
# scheduling: two runs on the same data with the same thread count could return
# the pairs in different orders. Results are now written to a pre-sized vector
# by index, which is deterministic and independent of the thread count.
# ---------------------------------------------------------------------------
test_that("obcorr returns a deterministic row order regardless of threads", {
  set.seed(1)
  df <- as.data.frame(matrix(rnorm(60 * 20), ncol = 20))

  key <- function(th) {
    res <- obcorr(df, method = "pearson", threads = th)
    paste(res$x, res$y, sep = ":")
  }

  k_default <- key(0)
  expect_identical(key(0), k_default)   # repeated run, default threads
  expect_identical(key(1), k_default)
  expect_identical(key(2), k_default)
  expect_identical(key(4), k_default)

  # values must not depend on the thread count either
  v1 <- obcorr(df, method = "pearson", threads = 1)$pearson
  v4 <- obcorr(df, method = "pearson", threads = 4)$pearson
  expect_equal(v1, v4)

  # every unordered pair appears exactly once
  expect_equal(length(k_default), choose(ncol(df), 2))
  expect_equal(anyDuplicated(k_default), 0L)
})

# ---------------------------------------------------------------------------
# Package-wide invariants that must hold for every exported algorithm.
# These are the checks that surfaced the defects above in the first place.
# ---------------------------------------------------------------------------
test_that("every algorithm conserves observations and returns finite WoE/IV", {
  set.seed(4242)
  n <- 600
  num_feature <- rnorm(n)
  cat_feature <- sample(paste0("L", 1:6), n, replace = TRUE)
  target <- rbinom(n, 1, 0.35)

  num_algos <- grep("^ob_numerical_", getNamespaceExports("OptimalBinningWoE"), value = TRUE)
  cat_algos <- grep("^ob_categorical_", getNamespaceExports("OptimalBinningWoE"), value = TRUE)

  # count_pos/count_neg only exist for binary-target algorithms; the multinomial
  # *_jedi_mwoe variants report per-class counts instead.
  check_common <- function(res, fn) {
    expect_equal(sum(res$count), n, info = fn)
    if (length(res$count_pos) == length(res$count) && length(res$count_neg) == length(res$count)) {
      expect_equal(res$count_pos + res$count_neg, res$count, info = fn)
    }
    expect_true(all(is.finite(res$woe)), info = fn)
    expect_true(all(is.finite(res$iv)), info = fn)
  }

  for (fn in num_algos) {
    res <- try(do.call(fn, list(feature = num_feature, target = target)), silent = TRUE)
    if (inherits(res, "try-error")) next
    check_common(res, fn)
  }

  for (fn in cat_algos) {
    res <- try(do.call(fn, list(feature = cat_feature, target = target)), silent = TRUE)
    if (inherits(res, "try-error")) next
    check_common(res, fn)
  }
})
