# Regression tests for the defects found in the 2026-08 audit of 1.13.2.
#
# Every test here FAILS on the unfixed sources and passes after the
# corresponding fix, so each one pins behaviour rather than merely exercising
# it. The numbering matches the audit report.

# ---------------------------------------------------------------------------
# FIX 1: ob_categorical_sketch passed (total_neg, total_pos) to helpers that
# expect (total_pos, total_neg), so every WoE was computed against the wrong
# marginal. Reproduced as max|woe - reference| = 2.7721 and total_iv = 10.4021
# where the correct values are ~0.0001 and ~0.0043.
# ---------------------------------------------------------------------------
test_that("sketch categorical WoE matches the reference definition", {
  set.seed(9)
  n <- 20000
  y <- rbinom(n, 1, 0.2)
  cc <- sample(letters[1:8], n, TRUE)

  r <- suppressWarnings(suppressMessages(
    obwoe(data.frame(target = y, cc = cc, stringsAsFactors = FALSE),
      target = "target", feature = "cc", algorithm = "sketch"
    )
  ))$results$cc

  tp <- sum(r$count_pos)
  tn <- sum(r$count_neg)
  ref <- log((r$count_pos / tp) / (r$count_neg / tn))

  # The bug produced 2.7721 here; smoothing alone accounts for < 0.01.
  expect_lt(max(abs(r$woe - ref)), 0.05)
  # The bug produced 10.4021; the true IV of a random category is ~0.004.
  expect_lt(sum(r$iv), 0.05)
})

# ---------------------------------------------------------------------------
# FIX 2: ob_categorical_sab only inserted a positive_counts key when it saw a
# positive target, but read the map with .at() at seven sites, so any category
# with zero events aborted the fit with "unordered_map::at".
# ---------------------------------------------------------------------------
test_that("sab categorical handles a category with zero events", {
  set.seed(1)
  n <- 3000
  f <- rep(c("a", "b", "c"), length.out = n)
  y <- integer(n)
  y[f == "a"] <- rbinom(sum(f == "a"), 1, .30)
  y[f == "b"] <- rbinom(sum(f == "b"), 1, .10)
  y[f == "c"] <- 0L

  r <- suppressWarnings(suppressMessages(
    obwoe(data.frame(target = y, f = f, stringsAsFactors = FALSE),
      target = "target", feature = "f", algorithm = "sab"
    )
  ))

  expect_null(r$results$f$error)
  expect_false(is.na(r$summary$total_iv))
})

# ---------------------------------------------------------------------------
# FIX 3: both DMIV engines returned only `divergence`/`total_divergence`, so
# obwoe()'s summary left total_iv as NA for every dmiv feature.
# ---------------------------------------------------------------------------
test_that("dmiv reports iv and total_iv, consistent with an independent algorithm", {
  set.seed(31)
  n <- 10000
  y <- rbinom(n, 1, 0.2)
  d <- data.frame(target = y, a = rnorm(n) + y, b = rnorm(n) + 0.5 * y)

  f <- suppressWarnings(suppressMessages(
    obwoe(d, target = "target", feature = c("a", "b"), algorithm = "dmiv")
  ))
  expect_false(any(is.na(f$summary$total_iv)))
  expect_true(all(is.finite(f$results$a$iv)))
  expect_equal(length(f$results$a$iv), length(f$results$a$bin))
  expect_equal(sum(f$results$a$iv), f$results$a$total_iv, tolerance = 1e-8)

  # Cross-check against an independent algorithm on the same data: standard IV
  # is a property of the partition, so two sensible partitions must agree to
  # within an order of magnitude.
  g <- suppressWarnings(suppressMessages(
    obwoe(d, target = "target", feature = c("a", "b"), algorithm = "jedi")
  ))
  expect_equal(f$summary$total_iv, g$summary$total_iv, tolerance = 0.5)

  # IV must NOT be derived from bin.woe: the default bin_method "woe1" is a
  # log-odds offset from standard WoE by log(TP/TN), so a woe-derived IV would
  # be badly wrong. Confirm the reported IV matches the counts instead.
  res <- f$results$a
  k <- length(res$count_pos)
  dp <- (res$count_pos + 0.5) / (sum(res$count_pos) + 0.5 * k)
  dn <- (res$count_neg + 0.5) / (sum(res$count_neg) + 0.5 * k)
  expect_equal(res$iv, (dp - dn) * log(dp / dn), tolerance = 1e-8)
})

test_that("categorical dmiv also reports total_iv", {
  set.seed(7)
  n <- 10000
  cc <- sample(letters[1:6], n, TRUE)
  y <- rbinom(n, 1, plogis(-1 + 0.6 * (cc %in% c("a", "b"))))

  f <- suppressWarnings(suppressMessages(
    obwoe(data.frame(target = y, cc = cc, stringsAsFactors = FALSE),
      target = "target", feature = "cc", algorithm = "dmiv"
    )
  ))
  expect_false(is.na(f$summary$total_iv))
  expect_gt(f$summary$total_iv, 0)
})

# ---------------------------------------------------------------------------
# FIX 4 and FIX 5: `converged` was effectively inverted in the numerical dmiv,
# bb and ir engines -- ordinary well-binned features reported FALSE while only
# degenerate ones reported TRUE.
# ---------------------------------------------------------------------------
test_that("numerical engines report converged for successful binnings", {
  set.seed(21)
  n <- 20000
  y <- rbinom(n, 1, 0.2)
  d <- data.frame(
    target = y,
    bin01 = rbinom(n, 1, plogis(-.5 + .8 * y)),
    tri = sample(0:2, n, TRUE),
    cont = rnorm(n) + 1.1 * y,
    noisy = rnorm(n) + 0.05 * y
  )
  feats <- c("bin01", "tri", "cont", "noisy")

  for (a in c("dmiv", "bb", "ir", "jedi")) {
    s <- suppressWarnings(suppressMessages(
      obwoe(d, target = "target", feature = feats, algorithm = a)
    ))$summary
    expect_true(all(s$converged),
      info = sprintf(
        "%s reported converged = FALSE for: %s", a,
        paste(s$feature[!s$converged], collapse = ", ")
      )
    )
  }
})

# ---------------------------------------------------------------------------
# FIX 6: the same defect in six categorical engines, which broke out of their
# merge loop (or took a fast path) without setting the flag.
# ---------------------------------------------------------------------------
test_that("categorical engines report converged for successful binnings", {
  set.seed(5)
  n <- 20000
  y <- rbinom(n, 1, 0.2)
  d <- data.frame(
    target = y,
    cc = sample(letters[1:12], n, TRUE),
    cc2 = sample(letters[1:3], n, TRUE),
    stringsAsFactors = FALSE
  )

  for (a in c("dp", "gmb", "mba", "sketch", "cm", "dmiv")) {
    s <- suppressWarnings(suppressMessages(
      obwoe(d, target = "target", feature = c("cc", "cc2"), algorithm = a)
    ))$summary
    expect_true(all(s$converged),
      info = sprintf(
        "%s reported converged = FALSE for: %s", a,
        paste(s$feature[!s$converged], collapse = ", ")
      )
    )
  }
})

# ---------------------------------------------------------------------------
# FIX 7: obwoe_apply() silently applied only class 1's WoE for a multiclass
# model, discarding the other classes -- no error, no warning.
# ---------------------------------------------------------------------------
test_that("obwoe_apply refuses a multiclass model instead of returning class 1", {
  set.seed(31)
  n <- 10000
  y3 <- sample(0:2, n, TRUE)
  d3 <- data.frame(target = y3, a = rnorm(n) + y3 * 0.4)

  m <- suppressWarnings(suppressMessages(
    obwoe(d3, target = "target", feature = "a", algorithm = "auto")
  ))
  expect_identical(m$target_type, "multinomial")
  expect_true(is.matrix(m$results$a$woe))

  expect_error(obwoe_apply(d3, m), "[Mm]ulticlass")
})

test_that("obwoe_apply still works for binary models", {
  set.seed(3)
  n <- 5000
  y <- rbinom(n, 1, 0.3)
  d <- data.frame(target = y, a = rnorm(n) + y)
  m <- suppressWarnings(suppressMessages(
    obwoe(d, target = "target", feature = "a", algorithm = "jedi")
  ))
  out <- obwoe_apply(d, m)
  expect_true(all(c("a_bin", "a_woe") %in% names(out)))
  expect_true(all(is.finite(out$a_woe)))
})

# ---------------------------------------------------------------------------
# FIX 8: an all-NA total_iv column broke three consumers, each by letting
# na.rm = TRUE turn "never measured" into 0.
# ---------------------------------------------------------------------------
test_that("an all-NA total_iv is reported as NA, not as an IV of zero", {
  set.seed(31)
  n <- 5000
  y <- rbinom(n, 1, 0.2)
  d <- data.frame(target = y, a = rnorm(n) + y, b = rnorm(n) + 0.5 * y)
  f <- suppressWarnings(suppressMessages(
    obwoe(d, target = "target", feature = c("a", "b"), algorithm = "jedi")
  ))
  f$summary$total_iv <- NA_real_

  s <- summary(f)
  expect_true(is.na(s$aggregate$total_iv_sum))
  expect_true(is.na(s$aggregate$mean_iv))
  expect_true(all(is.na(s$aggregate$iv_range)))

  txt <- paste(utils::capture.output(print(s)), collapse = "\n")
  expect_false(grepl("Total IV: 0.0000", txt, fixed = TRUE))
  expect_false(grepl("[Inf, -Inf]", txt, fixed = TRUE))
  expect_true(grepl("Total IV: NA", txt, fixed = TRUE))
})

test_that("obwoe_gains and plot(type = 'iv') fail informatively on an all-NA IV", {
  set.seed(31)
  n <- 5000
  y <- rbinom(n, 1, 0.2)
  d <- data.frame(target = y, a = rnorm(n) + y, b = rnorm(n) + 0.5 * y)
  f <- suppressWarnings(suppressMessages(
    obwoe(d, target = "target", feature = c("a", "b"), algorithm = "jedi")
  ))
  f$summary$total_iv <- NA_real_

  # Was: "attempt to select less than one element in get1index"
  expect_error(obwoe_gains(f), "finite Information Value")

  # Was: "need finite 'xlim' values"
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  expect_message(plot(f, type = "iv"), "No finite Information Value")
})

test_that("dmiv models are usable end to end now that they report IV", {
  set.seed(31)
  n <- 5000
  y <- rbinom(n, 1, 0.2)
  d <- data.frame(target = y, a = rnorm(n) + y, b = rnorm(n) + 0.5 * y)
  f <- suppressWarnings(suppressMessages(
    obwoe(d, target = "target", feature = c("a", "b"), algorithm = "dmiv")
  ))

  expect_no_error(obwoe_gains(f))
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  expect_no_error(plot(f, type = "iv"))
})

# ---------------------------------------------------------------------------
# FIX 9: obwoe_scorecard() checked the development frame's target for NA but
# not the validation samples', which failed deep inside .ob_score_metrics()
# with "missing value where TRUE/FALSE needed".
# ---------------------------------------------------------------------------
test_that("obwoe_scorecard names the sample whose target contains NA", {
  set.seed(41)
  mk <- function(nn) {
    yy <- rbinom(nn, 1, 0.25)
    data.frame(
      target = yy, v1 = rnorm(nn) + yy,
      v2 = rnorm(nn) + 0.7 * yy, v3 = rnorm(nn) + 0.4 * yy
    )
  }
  dev <- mk(4000)
  val <- mk(1500)
  val$target[sample(nrow(val), 25)] <- NA_integer_

  expect_error(
    suppressWarnings(suppressMessages(
      obwoe_scorecard(dev, target = "target", validation = list(oot = val), split = NULL)
    )),
    "oot"
  )
})

# ---------------------------------------------------------------------------
# FIX 10: .ob_cutoff_table() used sum() without na.rm inside if(), so an NA in
# score or y raised "missing value where TRUE/FALSE needed".
# ---------------------------------------------------------------------------
test_that(".ob_cutoff_table rejects NA score or target with a clear message", {
  ct <- .ob_cutoff_table
  set.seed(2)
  n <- 500
  sc <- rnorm(n, 600, 50)
  y <- rbinom(n, 1, 0.2)

  expect_equal(nrow(ct(sc, y)), 20L)

  y2 <- y
  y2[c(3, 9)] <- NA_integer_
  expect_error(ct(sc, y2), "target value\\(s\\) are missing")

  sc2 <- sc
  sc2[c(1, 5, 7)] <- NA_real_
  expect_error(ct(sc2, y), "score\\(s\\) are missing")
})

# ---------------------------------------------------------------------------
# FIX 11: mdlp (18 bins), gmb (11) and fetb (10) silently ignored max_bins = 5.
# ---------------------------------------------------------------------------
test_that("every algorithm honours max_bins", {
  set.seed(21)
  n <- 20000
  y <- rbinom(n, 1, 0.2)
  d <- data.frame(
    target = y, cont = rnorm(n) + 1.1 * y,
    cc = sample(letters[1:12], n, TRUE), stringsAsFactors = FALSE
  )

  alg <- obwoe_algorithms()
  violations <- character()

  for (i in seq_len(nrow(alg))) {
    a <- alg$algorithm[i]
    for (tp in c("numerical", "categorical")) {
      if (!alg[[tp]][i]) next
      f <- if (tp == "numerical") "cont" else "cc"
      r <- try(suppressWarnings(suppressMessages(
        obwoe(d,
          target = "target", feature = f, algorithm = a,
          min_bins = 2, max_bins = 5
        )
      )), silent = TRUE)
      if (inherits(r, "try-error")) next
      nb <- r$summary$n_bins
      if (!is.na(nb) && nb > 5) {
        violations <- c(violations, sprintf("%s/%s=%d", a, tp, nb))
      }
    }
  }

  expect_equal(violations, character(0),
    info = paste("max_bins violated by:", paste(violations, collapse = " "))
  )
})

# ---------------------------------------------------------------------------
# FIX 12
# ---------------------------------------------------------------------------
test_that("sketch numerical returns a bin label for a constant feature", {
  set.seed(1)
  n <- 2000
  y <- rbinom(n, 1, .3)
  r <- suppressWarnings(suppressMessages(
    obwoe(data.frame(target = y, k = rep(1, n)),
      target = "target", feature = "k", algorithm = "sketch"
    )
  ))
  expect_equal(r$summary$n_bins, 1L)
  expect_equal(length(r$results$k$bin), 1L)
})

test_that("udt keeps WoE finite with laplace_smoothing = 0", {
  set.seed(4)
  x <- c(rnorm(1000, -3), rnorm(1000, 3))
  y <- c(rep(0L, 1000), rbinom(1000, 1, 0.5))

  r <- suppressWarnings(suppressMessages(
    ob_numerical_udt(target = y, feature = x, laplace_smoothing = 0)
  ))
  expect_true(all(is.finite(r$woe)))
  expect_true(all(is.finite(r$iv)))
})

test_that("cm categorical exposes total_iv at top level", {
  set.seed(6)
  n <- 2000
  y <- rbinom(n, 1, .3)
  cc <- sample(letters[1:8], n, TRUE)
  r <- suppressWarnings(suppressMessages(
    obwoe(data.frame(target = y, cc = cc, stringsAsFactors = FALSE),
      target = "target", feature = "cc", algorithm = "cm"
    )
  ))$results$cc

  expect_true(is.numeric(r$total_iv))
  expect_equal(length(r$total_iv), 1L)
  # metadata is kept for backward compatibility
  expect_equal(r$total_iv, r$metadata$total_iv)
})

# ---------------------------------------------------------------------------
# FIX 13: numerical vs categorical parity.
#
# FIX 1 and the `converged` family were both cases of a defect fixed in one
# engine and never replicated to its twin. This test is what stops that
# recurring: for every algorithm with both variants, the reported WoE must
# match the reference log((pos_i/TP)/(neg_i/TN)) within a smoothing tolerance,
# on BOTH sides.
#
# dmiv is excluded deliberately: its default bin_method is "woe1", Zeng's
# log-odds ln((pos+0.5)/(neg+0.5)), which is standard WoE offset by the
# constant ln(TP/TN). It is checked against its own definition instead.
# jedi_mwoe is excluded because it returns a bins x classes matrix, not a
# vector, so there is no single WoE column to compare.
# ---------------------------------------------------------------------------
test_that("numerical and categorical variants agree on the WoE definition", {
  set.seed(101)
  n <- 20000
  y <- rbinom(n, 1, 0.25)
  xc <- rnorm(n) + 0.9 * y
  cc <- sample(letters[1:8], n, TRUE)
  yc <- rbinom(n, 1, plogis(-1 + 0.9 * (cc %in% c("a", "b", "c"))))

  max_dev <- function(a, tp) {
    d <- if (tp == "numerical") {
      data.frame(target = y, v = xc)
    } else {
      data.frame(target = yc, v = cc, stringsAsFactors = FALSE)
    }
    res <- suppressWarnings(suppressMessages(
      obwoe(d, target = "target", feature = "v", algorithm = a)
    ))$results$v

    k <- length(res$count_pos)
    # Reference with the 0.5 pseudo-counts the engines use, so the tolerance
    # only has to absorb differences between smoothing schemes, not smoothing
    # itself.
    ref <- log(
      ((res$count_pos + 0.5) / (sum(res$count_pos) + 0.5 * k)) /
        ((res$count_neg + 0.5) / (sum(res$count_neg) + 0.5 * k))
    )
    max(abs(res$woe - ref))
  }

  both <- obwoe_algorithms()
  both <- both$algorithm[both$numerical & both$categorical]
  both <- setdiff(both, c("dmiv", "jedi_mwoe"))

  for (a in both) {
    for (tp in c("numerical", "categorical")) {
      # The FIX 1 bug showed up as 2.7721 here; every correct engine is
      # below 0.02, so 0.05 catches the defect with a wide margin while
      # tolerating differing smoothing schemes.
      expect_lt(max_dev(a, tp), 0.05)
    }
  }
})

test_that("dmiv matches its own woe1 definition on both variants", {
  set.seed(101)
  n <- 20000
  y <- rbinom(n, 1, 0.25)
  xc <- rnorm(n) + 0.9 * y
  cc <- sample(letters[1:8], n, TRUE)
  yc <- rbinom(n, 1, plogis(-1 + 0.9 * (cc %in% c("a", "b", "c"))))

  for (tp in c("numerical", "categorical")) {
    d <- if (tp == "numerical") {
      data.frame(target = y, v = xc)
    } else {
      data.frame(target = yc, v = cc, stringsAsFactors = FALSE)
    }
    res <- suppressWarnings(suppressMessages(
      obwoe(d, target = "target", feature = "v", algorithm = "dmiv")
    ))$results$v

    # bin_method = "woe1": ln((pos + 0.5) / (neg + 0.5))
    expect_equal(
      res$woe,
      log((res$count_pos + 0.5) / (res$count_neg + 0.5)),
      tolerance = 1e-6
    )
  }
})

test_that("lpdb, ldb and udt scale linearly rather than quadratically", {
  # OBN_LDB and OBN_LPDB each estimated the density with a naive O(n^2) double
  # loop, and OBN_UDT rescanned every observation once per candidate split.
  # Measured scaling was n^2.00, n^2.00 and n^2.30, which put a single
  # 10^6-row variable at roughly 72 minutes and made all three unusable.
  #
  # A wall-clock budget is the only way to pin this: the defect is asymptotic,
  # not a wrong value. At n = 50,000 the old code took 7-11 s per algorithm and
  # the fixed code takes 0.01-0.02 s, so a two-second budget sits two orders of
  # magnitude above the fix and five times below the defect. Skipped on CRAN,
  # whose machines are deliberately slow and heavily shared.
  skip_on_cran()

  set.seed(11)
  n <- 50000L
  y <- rbinom(n, 1, 0.2)
  d <- data.frame(target = y, x = rnorm(n) + y)

  for (a in c("lpdb", "ldb", "udt")) {
    elapsed <- system.time(
      res <- suppressWarnings(suppressMessages(
        obwoe(d, target = "target", feature = "x", algorithm = a,
              min_bins = 2, max_bins = 5)
      ))
    )[["elapsed"]]

    expect_lt(elapsed, 2)
    expect_false(res$summary$error)
    expect_true(is.finite(res$summary$total_iv))
    expect_lte(res$summary$n_bins, 5L)
  }
})

test_that("the shared KDE keeps ldb and udt bit-identical to the double loop", {
  # The udt fix is an exact rewrite -- information gain depends only on integer
  # counts, so sweeping prefix totals reproduces the per-candidate rescan
  # exactly. ldb's density feeds a local-minimum search that the grid estimator
  # resolves identically at this size. Both were verified against a build of
  # the previous revision; these values pin them so a future change to the
  # shared estimator cannot move them silently.
  #
  # Unlike the scaling guard above this asserts values, not wall-clock time,
  # so it runs everywhere, CRAN included.

  set.seed(5)
  n <- 25000L
  y <- rbinom(n, 1, 0.2)
  d <- data.frame(target = y, x = rnorm(n) + y)

  udt <- suppressWarnings(suppressMessages(
    obwoe(d, target = "target", feature = "x", algorithm = "udt")))$results$x
  expect_equal(length(udt$bin), 3L)
  expect_equal(sum(udt$iv), 0.6877321502, tolerance = 1e-8)

  ldb <- suppressWarnings(suppressMessages(
    obwoe(d, target = "target", feature = "x", algorithm = "ldb")))$results$x
  expect_equal(length(ldb$bin), 2L)
  expect_equal(sum(ldb$iv), 0.6267022265, tolerance = 1e-8)
})
