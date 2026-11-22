# Regression tests for the forensic audit fixes of group G4 (partition
# structure of the numerical engines and honest monotonicity flags).
# See audit/AUDITORIA_FORENSE_BINNING_CPP.md.

g4_ref_iv <- function(pos, neg, alpha = 0.5) {
  K <- length(pos)
  dp <- (pos + alpha) / (sum(pos) + K * alpha)
  dn <- (neg + alpha) / (sum(neg) + K * alpha)
  sum((dp - dn) * log(dp / dn))
}

g4_bic <- function(pos, neg) {
  n <- pos + neg
  ll <- sum(ifelse(pos > 0, pos * log(pos / n), 0) + ifelse(neg > 0, neg * log(neg / n), 0))
  ll - (length(n) - 1) / 2 * log(sum(n))
}

# ---- X-09 / E-01 (+E-05): OBN_Sketch cutpoints by exact DP -------------------

test_that("ob_numerical_sketch selects cutpoints by an exact DP, not by single-cut IV (X-09, E-01)", {
  set.seed(1)
  n <- 20000
  x <- rnorm(n)
  y <- rbinom(n, 1, plogis(1.5 * x - 1))
  r <- ob_numerical_sketch(x, y, max_bins = 5, monotonic = FALSE)
  # 1.14.1 kept adjacent cuts (counts 9012 / 2956 / 8032 after the cutoff
  # merges) with reference IV 1.090; the exact DP over the same candidates
  # spreads the cuts over the range (reference IV 1.428).
  expect_gt(g4_ref_iv(r$count_pos, r$count_neg), 1.4)
  expect_length(r$count, 5L)
  expect_true(all(r$count >= ceiling(0.05 * n)))
})

test_that("ob_numerical_sketch small-sample DP respects the minimum bin support (E-05)", {
  set.seed(2)
  x <- rnorm(30)
  y <- rbinom(30, 1, plogis(0.3 * x))
  r <- ob_numerical_sketch(x, y, max_bins = 5, bin_cutoff = 0.1, monotonic = FALSE)
  expect_true(all(r$count >= 3L))
  expect_equal(sum(r$count), 30L)
})

test_that("ob_numerical_sketch chooses the number of bins by BIC; max_bins is a ceiling (Section 15.5)", {
  # Null signal: no extra bin raises the log-likelihood by log(n) / 2, so the
  # BIC keeps min_bins. Choosing K by the largest training IV (it never falls
  # as K grows) returned max_bins = 8 noise bins.
  set.seed(21)
  x <- rnorm(3000)
  y <- rbinom(3000, 1, 0.3)
  r <- ob_numerical_sketch(x, y, min_bins = 2, max_bins = 8, monotonic = FALSE)
  expect_length(r$count, 2L)
  expect_equal(sum(r$count), 3000L)
  # A strong signal on a large sample still earns max_bins (see the X-09 test).
  set.seed(1)
  x <- rnorm(20000)
  y <- rbinom(20000, 1, plogis(1.5 * x - 1))
  r5 <- ob_numerical_sketch(x, y, min_bins = 2, max_bins = 5, monotonic = FALSE)
  r3 <- ob_numerical_sketch(x, y, min_bins = 3, max_bins = 3, monotonic = FALSE)
  expect_length(r5$count, 5L)
  expect_gt(g4_bic(r5$count_pos, r5$count_neg), g4_bic(r3$count_pos, r3$count_neg))
})

test_that("ob_numerical_sketch's monotone DP respects the order of the reported WoE (X-09, X-11)", {
  # 0.2% events (corpus imbalance/rate_02pct, seed 1001). The DP's alpha = 0.5
  # order alone accepted 0/1369 | 2/609 | 0/120, whose reported WoE (prior
  # centred on the overall rate) is -8.1 / 1.0 / -5.7: flagged non-monotone
  # with a warning although monotonic = TRUE was requested.
  set.seed(1001)
  x <- rnorm(3000)
  y <- rbinom(3000, 1, plogis(-6.8 + 1.0 * x))
  idx <- sample.int(3000, 2100)
  expect_no_warning(r <- ob_numerical_sketch(x[idx], y[idx]))
  expect_true(r$monotonic)
  expect_true(all(diff(r$woe) >= -1e-9) || all(diff(r$woe) <= 1e-9))
})

# ---- X-14 / C-06: OBN_EWB restores min_bins after an outlier ---------------

test_that("ob_numerical_ewb keeps min_bins when one outlier empties the intervals (X-14, C-06)", {
  set.seed(3)
  x <- c(rnorm(999), 1000)
  y <- rbinom(1000, 1, plogis(pmin(x, 3)))
  r <- ob_numerical_ewb(x, y)
  # 1.14.1 returned 2 bins (999 / 1) with IV ~ 0.003
  expect_gte(length(r$count), 3L)
  expect_equal(sum(r$count), 1000L)
  expect_true(all(r$count > 0))
  expect_gt(g4_ref_iv(r$count_pos, r$count_neg), 0.3)
})

# ---- X-13 / C-01: OBN_UDT best-first recursive pre-binning -----------------

test_that("ob_numerical_udt finds every change point of a multi-step signal (X-13, C-01)", {
  set.seed(1)
  x <- runif(5000)
  y <- rbinom(5000, 1, ifelse(x < .3, .1, ifelse(x < .7, .5, .9)))
  r <- ob_numerical_udt(x, y)
  # 1.14.1: all 20 pre-bin cuts in [0.697, 0.701]; output 1545 / 2 / 3453 rows
  # with the change point at 0.7 lost
  expect_true(all(r$count / sum(r$count) >= 0.05))
  expect_true(any(abs(r$cutpoints - 0.3) < 0.05))
  expect_true(any(abs(r$cutpoints - 0.7) < 0.05))
})

test_that("ob_numerical_udt still reaches min_bins on perfectly separated data (X-13)", {
  set.seed(4)
  x <- rnorm(2000)
  y <- as.integer(x > 0)
  r <- ob_numerical_udt(x, y, min_bins = 3, max_bins = 5)
  expect_gte(length(r$count), 3L)
  expect_true(all(r$count >= ceiling(0.05 * 2000)))
})

test_that("ob_numerical_udt picks K by BIC and grows no sliver leaves on small samples (X-13, Section 15.5)", {
  # Null signal: the BIC keeps min_bins bins instead of max_bins.
  set.seed(23)
  x <- rnorm(3000)
  y <- rbinom(3000, 1, 0.3)
  r <- ob_numerical_udt(x, y, min_bins = 2, max_bins = 8)
  expect_length(r$count, 2L)
  # n = 42 (corpus size/n60): ceil(0.05 * 42) = 3 rows per leaf let the tree
  # cut off pure 3/0 or 0/3 slivers by chance; leaves now hold >= 7 rows
  # (CART minbucket) and K <= max_bins is chosen by BIC. 1.14.1 returned 7
  # bins of 4-10 rows for seed 1004.
  for (s in 1:5) {
    set.seed(1000 + s)
    x <- rnorm(60)
    y <- rbinom(60, 1, plogis(x))
    idx <- sample.int(60, 42)
    r <- ob_numerical_udt(x[idx], y[idx], min_bins = 3, max_bins = 8)
    expect_true(all(r$count >= 7L), info = s)
    expect_gte(length(r$count), 3L)
    expect_lte(length(r$count), 4L)
  }
})

# ---- X-11 (B-05, C-09, E-14): honest monotonicity flag ----------------------

g4_is_mono <- function(r) {
  reg <- r$bin != "NA"
  w <- r$woe[reg]
  rw <- log((r$count_pos[reg] + 0.5) / (r$count_neg[reg] + 0.5))
  (all(diff(w) >= -1e-9) && all(diff(rw) >= -1e-9)) ||
    (all(diff(w) <= 1e-9) && all(diff(rw) <= 1e-9))
}

test_that("monotonicity that min_bins blocks is flagged and warned (X-11, B-05)", {
  set.seed(11)
  x <- runif(3000)
  y <- rbinom(3000, 1, ifelse(x < .2 | x > .8, .6, .1))
  # U-shaped signal: MBLP's merges stop at min_bins = 3 with a U-shaped WoE,
  # which 1.14.1 returned silently as converged and "increasing"/"decreasing".
  expect_warning(r <- ob_numerical_mblp(x, y, min_bins = 3, max_bins = 5),
                 "monotonic WoE was requested")
  expect_false(r$monotonic)
  expect_false(g4_is_mono(r))
})

test_that("the monotonic flag of every G4 engine agrees with an independent check (X-11)", {
  set.seed(12)
  x <- rnorm(2000)
  y <- rbinom(2000, 1, plogis(-1 + x))
  xu <- runif(2000)
  yu <- rbinom(2000, 1, ifelse(xu < .2 | xu > .8, .6, .1))
  fits <- list(
    mblp = ob_numerical_mblp, dp = ob_numerical_dp, lpdb = ob_numerical_lpdb,
    udt = ob_numerical_udt, kmb = ob_numerical_kmb, ewb = ob_numerical_ewb,
    ldb = ob_numerical_ldb, sketch = ob_numerical_sketch, ir = ob_numerical_ir)
  for (nm in names(fits)) for (d in list(list(x, y), list(xu, yu))) {
    r <- suppressWarnings(fits[[nm]](d[[1]], d[[2]]))
    expect_type(r$monotonic, "logical")
    expect_identical(r$monotonic, g4_is_mono(r), label = nm)
  }
})

# ---- E-06 / C-14 / B-09: IV with the same smoothed shares as the WoE --------

test_that("dp, kmb, ldb and lpdb never report a negative bin IV (E-06, C-14, B-09)", {
  # 1.14.1 computed IV as (raw share of events - raw share of non-events)
  # times the smoothed WoE: with a handful of rows a bin could get IV < 0.
  fns <- list(dp = ob_numerical_dp, kmb = ob_numerical_kmb,
              ldb = ob_numerical_ldb, lpdb = ob_numerical_lpdb)
  neg <- character()
  for (s in 1:300) {
    set.seed(s)
    n <- sample(c(10, 30), 1)
    x <- switch(sample(3, 1), round(rnorm(n)), rnorm(n), c(rnorm(n - 1), 1e15))
    y <- rbinom(n, 1, 0.3)
    if (length(unique(y)) < 2) y[1:2] <- 0:1
    for (a in names(fns)) {
      r <- tryCatch(suppressWarnings(fns[[a]](x, y)), error = function(e) NULL)
      if (!is.null(r) && any(r$iv < -1e-12)) neg <- c(neg, paste(a, s))
      if (!is.null(r)) {
        expect_equal(r$total_iv, sum(r$iv), tolerance = 1e-10)
      }
    }
  }
  expect_identical(neg, character())
})

# ---- C-08: KMB / EWB monotone direction ------------------------------------

test_that("kmb and ewb pick the monotone direction by result, not by two bins (C-08)", {
  # Weak increasing trend: 1.14.1 took the direction from the first two bins
  # and returned a non-monotone WoE in 1 (kmb) and 8 (ewb) of these 40 runs.
  nonmono <- c(kmb = 0L, ewb = 0L)
  for (s in 1:40) {
    set.seed(s)
    x <- runif(1500)
    y <- rbinom(1500, 1, plogis(-1 + 0.8 * x))
    r1 <- suppressWarnings(ob_numerical_kmb(x, y, min_bins = 3, max_bins = 6))
    r2 <- suppressWarnings(ob_numerical_ewb(x, y, min_bins = 3, max_bins = 6))
    nonmono["kmb"] <- nonmono["kmb"] + !g4_is_mono(r1)
    nonmono["ewb"] <- nonmono["ewb"] + !g4_is_mono(r2)
    expect_identical(r1$monotonic, g4_is_mono(r1))
    expect_identical(r2$monotonic, g4_is_mono(r2))
  }
  expect_equal(unname(nonmono), c(0L, 0L))
})
