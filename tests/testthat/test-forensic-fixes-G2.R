# Permanent regression tests for the G2 fixes of the forensic audit
# (audit/AUDITORIA_FORENSE_BINNING_CPP.md): merge criterion of the greedy
# numerical binners (X-08 / T1) and multiclass monotonicity (X-17 / D-03).
# Each test encodes the minimal failing scenario with its expected result.

# Reference IV of a partition: normalized Jeffreys smoothing (alpha = 0.5,
# K = number of bins), the estimator of audit Section 7.1 / 9.5.
g2_ref_iv <- function(pos, neg, a = 0.5) {
  K <- length(pos)
  dp <- (pos + a) / (sum(pos) + K * a)
  dn <- (neg + a) / (sum(neg) + K * a)
  sum((dp - dn) * log(dp / dn))
}

# Best contiguous partition of m ordered groups into k bins (exhaustive).
g2_brute_iv <- function(pos, neg, k) {
  m <- length(pos)
  best <- -Inf
  for (cs in utils::combn(m - 1, k - 1, simplify = FALSE)) {
    g <- findInterval(seq_len(m), cs + 1) + 1
    best <- max(best, g2_ref_iv(tapply(pos, g, sum), tapply(neg, g, sum)))
  }
  best
}

# Six groups of 200 observations (x = 1..6) with the given event counts.
g2_groups <- function(ev, n = 200) {
  list(x = rep(seq_along(ev), each = n),
       y = unlist(lapply(ev, function(e) c(rep(1, e), rep(0, n - e)))))
}

# ---- D-03 / D-02: OBN_JEDIMWoE ---------------------------------------------------
test_that("D-03: ob_numerical_jedi_mwoe does not force K = 3 WoE curves to be monotone", {
  # Three classes centred at 0, 1.5 and 3 (deterministic normal quantiles).
  # Forcing the unimodal one-vs-rest WoE of the middle class to be monotone,
  # together with the other two, merged bins until one held ~70% of the data.
  z <- qnorm(ppoints(2000))
  x <- c(z, z + 1.5, z + 3)
  y <- rep(0:2, each = 2000)
  r <- ob_numerical_jedi_mwoe(x, y, min_bins = 3, max_bins = 8, max_n_prebins = 30)
  expect_equal(length(r$count), 8L)
  expect_true(max(r$count) / sum(r$count) <= 0.4)
  # the middle class is genuinely unimodal: its WoE rises and then falls
  w1 <- r$woe[, 2]
  expect_true(is.unsorted(w1) && is.unsorted(rev(w1)))
})

test_that("D-03: ob_numerical_jedi_mwoe keeps monotone WoE for a binary target", {
  z <- qnorm(ppoints(3000))
  x <- c(z, z + 1)
  y <- rep(0:1, each = 3000)
  r <- ob_numerical_jedi_mwoe(x, y, min_bins = 3, max_bins = 5)
  w <- r$woe[, 2]
  expect_true(!is.unsorted(w) || !is.unsorted(rev(w)))
})

# ---- C-05: OBN_BB merge criterion ----------------------------------------------
test_that("C-05: ob_numerical_bb merges the adjacent pair with the smallest IV loss", {
  # Events per group 20, 100, 40, 180, 100, 30 (of 200), reduced to 3 bins.
  # The baseline folded the smallest-IV bin into its LEFT neighbour and
  # returned {1,2,3} {4,5} {6} (ref IV 0.918); the optimum is {1,2,3} {4} {5,6}.
  ev <- c(20, 100, 40, 180, 100, 30)
  d <- g2_groups(ev)
  r <- ob_numerical_bb(d$x, d$y, min_bins = 2, max_bins = 3, max_n_prebins = 6,
                       is_monotonic = FALSE, bin_cutoff = 0.01)
  expect_equal(r$count_pos, c(160, 180, 130))
  expect_equal(g2_ref_iv(r$count_pos, r$count_neg), g2_brute_iv(ev, 200 - ev, 3),
               tolerance = 1e-10)
})

test_that("C-05: ob_numerical_bb keeps a near-separable split at the class boundary", {
  # y = 1[x > 1000] with three flipped labels. The baseline merged the first
  # positive pre-bin (100 events) into the negative block: bin 1 = 101/999.
  x <- 1:2000
  y <- as.integer(x > 1000)
  y[c(150, 1700, 1900)] <- 1L - y[c(150, 1700, 1900)]
  r <- ob_numerical_bb(x, y)
  expect_true(r$cutpoints[1] >= 1000 && r$cutpoints[1] < 1001)
  expect_equal(r$count_pos[1], 1)
  expect_equal(r$count_neg[1], 999)
  expect_false(is.unsorted(r$woe))
})

# ---- D-02: OBN_JEDI merge criterion --------------------------------------------
test_that("D-02: ob_numerical_jedi does not swallow the informative range in one bin", {
  # x = 1..1000, event rate k/10 in the k-th block of 100 (no events below
  # 100). The smallest-SUM rule never merged the pure leading bins (huge
  # EPS-floored IV each) and pooled x in (100, 950] into one bin of 850.
  x <- 1:1000
  y <- as.integer((x %% 10) < (x %/% 100))
  r <- ob_numerical_jedi(x, y, max_bins = 4)
  # D-02 readmitted with the empirical-Bayes merge decision (audit Section
  # 15.5, manifest 50): the raw-IV greedy of the held patch isolated the 50
  # leading rows without events (50/350/400/200, ref IV 1.606); with shrinkage
  # a 50-row pure bin is worth little and the blocks are cut evenly
  # (300/200/200/300, ref IV 1.548). Either way far above the baseline
  # (50/50/850/50, ref IV 0.903).
  expect_equal(r$count, c(300, 200, 200, 300))
  expect_equal(r$cutpoints, c(300, 500, 700))
  expect_true(max(r$count) <= 400)
  expect_gt(g2_ref_iv(r$count_pos, r$count_neg), 1.5)
})

# ---- D-12: OBN_MOB rare-bin neighbour -------------------------------------------
test_that("D-12: ob_numerical_mob merges a rare bin with the neighbour that loses less IV", {
  # Differential-corpus scenario missing/na_30pct, seed 1 (70% training split),
  # min_bins = 2, max_bins = 3. The rare-bin direction used IV_a + IV_b, which
  # never subtracts the merged bin; the baseline ended at ref IV 0.611, the
  # actual loss of reported IV gives 0.708 with the same number of bins.
  set.seed(1001)
  x <- rnorm(3000)
  y <- rbinom(3000, 1, plogis(x))
  x[runif(3000) < .3] <- NA
  idx <- sample.int(3000, 2100)
  r <- ob_numerical_mob(x[idx], y[idx], min_bins = 2, max_bins = 3)
  expect_equal(length(r$count), 3L)
  # Readmitted with the empirical-Bayes direction (audit Section 15.5,
  # manifest 51): the raw-IV direction of the held patch reached 0.708 by
  # leaving a small top bin; with shrinkage the bins are 432 / 576 / 447
  # (ref IV 0.679), still above the baseline's 432 / 936 / 87 (0.611).
  expect_equal(r$count, c(432, 576, 447))
  expect_gt(g2_ref_iv(r$count_pos, r$count_neg), 0.65)
})

# ---- B-03: OBN_OSLP merge criterion --------------------------------------------
test_that("B-03: ob_numerical_oslp merges two pure neighbours before informative bins", {
  # Events 0, 0, 30, 60, 100, 200 (of 200). The two groups without events have
  # the largest IV each, so the smallest-SUM rule kept them apart and merged
  # the informative middle groups instead (ref IV 4.85); merging the two pure
  # groups loses nothing and the result is the 3-bin optimum (6.37).
  ev <- c(0, 0, 30, 60, 100, 200)
  d <- g2_groups(ev)
  r <- ob_numerical_oslp(d$x, d$y, min_bins = 2, max_bins = 3, max_n_prebins = 6,
                         bin_cutoff = 0.01)
  expect_equal(r$count_pos, c(0, 190, 200))
  expect_equal(r$count_neg, c(400, 410, 0))
  expect_equal(g2_ref_iv(r$count_pos, r$count_neg), g2_brute_iv(ev, 200 - ev, 3),
               tolerance = 1e-10)
})

# ---- B-04: OBN_MRBLP merge criterion -------------------------------------------
test_that("B-04: ob_numerical_mrblp no longer merges by the smallest |IV_i - IV_i+1|", {
  # Events 10, 30, 90, 110, 170, 190 (of 200): |dIV| merged the bins around the
  # average rate into one block {2..5} (ref IV 1.73). The greedy step on the
  # reported IV reaches the 3-bin optimum {1,2} {3,4} {5,6} (2.33).
  ev <- c(10, 30, 90, 110, 170, 190)
  d <- g2_groups(ev)
  r <- ob_numerical_mrblp(d$x, d$y, min_bins = 2, max_bins = 3, max_n_prebins = 6,
                          bin_cutoff = 0.01)
  expect_equal(r$count_pos, c(40, 200, 360))
  expect_equal(g2_ref_iv(r$count_pos, r$count_neg), g2_brute_iv(ev, 200 - ev, 3),
               tolerance = 1e-10)
})

# ---- B-03 / B-04 readmission: evidence for pure bins (manifests 54, 55) ------------
# A pure bin of n rows may stay alone only if the exact binomial test of zero
# occurrences under the global rate r0 rejects chance: (1 - r0)^n <= 0.01
# (r0^n without non-events). Adjacent same-class pure bins are always merged.
g2_pure_p <- function(r) {
  tp <- sum(r$count_pos); tn <- sum(r$count_neg); r0 <- tp / (tp + tn)
  ifelse(r$count_pos == 0, (1 - r0)^r$count,
         ifelse(r$count_neg == 0, r0^r$count, 0))
}

test_that("B-03/B-04: a block without events by chance is not kept as a bin", {
  # 2000 rows, 6 events (r0 = 0.003). The first 100-row pre-bin has no event,
  # which happens by chance with probability 0.997^100 = 0.74; its smoothed IV
  # is large, so the IV-loss criterion kept it apart (0/100 1/299 5/1595, as
  # did version 1.14.1).
  x <- 1:2000
  y <- integer(2000)
  y[c(130, 520, 1210, 1650, 1880, 1990)] <- 1L
  for (fn in list(ob_numerical_oslp, ob_numerical_mrblp)) {
    r <- fn(x, y)
    expect_equal(r$count_pos, c(2, 2, 1, 1))
    expect_equal(r$count_neg, c(1198, 598, 99, 99))
    expect_true(all(g2_pure_p(r) <= 0.01))
    r <- fn(x, y, min_bins = 2, max_bins = 3)
    expect_equal(r$count_pos, c(4, 1, 1))
    expect_true(all(r$count_pos > 0))
  }
})

test_that("B-03/B-04: runs of pure bins of one class are merged under a wide max_bins", {
  # Perfect separator with one mixed pre-bin. The IV-loss criterion only acts
  # above max_bins = 8 and left six pure bins without events side by side
  # (0/525 0/105 0/105 0/105 0/105 0/105 21/84 945/0); merging them keeps the
  # binomial likelihood (G^2 = 0) and the separation.
  z <- qnorm(ppoints(2100))
  y <- as.integer(z > 0.1)
  for (fn in list(ob_numerical_oslp, ob_numerical_mrblp)) {
    r <- fn(z, y, min_bins = 3, max_bins = 8)
    expect_equal(r$count_pos, c(0, 21, 945))
    expect_equal(r$count_neg, c(1050, 84, 0))
  }
  # A significant pure bin survives (perfect separation is not merged away).
  for (fn in list(ob_numerical_oslp, ob_numerical_mrblp)) {
    r <- fn(z, as.integer(z > 0), min_bins = 2, max_bins = 8)
    expect_equal(r$count_pos, c(0, 1050))
    expect_equal(r$count_neg, c(1050, 0))
  }
})

# ---- D-01: OBN_DMIV merge criterion --------------------------------------------
test_that("D-01: ob_numerical_dmiv merges by the loss of the selected divergence", {
  # Audit scenario: N = 20000, x ~ U(0, 10), logit(p) = -3 + 0.5 x, 3-5 bins.
  # The smallest-|contribution| bin folded into its left neighbour gave about
  # 46% of the MOB IV for he/kl/tr/klj/sc/js/ln and 55% for l1/l2.
  set.seed(2013)
  x <- runif(20000, 0, 10)
  y <- rbinom(20000, 1, plogis(-3 + 0.5 * x))
  mob_iv <- ob_numerical_mob(x, y, min_bins = 3, max_bins = 5)$total_iv
  fit <- function(d) ob_numerical_dmiv(x, y, min_bins = 3, max_bins = 5,
                                       bin_method = "woe", divergence_method = d)
  for (d in c("he", "kl", "tr", "klj", "sc", "js")) {
    expect_gte(fit(d)$total_iv, 0.95 * mob_iv)
  }
  for (d in c("l1", "l2", "ln")) {
    expect_gte(fit(d)$total_iv, 0.80 * mob_iv)
  }
  # the selected measure matters: Hellinger and L-infinity cut differently
  expect_false(identical(fit("he")$cutpoints, fit("ln")$cutpoints))
})

# ---- D-01 readmission (audit 15.5): minimum evidence in OBN_DMIV --------------
test_that("D-01/15.5: ob_numerical_dmiv keeps a pure bin only if its purity is significant", {
  # 40 rows, x ~ N(0, 1), logit(p) = x (corpus family size/small_n40). The
  # held D-01 patch kept pure bins of 2-3 rows isolated: the f-divergences
  # reward a pure bin whatever its size, and the hold-out log-loss was
  # significantly worse (+4.6% to +4.9% for he/js/klj with min_bins = 2,
  # max_bins = 3). A bin without events (non-events) may stay only if
  # (1 - r0)^n <= 0.05 (r0^n <= 0.05), unless min_bins blocks the merge.
  for (s in 1:20) {
    set.seed(1000 + s)
    x <- rnorm(40)
    y <- rbinom(40, 1, plogis(x))
    r0 <- mean(y)
    for (d in c("he", "kl", "js", "klj", "l2")) {
      for (mm in list(c(2, 3), c(3, 5), c(3, 8))) {
        r <- ob_numerical_dmiv(x, y, min_bins = mm[1], max_bins = mm[2],
                               divergence_method = d)
        n <- r$count
        p_pure <- ifelse(r$count_pos == 0, (1 - r0)^n,
                         ifelse(r$count_neg == 0, r0^n, 0))
        if (length(n) > mm[1]) {
          expect_true(all(p_pure <= 0.05),
                      info = sprintf("seed %d, %s, %d-%d bins", s, d, mm[1], mm[2]))
        }
      }
    }
  }
})

test_that("D-01/15.5: l1/l2/ln merge same-side bins by the smallest chi-square", {
  # Audit scenario of D-01. Merging two bins on the same side of p = q never
  # lowers an L1/L2/L-infinity distance, so maximizing the distance (held
  # patch) merged the LARGEST same-side bins first: 8000 / 1000 / 1000 /
  # 1000 / 9000 rows for l2 and ln, 86.6% of the MOB IV, and significantly
  # worse hold-out log-loss in four corpus families. Same-side pairs now go
  # by the smallest Pearson chi-square: 99.7% of the MOB IV, no bin below
  # 10% of the rows.
  set.seed(2013)
  x <- runif(20000, 0, 10)
  y <- rbinom(20000, 1, plogis(-3 + 0.5 * x))
  mob_iv <- ob_numerical_mob(x, y, min_bins = 3, max_bins = 5)$total_iv
  for (d in c("l1", "l2", "ln")) {
    r <- ob_numerical_dmiv(x, y, min_bins = 3, max_bins = 5,
                           bin_method = "woe", divergence_method = d)
    expect_length(r$count, 5)
    expect_gte(r$total_iv, 0.95 * mob_iv)
    expect_gte(min(r$count), 0.1 * length(x))
  }
})
