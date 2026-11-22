# Permanent regression tests for the categorical-engine fixes of the forensic
# audit, group G3 (audit/AUDITORIA_FORENSE_BINNING_CPP.md, Section 9.8).
# Each test encodes the minimal failing scenario of a finding with its
# expected result stated explicitly.

# Build a feature/target pair from per-category (count, events) pairs.
g3_make <- function(spec) {
  x <- unlist(lapply(names(spec), function(k) rep(k, spec[[k]][1])))
  y <- unlist(lapply(names(spec), function(k) {
    c(rep(1, spec[[k]][2]), rep(0, spec[[k]][1] - spec[[k]][2]))
  }))
  list(x = x, y = y)
}
g3_cats <- function(label) strsplit(label, "%;%", fixed = TRUE)[[1]]

# ---- D-05: OBC_MOB ignores bin_cutoff when ncat <= max_bins -------------------
test_that("D-05: ob_categorical_mob merges a rare category of a low-cardinality feature", {
  # 4 categories and max_bins = 5: the baseline took the "one bin per
  # category" branch and kept R (10 rows of 10 000, 0.1%) as its own bin.
  d <- g3_make(list(A = c(4000, 400), B = c(3000, 600), C = c(2990, 900), R = c(10, 5)))
  r <- ob_categorical_mob(d$x, d$y, min_bins = 3, max_bins = 5, bin_cutoff = 0.05)
  expect_false("R" %in% r$bin)
  expect_equal(length(r$bin), 3L)
  expect_true(all(r$count / sum(r$count) >= 0.05))
  expect_equal(sum(r$count), 10000)
  # R (rate 0.5) joins the category with the closest event rate, C (0.30).
  expect_true(any(vapply(r$bin, function(b) setequal(g3_cats(b), c("C", "R")), TRUE)))
  # min_bins still wins: with min_bins = 4 nothing can be merged.
  r4 <- ob_categorical_mob(d$x, d$y, min_bins = 4, max_bins = 5, bin_cutoff = 0.05)
  expect_equal(length(r4$bin), 4L)
})

# ---- D-06: OBC_DMIV bin_cutoff was a dead parameter ----------------------------
test_that("D-06: ob_categorical_dmiv applies bin_cutoff to rare categories", {
  # r1 and r2 hold 0.5% of the rows each. The baseline ignored bin_cutoff
  # (only < 5 rows were pooled, and only above max_n_prebins categories), so
  # both stayed as separate bins whatever bin_cutoff said.
  d <- g3_make(list(A = c(3000, 300), B = c(3000, 600), C = c(2000, 600),
                    D = c(1900, 800), r1 = c(50, 5), r2 = c(50, 40)))
  r <- ob_categorical_dmiv(d$x, d$y, min_bins = 2, max_bins = 6, bin_cutoff = 0.05)
  expect_false(any(r$bin %in% c("r1", "r2")))
  expect_true(all(r$count >= 0.05 * 10000))
  expect_equal(sum(r$count), 10000)
  # Each rare category joins its event-rate neighbour: r1 (0.10) with A
  # (0.10), r2 (0.80) with D (0.42), the highest-rate frequent category.
  expect_true(any(vapply(r$bin, function(b) setequal(g3_cats(b), c("A", "r1")), TRUE)))
  expect_true(any(vapply(r$bin, function(b) setequal(g3_cats(b), c("D", "r2")), TRUE)))
  # bin_cutoff now matters: with a tiny cutoff both rare levels stay apart.
  r0 <- ob_categorical_dmiv(d$x, d$y, min_bins = 2, max_bins = 6, bin_cutoff = 0.001)
  expect_true(all(c("r1", "r2") %in% r0$bin))
})

test_that("D-06: ob_categorical_dmiv bounds the bins entering the merge phase", {
  # 300 levels of 20 rows each (0.33% < bin_cutoff): pooled in rate order to
  # at most max_n_prebins bins before the divergence merges.
  set.seed(11)
  lv <- sprintf("L%03d", 1:300)
  x <- rep(lv, each = 20)
  y <- rbinom(6000, 1, rep(seq(0.05, 0.7, length.out = 300), each = 20))
  r <- ob_categorical_dmiv(x, y, min_bins = 3, max_bins = 5, max_n_prebins = 10)
  expect_lte(length(r$bin), 5L)
  expect_equal(sum(r$count), 6000)
  expect_lte(r$iterations, 10L)   # <= max_n_prebins - min_bins merges remain
})

# ---- C-02 / C-03 / C-04: OBC_SAB is exact -------------------------------------
# All set partitions of 1..m into exactly k non-empty blocks (restricted growth).
g3_set_partitions <- function(m, k) {
  out <- list()
  rec <- function(a, i, mx) {
    if (i > m) {
      if (mx == k) out[[length(out) + 1L]] <<- a
      return(invisible())
    }
    if (mx + (m - i + 1) < k) return(invisible())
    for (b in seq_len(min(mx + 1L, k))) {
      a[i] <- b
      rec(a, i + 1L, max(mx, b))
    }
  }
  rec(integer(m), 1L, 0L)
  out
}
# SAB's reported estimator: Bayesian prior of strength 0.5 split by the base rate.
g3_sab_iv <- function(pos, cnt, tp, tn) {
  w <- 0.5
  r <- tp / (tp + tn)
  p <- (pos + w * r) / (tp + w)
  q <- (cnt - pos + w * (1 - r)) / (tn + w)
  sum((p - q) * log(p / q))
}

# ---- B-02: OBC_SBLP's DP optimises the reported IV ------------------------------
test_that("B-02: ob_categorical_sblp returns the best reported IV over K and cut points", {
  # The DP normalised the smoothing by the number of categories in each
  # segment (e - s) instead of the final K, ordered by a smoothed rate and only
  # tried K = max_bins, so its partition was not the optimum of the reported
  # IV in ~25% of small-sample cases. Brute force over every K in
  # [min_bins, max_bins] and every contiguous partition in exact rate order,
  # scored with the reported estimator (alpha, K).
  sblp_iv <- function(pos, neg, alpha = 0.5) {
    K <- length(pos)
    p <- (pos + alpha) / (sum(pos) + alpha * K)
    q <- (neg + alpha) / (sum(neg) + alpha * K)
    sum((p - q) * log(p / q))
  }
  set.seed(21)
  for (rep in 1:5) {
    lv <- LETTERS[1:8]
    n_cat <- sample(12:30, 8, TRUE)
    x <- rep(lv, n_cat)
    y <- unlist(lapply(seq_along(lv), function(i) rbinom(n_cat[i], 1, runif(1, 0.05, 0.8))))
    if (length(unique(y)) < 2) next
    r <- ob_categorical_sblp(x, y, min_bins = 2, max_bins = 4,
                             bin_cutoff = 0.01, max_n_prebins = 20)
    pos <- as.numeric(tapply(y, x, sum)[lv])
    cnt <- as.numeric(tapply(y, x, length)[lv])
    o <- order(pos / cnt, cnt)
    pos <- pos[o]
    neg <- cnt[o] - pos
    best <- -Inf
    for (k in 2:4) for (cuts in combn(7, k - 1, simplify = FALSE)) {
      g <- findInterval(seq_len(8), cuts + 1) + 1
      best <- max(best, sblp_iv(tapply(pos, g, sum), tapply(neg, g, sum)))
    }
    expect_equal(r$total_iv, best, tolerance = 1e-10)
    expect_true(r$converged)
    expect_false(is.unsorted(r$woe))
  }
})

# ---- E-04: OBC_GMB pre-bin limitation ignores the event rate -------------------
test_that("E-04: ob_categorical_gmb reduces pre-bins along the event-rate order", {
  # 60 equal-size categories, event rates linear from 2% to 60%. With
  # max_n_prebins = 20 the baseline kept the 20 lowest-rate categories (stable
  # sort by count) and folded the 40 others -- two thirds of the data -- into
  # one bin: IV 0.605 against 0.791 for the 60 categories.
  lv <- sprintf("c%02d", 1:60)
  ev <- round(seq(0.02, 0.60, length.out = 60) * 1000)
  x <- rep(lv, each = 1000)
  y <- unlist(lapply(ev, function(e) c(rep(1, e), rep(0, 1000 - e))))
  r <- ob_categorical_gmb(x, y, bin_cutoff = 0.01, max_n_prebins = 20,
                          min_bins = 3, max_bins = 5)
  a <- 0.5
  K <- 60
  dp <- (ev + a) / (sum(ev) + K * a)
  dn <- (1000 - ev + a) / (sum(1000 - ev) + K * a)
  iv60 <- sum((dp - dn) * log(dp / dn))
  expect_gte(r$total_iv, 0.9 * iv60)
  expect_lt(max(r$count), 30000)            # baseline: one bin of 41 000 rows
  expect_equal(sum(r$count), 60000)
  # Every bin is a run of consecutive categories in event-rate order.
  for (b in r$bin) {
    idx <- sort(match(g3_cats(b), lv))
    expect_true(all(diff(idx) == 1L))
  }
})

# ---- E-03: bin_cutoff applies to observations, not to events -----------------
test_that("E-03: a large low-risk category keeps its own bin (mba, sketch)", {
  # LOW holds 50% of the rows and 20 of the 1220 events (rate 0.2%). With
  # bin_cutoff = 0.05 the baseline also demanded count_pos >= 0.05 * 1220 = 61
  # and merged LOW, the most informative bin, with M1.
  d <- g3_make(list(LOW = c(10000, 20), M1 = c(4000, 200),
                    M2 = c(4000, 400), HI = c(2000, 600)))
  for (f in list(ob_categorical_mba, ob_categorical_sketch)) {
    r <- f(d$x, d$y, bin_cutoff = 0.05)
    expect_true("LOW" %in% r$bin)
    expect_equal(length(r$bin), 4L)
    expect_equal(r$count[r$bin == "LOW"], 10000)
    expect_equal(which.min(r$woe), which(r$bin == "LOW"))
  }
})


# ---- E-03 follow-up: minimum evidence (empirical-Bayes order and merges) ------
# Posterior-mean event rate under a beta prior centred on the overall rate with
# m = 20 pseudo-observations (the constant of OBC_MBA / OBC_Sketch / OBC_SAB).
g3_shrunk <- function(x, y, m = 20) {
  r0 <- mean(y)
  (tapply(y, x, sum) + m * r0) / (tapply(y, x, length) + m)
}

test_that("E-03: a pure category joins its nearest neighbour when bins must be merged (mba, sketch)", {
  # A has no event (10% of the rows). Reducing 4 bins to max_bins = 3, the
  # held E-03 patch merged C and D (rates 0.30 and 0.63) because the pure bin
  # looked too valuable to merge: hold-out log-loss +2.5% (MBA) and +5.3%
  # (Sketch) over 20 seeds. (The baseline avoided it only because its events
  # rule had already merged A.) Under the empirical-Bayes IV the cheapest
  # merge is A with B (rate 0.057), which is also the maximum-likelihood one.
  d <- g3_make(list(A = c(204, 0), B = c(653, 37), C = c(606, 180), D = c(637, 402)))
  for (f in list(ob_categorical_mba, ob_categorical_sketch)) {
    r <- f(d$x, d$y, min_bins = 2, max_bins = 3)
    expect_equal(length(r$bin), 3L)
    expect_true(any(vapply(r$bin, function(b) setequal(g3_cats(b), c("A", "B")), TRUE)))
    expect_true(all(c("C", "D") %in% r$bin))
  }
})

test_that("E-03: ob_categorical_mba bins are contiguous runs in the shrunk-rate order", {
  # 200 sparse categories (about 10 rows each), the cat/k200_signal family of
  # the audit harness. Ordering by the raw rate pooled the categories that had
  # no event by chance (training rate 0.3%, hold-out rate 10%): +7% hold-out
  # log-loss in 20/20 seeds once the undocumented events rule was removed.
  set.seed(2001)
  lv <- sprintf("c%03d", 1:200)
  x <- sample(lv, 2100, TRUE, prob = rexp(200))
  y <- rbinom(2100, 1, setNames(plogis(rnorm(200, -1.5, 1)), lv)[x])
  r <- ob_categorical_mba(x, y)
  expect_equal(sum(r$count), 2100)
  s <- g3_shrunk(x, y)
  rng <- t(vapply(r$bin, function(b) range(s[g3_cats(b)]), numeric(2)))
  rng <- rng[order(rng[, 1]), , drop = FALSE]
  expect_true(all(rng[-1, 1] >= rng[-nrow(rng), 2]))
  # No bin is made only of categories without events.
  expect_true(all(r$count_pos > 0))
})

# ---- E-02: stale indices in OBC_Sketch::enforce_bin_cutoff ---------------------
test_that("E-02: ob_categorical_sketch leaves no bin below bin_cutoff", {
  # D (30 rows) and the pooled rare categories (10 rows) are both below the
  # 5% cutoff (50 rows). D merged with the pool (the higher index), but the
  # pending entry of the erased bin was not redirected to the merged bin, so
  # a 40-row bin survived with 4 bins > min_bins = 3.
  d <- g3_make(list(A = c(400, 40), B = c(300, 60), C = c(260, 78),
                    D = c(30, 15), r1 = c(5, 3), r2 = c(5, 2)))
  r <- ob_categorical_sketch(d$x, d$y, bin_cutoff = 0.05, min_bins = 3)
  expect_true(all(r$count >= ceiling(0.05 * length(d$x))) ||
                length(r$bin) == 3L)
  expect_equal(sum(r$count), length(d$x))
  expect_false(any(vapply(r$bin, function(b) setequal(g3_cats(b), c("D", "r1", "r2")), TRUE)))
})

# ---- C-02 / C-03 / C-04 (X-12): OBC_SAB is an exact DP, with minimum evidence --
# Empirical-Bayes IV of a partition (ob::eb_woe_iv, m pseudo-observations per
# bin, shares normalized over the K bins of the partition).
g3_eb_iv <- function(pos, cnt, tp, tn, m = 20) {
  K <- length(pos)
  r0 <- tp / (tp + tn)
  p <- (pos + m * r0) / (tp + K * m * r0)
  q <- (cnt - pos + m * (1 - r0)) / (tn + K * m * (1 - r0))
  sum((p - q) * log(p / q))
}

test_that("C-02: ob_categorical_sab returns the exact optimum of its documented objective (brute force)", {
  # The annealing stopped ~200 iterations after its last improvement and
  # penalised arbitrary bin labels by -500; on few categories it was 2.6%
  # below the optimum in 161/200 instances. The search is now exact: for each
  # k in [min_bins, max_bins] the best contiguous partition of the categories
  # in shrunk-rate order under the empirical-Bayes IV, then the k with the
  # largest reported IV. Checked here against enumeration of all cut points.
  set.seed(7)
  for (rep in 1:4) {
    lv <- LETTERS[1:7]
    n_cat <- sample(60:140, 7, TRUE)
    rate <- runif(7, 0.02, 0.6)
    x <- rep(lv, n_cat)
    y <- unlist(lapply(seq_along(lv), function(i) rbinom(n_cat[i], 1, rate[i])))
    r <- ob_categorical_sab(x, y, min_bins = 2, max_bins = 4, bin_cutoff = 0.01)
    pos <- tapply(y, x, sum)[lv]
    cnt <- tapply(y, x, length)[lv]
    tp <- sum(y)
    tn <- length(y) - tp
    o <- order(g3_shrunk(x, y)[lv], cnt)
    pos <- pos[o]
    cnt <- cnt[o]
    best <- -Inf
    for (k in 2:4) {
      cuts <- combn(6, k - 1)
      val <- apply(cuts, 2, function(cc) {
        g <- findInterval(1:7, cc + 1) + 1
        g3_eb_iv(tapply(pos, g, sum), tapply(cnt, g, sum), tp, tn)
      })
      cc <- cuts[, which.max(val)]
      g <- findInterval(1:7, cc + 1) + 1
      best <- max(best, g3_sab_iv(tapply(pos, g, sum), tapply(cnt, g, sum), tp, tn))
    }
    expect_equal(r$total_iv, best, tolerance = 1e-10)
    expect_true(r$converged)
    expect_identical(r$iterations, 0L)
  }
})

test_that("C-02: ob_categorical_sab does not depend on set.seed() and keeps bin_cutoff", {
  set.seed(3)
  lv <- sprintf("c%02d", 1:40)
  x <- sample(lv, 4000, TRUE)
  y <- rbinom(4000, 1, setNames(seq(0.05, 0.6, length.out = 40), lv)[x])
  set.seed(1)
  r1 <- ob_categorical_sab(x, y)
  set.seed(99)
  r2 <- ob_categorical_sab(x, y)
  expect_identical(r1, r2)
  expect_true(all(r1$count / sum(r1$count) >= 0.05))
  expect_equal(sum(r1$count), 4000)
  # Bins are contiguous in the shrunk event rate: their ranges never overlap.
  s <- g3_shrunk(x, y)
  rng <- t(vapply(r1$bin, function(b) range(s[g3_cats(b)]), numeric(2)))
  rng <- rng[order(rng[, 1]), , drop = FALSE]
  expect_true(all(rng[-1, 1] >= rng[-nrow(rng), 2]))
})

test_that("X-12: ob_categorical_sab does not isolate sparse categories without events", {
  # 200 sparse categories (about 10 rows each; cat/k200_signal of the audit
  # harness). With the raw-rate order the exact DP isolated the categories
  # that had no event by chance (a bin with 0 events among > 105 rows), which
  # maximised training IV and raised the hold-out log-loss by 5-8% in 20/20
  # seeds. In the shrunk-rate order a tiny category is ranked near the
  # overall rate unless its data say otherwise.
  set.seed(2001)
  lv <- sprintf("c%03d", 1:200)
  x <- sample(lv, 2100, TRUE, prob = rexp(200))
  y <- rbinom(2100, 1, setNames(plogis(rnorm(200, -1.5, 1)), lv)[x])
  r <- ob_categorical_sab(x, y)
  expect_equal(sum(r$count), 2100)
  expect_true(all(r$count_pos > 0))
  expect_true(all(r$count / 2100 >= 0.05))
  zero <- names(which(tapply(y, x, sum) == 0))
  for (b in r$bin) expect_false(all(g3_cats(b) %in% zero))
})
