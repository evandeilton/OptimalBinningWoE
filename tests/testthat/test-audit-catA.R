# Audit regressions and invariants for the categorical algorithms
# cm, dmiv, dp, fetb, gmb and ivb (src/OBC_{CM,DMIV,DP,FETB,GMB,IVB}_v5.cpp).
#
# Every "regression" block below fails on the code before the audit and passes
# after it; the invariant blocks pin the contract each algorithm documents.

catA_algs <- c("cm", "dmiv", "dp", "fetb", "gmb", "ivb")

catA_fn <- function(a) {
  get(paste0("ob_categorical_", a), envir = asNamespace("OptimalBinningWoE"))
}

# Split bin labels back into categories, keeping empty-string categories
# (strsplit() drops a trailing empty field).
catA_split <- function(bins, sep = "%;%") {
  unlist(lapply(bins, function(b) {
    if (identical(b, "")) {
      return("")
    }
    p <- strsplit(b, sep, fixed = TRUE)[[1]]
    if (endsWith(b, sep)) p <- c(p, "")
    p
  }))
}

# The contract shared by all six algorithms.
catA_check <- function(res, x, y, max_bins, min_bins, sep = "%;%") {
  x <- as.character(x)
  x[is.na(x)] <- "NA"
  nb <- length(res$bin)
  expect_equal(sum(res$count), length(y))
  expect_equal(sum(res$count_pos), sum(y))
  expect_equal(res$count, res$count_pos + res$count_neg)
  cats <- catA_split(res$bin, sep)
  expect_setequal(cats, unique(x))
  expect_false(anyDuplicated(cats) > 0)
  expect_lte(nb, max_bins)
  expect_gte(nb, min(min_bins, length(unique(x))))
  expect_true(all(is.finite(res$woe)))
  expect_true(all(is.finite(res$iv)))
  invisible(res)
}

catA_data <- function(n, k, seed, sd = 1, prefix = "c") {
  set.seed(seed)
  lev <- sprintf("%s%03d", prefix, seq_len(k))
  x <- sample(lev, n, TRUE)
  eff <- rnorm(k, sd = sd)
  names(eff) <- lev
  y <- rbinom(n, 1, plogis(-0.7 + eff[x]))
  list(x = x, y = as.integer(y))
}

# ============================================================================
# Invariants: silent, complete, within the bin limits
# ============================================================================
test_that("all catA algorithms satisfy the binning contract silently", {
  sets <- list(
    catA_data(600, 6, 1),
    catA_data(2000, 25, 2, sd = 2),
    catA_data(300, 3, 3),
    catA_data(5000, 400, 4) # high cardinality, every level below bin_cutoff
  )
  for (d in sets) {
    for (a in catA_algs) {
      expect_silent(res <- catA_fn(a)(d$x, d$y))
      catA_check(res, d$x, d$y, max_bins = 5, min_bins = 3)
    }
  }
})

test_that("catA algorithms satisfy the contract on German credit", {
  skip_if_no_german()
  gc <- german_credit()
  vars <- c("purpose", "credit_history", "savings")
  for (v in vars) {
    for (a in catA_algs) {
      expect_silent(res <- catA_fn(a)(gc[[v]], gc$target))
      catA_check(res, gc[[v]], gc$target, max_bins = 5, min_bins = 3)
    }
  }
})

test_that("empty-string and NA categories are kept and counted", {
  set.seed(11)
  x <- sample(c("", "a", "b", "c", "d", "e", "f", NA), 3000, TRUE)
  y <- rbinom(3000, 1, 0.3)
  for (a in c("cm", "dmiv", "dp", "fetb")) {
    expect_silent(res <- catA_fn(a)(x, y))
    catA_check(res, x, y, max_bins = 5, min_bins = 3)
  }
})

# ============================================================================
# Bin separator inside category names
# ============================================================================
test_that("a category containing bin_separator triggers a warning", {
  set.seed(12)
  x <- sample(c("a%;%b", "a", "b", "c", "d", "e", "f"), 2000, TRUE)
  y <- rbinom(2000, 1, 0.3)
  for (a in catA_algs) {
    expect_warning(catA_fn(a)(x, y), "bin_separator", fixed = TRUE)
    # With a separator that does not occur in the names the labels are exact.
    expect_silent(res <- catA_fn(a)(x, y, bin_separator = "|"))
    catA_check(res, x, y, max_bins = 5, min_bins = 3, sep = "|")
  }
})

test_that("dp counts are right when a category contains the separator", {
  # dp used to recover a pre-bin's counts by splitting its label on the
  # separator: "a%;%b" was read as "a" + "b" (2292 rows reported for 2000).
  set.seed(12)
  x <- sample(c("a%;%b", "a", "b", "c", "d", "e", "f"), 2000, TRUE)
  y <- rbinom(2000, 1, 0.3)
  res <- suppressWarnings(ob_categorical_dp(x, y))
  expect_equal(sum(res$count), 2000)
  expect_equal(sum(res$count_pos), sum(y))
})

# ============================================================================
# cm: ChiMerge significance level, chi-square overflow
# ============================================================================
test_that("cm uses the upper-tail chi-square quantile as the merge threshold", {
  # A and B differ by chi2 ~ 0.52 (< 3.841, not significant at 5%): ChiMerge
  # merges them. The threshold used to be 0.004 for alpha = 0.05, so they
  # were kept apart.
  x <- rep(c("A", "B", "C", "D"), each = 400)
  y <- c(
    rep(1:0, c(100, 300)), rep(1:0, c(110, 290)),
    rep(1:0, c(200, 200)), rep(1:0, c(320, 80))
  )
  res <- ob_categorical_cm(x, y, min_bins = 2, max_bins = 4)
  expect_setequal(res$bin, c("A%;%B", "C", "D"))
  # alpha = 0.5 -> threshold qchisq(0.5, 1) = 0.455 < 0.52: nothing merges.
  res2 <- ob_categorical_cm(x, y,
    min_bins = 2, max_bins = 4,
    chi_merge_threshold = 0.5
  )
  expect_length(res2$bin, 4)
})

test_that("cm chi-square does not overflow on large bins", {
  # r * c products above 2^31 overflowed int: B (31%) was merged with C (50%)
  # instead of A (30%).
  m <- 60000L
  x <- rep(c("A", "B", "C"), each = m)
  y <- c(
    rep(1:0, c(18000L, m - 18000L)), rep(1:0, c(18600L, m - 18600L)),
    rep(1:0, c(30000L, m - 30000L))
  )
  res <- ob_categorical_cm(x, y, min_bins = 2, max_bins = 2)
  expect_setequal(res$bin, c("A%;%B", "C"))
})

test_that("cm Chi2 variant runs and respects the contract", {
  d <- catA_data(3000, 12, 21)
  expect_silent(res <- ob_categorical_cm(d$x, d$y, use_chi2_algorithm = TRUE))
  catA_check(res, d$x, d$y, max_bins = 5, min_bins = 3)
  expect_equal(res$algorithm, "Chi2")
})

# ============================================================================
# dmiv: console output, pre-bin labels, max_bins, high cardinality
# ============================================================================
test_that("dmiv prints nothing to the console", {
  set.seed(31)
  x <- sample(c("a", "b", "c"), 500, TRUE) # already <= max_bins
  y <- rbinom(500, 1, 0.4)
  expect_silent(ob_categorical_dmiv(x, y))
  skip_if_no_german()
  gc <- german_credit()
  out <- capture.output(
    fit <- obwoe(gc[, c("purpose", "housing", "job", "target")],
      target = "target", algorithm = "dmiv"
    )
  )
  expect_length(out, 0)
})

test_that("dmiv pre-binned rare categories appear in the bin labels", {
  # They used to be pooled into a bin labelled "PREBIN_OTHER", so they could
  # not be mapped back when the binning was applied.
  set.seed(32)
  x <- c(sample(sprintf("big%02d", 1:10), 3000, TRUE), sprintf("rare%02d", 1:30))
  y <- rbinom(length(x), 1, 0.3)
  res <- ob_categorical_dmiv(x, y)
  expect_false(any(grepl("PREBIN_OTHER", res$bin, fixed = TRUE)))
  catA_check(res, x, y, max_bins = 5, min_bins = 3)
})

test_that("dmiv honours min_bins when pre-binning would collapse the data", {
  set.seed(33)
  x <- c(rep("big", 900), sprintf("r%02d", sample.int(40, 100, TRUE)))
  y <- rbinom(1000, 1, ifelse(x == "big", 0.2, 0.6))
  res <- ob_categorical_dmiv(x, y, min_bins = 3, max_bins = 5)
  catA_check(res, x, y, max_bins = 5, min_bins = 3)
})

test_that("dmiv reaches max_bins on high-cardinality data", {
  # 1,500 levels need 1,495 merges; merging used to stop at
  # max_iterations = 1000 and return ~500 bins (and took O(k^3) time).
  set.seed(34)
  x <- sprintf("z%04d", sample.int(1500, 15000, TRUE))
  y <- rbinom(15000, 1, 0.25)
  res <- ob_categorical_dmiv(x, y)
  catA_check(res, x, y, max_bins = 5, min_bins = 3)
  expect_true(res$converged)
  # A budget that runs out before the tolerance is met: still max_bins, but
  # reported as not converged. (Levels of distinct sizes, so consecutive
  # minimum divergences differ and the tolerance is not met by a tie.)
  set.seed(34)
  lev <- sprintf("z%03d", 1:80)
  x2 <- rep(lev, times = 20 + 3 * seq_along(lev))
  y2 <- rbinom(length(x2), 1, plogis(rnorm(80))[match(x2, lev)])
  res2 <- ob_categorical_dmiv(x2, y2, max_iterations = 3, convergence_threshold = 1e-12)
  catA_check(res2, x2, y2, max_bins = 5, min_bins = 3)
  expect_false(res2$converged)
})

test_that("dmiv runs every divergence and WoE method", {
  d <- catA_data(800, 15, 35)
  for (m in c("he", "kl", "tr", "klj", "sc", "js", "l1", "l2", "ln")) {
    for (bm in c("woe", "woe1")) {
      expect_silent(res <- ob_categorical_dmiv(d$x, d$y,
        divergence_method = m, bin_method = bm
      ))
      catA_check(res, d$x, d$y, max_bins = 5, min_bins = 3)
      expect_true(is.finite(res$total_divergence) || m == "sc")
    }
  }
  expect_error(ob_categorical_dmiv(d$x, d$y, divergence_method = "xx"))
  expect_error(ob_categorical_dmiv(d$x, d$y, bin_method = "xx"))
})

# ============================================================================
# dp: empty-string category, large bin_cutoff, optimality
# ============================================================================
test_that("dp keeps an empty-string category that starts a rare group", {
  # The group label was built with `if (label.empty()) label = cat`, so an
  # empty first category vanished from the label -- and from the counts.
  # (55 of these 60 rows used to be reported.)
  x <- c(
    "L6", "L1", "", "L6", "L8", "L6", "L7", "L6", "L7", "L4", "L1",
    "L8", "L3", "L5", "L1", "L8", "L4", "L8", "L3", "L6", "L6", "L2",
    "L6", "", "L2", "L8", "L8", "L6", "L5", "L1", "L8", "L2", "L3",
    "L5", "L1", "", "L2", "", "L8", "L1", "L7", "L4", "L5", "L6",
    "L3", "L3", "L8", "L2", "L3", "L7", "L3", "L7", "L4", "L6", "L1",
    "L1", "L4", "L7", "L2", ""
  )
  y <- c(
    0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 1L, 1L, 1L, 0L, 1L, 0L,
    0L, 0L, 0L, 1L, 1L, 1L, 0L, 0L, 0L, 1L, 1L, 0L, 1L, 0L, 0L, 0L,
    0L, 0L, 0L, 0L, 1L, 1L, 0L, 0L, 1L, 0L, 0L, 0L, 1L, 1L, 0L, 0L,
    0L, 0L, 0L, 0L, 1L, 1L, 0L, 0L, 0L, 1L, 0L, 1L, 0L
  )
  res <- ob_categorical_dp(x, y,
    min_bins = 3, max_bins = 8, bin_cutoff = 0.1,
    max_n_prebins = 30, monotonic_trend = "descending"
  )
  catA_check(res, x, y, max_bins = 8, min_bins = 3)
})

test_that("dp no longer fails when rare grouping leaves too few pre-bins", {
  # One dominant level and nine 1% levels with bin_cutoff = 0.2 used to end
  # in "Failed to find optimal binning with the given constraints."
  set.seed(41)
  x <- c(rep("A", 910), rep(sprintf("r%d", 1:9), each = 10))
  y <- rbinom(1000, 1, ifelse(x == "A", 0.2, 0.5))
  expect_silent(res <- ob_categorical_dp(x, y, bin_cutoff = 0.2))
  catA_check(res, x, y, max_bins = 5, min_bins = 3)
})

test_that("dp returns the IV-optimal contiguous partition", {
  # Brute force over all partitions of the event-rate-sorted categories.
  set.seed(42)
  lev <- LETTERS[1:7]
  x <- sample(lev, 4000, TRUE)
  y <- rbinom(4000, 1, plogis(seq(-2, 2, length.out = 7))[match(x, lev)])
  res <- ob_categorical_dp(x, y, min_bins = 2, max_bins = 4, bin_cutoff = 0.01)
  tp <- sum(y)
  tn <- sum(1 - y)
  pos <- tapply(y, x, sum)
  cnt <- table(x)
  o <- order(pos / cnt)
  p <- as.numeric(pos[o])
  q <- as.numeric(cnt[o]) - p
  iv_seg <- function(i, j) {
    a <- sum(p[i:j]) / tp
    b <- sum(q[i:j]) / tn
    (a - b) * log(a / b)
  }
  best <- -Inf
  for (k in 2:4) {
    for (cuts in combn(6, k - 1, simplify = FALSE)) {
      e <- c(0, cuts, 7)
      best <- max(best, sum(vapply(seq_len(k), function(s) iv_seg(e[s] + 1, e[s + 1]), 0)))
    }
  }
  expect_equal(res$total_iv, best, tolerance = 1e-10)
})

test_that("dp does not discard partitions because of pure pre-bins", {
  # The in-recurrence monotonicity test compared WoE values that are reported
  # as 0 for pure bins, so valid partitions were discarded (IV 1.22 here
  # instead of 2.03).
  x <- c(
    "L4", "L3", "L6", "L4", "L2", "L3", "L8", "L2", "L3", "L3",
    "L2", "L8", "L1", "L5", "L2", "L4", "L7", "L8", "L3", "L2"
  )
  y <- c(
    0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 1L, 0L, 0L, 0L, 1L,
    1L, 1L, 0L, 0L, 1L
  )
  res <- ob_categorical_dp(x, y)
  catA_check(res, x, y, max_bins = 5, min_bins = 3)
  expect_gt(res$total_iv, 2)
})

test_that("dp validates its inputs", {
  d <- catA_data(500, 8, 43)
  expect_error(ob_categorical_dp(d$x, d$y, max_n_prebins = 0), "max_n_prebins")
  y2 <- d$y
  y2[length(y2)] <- 2L # after both classes were seen: used to be counted
  expect_error(ob_categorical_dp(d$x, y2), "only values 0 and 1")
  for (tr in c("ascending", "descending", "none")) {
    expect_silent(res <- ob_categorical_dp(d$x, d$y, monotonic_trend = tr))
    catA_check(res, d$x, d$y, max_bins = 5, min_bins = 3)
  }
})

# ============================================================================
# fetb: Fisher's exact test p-value, rare categories, max_bins
# ============================================================================
test_that("fetb merges the adjacent pair with the highest Fisher p-value", {
  # The criterion used to be the point probability of the observed table,
  # which picked the other pair in both of these samples.
  pick <- function(seed) {
    set.seed(seed)
    n <- c(A = sample(20:200, 1), B = sample(20:200, 1), C = sample(20:200, 1))
    rate <- c(A = 0.2, B = 0.3, C = 0.45) + runif(3, -0.1, 0.1)
    x <- rep(names(n), n)
    y <- rbinom(length(x), 1, rate[x])
    list(x = x, y = y)
  }
  fisher_choice <- function(x, y) {
    pos <- tapply(y, x, sum)
    neg <- tapply(1 - y, x, sum)
    o <- order(log((pos / sum(pos)) / (neg / sum(neg))))
    pv <- vapply(1:2, function(i) {
      stats::fisher.test(rbind(
        c(pos[o[i]], neg[o[i]]),
        c(pos[o[i + 1]], neg[o[i + 1]])
      ))$p.value
    }, 0)
    i <- which.max(pv)
    sort(names(pos)[o[c(i, i + 1)]])
  }
  for (s in c(105, 177, 364, 371)) {
    d <- pick(s)
    res <- ob_categorical_fetb(d$x, d$y, min_bins = 2, max_bins = 2, bin_cutoff = 0.01)
    merged <- res$bin[grepl("%;%", res$bin, fixed = TRUE)]
    expect_equal(sort(strsplit(merged, "%;%", fixed = TRUE)[[1]]), fisher_choice(d$x, d$y))
  }
})

test_that("fetb agrees with fisher.test on random three-level samples", {
  for (s in 1:25) {
    set.seed(500 + s)
    n <- sample(15:120, 3)
    x <- rep(c("A", "B", "C"), n)
    y <- rbinom(length(x), 1, runif(1, 0.1, 0.6))
    pos <- tapply(y, x, sum)
    neg <- tapply(1 - y, x, sum)
    if (any(pos == 0 | neg == 0)) next
    o <- order(log((pos / sum(pos)) / (neg / sum(neg))))
    pv <- vapply(1:2, function(i) {
      stats::fisher.test(rbind(
        c(pos[o[i]], neg[o[i]]),
        c(pos[o[i + 1]], neg[o[i + 1]])
      ))$p.value
    }, 0)
    if (abs(diff(pv)) < 1e-6) next
    i <- which.max(pv)
    res <- ob_categorical_fetb(x, y, min_bins = 2, max_bins = 2, bin_cutoff = 0.01)
    merged <- res$bin[grepl("%;%", res$bin, fixed = TRUE)]
    expect_equal(sort(strsplit(merged, "%;%", fixed = TRUE)[[1]]),
      sort(names(pos)[o[c(i, i + 1)]]),
      info = paste("seed", s)
    )
  }
})

test_that("fetb keeps min_bins on high-cardinality data", {
  # Every level below bin_cutoff used to be pooled into one bin: 1 bin, IV 0.
  d <- catA_data(20000, 500, 51)
  res <- ob_categorical_fetb(d$x, d$y)
  catA_check(res, d$x, d$y, max_bins = 5, min_bins = 3)
  expect_gt(sum(res$iv), 0)
  # A bin_cutoff so large that even the pools are too few.
  res2 <- ob_categorical_fetb(d$x, d$y, bin_cutoff = 0.6)
  catA_check(res2, d$x, d$y, max_bins = 5, min_bins = 3)
})

test_that("fetb reaches max_bins even with a tiny iteration budget", {
  d <- catA_data(5000, 30, 52, sd = 2)
  res <- ob_categorical_fetb(d$x, d$y, bin_cutoff = 0.001, max_iterations = 3)
  catA_check(res, d$x, d$y, max_bins = 5, min_bins = 3)
  expect_false(res$converged)
  res2 <- ob_categorical_fetb(d$x, d$y, bin_cutoff = 0.001)
  expect_true(res2$converged)
})

test_that("fetb returns non-decreasing WoE", {
  for (s in 1:10) {
    d <- catA_data(3000, 20, 600 + s, sd = 1.5)
    res <- ob_categorical_fetb(d$x, d$y, bin_cutoff = 0.001, max_bins = 6)
    expect_false(is.unsorted(res$woe), info = paste("seed", s))
  }
})

test_that("fetb validates its inputs", {
  d <- catA_data(300, 5, 53)
  y2 <- d$y
  y2[1] <- 2L
  expect_error(ob_categorical_fetb(d$x, y2), "binary")
  expect_error(ob_categorical_fetb(d$x, d$y, max_iterations = 0), "max_iterations")
  expect_error(ob_categorical_fetb(d$x, d$y, min_bins = 1), "min_bins")
  # Direct call with a missing category (the wrapper maps NA to "NA" first).
  x <- d$x
  x[1:5] <- NA
  res <- OptimalBinningWoE:::optimal_binning_categorical_fetb(d$y, x)
  expect_true(any(grepl("__NA__", res$bin, fixed = TRUE)))
})

# ============================================================================
# gmb: greedy criterion, rare categories
# ============================================================================
test_that("gmb follows its documented greedy criterion", {
  # Reference: repeatedly merge the adjacent pair (event-rate order) whose
  # merge leaves the highest total Bayesian-smoothed IV. Above 10 bins the
  # scores used to be read from a position-keyed cache that went stale after
  # every merge; 56 of 60 such samples disagreed with the reference.
  gmb_ref <- function(x, y, max_bins) {
    tp <- sum(y)
    tn <- sum(1 - y)
    pr_pos <- 0.5 * tp / (tp + tn)
    pr_neg <- 0.5 - pr_pos
    iv1 <- function(p, n) {
      dp <- (p + pr_pos) / (tp + 0.5)
      dn <- (n + pr_neg) / (tn + 0.5)
      (dp - dn) * log(dp / dn)
    }
    cats <- sort(unique(x))
    pos <- tapply(y, factor(x, cats), sum)
    cnt <- as.numeric(table(factor(x, cats)))
    o <- order(pos / cnt)
    g <- as.list(cats[o])
    gp <- as.numeric(pos[o])
    gn <- cnt[o] - gp
    while (length(g) > max_bins) {
      ivs <- mapply(iv1, gp, gn)
      sc <- vapply(seq_len(length(g) - 1), function(i) {
        sum(ivs[-c(i, i + 1)]) + iv1(gp[i] + gp[i + 1], gn[i] + gn[i + 1])
      }, 0)
      i <- which.max(sc)
      g[[i]] <- c(g[[i]], g[[i + 1]])
      g[[i + 1]] <- NULL
      gp[i] <- gp[i] + gp[i + 1]
      gp <- gp[-(i + 1)]
      gn[i] <- gn[i] + gn[i + 1]
      gn <- gn[-(i + 1)]
    }
    vapply(g, function(z) paste(sort(z), collapse = "|"), "")
  }
  norm <- function(b) {
    vapply(strsplit(b, "%;%", fixed = TRUE), function(z) paste(sort(z), collapse = "|"), "")
  }
  for (s in 1:5) {
    d <- catA_data(3000, 14, s, prefix = "g")
    res <- ob_categorical_gmb(d$x, d$y, min_bins = 2, max_bins = 5, bin_cutoff = 0.01)
    expect_setequal(norm(res$bin), gmb_ref(d$x, d$y, 5))
  }
})

test_that("gmb keeps min_bins when every level is rare", {
  d <- catA_data(20000, 500, 61)
  res <- ob_categorical_gmb(d$x, d$y)
  catA_check(res, d$x, d$y, max_bins = 5, min_bins = 3)
  res2 <- ob_categorical_gmb(d$x, d$y, bin_cutoff = 0.6)
  catA_check(res2, d$x, d$y, max_bins = 5, min_bins = 3)
})

test_that("gmb warns on degenerate classes and validates targets", {
  set.seed(62)
  x <- sample(letters[1:6], 200, TRUE)
  y <- c(1L, 1L, integer(198))
  expect_warning(ob_categorical_gmb(x, y), "fewer than 5 samples")
  y2 <- rbinom(200, 1, 0.5)
  y2[7] <- 2L
  expect_error(ob_categorical_gmb(x, y2), "binary")
  x2 <- x
  x2[1:3] <- NA
  expect_warning(
    OptimalBinningWoE:::optimal_binning_categorical_gmb(rbinom(200, 1, 0.5), x2),
    "missing values"
  )
})

# ============================================================================
# ivb: rare categories, custom separator, DP optimality, monotonic repair
# ============================================================================
test_that("ivb labels pooled rare categories with the caller's separator", {
  set.seed(71)
  x <- c(sample(letters[1:6], 3000, TRUE), sample(sprintf("r%d", 1:10), 100, TRUE))
  y <- rbinom(length(x), 1, 0.3)
  res <- ob_categorical_ivb(x, y, bin_separator = "|")
  catA_check(res, x, y, max_bins = 5, min_bins = 3, sep = "|")
  expect_false(any(grepl("%;%", res$bin, fixed = TRUE)))
})

test_that("ivb keeps min_bins when every level is rare", {
  d <- catA_data(20000, 500, 72)
  res <- ob_categorical_ivb(d$x, d$y)
  catA_check(res, d$x, d$y, max_bins = 5, min_bins = 3)
  res2 <- ob_categorical_ivb(d$x, d$y, bin_cutoff = 0.6)
  catA_check(res2, d$x, d$y, max_bins = 5, min_bins = 3)
})

test_that("ivb evaluates the full DP recurrence", {
  # A "banded" j range capped the size of the last bin, so partitions with
  # one large bin were never evaluated. Brute force over all partitions.
  set.seed(73)
  lev <- sprintf("L%02d", 1:10)
  x <- sample(lev, 5000, TRUE, prob = c(0.3, rep(0.7 / 9, 9)))
  y <- rbinom(5000, 1, plogis(c(-2, rep(0.2, 8), 1.5))[match(x, lev)])
  res <- ob_categorical_ivb(x, y, min_bins = 2, max_bins = 3, bin_cutoff = 0.01)
  tp <- sum(y)
  tn <- sum(1 - y)
  pr_pos <- 0.5 * tp / (tp + tn)
  pr_neg <- 0.5 - pr_pos
  pos <- tapply(y, x, sum)
  cnt <- table(x)
  o <- order(pos / cnt)
  p <- as.numeric(pos[o])
  q <- as.numeric(cnt[o]) - p
  iv_seg <- function(i, j) {
    a <- (sum(p[i:j]) + pr_pos) / (tp + 0.5)
    b <- (sum(q[i:j]) + pr_neg) / (tn + 0.5)
    (a - b) * log(a / b)
  }
  best <- -Inf
  for (k in 2:3) {
    for (cuts in combn(9, k - 1, simplify = FALSE)) {
      e <- c(0, cuts, 10)
      best <- max(best, sum(vapply(seq_len(k), function(s) iv_seg(e[s] + 1, e[s + 1]), 0)))
    }
  }
  expect_equal(res$total_iv, best, tolerance = 1e-10)
})

test_that("ivb monotonic repair never drops categories", {
  # A tiny all-positive level after a large high-rate level: after Bayesian
  # smoothing its WoE falls below its neighbour's although its event rate is
  # higher, so the repair runs. It used to erase the wrong boundary --
  # sometimes the last one, and the final bin's categories vanished.
  x <- c(
    rep("low", 400), rep("mid1", 300), rep("mid2", 300), rep("high", 300),
    rep("top", 200), rep("tinytop", 2)
  )
  y <- c(
    rep(1:0, c(20, 380)), rep(1:0, c(90, 210)), rep(1:0, c(90, 210)),
    rep(1:0, c(150, 150)), rep(1:0, c(190, 10)), 1L, 1L
  )
  res <- ob_categorical_ivb(x, y, min_bins = 2, max_bins = 5, bin_cutoff = 0.001)
  catA_check(res, x, y, max_bins = 5, min_bins = 2)
  expect_false(is.unsorted(res$woe))
  # Perfectly separating sample on which the old code lost level L15.
  x2 <- c(
    "L15", "L30", "L33", "L13", "L22", "L48", "L46", "L28", "L8",
    "L34", "L22", "L46", "L17", "L28", "L5", "L30", "L5", "L48",
    "L32", "L59"
  )
  y2 <- c(
    1L, 1L, 0L, 1L, 1L, 0L, 0L, 1L, 1L, 0L, 1L, 0L, 1L, 1L, 1L,
    1L, 1L, 0L, 0L, 0L
  )
  res2 <- ob_categorical_ivb(x2, y2,
    min_bins = 2, max_bins = 4, bin_cutoff = 0.02,
    max_n_prebins = 10
  )
  catA_check(res2, x2, y2, max_bins = 4, min_bins = 2)
})

test_that("ivb input handling", {
  d <- catA_data(400, 6, 75)
  expect_error(ob_categorical_ivb(rep("a", 50), rbinom(50, 1, 0.5)), "at least 2 distinct")
  f <- factor(d$x)
  f[1:4] <- NA
  expect_warning(
    res <- OptimalBinningWoE:::optimal_binning_categorical_ivb(d$y, f),
    "missing values found in feature"
  )
  expect_equal(sum(res$count), 400)
  yna <- d$y
  yna[1:3] <- NA
  expect_warning(
    res2 <- OptimalBinningWoE:::optimal_binning_categorical_ivb(yna, d$x),
    "missing values found in target"
  )
  expect_equal(sum(res2$count), 397)
  expect_error(
    OptimalBinningWoE:::optimal_binning_categorical_ivb(d$y, d$x[-1]),
    "same length"
  )
  expect_error(
    OptimalBinningWoE:::optimal_binning_categorical_ivb(d$y, seq_along(d$y)),
    "factor or character"
  )
})

# ============================================================================
# Input validation and remaining branches
# ============================================================================
test_that("catA algorithms reject invalid inputs with an error", {
  d <- catA_data(300, 6, 81)
  x <- d$x
  y <- d$y
  y2 <- y
  y2[length(y2)] <- 2L
  yna <- y
  yna[1] <- NA
  for (a in catA_algs) {
    f <- catA_fn(a)
    if (a != "ivb") expect_error(f(x, y, min_bins = 4, max_bins = 3), info = a)
    expect_error(f(x, y, bin_cutoff = -0.1), info = a)
    expect_error(f(x, y2), info = a)
    expect_error(f(x, rep(0L, length(y))), info = a)
    if (a != "ivb") expect_error(f(x, yna), info = a)
  }
  for (a in c("cm", "dmiv", "dp")) {
    f <- catA_fn(a)
    expect_error(f(x, y, bin_cutoff = 1), info = a)
    expect_error(f(x, y, max_n_prebins = 1), info = a)
    expect_error(f(x, y, convergence_threshold = 0), info = a)
    expect_error(f(x, y, max_iterations = 0), info = a)
  }
  for (a in c("cm", "dp", "gmb", "ivb")) {
    expect_error(catA_fn(a)(x, y, min_bins = 1), info = a)
  }
  expect_error(ob_categorical_dmiv(x, y, min_bins = 0))
  expect_error(ob_categorical_cm(x, y, chi_merge_threshold = 1))
  expect_error(ob_categorical_dp(x, y, monotonic_trend = "up"))
  expect_error(ob_categorical_gmb(x, y, max_n_prebins = 2))
  for (a in c("cm", "dmiv")) {
    expect_error(catA_fn(a)(rep("a", 50), rbinom(50, 1, 0.5)), info = a)
  }
  xe <- x
  xe[1] <- ""
  for (a in c("gmb", "ivb")) expect_error(catA_fn(a)(xe, y), "empty strings")
})

test_that("internal entry points validate lengths and map NA categories", {
  d <- catA_data(200, 5, 82)
  x <- d$x
  x[1:3] <- NA
  ns <- asNamespace("OptimalBinningWoE")
  for (nm in c("cm", "dmiv", "dp", "fetb", "gmb")) {
    fn <- get(paste0("optimal_binning_categorical_", nm), envir = ns)
    expect_error(fn(d$y, d$x[-1]), info = nm)
    expect_error(fn(integer(0), character(0)), info = nm)
  }
  for (nm in c("cm", "dmiv", "dp")) {
    fn <- get(paste0("optimal_binning_categorical_", nm), envir = ns)
    res <- fn(d$y, x)
    expect_true(any(grepl("NA", res$bin, fixed = TRUE)), info = nm)
    expect_equal(sum(res$count), 200, info = nm)
  }
  expect_error(ns$optimal_binning_categorical_ivb(integer(0), character(0)))
  expect_error(suppressWarnings(
    ns$optimal_binning_categorical_ivb(rep(NA_integer_, 5), letters[1:5])
  ))
})

test_that("cm edge paths: few categories, iteration cap, tolerance", {
  set.seed(83)
  x <- sample(c("a", "b"), 300, TRUE)
  y <- rbinom(300, 1, 0.4)
  res <- ob_categorical_cm(x, y) # min_bins 3 > 2 categories
  expect_length(res$bin, 2)
  expect_true(any(grepl("Adjusted min_bins", res$warnings)))
  d <- catA_data(4000, 30, 84, sd = 2)
  res2 <- ob_categorical_cm(d$x, d$y,
    bin_cutoff = 0.001, max_n_prebins = 40,
    max_bins = 25, min_bins = 2, max_iterations = 1, chi_merge_threshold = 0.001
  )
  expect_false(res2$converged)
  expect_true(any(grepl("max_iterations", res2$warnings)))
  res3 <- ob_categorical_cm(d$x, d$y,
    bin_cutoff = 0.001, max_n_prebins = 10,
    max_bins = 8, min_bins = 2, convergence_threshold = 10
  )
  catA_check(res3, d$x, d$y, max_bins = 8, min_bins = 2)
})

test_that("dmiv edge paths: max_bins = 1 and pure categories", {
  set.seed(85)
  x <- sample(letters[1:6], 600, TRUE)
  y <- as.integer(x %in% c("a", "b"))
  y[x == "c"] <- rbinom(sum(x == "c"), 1, 0.5)
  for (m in c("sc", "tr", "kl", "js", "he", "klj", "l1", "l2", "ln")) {
    expect_silent(res <- ob_categorical_dmiv(x, y, divergence_method = m))
    catA_check(res, x, y, max_bins = 5, min_bins = 3)
  }
  res1 <- ob_categorical_dmiv(x, y, min_bins = 1, max_bins = 1)
  expect_length(res1$bin, 1)
  expect_equal(sum(res1$count), 600)
})

test_that("dp limits pre-bins to max_n_prebins", {
  d <- catA_data(5000, 25, 86)
  res <- ob_categorical_dp(d$x, d$y, bin_cutoff = 0.001, max_n_prebins = 6)
  catA_check(res, d$x, d$y, max_bins = 5, min_bins = 3)
  # max_n_prebins below min_bins: every pre-bin becomes a bin
  res2 <- ob_categorical_dp(d$x, d$y, bin_cutoff = 0.001, max_n_prebins = 2)
  expect_length(res2$bin, 2)
  expect_equal(sum(res2$count), 5000)
})

test_that("fetb repairs WoE order around pure bins", {
  set.seed(1)
  k <- sample(4:12, 1)
  lev <- sprintf("L%02d", 1:k)
  sz <- sample(c(1, 2, 3, 5, 20, 100, 300), k, TRUE)
  x <- rep(lev, sz)
  y <- rbinom(length(x), 1, runif(k)[match(x, lev)])
  mb <- sample(2:4, 1)
  res <- ob_categorical_fetb(x, y, min_bins = 2, max_bins = mb, bin_cutoff = 0.001)
  catA_check(res, x, y, max_bins = mb, min_bins = 2)
  expect_false(is.unsorted(res$woe))
})

test_that("gmb scores bins whose smoothed IV is exactly zero", {
  # Balanced classes and levels with equal counts of both classes give a
  # bin IV of exactly 0, which the total-IV sum recomputes from the counts.
  x <- rep(sprintf("q%02d", 1:12), each = 40)
  y <- rep(c(0L, 1L), 240)
  y[x %in% c("q01", "q02")] <- 0L
  y[x %in% c("q11", "q12")] <- 1L
  res <- ob_categorical_gmb(x, y, bin_cutoff = 0.01)
  catA_check(res, x, y, max_bins = 5, min_bins = 3)
})

test_that("ivb large pre-bin sets and inverted bin limits", {
  d <- catA_data(8000, 30, 87)
  res <- ob_categorical_ivb(d$x, d$y, bin_cutoff = 0.001, max_n_prebins = 40)
  catA_check(res, d$x, d$y, max_bins = 5, min_bins = 3)
  # max_bins < min_bins is corrected to max_bins = min_bins
  res2 <- ob_categorical_ivb(d$x, d$y, min_bins = 4, max_bins = 3, bin_cutoff = 0.001)
  expect_length(res2$bin, 4)
})
