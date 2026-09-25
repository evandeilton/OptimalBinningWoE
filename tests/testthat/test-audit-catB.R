# Audit of the categorical engines jedi, jedi_mwoe, mba, milp, mob and sab:
# regression tests for the bugs fixed in that audit, and invariants that every
# result must satisfy (counts add up, at most max_bins bins, finite and
# monotone WoE, no warnings on valid input).

# 8 equally frequent categories whose event rate rises from 0.1 to 0.8.
catb_separated <- function(n_per = 200, seed = 1) {
  set.seed(seed)
  k <- 8
  f <- rep(paste0("c", seq_len(k)), each = n_per)
  y <- rbinom(length(f), 1, rep(seq(0.1, 0.8, length.out = k), each = n_per))
  list(f = f, y = y)
}

# Binary-result invariants shared by every binary engine; returns the names of
# the violated ones (character(0) when the result is valid).
catb_violations <- function(r, f, y, max_bins) {
  v <- character()
  if (sum(r$count) != length(f)) v <- c(v, "count_sum")
  if (any(r$count != r$count_pos + r$count_neg)) v <- c(v, "count_pos_neg")
  if (sum(r$count_pos) != sum(y)) v <- c(v, "pos_sum")
  if (any(r$count <= 0)) v <- c(v, "empty_bin")
  if (length(r$bin) > max_bins) v <- c(v, "max_bins")
  if (!all(is.finite(r$woe)) || !all(is.finite(r$iv))) v <- c(v, "non_finite")
  if (abs(r$total_iv - sum(abs(r$iv))) > 1e-10 * max(1, r$total_iv)) v <- c(v, "total_iv")
  if (length(r$woe) > 1) {
    d <- diff(r$woe)
    if (!(all(d >= -1e-12) || all(d <= 1e-12))) v <- c(v, "non_monotone")
  }
  v
}

catb_check_binary <- function(r, f, y, max_bins) {
  expect_identical(catb_violations(r, f, y, max_bins), character())
}

# Run an engine, collecting warnings instead of letting them escape.
catb_run <- function(fn, ...) {
  warns <- character()
  r <- withCallingHandlers(fn(...), warning = function(w) {
    warns <<- c(warns, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(r = r, warns = warns)
}

catb_binary_algos <- c("jedi", "mba", "milp", "mob", "sab")

# ---------------------------------------------------------------------------
# Invariants on random and adversarial inputs
# ---------------------------------------------------------------------------
test_that("binary engines satisfy the output invariants on fuzzed inputs", {
  problems <- character()
  for (s in 1:40) {
    set.seed(s)
    k <- sample(c(4, 9, 30, 120), 1)
    n <- sample(c(60, 400, 1500), 1)
    lv <- sprintf("c%03d", seq_len(k))
    f <- sample(lv, n, TRUE, prob = if (s %% 2) rep(1, k) else 1 / seq_len(k))
    y <- rbinom(n, 1, plogis(rnorm(k, -1, 1.5)[match(f, lv)]))
    if (sum(y) < 5 || sum(1 - y) < 5) next
    mx <- sample(3:6, 1)
    for (a in catb_binary_algos) {
      set.seed(1)
      out <- catb_run(get(paste0("ob_categorical_", a)), f, y,
                      min_bins = 2, max_bins = mx, bin_cutoff = 0.02)
      cats <- unlist(strsplit(out$r$bin, "%;%", fixed = TRUE))
      v <- c(catb_violations(out$r, f, y, mx),
             if (length(out$warns)) "warning",
             if (!setequal(cats, f) || anyDuplicated(cats)) "partition")
      if (length(v)) problems <- c(problems, paste(s, a, v))
    }
  }
  expect_identical(problems, character())
})

test_that("binary engines handle high-cardinality features (700 levels)", {
  set.seed(3)
  k <- 700
  lv <- sprintf("lvl%04d", seq_len(k))
  f <- sample(lv, 6000, TRUE, prob = 1 / seq_len(k))
  y <- rbinom(6000, 1, plogis(rnorm(k, -1, 1)[match(f, lv)]))
  for (a in catb_binary_algos) {
    fn <- get(paste0("ob_categorical_", a))
    for (cut in c(0.05, 1e-4)) {
      set.seed(1)
      expect_no_warning(r <- fn(f, y, bin_cutoff = cut))
      catb_check_binary(r, f, y, 5)
      expect_setequal(unlist(strsplit(r$bin, "%;%", fixed = TRUE)), unique(f))
    }
  }
})

test_that("category names containing the separator are kept intact", {
  set.seed(5)
  lv <- c("a%;%b", "c", "d%;%", "%;%e", "f g", "h", "i", "j")
  f <- sample(lv, 800, TRUE)
  y <- rbinom(800, 1, c(.1, .2, .3, .4, .5, .6, .7, .8)[match(f, lv)])
  for (a in catb_binary_algos) {
    fn <- get(paste0("ob_categorical_", a))
    set.seed(1)
    r <- fn(f, y, bin_separator = "|||")
    catb_check_binary(r, f, y, 5)
    expect_setequal(unlist(strsplit(r$bin, "|||", fixed = TRUE)), lv)
  }
})

test_that("NA in the feature becomes the 'NA' category", {
  set.seed(6)
  f <- sample(c("a", "b", "c", "d", NA), 500, TRUE)
  y <- rbinom(500, 1, 0.3)
  for (a in catb_binary_algos) {
    fn <- get(paste0("ob_categorical_", a))
    set.seed(1)
    r <- fn(f, y, max_bins = 5)
    expect_true("NA" %in% unlist(strsplit(r$bin, "%;%", fixed = TRUE)))
    expect_equal(sum(r$count), 500)
  }
})

test_that("tiny and degenerate inputs give valid results or the documented warning", {
  # one or two categories
  r <- ob_categorical_jedi(rep(c("a", "b"), 10), rep(0:1, 10))
  expect_equal(sum(r$count), 20)
  r <- ob_categorical_mob(rep("a", 20), rep(0:1, 10))
  expect_equal(r$count, 20L)
  # fewer than 5 events: documented warning
  f <- rep(letters[1:6], 5)
  y <- c(1L, 1L, rep(0L, 28))
  for (a in catb_binary_algos) {
    fn <- get(paste0("ob_categorical_", a))
    set.seed(1)
    expect_warning(r <- fn(f, y), "fewer than 5")
    catb_check_binary(r, f, y, 5)
  }
})

test_that("a target outside {0, 1} is rejected, not counted", {
  f <- rep(letters[1:6], 20)
  set.seed(2)
  y <- rbinom(120, 1, 0.4)
  y[c(50, 70)] <- 2L
  for (a in catb_binary_algos) {
    fn <- get(paste0("ob_categorical_", a))
    expect_error(fn(f, y))
  }
})

# ---------------------------------------------------------------------------
# jedi
# ---------------------------------------------------------------------------
test_that("jedi keeps the most frequent rare categories, not the rarest", {
  # four frequent categories and 20 rare ones of sizes 1..20. Up to min_bins
  # rare categories stay separate; the old pre-binning picked the rarest
  # (1, 2 and 3 observations) and they survived as bins of their own.
  set.seed(8)
  sz <- 1:20
  f <- c(rep(c("A", "B", "C", "D"), each = 240), rep(sprintf("r%02d", sz), sz))
  y <- c(rbinom(960, 1, rep(c(.1, .3, .5, .7), each = 240)),
         unlist(lapply(sz, function(s) rep(c(1L, 0L), length.out = s))))
  y[961:966] <- 1L
  r <- ob_categorical_jedi(f, y, max_bins = 8)
  catb_check_binary(r, f, y, 8)
  expect_gte(min(r$count), 18)
})

test_that("jedi merges the WoE-adjacent pair with the smallest IV loss", {
  # With more than 10 pre-bins the old IV cache returned stale totals.
  set.seed(9)
  k <- 25
  f <- rep(sprintf("c%02d", 1:k), each = 80)
  y <- rbinom(length(f), 1, rep(seq(0.05, 0.9, length.out = k), each = 80))
  r <- ob_categorical_jedi(f, y, max_bins = 5, bin_cutoff = 0.01)
  # reference: greedy minimum IV loss on the WoE-sorted categories
  tp <- sum(y); tn <- length(y) - tp
  pp <- 0.5 * tp / (tp + tn); pn <- 0.5 - pp
  ivt <- function(p, q) {
    a <- (p + pp) / (tp + 0.5); b <- (q + pn) / (tn + 0.5)
    (a - b) * log(a / b)
  }
  pos <- tapply(y, f, sum); neg <- tapply(1 - y, f, sum)
  o <- order(log(((pos + pp) / (tp + .5)) / ((neg + pn) / (tn + .5))))
  P <- as.numeric(pos[o]); N <- as.numeric(neg[o])
  while (length(P) > 5) {
    loss <- sapply(seq_len(length(P) - 1), function(i)
      ivt(P[i], N[i]) + ivt(P[i + 1], N[i + 1]) - ivt(P[i] + P[i + 1], N[i] + N[i + 1]))
    i <- which.min(loss)
    P[i] <- P[i] + P[i + 1]; N[i] <- N[i] + N[i + 1]
    P <- P[-(i + 1)]; N <- N[-(i + 1)]
  }
  expect_equal(sort(r$count), sort(as.integer(P + N)))
})

test_that("jedi rejects a non-positive min_bins", {
  expect_error(ob_categorical_jedi(rep(letters[1:5], 20), rep(0:1, 50), min_bins = 0))
})

# ---------------------------------------------------------------------------
# jedi_mwoe
# ---------------------------------------------------------------------------
test_that("jedi_mwoe returns consistent multiclass results (3-5 classes)", {
  for (K in 3:5) {
    set.seed(10 + K)
    lv <- sprintf("c%03d", 1:60)
    f <- sample(lv, 3000, TRUE, prob = 1 / seq_along(lv))
    ym <- sample(0:(K - 1), 3000, TRUE)
    ym[f %in% lv[1:3] & ym == K - 1] <- 0L # a class absent from some categories
    for (cut in c(0.05, 0.001)) {
      expect_no_warning(r <- ob_categorical_jedi_mwoe(f, ym, bin_cutoff = cut))
      expect_equal(r$n_classes, K)
      expect_equal(dim(r$woe), c(length(r$bin), K))
      expect_equal(sum(r$count), 3000)
      expect_equal(rowSums(r$class_counts), r$count)
      expect_equal(colSums(r$class_counts), as.vector(table(factor(ym, 0:(K - 1)))))
      expect_true(all(is.finite(r$woe)) && all(is.finite(r$iv)))
      expect_lte(length(r$bin), 5)
      expect_setequal(unlist(strsplit(r$bin, "%;%", fixed = TRUE)), unique(f))
    }
  }
})

test_that("jedi_mwoe with max_bins = 1 terminates", {
  # merge_adjacent_bins() was a no-op on two bins, so the final
  # `while (bins > max_bins)` loop spun forever for these category counts.
  for (k in c(3, 5, 7)) {
    set.seed(4)
    f <- sample(sprintf("c%02d", 1:k), 800, TRUE)
    ym <- sample(0:2, 800, TRUE)
    r <- ob_categorical_jedi_mwoe(f, ym, min_bins = 1, max_bins = 1)
    expect_length(r$bin, 1)
    expect_equal(r$count, 800L)
  }
})

test_that("jedi_mwoe keeps the most frequent rare categories, not the rarest", {
  set.seed(4)
  f <- sample(sprintf("c%03d", 1:40), 3000, TRUE)
  ym <- sample(0:3, 3000, TRUE)
  r <- ob_categorical_jedi_mwoe(f, ym)
  cnt <- table(f)
  single <- r$bin[!grepl("%;%", r$bin, fixed = TRUE)]
  # the categories left on their own are among the most frequent ones
  expect_true(all(cnt[single] >= sort(cnt, decreasing = TRUE)[3]))
})

test_that("jedi_mwoe pre-binning handles 1000 categories", {
  set.seed(12)
  lv <- sprintf("c%04d", 1:1000)
  f <- sample(lv, 20000, TRUE)
  ym <- sample(0:3, 20000, TRUE)
  r <- ob_categorical_jedi_mwoe(f, ym, bin_cutoff = 1e-4)
  expect_equal(sum(r$count), 20000)
  expect_lte(length(r$bin), 5)
  expect_setequal(unlist(strsplit(r$bin, "%;%", fixed = TRUE)), lv)
})

# ---------------------------------------------------------------------------
# mba
# ---------------------------------------------------------------------------
test_that("mba reduces to max_bins without spurious warnings", {
  # identical event rates: merges that cost (almost) no IV used to stop the
  # reduction and raise "Could not reduce number of bins" once per merge.
  f <- rep(sprintf("c%02d", 1:30), each = 100)
  y <- rep(c(rep(1L, 30), rep(0L, 70)), 30)
  expect_no_warning(r <- ob_categorical_mba(f, y, bin_cutoff = 0.01))
  catb_check_binary(r, f, y, 5)
})

test_that("mba enforces bin_cutoff", {
  set.seed(24)
  k <- 40
  f <- sample(sprintf("c%02d", 1:k), 600, TRUE, prob = 1 / (1:k))
  y <- rbinom(600, 1, plogis(rnorm(k, -1, 1.5)[as.integer(sub("c", "", f))]))
  expect_no_warning(r <- ob_categorical_mba(f, y))
  catb_check_binary(r, f, y, 5)
  expect_true(all(r$count >= 0.05 * 600))
})

# ---------------------------------------------------------------------------
# milp
# ---------------------------------------------------------------------------
test_that("milp keeps well-separated categories apart and uses max_bins", {
  d <- catb_separated()
  r <- ob_categorical_milp(d$f, d$y)
  catb_check_binary(r, d$f, d$y, 5)
  expect_length(r$bin, 5)
  # the two extremes were merged into one bin (IV 0.007)
  expect_false(any(grepl("c1", r$bin) & grepl("c8", r$bin)))
  expect_gt(r$total_iv, 0.9)
})

test_that("milp never returns more than max_bins bins", {
  f <- rep(sprintf("c%02d", 1:30), each = 100)
  y <- rep(c(rep(1L, 30), rep(0L, 70)), 30)
  r <- ob_categorical_milp(f, y, bin_cutoff = 0.01)
  catb_check_binary(r, f, y, 5)
})

# ---------------------------------------------------------------------------
# mob
# ---------------------------------------------------------------------------
test_that("mob does not treat every category as rare", {
  # every initial bin had count 1, so all categories were merged down towards
  # min_bins whatever bin_cutoff said
  d <- catb_separated()
  r <- ob_categorical_mob(d$f, d$y)
  catb_check_binary(r, d$f, d$y, 5)
  expect_length(r$bin, 5)
  expect_gt(r$total_iv, 0.9)
})

# ---------------------------------------------------------------------------
# sab
# ---------------------------------------------------------------------------
test_that("sab actually searches and is reproducible under set.seed()", {
  d <- catb_separated()
  set.seed(1)
  r1 <- ob_categorical_sab(d$f, d$y)
  set.seed(1)
  r2 <- ob_categorical_sab(d$f, d$y)
  expect_identical(r1, r2)
  catb_check_binary(r1, d$f, d$y, 5)
  # the convergence check ran from iteration 0 and on a 10-iteration window,
  # so the search typically stopped after 1-120 iterations
  expect_gt(r1$iterations, 200)
  expect_gt(r1$total_iv, 0.95)
})

test_that("sab honours min_bins when every category can be its own bin", {
  # three categories, each above bin_cutoff; merging the two nearly pure ones
  # raises the smoothed IV, and 2 bins came back for about 1 seed in 5
  f <- c(rep("A", 222), rep("B", 101), rep("C", 77))
  y <- c(rep(1L, 39), rep(0L, 183), 1L, rep(0L, 100), 1L, 1L, rep(0L, 75))
  set.seed(161)
  o <- sample(400)
  f <- f[o]
  y <- y[o]
  nb <- vapply(1:12, function(s) {
    set.seed(s)
    length(ob_categorical_sab(f, y)$bin)
  }, integer(1))
  expect_true(all(nb == 3L))
})

test_that("sab validates its annealing parameters", {
  f <- rep(letters[1:6], 20)
  y <- rep(0:1, 60)
  expect_error(ob_categorical_sab(f, y, initial_temperature = 0))
  expect_error(ob_categorical_sab(f, y, cooling_rate = 1.5))
})
