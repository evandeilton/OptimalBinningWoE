# Regression and invariant tests for the "core" audit: gains tables
# (OB_Utils.cpp), obcorr() (OB_Correlation.cpp), the internal L-BFGS logistic
# regression (OB_LogisticRegression.cpp), ob_numerical_udt() (OBN_UDT_v5.cpp)
# and the shared header src/common/optimal_binning_common.h.

# ---------------------------------------------------------------------------
# Gains tables
# ---------------------------------------------------------------------------
gains_reference <- function(id, cnt, pos, neg) {
  o <- order(id, method = "radix") # stable, NA last
  cnt <- cnt[o]; pos <- pos[o]; neg <- neg[o]
  pp <- pos / sum(pos); np <- neg / sum(neg)
  woe <- ifelse(pp > 0 & np > 0, log(pp / np), 0)
  list(
    woe = woe, iv = ifelse(pp > 0 & np > 0, (pp - np) * woe, 0),
    ks = abs(cumsum(pos) / sum(pos) - cumsum(neg) / sum(neg)),
    lift = (pos / cnt) / (sum(pos) / sum(cnt))
  )
}

test_that("obwoe_gains_score matches an independent computation, without warnings", {
  set.seed(1)
  x <- rnorm(500)
  y <- rbinom(500, 1, plogis(x))
  b <- ob_numerical_kmb(feature = x, target = y, min_bins = 3, max_bins = 5)
  expect_no_warning(g <- obwoe_gains_score(b))
  ref <- gains_reference(b$id, b$count, b$count_pos, b$count_neg)
  for (nm in names(ref)) expect_equal(g[[nm]], ref[[nm]], tolerance = 1e-12)
  expect_equal(sum(g$count), 500)

  # Shuffled ids, integer storage, zero cells
  for (k in 1:20) {
    set.seed(100 + k)
    m <- sample(2:25, 1)
    pos <- rpois(m, 10); neg <- rpois(m, 10)
    pos[1] <- 0L
    id <- sample(m)
    expect_no_warning(g <- obwoe_gains_score(list(
      id = id, bin = paste0("B", id), count = pos + neg,
      count_pos = pos, count_neg = neg
    )))
    ref <- gains_reference(id, pos + neg, pos, neg)
    expect_equal(g$id, sort(as.numeric(id)))
    for (nm in c("woe", "iv", "ks")) expect_equal(g[[nm]], ref[[nm]], tolerance = 1e-12)
  }
})

test_that("obwoe_gains_score sorts a missing id last and keeps tied ids in input order", {
  g <- obwoe_gains_score(list(
    id = c(2, NA, 1), bin = c("b", "na", "a"), count = c(10, 5, 10),
    count_pos = c(3, 1, 6), count_neg = c(7, 4, 4)
  ))
  expect_identical(g$bin, c("a", "b", "na"))

  m <- 40L
  id <- c(1L, 1L, 3:m) # 40 rows, first two share an id
  g <- obwoe_gains_score(list(
    id = id, bin = paste0("B", seq_len(m)), count = rep(10, m),
    count_pos = rep(4, m), count_neg = rep(6, m)
  ))
  expect_identical(g$bin[1:2], c("B1", "B2"))
})

test_that("obwoe_gains_variable no longer pools WoE values equal to 6 decimals", {
  df <- data.frame(
    feature = 1:4, bin = c("a", "a", "b", "b"),
    woe = c(0.1234561, 0.1234561, 0.1234564, 0.1234564), idbin = c(1, 1, 2, 2)
  )
  g <- obwoe_gains_variable(df, c(0, 1, 1, 1), group_var = "woe")
  expect_equal(nrow(g), 2L)
  expect_equal(g$count, c(2, 2))
  expect_equal(g$pos, c(1, 2))
  expect_false(anyDuplicated(g$bin) > 0)
})

test_that("obwoe_gains_variable aggregates exactly like tapply", {
  set.seed(7)
  n <- 3000
  idb <- sample(8, n, TRUE)
  df <- data.frame(feature = 0, bin = paste0("b", idb), woe = rnorm(8)[idb], idbin = idb)
  y <- rbinom(n, 1, 0.3)
  for (gv in c("bin", "woe", "idbin")) {
    expect_no_warning(g <- obwoe_gains_variable(df, y, group_var = gv))
    expect_equal(g$id, as.numeric(1:8))
    expect_equal(g$pos, as.numeric(tapply(y, idb, sum)))
    expect_equal(g$count, as.numeric(tabulate(idb)))
  }
})

# ---------------------------------------------------------------------------
# obcorr()
# ---------------------------------------------------------------------------
test_that("distance correlation is the Szekely-Rizzo-Bakirov statistic", {
  dcor_ref <- function(x, y) {
    A <- as.matrix(dist(x)); B <- as.matrix(dist(y))
    A <- A - outer(rowMeans(A), colMeans(A), "+") + mean(A)
    B <- B - outer(rowMeans(B), colMeans(B), "+") + mean(B)
    sqrt(mean(A * B) / sqrt(mean(A * A) * mean(B * B)))
  }
  set.seed(1)
  x <- rnorm(120)
  df <- data.frame(a = x, b = x, c = 10 * x, d = rnorm(120), e = exp(x))
  r <- obcorr(df, method = "distance", threads = 1)
  # A variable with itself (or a rescaled copy) has dCor = 1; it used to be
  # 1.69 and 0.53 here.
  expect_equal(r$distance[r$x == "a" & r$y == "b"], 1, tolerance = 1e-12)
  expect_equal(r$distance[r$x == "a" & r$y == "c"], 1, tolerance = 1e-12)
  for (k in seq_len(nrow(r))) {
    expect_equal(r$distance[k], dcor_ref(df[[r$x[k]]], df[[r$y[k]]]), tolerance = 1e-10)
  }
  expect_true(all(r$distance >= 0 & r$distance <= 1))
})

test_that("rank and robust correlations are invariant to the unit of measurement", {
  set.seed(2)
  x <- rnorm(60); y <- x + rnorm(60)
  r1 <- obcorr(data.frame(x = x, y = y), threads = 1)
  r2 <- obcorr(data.frame(x = x * 1e-12, y = y * 1e-12), threads = 1)
  r3 <- obcorr(data.frame(x = x * 1e9, y = y), threads = 1)
  for (m in c("pearson", "spearman", "kendall", "hoeffding", "distance", "biweight", "pbend")) {
    expect_false(is.na(r2[[m]]), info = m)
    expect_equal(r2[[m]], r1[[m]], tolerance = 1e-9, info = m)
    expect_equal(r3[[m]], r1[[m]], tolerance = 1e-9, info = m)
  }
})

test_that("obcorr agrees with stats::cor (pairwise complete) and Hmisc::hoeffd", {
  set.seed(3)
  n <- 80
  x <- rnorm(n); y <- round(x + rnorm(n), 1); z <- rexp(n)
  y[c(3, 9, 40)] <- NA
  x[c(5, 9)] <- NA
  df <- data.frame(x = x, y = y, z = z)
  r <- obcorr(df, threads = 1)
  for (k in seq_len(nrow(r))) {
    a <- df[[r$x[k]]]; b <- df[[r$y[k]]]
    cc <- complete.cases(a, b)
    expect_equal(r$pearson[k], cor(a[cc], b[cc]), tolerance = 1e-12)
    # Spearman ranks the complete pairs only (it used to rank every
    # non-missing value of each column)
    expect_equal(r$spearman[k], cor(a[cc], b[cc], method = "spearman"), tolerance = 1e-12)
    expect_equal(r$kendall[k], cor(a[cc], b[cc], method = "kendall"), tolerance = 1e-12)
    if (requireNamespace("Hmisc", quietly = TRUE)) {
      expect_equal(r$hoeffding[k], Hmisc::hoeffd(a[cc], b[cc])$D[1, 2], tolerance = 1e-10)
    }
  }
})

test_that("Pearson survives a large offset (no catastrophic cancellation)", {
  set.seed(4)
  x <- rnorm(100); y <- x + rnorm(100)
  r <- obcorr(data.frame(x = x + 1e9, y = y + 1e9), method = "pearson", threads = 1)
  expect_equal(r$pearson, cor(x, y), tolerance = 1e-8)
})

test_that("obcorr rejects an unknown method, skips factors and is thread-invariant", {
  df <- data.frame(a = rnorm(30), b = rnorm(30), f = factor(sample(letters[1:3], 30, TRUE)), c = rnorm(30))
  expect_error(obcorr(df, method = "foo"), "method must be one of")
  r <- obcorr(df, threads = 1)
  expect_false(any(c(r$x, r$y) == "f"))
  expect_equal(nrow(r), 3L)
  expect_identical(obcorr(df, threads = 1), obcorr(df, threads = 2))
  expect_no_warning(obcorr(df))
})

test_that("obcorr handles integer, logical and constant columns and every method shape", {
  set.seed(8)
  n <- 40
  df <- data.frame(
    i = sample(1:5, n, TRUE), l = sample(c(TRUE, FALSE, NA), n, TRUE),
    d = rnorm(n), k = rep(3, n)
  )
  df$i[2] <- NA
  r <- obcorr(df, threads = 1)
  expect_equal(nrow(r), 6L)
  cc <- complete.cases(df$i, df$l)
  expect_equal(r$pearson[r$x == "i" & r$y == "l"],
               cor(df$i[cc], as.numeric(df$l[cc])), tolerance = 1e-12)
  # a constant column: undefined correlations, zero distance correlation
  ck <- r[r$y == "k", ]
  expect_true(all(is.na(ck$pearson)) && all(is.na(ck$spearman)) && all(is.na(ck$kendall)))
  expect_equal(ck$distance, rep(0, 3))
  shapes <- list(
    pearson = "pearson", spearman = "spearman", kendall = "kendall",
    hoeffding = "hoeffding", distance = "distance", biweight = "biweight",
    pbend = "pbend", robust = c("biweight", "pbend"),
    alternative = c("hoeffding", "distance")
  )
  for (m in names(shapes)) {
    rm <- obcorr(df, method = m, threads = 1)
    expect_identical(names(rm), c("x", "y", shapes[[m]]), info = m)
    for (cl in shapes[[m]]) expect_identical(rm[[cl]], r[[cl]], info = m)
  }
})

test_that("gains by bin merges identical labels carried by distinct CHARSXPs", {
  a <- "caf\xe9"
  Encoding(a) <- "latin1"
  b <- a
  Encoding(b) <- "bytes"
  df <- data.frame(feature = 1:4, woe = 0, idbin = c(1, 1, 2, 2))
  df$bin <- c(a, b, "z", "z")
  g <- obwoe_gains_variable(df, c(0, 1, 1, 0), group_var = "bin")
  expect_equal(nrow(g), 2L)
  expect_equal(g$count, c(2, 2))
})

# ---------------------------------------------------------------------------
# Internal logistic regression
# ---------------------------------------------------------------------------
test_that("sparse and dense designs give the same logistic fit", {
  skip_if_not_installed("Matrix")
  set.seed(12)
  X <- cbind(1, matrix(rbinom(400 * 3, 1, 0.3), 400, 3))
  y <- rbinom(400, 1, plogis(X %*% c(-0.5, 1, -1, 0.5)))
  fd <- OptimalBinningWoE:::.ob_fit_logistic_regression(X, y)
  fs <- OptimalBinningWoE:::.ob_fit_logistic_regression(methods::as(X, "dgCMatrix"), y)
  expect_equal(fs$coefficients, fd$coefficients, tolerance = 1e-8)
  expect_equal(fs$se, fd$se, tolerance = 1e-6)
})

test_that("logistic regression reports real iterations and convergence", {
  set.seed(5)
  X <- cbind(1, matrix(rnorm(300 * 3), 300, 3))
  y <- rbinom(300, 1, plogis(X %*% c(-0.5, 1, -1, 0.5)))
  expect_no_warning(f <- OptimalBinningWoE:::.ob_fit_logistic_regression(X, y))
  expect_true(f$convergence)
  expect_true(f$iterations >= 1L && f$iterations < 300L)
  g <- glm.fit(X, y, family = binomial())
  expect_equal(as.numeric(f$coefficients), as.numeric(g$coefficients), tolerance = 1e-4)
})

test_that("standard errors are not dropped for well-conditioned small-scale designs", {
  set.seed(5)
  n <- 500; p <- 12
  X <- cbind(1, matrix(rnorm(n * p), n, p) * 0.01)
  y <- rbinom(n, 1, plogis(X %*% c(0.2, rep(30, p))))
  f <- OptimalBinningWoE:::.ob_fit_logistic_regression(X, y)
  g <- glm.fit(X, y, family = binomial())
  se_glm <- sqrt(diag(chol2inv(qr.R(g$qr))))
  expect_false(anyNA(f$se))
  expect_equal(as.numeric(f$se), se_glm, tolerance = 1e-3)
})

test_that("logistic regression accepts integer designs and rejects missing values", {
  set.seed(6)
  Xi <- cbind(1L, sample(0:3, 100, TRUE))
  yi <- rbinom(100, 1, 0.4)
  expect_no_error(f <- OptimalBinningWoE:::.ob_fit_logistic_regression(Xi, yi))
  expect_length(f$coefficients, 2L)
  Xn <- cbind(1, c(NA, rnorm(99)))
  expect_error(OptimalBinningWoE:::.ob_fit_logistic_regression(Xn, yi), "missing")
})

test_that("separated data give a finite log-likelihood and no R warning", {
  set.seed(3)
  x <- rnorm(100) * 1e3
  y <- as.numeric(x > 0)
  expect_no_warning(f <- OptimalBinningWoE:::.ob_fit_logistic_regression(cbind(1, x), y))
  expect_true(is.finite(f$loglikelihood))
  expect_true(is.logical(f$convergence) && is.integer(f$iterations))
})

# ---------------------------------------------------------------------------
# ob_numerical_udt()
# ---------------------------------------------------------------------------
udt_invariants <- function(r, n, max_bins) {
  reg <- r$bin != "NA"
  expect_equal(sum(r$count), n)
  expect_equal(r$count, r$count_pos + r$count_neg)
  expect_lte(sum(reg), max_bins)
  expect_true(all(is.finite(r$woe)))
  expect_true(all(is.finite(r$iv)))
  expect_true(!is.unsorted(r$cutpoints, strictly = TRUE))
  expect_length(r$cutpoints, max(0L, sum(reg) - 1L))
  expect_true(r$gini >= -1 - 1e-12 && r$gini <= 1 + 1e-12)
}

test_that("UDT terminates when min_bins exceeds the number of distinct values", {
  # 3 <= #distinct < min_bins used to spin forever in the min_bins split loop
  set.seed(1)
  x <- sample(1:3, 300, TRUE)
  y <- rbinom(300, 1, 0.3)
  r <- ob_numerical_udt(x, y, min_bins = 5, max_bins = 6)
  expect_equal(length(r$bin), 3L)
  udt_invariants(r, 300, 6)
  x4 <- sample(1:4, 300, TRUE)
  r4 <- ob_numerical_udt(x4, y, min_bins = 6, max_bins = 8)
  udt_invariants(r4, 300, 8)
})

test_that("UDT Gini is 2 * AUC - 1 of the binned WoE", {
  set.seed(3)
  x <- rnorm(2000)
  y <- rbinom(2000, 1, plogis(2 * x))
  r <- ob_numerical_udt(x, y)
  s <- r$woe[findInterval(x, r$cutpoints, left.open = TRUE) + 1]
  auc <- mean(outer(s[y == 1], s[y == 0], ">") + 0.5 * outer(s[y == 1], s[y == 0], "=="))
  expect_equal(r$gini, 2 * auc - 1, tolerance = 1e-12)
  expect_lt(r$gini, 1)
})

test_that("UDT merges the adjacent pair with the smallest IV loss", {
  # Bins 3 and 4 have identical event rates, so merging them loses (almost) no
  # IV; the old criterion merged the pair with the smallest IV *sum* (1 and 2).
  v <- rep(1:5, each = 200)
  rate <- c(0.50, 0.52, 0.10, 0.10, 0.90)
  y <- unlist(lapply(rate, function(r) c(rep(1, r * 200), rep(0, 200 - r * 200))))
  r <- ob_numerical_udt(v, y, min_bins = 2, max_bins = 4, bin_cutoff = 0.01)
  expect_equal(r$cutpoints, c(1.5, 2.5, 4.5))
  udt_invariants(r, 1000, 4)
})

test_that("UDT handles an all-missing feature and zero smoothing", {
  set.seed(4)
  r <- ob_numerical_udt(rep(NA_real_, 50), rbinom(50, 1, 0.5))
  expect_identical(r$bin, "NA")
  expect_equal(sum(r$count), 50) # was 100 (a "(nan;nan]" bin plus the NA bin)

  x <- rep(c(1, 2), each = 50)
  y <- c(rep(0, 50), rep(0:1, 25))
  r <- ob_numerical_udt(x, y, laplace_smoothing = 0)
  expect_true(all(is.finite(r$woe)))
  expect_true(all(is.finite(r$iv)))
})

test_that("UDT puts +-Inf in the edge bins and only NaN in the NA bin", {
  set.seed(9)
  x <- c(rnorm(400), -Inf, -Inf, Inf, NA, NaN)
  y <- c(rbinom(400, 1, 0.4), 1L, 0L, 1L, 0L, 1L)
  r <- ob_numerical_udt(x, y, min_bins = 3, max_bins = 5)
  reg <- r$bin != "NA"
  expect_equal(r$count[!reg], 2L) # NA and NaN only (the Infs used to land here)
  expect_true(all(is.finite(r$cutpoints)))
  # counts agree with routing every finite/infinite value through the cut points
  b <- findInterval(x[!is.na(x)], r$cutpoints, left.open = TRUE) + 1L
  expect_equal(r$count[reg], tabulate(b, nbins = sum(reg)))
  expect_equal(sum(r$count), length(x))
  expect_true(all(is.finite(r$woe)))

  # few-distinct paths
  r2 <- ob_numerical_udt(c(rep(1, 50), rep(2, 50), -Inf, Inf, NA), c(rep(0:1, 50), 1L, 0L, 1L))
  expect_equal(r2$count, c(51L, 51L, 1L))
  r3 <- ob_numerical_udt(c(rep(Inf, 10), rep(-Inf, 10), NA), c(rep(0:1, 10), 1L))
  expect_equal(sum(r3$count), 21L)
  expect_equal(r3$count[r3$bin == "NA"], 1L)
})

test_that("UDT invariants hold across random and adversarial inputs", {
  for (k in 1:30) {
    set.seed(700 + k)
    n <- sample(c(20, 100, 400), 1)
    x <- switch(k %% 6 + 1,
      rnorm(n), rlnorm(n, 0, 2), round(rnorm(n), 1), sample(1:6, n, TRUE),
      c(rnorm(n - 4), 1e300, -1e300, Inf, NA), rcauchy(n)
    )
    y <- rbinom(n, 1, 0.35); y[1:2] <- c(0L, 1L)
    mb <- sample(3:6, 1)
    expect_no_warning(r <- ob_numerical_udt(x, y, min_bins = 2, max_bins = mb,
      monotonicity_direction = c("none", "auto", "increasing")[k %% 3 + 1]))
    udt_invariants(r, n, mb)
  }
})

# ---------------------------------------------------------------------------
# Shared header: gaussian_kde_grid()/gaussian_kde_sorted() on an overflowing span
# ---------------------------------------------------------------------------
test_that("KDE-based binning copes with a feature range that overflows", {
  # hi - lo = Inf used to give dx = Inf and a size_t conversion of NaN
  # (undefined behaviour) inside the shared KDE helpers.
  set.seed(11)
  x <- c(rnorm(300), -1.5e308, 1.5e308)
  y <- rbinom(302, 1, 0.4)
  for (f in list(ob_numerical_ldb, ob_numerical_lpdb)) {
    expect_no_warning(r <- f(x, y))
    expect_equal(sum(r$count), 302)
  }
})
