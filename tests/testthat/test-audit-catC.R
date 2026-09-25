# Audit (group catC): ob_categorical_sblp / swb / sketch / udt,
# ob_apply_woe_cat / ob_apply_woe_num, ob_check_distincts, ob_cutpoints_*,
# ob_preprocess. Regression tests for the bugs fixed in this audit plus
# invariant tests for every algorithm.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
catC_split <- function(label, sep = "%;%") {
  strsplit(paste0(label, sep), sep, fixed = TRUE)[[1]]
}

catC_data <- function(n, L, seed, na = FALSE, zipf = FALSE) {
  set.seed(seed)
  lv <- sprintf("L%03d", seq_len(L))
  p <- if (zipf) 1 / seq_len(L)^1.1 else rep(1, L)
  f <- sample(lv, n, replace = TRUE, prob = p)
  eta <- rnorm(L)[match(f, lv)]
  if (na) f[sample(n, max(1L, n %/% 10))] <- NA
  eta[is.na(eta)] <- 0.3
  y <- rbinom(n, 1, plogis(eta))
  y[1:2] <- c(0L, 1L)
  list(f = f, y = as.integer(y))
}

catC_check_fit <- function(r, f, y, max_bins) {
  ff <- f
  ff[is.na(ff)] <- "NA"
  expect_equal(sum(r$count), length(f))
  expect_equal(sum(r$count_pos), sum(y))
  expect_equal(r$count_pos + r$count_neg, r$count)
  expect_lte(length(r$bin), max_bins)
  expect_true(all(r$count > 0))
  expect_true(all(is.finite(r$woe)))
  expect_true(all(is.finite(r$iv)))
  if (length(r$woe) > 2) {
    d <- diff(r$woe)
    expect_true(all(d >= -1e-10) || all(d <= 1e-10))
  }
  cats <- unlist(lapply(r$bin, catC_split))
  expect_setequal(cats, unique(ff))
  expect_false(anyDuplicated(cats) > 0)
  # Scoring the training data reproduces the fitted WoE of each category
  ap <- ob_apply_woe_cat(r, f)
  bin_of <- setNames(rep(r$bin, lengths(lapply(r$bin, catC_split))), cats)
  expect_equal(ap$woe, unname(r$woe[match(bin_of[ff], r$bin)]))
}

algos_catC <- list(
  sblp = ob_categorical_sblp, swb = ob_categorical_swb,
  sketch = ob_categorical_sketch, udt = ob_categorical_udt
)

# ---------------------------------------------------------------------------
# Invariants for the four categorical algorithms
# ---------------------------------------------------------------------------
test_that("catC algorithms satisfy the binning invariants on varied inputs", {
  cases <- list(
    catC_data(300, 6, 1),
    catC_data(1500, 40, 2, zipf = TRUE),
    catC_data(500, 12, 3, na = TRUE),
    catC_data(40, 30, 4),
    catC_data(3000, 600, 5, zipf = TRUE)
  )
  for (d in cases) {
    for (nm in names(algos_catC)) {
      r <- expect_no_warning(algos_catC[[nm]](d$f, d$y))
      catC_check_fit(r, d$f, d$y, max_bins = 5)
      r2 <- expect_no_warning(algos_catC[[nm]](d$f, d$y,
        min_bins = 2, max_bins = 4,
        bin_cutoff = 0.02, max_n_prebins = 10
      ))
      catC_check_fit(r2, d$f, d$y, max_bins = 4)
    }
  }
})

test_that("catC algorithms handle empty-string and separator-bearing categories", {
  set.seed(6)
  lv <- c("", "a%;%b", "a%;%c", " pad ", "x", "y", "z")
  f <- sample(lv, 700, replace = TRUE)
  y <- rbinom(700, 1, plogis(c(-1, 0.5, 1, -0.5, 0, 1.5, -2)[match(f, lv)]))
  for (nm in names(algos_catC)) {
    r <- expect_no_warning(algos_catC[[nm]](f, y, bin_cutoff = 0.01))
    expect_equal(sum(r$count), 700)
    ap <- ob_apply_woe_cat(r, f)
    expect_false(anyNA(ap$woe))
    # every row gets the WoE of the bin that counted it
    lab <- vapply(f, function(v) {
      r$bin[which(vapply(r$bin, function(b) {
        grepl(paste0("%;%", v, "%;%"), paste0("%;%", b, "%;%"), fixed = TRUE)
      }, logical(1)))[1]]
    }, character(1))
    expect_equal(ap$woe, unname(r$woe[match(lab, r$bin)]))
  }
})

# ---------------------------------------------------------------------------
# SBLP
# ---------------------------------------------------------------------------
test_that("[catC] SBLP keeps a genuine empty-string category and its counts", {
  set.seed(7)
  f <- c("", "", "", sample(sprintf("c%02d", 1:24), 57, replace = TRUE))
  y <- rbinom(60, 1, 0.4)
  y[1:2] <- 0:1
  r <- ob_categorical_sblp(f, y)
  expect_equal(sum(r$count), 60)
  expect_true("" %in% unlist(lapply(r$bin, catC_split)))
})

test_that("[catC] SBLP keeps category names that contain the separator intact", {
  f <- rep(c("a%;%b", "c", "d", "e", "f", "g"), each = 20)
  y <- rep(c(1, 0, 1, 1, 0, 0, 1, 0, 0, 1), 12)
  r <- ob_categorical_sblp(f, y)
  expect_true(any(grepl("a%;%b", r$bin, fixed = TRUE)))
  ap <- ob_apply_woe_cat(r, "a%;%b")
  expect_false(is.na(ap$woe))
})

test_that("[catC] SBLP never exceeds max_bins and reports monotonic WoE", {
  f <- c("g", "g", "b", "b", "d", "d", "c", "e", "d", "f", "d", "h")
  y <- c(0L, 0L, 1L, 0L, 0L, 1L, 0L, 1L, 1L, 0L, 1L, 0L)
  r <- ob_categorical_sblp(f, y, min_bins = 4, max_bins = 5, alpha = 2, bin_cutoff = 0.01)
  expect_lte(length(r$bin), 5)
  expect_false(is.unsorted(r$woe))
  expect_equal(sum(r$count), 12)
})

test_that("[catC] SBLP rejects NA and single-class targets", {
  expect_error(ob_categorical_sblp(c("a", "b", "c"), c(0, NA, 1)), "missing")
  expect_error(ob_categorical_sblp(c("a", "b", "c"), c(0, 0, 0)), "both")
})

# ---------------------------------------------------------------------------
# SWB
# ---------------------------------------------------------------------------
test_that("[catC] SWB errors (instead of looping forever) when min_bins > max_bins", {
  set.seed(8)
  f <- sample(letters[1:10], 500, replace = TRUE)
  y <- rbinom(500, 1, 0.4)
  expect_error(ob_categorical_swb(f, y, min_bins = 4, max_bins = 3), "max_bins")
})

test_that("[catC] SWB caps min_bins at the number of categories", {
  f <- rep(c("c1", "c2", "c3"), c(3, 2, 3))
  y <- c(0L, 0L, 0L, 1L, 1L, 1L, 0L, 0L)
  r <- ob_categorical_swb(f, y, min_bins = 4, max_bins = 6)
  expect_true(r$converged)
  expect_equal(sum(r$count), 8)
})

test_that("[catC] SWB/sketch reach min_bins when no category passes bin_cutoff", {
  set.seed(14)
  f <- sample(sprintf("c%02d", 1:40), 400, replace = TRUE)
  y <- rbinom(400, 1, plogis(rnorm(40)[match(f, sprintf("c%02d", 1:40))]))
  for (fn in list(ob_categorical_swb, ob_categorical_sketch)) {
    r <- fn(f, y, min_bins = 3, max_bins = 5, bin_cutoff = 0.2)
    expect_gte(length(r$bin), 3)
    expect_lte(length(r$bin), 5)
    expect_equal(sum(r$count), 400)
  }
})

test_that("[catC] UDT/SWB/sketch merge thousands of frequent pre-bins quickly", {
  set.seed(15)
  L <- 1500
  lv <- sprintf("lv%04d", seq_len(L))
  f <- sample(rep(lv, 20))
  y <- rbinom(length(f), 1, plogis(rnorm(L)[match(f, lv)]))
  for (fn in list(ob_categorical_udt, ob_categorical_swb, ob_categorical_sketch)) {
    t <- system.time(r <- fn(f, y, bin_cutoff = 1e-5, max_n_prebins = 20))[["elapsed"]]
    expect_lt(t, 10)
    expect_lte(length(r$bin), 5)
    expect_equal(sum(r$count), length(f))
  }
})

# ---------------------------------------------------------------------------
# Sketch
# ---------------------------------------------------------------------------
test_that("[catC] sketch bin counts are exact even with many colliding categories", {
  set.seed(9)
  L <- 3000
  lv <- sprintf("lvl%04d", seq_len(L))
  f <- sample(lv, 20000, replace = TRUE, prob = 1 / seq_len(L)^0.8)
  y <- rbinom(20000, 1, plogis(rnorm(L)[match(f, lv)]))
  r <- expect_no_warning(ob_categorical_sketch(f, y))
  expect_equal(sum(r$count), 20000)
  expect_equal(sum(r$count_pos), sum(y))
  expect_equal(r$count_pos + r$count_neg, r$count)
})

test_that("[catC] sketch WoE is monotonic after optimisation, without spurious warnings", {
  set.seed(10)
  f <- sample(sprintf("c%02d", 1:30), 30, replace = TRUE)
  y <- rbinom(30, 1, 0.5)
  y[1:2] <- 0:1
  r <- expect_no_warning(ob_categorical_sketch(f, y))
  d <- diff(r$woe)
  expect_true(all(d >= 0) || all(d <= 0))
})

# ---------------------------------------------------------------------------
# UDT
# ---------------------------------------------------------------------------
test_that("[catC] UDT pools rare categories into one bin (documented behaviour)", {
  f <- rep(c("A", "B", "C", "D", "E", "F"), c(100, 100, 100, 1, 1, 2))
  y <- c(rep(0:1, 50), rep(c(0, 0, 1, 1), 25), rep(c(1, 1, 1, 0), 25), 0, 1, 0, 1)
  r <- ob_categorical_udt(f, y)
  pooled <- r$bin[grepl("%;%", r$bin, fixed = TRUE)]
  expect_length(pooled, 1)
  expect_setequal(catC_split(pooled), c("D", "E", "F"))
  expect_equal(sum(r$count), 304)
})

# ---------------------------------------------------------------------------
# ob_apply_woe_cat
# ---------------------------------------------------------------------------
test_that("[catC] ob_apply_woe_cat scores NA with the fitted missing-value bin", {
  fit <- list(id = 1:3, bin = c("a%;%NA", "b", "c"), woe = c(-1, 0.5, 2))
  r <- ob_apply_woe_cat(fit, c("a", NA, "NA", "", "b", "zz"))
  expect_equal(r$woe, c(-1, -1, -1, -1, 0.5, NA))
  expect_equal(r$bin, c("a%;%NA", "a%;%NA", "a%;%NA", "a%;%NA", "b", "Special"))
  expect_equal(r$ismissing, c(0L, 1L, 1L, 1L, 0L, 1L))
  expect_equal(r$idbin, c(1, 1, 1, 1, 2, 4))
  # without a missing-value bin, missing values stay "Special"
  fit2 <- list(id = 1:2, bin = c("a", "b"), woe = c(1, 2))
  r2 <- ob_apply_woe_cat(fit2, c(NA, "NA", "a"))
  expect_equal(r2$bin, c("Special", "Special", "a"))
  expect_equal(r2$woe, c(NA, NA, 1))
})

test_that("[catC] ob_apply_woe_cat matches like obwoe_sql(): exact, first bin wins", {
  fit <- list(id = 1:3, bin = c(" A", "A", "x%;%A"), woe = c(1, 2, 3))
  r <- ob_apply_woe_cat(fit, c(" A", "A", "x", "  A  "))
  expect_equal(r$woe, c(1, 2, 3, 1))
  # separator inside a category name
  fit2 <- list(id = 1:2, bin = c("q%;%a%;%b", "a"), woe = c(-1, 1))
  r2 <- ob_apply_woe_cat(fit2, c("a%;%b", "a", "q"))
  expect_equal(r2$woe, c(-1, -1, -1))
})

test_that("[catC] ob_apply_woe_cat handles an empty-string category and a category 'NA'", {
  # "" as a trailing component, "" as a whole label, "NA" as a real category
  fit <- list(id = 1:3, bin = c("x%;%", "NA%;%y", "z"), woe = c(1, 2, 3))
  r <- ob_apply_woe_cat(fit, c("", "x", "NA", "y", NA, "z", " "))
  expect_equal(r$woe, c(1, 1, 2, 2, 1, 3, 1))
  expect_equal(r$bin, c("x%;%", "x%;%", "NA%;%y", "NA%;%y", "x%;%", "z", "x%;%"))
  fit2 <- list(id = 1:2, bin = c("", "b"), woe = c(-1, 1))
  r2 <- ob_apply_woe_cat(fit2, c("", "b", NA, "c"))
  expect_equal(r2$woe, c(-1, 1, -1, NA))
  # cutpoints: "b+" defines the bin {"b", ""}, NA is the category "NA"
  f <- c("a", "", "b", NA, "a", "b")
  y <- c(1, 0, 1, 0, 0, 1)
  rc <- ob_cutpoints_cat(f, y, c("a+NA", "b+"))
  expect_equal(rc$woebin$bin, c("a%;%NA", "b%;%"))
  expect_equal(rc$woebin$count, c(3, 3))
  expect_equal(ob_apply_woe_cat(rc$woebin, f)$woe, rc$woefeature)
})

test_that("[catC] ob_apply_woe_cat is linear in n with high-cardinality labels", {
  set.seed(12)
  lv <- sprintf("level_%04d", 1:800)
  f <- sample(lv, 1e5, replace = TRUE)
  fit <- list(
    id = 1:2, woe = c(-0.5, 0.5),
    bin = c(paste(lv[1:400], collapse = "%;%"), paste(lv[401:800], collapse = "%;%"))
  )
  newf <- c(f, "unseen", NA)
  t <- system.time(r <- ob_apply_woe_cat(fit, newf))[["elapsed"]]
  expect_lt(t, 2)
  expect_equal(r$woe[1:1e5], ifelse(match(f, lv) <= 400, -0.5, 0.5))
  expect_equal(r$bin[1e5 + 1:2], c("Special", "Special"))
})

# ---------------------------------------------------------------------------
# ob_apply_woe_num
# ---------------------------------------------------------------------------
test_that("[catC] ob_apply_woe_num uses a fitted missing-value bin (cutpoints + 2)", {
  fit <- list(cutpoints = c(0, 1), woe = c(-1, 0, 1, 9), id = 1:4,
              bin = c("(-Inf;0]", "(0;1]", "(1;+Inf]", "NA"))
  r <- ob_apply_woe_num(fit, c(-5, 0, 0.5, 1, 2, NA, NaN, -999))
  expect_equal(r$woe, c(-1, -1, 0, 0, 1, 9, 9, 9))
  expect_equal(r$ismissing, c(0L, 0L, 0L, 0L, 0L, 1L, 1L, 1L))
  expect_equal(r$idbin[6], 4)
  # the NA bin can sit anywhere when it is labelled
  fit2 <- list(cutpoints = 0, woe = c(5, -1, 1), id = c(9, 1, 2), bin = c("Missing", "a", "b"))
  r2 <- ob_apply_woe_num(fit2, c(-1, 1, NA))
  expect_equal(r2$woe, c(-1, 1, 5))
})

test_that("[catC] ob_apply_woe_num right-closed / left-closed semantics", {
  fit <- list(cutpoints = c(1, 2, 3), woe = c(10, 20, 30, 40), id = 1:4)
  x <- c(-Inf, 1, 1.5, 2, 3, 3.1, Inf)
  expect_equal(ob_apply_woe_num(fit, x)$woe, c(10, 10, 20, 20, 30, 40, 40))
  expect_equal(ob_apply_woe_num(fit, x, include_upper_bound = FALSE)$woe,
               c(10, 20, 20, 30, 40, 40, 40))
  # identical to cut(right = TRUE), the convention of obwoe_apply()/obwoe_sql()
  set.seed(13)
  z <- c(round(rnorm(5000), 1), 1, 2, 3)
  expect_equal(ob_apply_woe_num(fit, z)$idbin,
               as.numeric(cut(z, c(-Inf, 1, 2, 3, Inf), right = TRUE, labels = FALSE)))
})

# ---------------------------------------------------------------------------
# ob_cutpoints_num / ob_cutpoints_cat
# ---------------------------------------------------------------------------
test_that("[catC] ob_cutpoints_num leaves the caller's cutpoints untouched", {
  cp <- c(3, 1, 2)
  r <- ob_cutpoints_num(c(0.5, 1.5, 2.5, 3.5), c(0, 1, 0, 1), cp)
  expect_identical(cp, c(3, 1, 2))
  expect_equal(r$cutpoints, c(1, 2, 3))
})

test_that("[catC] ob_cutpoints_num does not bin missing feature values", {
  r <- ob_cutpoints_num(c(NA, 1, 2, 3, NaN, 4), c(1, 0, 1, 0, 1, 1), 2.5)
  expect_equal(sum(r$woebin$count), 4)
  expect_true(all(is.na(r$woefeature[c(1, 5)])))
  expect_false(anyNA(r$woefeature[-c(1, 5)]))
})

test_that("[catC] ob_cutpoints_* validate target and cutpoints", {
  expect_error(ob_cutpoints_num(1:3, c(0, 1), 2), "same length")
  expect_error(ob_cutpoints_num(1:3, c(0, NA, 1), 2), "only 0 and 1")
  expect_error(ob_cutpoints_num(1:3, c(0, 2, 1), 2), "only 0 and 1")
  expect_error(ob_cutpoints_num(1:3, c(0, 0, 0), 2), "both")
  expect_error(ob_cutpoints_num(1:3, c(0, 1, 1), c(2, NA)), "NA")
  expect_error(ob_cutpoints_cat(c("a", "b"), c(0, 1), character(0)), "at least one bin")
  expect_error(ob_cutpoints_cat(c("a", "b", "z"), c(0, 1, 1), c("a", "b")), "'z'")
  expect_error(ob_cutpoints_cat(c("a", "b"), c(0, 1), c("a+b", "b")), "more than one bin")
})

test_that("[catC] ob_cutpoints_cat matches NA as the category 'NA' and round-trips", {
  f <- c("a", NA, "b", "a", NA, "b")
  y <- c(1, 0, 1, 0, 0, 1)
  r <- ob_cutpoints_cat(f, y, c("a+NA", "b"))
  expect_equal(r$woebin$count, c(4, 2))
  expect_equal(ob_apply_woe_cat(r$woebin, f)$woe, r$woefeature)
})

# ---------------------------------------------------------------------------
# ob_check_distincts
# ---------------------------------------------------------------------------
test_that("[catC] ob_check_distincts compares numbers exactly", {
  y <- c(0, 1, 0, 1, 0, 1, 0)
  expect_equal(ob_check_distincts(c(1e-7, 2e-7, 3e-7, Inf, -Inf, -0, 0), y), c(4L, 2L))
  # used to be 1: std::to_string() printed all four as "0.000000"
  expect_equal(ob_check_distincts(c(1e-7, 2e-7, 3e-7, 0), c(0, 0, 0, 0))[1], 4L)
  # -0 and 0 are one value; +-Inf are excluded (as documented)
  expect_equal(ob_check_distincts(c(-0, 0, 1, Inf), c(0, 1, 0, 1))[1], 2L)
  expect_equal(ob_check_distincts(c("a", "b", NA, "a"), c(0, 1, 1, 1)), c(2L, 2L))
  expect_equal(ob_check_distincts(c(TRUE, FALSE, NA, TRUE), c(0, 1, 1, 1)), c(2L, 2L))
})

# ---------------------------------------------------------------------------
# ob_preprocess
# ---------------------------------------------------------------------------
test_that("[catC] ob_preprocess does not modify its input", {
  x <- c(1, NA, 3, 4, 100, NA, 5, 6)
  x_copy <- x + 0
  xc <- c("a", NA, "b")
  y <- c(0, 1, 0, 1, 0, 1, 0, 1)
  r <- ob_preprocess(x, y, outlier_process = TRUE)
  expect_identical(x, x_copy)
  expect_identical(r$preprocess$feature, x_copy)
  r2 <- ob_preprocess(xc, c(0, 1, 1))
  expect_identical(xc, c("a", NA, "b"))
  expect_identical(r2$preprocess$feature, c("a", NA, "b"))
  expect_identical(r2$preprocess$feature_preprocessed, c("a", "N/A", "b"))
})

test_that("[catC] ob_preprocess keeps the missing sentinel out of outlier treatment", {
  set.seed(123)
  x <- c(rnorm(95, 50, 10), NA, NA, 200, -100, 250)
  y <- sample(0:1, 100, replace = TRUE)
  for (m in c("iqr", "zscore", "grubbs")) {
    r <- ob_preprocess(x, y, outlier_process = TRUE, outlier_method = m)
    expect_equal(r$report$missing_count, 2L)
    expect_equal(r$preprocess$feature_preprocessed[96:97], c(-999, -999))
  }
  # the two NAs no longer count as outliers: same count as without them
  for (m in c("iqr", "zscore", "grubbs")) {
    r1 <- ob_preprocess(x, y, outlier_process = TRUE, outlier_method = m)
    r0 <- ob_preprocess(x[!is.na(x)], y[!is.na(x)], outlier_process = TRUE, outlier_method = m)
    expect_equal(r1$report$outlier_count, r0$report$outlier_count)
    expect_equal(r1$preprocess$feature_preprocessed[!is.na(x)], r0$preprocess$feature_preprocessed)
  }
})

test_that("[catC] ob_preprocess handles tiny samples without out-of-bounds reads", {
  for (m in c("iqr", "zscore", "grubbs")) {
    expect_no_warning(r <- ob_preprocess(c(1, 5), c(0, 1), outlier_process = TRUE, outlier_method = m))
    expect_equal(r$preprocess$feature_preprocessed, c(1, 5))
    expect_no_warning(ob_preprocess(c(NA, 5, NA), c(0, 1, 1), outlier_process = TRUE, outlier_method = m))
  }
})

test_that("[catC] Grubbs removes the extreme observations one at a time", {
  x <- c(1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 50, 2, 1, 3, 2, -40)
  y <- rep(0:1, 8)
  r <- ob_preprocess(x, y, outlier_process = TRUE, outlier_method = "grubbs")
  expect_equal(r$report$outlier_count, 2L)
  expect_equal(r$preprocess$feature_preprocessed[c(11, 16)], c(-999, -999))
  # ties at the extreme: the earliest observation goes first
  x2 <- c(rep(0, 20), 10, 10)
  r2 <- ob_preprocess(x2, rep(0:1, 11), outlier_process = TRUE, outlier_method = "grubbs")
  expect_true(r2$report$outlier_count >= 1L)
  expect_equal(r2$preprocess$feature_preprocessed[21], -999)
})

# ---------------------------------------------------------------------------
# Crashers of the old code (kept last: the old build segfaults on these)
# ---------------------------------------------------------------------------
test_that("[catC] ob_apply_woe_num works with zero cutpoints (single bin)", {
  r <- ob_apply_woe_num(list(cutpoints = numeric(0), woe = 0.5, id = 1), c(1, 2, NA))
  expect_equal(r$woe, c(0.5, 0.5, NA))
  expect_equal(r$bin[1], "(-Inf;+Inf]")
})
