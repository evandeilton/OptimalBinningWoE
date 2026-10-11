# ChiMerge significance level: numerical vs categorical.
#
# ob_numerical_cm() merges the initial bins down to max_bins before its
# significance test is reached, so chi_merge_threshold does not change the
# fit; ?ob_numerical_cm documents this. ob_categorical_cm() merges down to
# max_bins and then keeps merging while pairs are not significantly different,
# so the level decides the bin count there. If the numerical engine is later
# changed to honor the level, the first test must change with the docs.

test_that("numerical cm: chi_merge_threshold does not change the cut points", {
  alphas <- c(0.001, 0.01, 0.05, 0.2, 0.5, 0.99)
  for (seed in c(1, 2, 3)) {
    set.seed(20261010 + seed)
    n <- 2000
    x <- if (seed == 2) rexp(n) else rnorm(n)
    y <- rbinom(n, 1, plogis(-1 + 0.8 * x))
    for (chi2 in c(FALSE, TRUE)) {
      for (init in c("equal_frequency", "equal_width")) {
        fits <- lapply(alphas, function(a) {
          suppressWarnings(ob_numerical_cm(x, y,
            min_bins = 2, max_bins = 6, chi_merge_threshold = a,
            use_chi2_algorithm = chi2, init_method = init
          ))
        })
        info <- sprintf("seed %d, chi2 %s, init %s", seed, chi2, init)
        for (f in fits[-1]) {
          expect_identical(f$cutpoints, fits[[1]]$cutpoints, info = info)
        }
      }
    }
  }
})

test_that("numerical cm: max_bins, not the level, sets the number of bins", {
  set.seed(20261011)
  n <- 3000
  x <- rnorm(n)
  y <- rbinom(n, 1, plogis(0.8 * x))
  for (mb in c(3, 6)) {
    nb <- vapply(c(0.001, 0.99), function(a) {
      length(ob_numerical_cm(x, y,
        min_bins = 2, max_bins = mb, chi_merge_threshold = a
      )$woe)
    }, integer(1))
    expect_equal(nb[1], nb[2], info = sprintf("max_bins %d", mb))
    expect_lte(nb[1], mb)
  }
})

test_that("categorical cm: chi_merge_threshold is the stopping rule", {
  set.seed(2001)
  n <- 3000
  L <- sample(8:25, 1)
  lv <- paste0("L", seq_len(L))
  f <- sample(lv, n, TRUE, prob = runif(L, 0.5, 2))
  rate <- runif(L, 0.05, 0.4)
  names(rate) <- lv
  y <- rbinom(n, 1, rate[f])

  nbins <- vapply(c(0.001, 0.05, 0.99), function(a) {
    r <- suppressWarnings(ob_categorical_cm(f, y,
      min_bins = 2, max_bins = 6, bin_cutoff = 0.01, chi_merge_threshold = a
    ))
    length(r$woe)
  }, integer(1))

  # a smaller level merges more; a level near 1 never merges past max_bins
  expect_lte(nbins[1], nbins[2])
  expect_lte(nbins[2], nbins[3])
  expect_lt(nbins[1], nbins[3])
  expect_equal(nbins[3], 6L)
})
