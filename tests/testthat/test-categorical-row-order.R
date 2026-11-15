# Fitted bins of the categorical algorithms must not depend on the row order.
# Categories with equal counts or event rates are ordered by count, then by
# name, so neither the row order nor the C++ standard library changes the fit.

# 40 categories with counts in {2, 3, 4, 6}: many ties in count and event rate.
row_order_data <- function() {
  set.seed(42)
  # Mixed case, so the byte-wise name order differs from most locales.
  lev <- paste0(rep(c("C", "c"), 20), sprintf("%02d", 1:40))
  n_per <- sample(c(2, 3, 4, 6), 40, TRUE)
  pos_per <- vapply(n_per, function(n) sample(0:n, 1), 0)
  list(
    x = rep(lev, n_per),
    y = unlist(mapply(function(n, p) c(rep(1L, p), rep(0L, n - p)), n_per, pos_per))
  )
}

row_order_algs <- c(
  "cm", "fetb", "gmb", "ivb", "jedi", "jedi_mwoe", "sblp", "sketch", "swb", "udt"
)

for (alg in row_order_algs) {
  test_that(paste(alg, "result does not depend on the row order"), {
    fit <- get(paste0("ob_categorical_", alg), envir = asNamespace("OptimalBinningWoE"))
    d <- row_order_data()
    # Default settings, and a small max_n_prebins that forces pre-bin merging.
    for (args in list(list(), list(max_n_prebins = 5, bin_cutoff = 0.01))) {
      ref <- suppressWarnings(do.call(fit, c(list(d$x, d$y), args)))
      expect_gt(length(ref$bin), 1)
      for (s in 1:10) {
        set.seed(s)
        o <- sample(length(d$x))
        res <- suppressWarnings(do.call(fit, c(list(d$x[o], d$y[o]), args)))
        expect_identical(res$bin, ref$bin)
        expect_identical(res$woe, ref$woe)
        expect_identical(res$total_iv, ref$total_iv)
      }
    }
  })
}
