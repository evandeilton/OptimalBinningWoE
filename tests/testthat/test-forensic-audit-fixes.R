# Permanent regression tests for the fixes derived from the forensic audit of
# the C++ binning engines (audit/AUDITORIA_FORENSE_BINNING_CPP.md, Section 9.8).
# Each test encodes the minimal failing scenario of a finding with its
# expected result stated explicitly.

# Helper: does a categorical bin label contain exactly the given categories?
bin_has <- function(label, cats, sep = "%;%") {
  setequal(strsplit(label, sep, fixed = TRUE)[[1]], cats)
}

# ---- B-01: OBC_DP pure bins ---------------------------------------------------
test_that("B-01: ob_categorical_dp isolates pure categories and ranks them first", {
  # Minimal counter-example from the audit (pos/neg per category):
  # A 0/100, B 0/20, C 25/75, D 20/20, k = 2. The baseline cut after C
  # (pure bin valued at IV = 0); the optimum isolates the pure group {A, B}.
  feat <- c(rep("A", 100), rep("B", 20), rep("C", 100), rep("D", 40))
  y <- c(rep(0, 100), rep(0, 20), rep(1, 25), rep(0, 75), rep(1, 20), rep(0, 20))
  r <- ob_categorical_dp(feat, y, min_bins = 2, max_bins = 2)
  expect_equal(nrow(as.data.frame(r[c("bin", "woe")])), 2L)
  expect_true(bin_has(r$bin[1], c("A", "B")))
  expect_true(bin_has(r$bin[2], c("C", "D")))
  expect_true(all(r$iv >= 0))
  expect_true(r$woe[1] < r$woe[2])          # pure bin gets the extreme WoE, not 0
  expect_false(is.unsorted(r$woe))
})

test_that("B-01: no ob_categorical_dp bin has WoE exactly 0 because it is pure", {
  set.seed(101)
  feat <- sample(c("A", "B", "C", "D"), 1000, TRUE, prob = c(.1, .3, .3, .3))
  rate <- c(A = 0, B = .05, C = .3, D = .6)
  y <- rbinom(1000, 1, rate[feat])
  r <- ob_categorical_dp(feat, y, min_bins = 4, max_bins = 4)
  pure <- r$count_pos == 0 | r$count_neg == 0
  expect_true(any(pure))
  expect_true(all(r$woe[pure] != 0))
  expect_equal(which.min(r$woe), which(pure & r$count_pos == 0)[1])
})

# ---- A-09: OBC_FETB pure categories --------------------------------------------
test_that("A-09: ob_categorical_fetb gives a pure category the lowest WoE", {
  feat <- c(rep("A", 100), rep("B", 100), rep("C", 100), rep("D", 100))
  y <- c(rep(0, 100), rep(1, 5), rep(0, 95), rep(1, 20), rep(0, 80), rep(1, 60), rep(0, 40))
  r <- ob_categorical_fetb(feat, y, min_bins = 4, max_bins = 4)
  a <- unname(which(vapply(r$bin, function(b) "A" %in% strsplit(b, "%;%", fixed = TRUE)[[1]], TRUE)))
  expect_length(a, 1L)
  expect_equal(a, which.min(r$woe))
  expect_true(r$woe[a] < 0)
  expect_false(is.unsorted(r$woe))
  expect_true(all(r$iv >= 0))
})
