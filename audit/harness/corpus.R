# Scenario corpus for the differential (baseline x candidate) harness.
#
# Every scenario is generated deterministically from a fixed seed so that the
# baseline and the candidate builds see byte-identical inputs. The corpus
# deliberately covers the failure modes found by the forensic audit
# (audit/AUDITORIA_FORENSE_BINNING_CPP.md, Sections 4-5) as well as healthy
# cases that must stay healthy.
#
# Each scenario is a list with:
#   id      unique identifier "<family>/<name>[/s<seed>]"
#   type    "num" | "cat"
#   x, y    training feature / binary target
#   x_test, y_test  hold-out sample from the same generator (out-of-sample layer)

make_split <- function(gen, n, seed, test_frac = 0.3) {
  set.seed(seed)
  d <- gen(n)
  idx <- sample.int(length(d$x), floor(length(d$x) * (1 - test_frac)))
  list(x = d$x[idx], y = d$y[idx], x_test = d$x[-idx], y_test = d$y[-idx])
}

num_generators <- list(
  "signal/logistic_strong" = function(n) { x <- rnorm(n); list(x = x, y = rbinom(n, 1, plogis(-1 + 1.5 * x))) },
  "signal/logistic_weak"   = function(n) { x <- rnorm(n); list(x = x, y = rbinom(n, 1, plogis(-1 + 0.3 * x))) },
  "signal/step"            = function(n) { x <- runif(n); list(x = x, y = rbinom(n, 1, ifelse(x > 0.5, 0.4, 0.1))) },
  "signal/multi_step"      = function(n) { x <- runif(n); list(x = x, y = rbinom(n, 1, ifelse(x < .3, .1, ifelse(x < .7, .5, .9)))) },
  "signal/u_shape"         = function(n) { x <- runif(n); list(x = x, y = rbinom(n, 1, ifelse(x < .2 | x > .8, .6, .1))) },
  "signal/noise"           = function(n) { x <- rnorm(n); list(x = x, y = rbinom(n, 1, 0.3)) },
  "separation/perfect"     = function(n) { x <- rnorm(n); list(x = x, y = as.integer(x > 0)) },
  "separation/near"        = function(n) { x <- rnorm(n); y <- as.integer(x > 0); k <- sample.int(n, 3); y[k] <- 1L - y[k]; list(x = x, y = y) },
  "separation/flag_pure_level" = function(n) { x <- rbinom(n, 1, .5); list(x = x, y = ifelse(x == 0, 0L, rbinom(n, 1, .1))) },
  "imbalance/rate_1pct"    = function(n) { x <- rnorm(n); list(x = x, y = rbinom(n, 1, plogis(-5 + 1.2 * x))) },
  "imbalance/rate_02pct"   = function(n) { x <- rnorm(n); list(x = x, y = rbinom(n, 1, plogis(-6.8 + 1.0 * x))) },
  "discrete/ties"          = function(n) { x <- round(rnorm(n), 1); list(x = x, y = rbinom(n, 1, plogis(x))) },
  "discrete/three_values"  = function(n) { x <- sample(c(1, 2, 3), n, TRUE); list(x = x, y = rbinom(n, 1, c(.1, .3, .6)[x])) },
  "discrete/mass_at_zero"  = function(n) { x <- ifelse(runif(n) < .9, 0, rexp(n)); list(x = x, y = rbinom(n, 1, plogis(-1 + x))) },
  "scale/tiny"             = function(n) { x <- rnorm(n) * 1e-12; list(x = x, y = rbinom(n, 1, plogis(x * 1e12))) },
  "scale/huge"             = function(n) { x <- rnorm(n) * 1e12; list(x = x, y = rbinom(n, 1, plogis(x / 1e12))) },
  "outlier/single_extreme" = function(n) { x <- c(rnorm(n - 1), 1e5); list(x = x, y = rbinom(n, 1, plogis(pmin(x, 3)))) },
  "missing/na_30pct"       = function(n) { x <- rnorm(n); y <- rbinom(n, 1, plogis(x)); x[runif(n) < .3] <- NA; list(x = x, y = y) },
  "size/small_n40"         = function(n) { x <- rnorm(40); list(x = x, y = rbinom(40, 1, plogis(x))) },
  "size/n60"               = function(n) { x <- rnorm(60); list(x = x, y = rbinom(60, 1, plogis(x))) }
)

cat_generators <- list(
  "cat/k5_signal"     = function(n) { f <- sample(LETTERS[1:5], n, TRUE); r <- c(A = .05, B = .1, C = .2, D = .35, E = .6); list(x = f, y = rbinom(n, 1, r[f])) },
  "cat/k20_signal"    = function(n) { lv <- sprintf("c%02d", 1:20); f <- sample(lv, n, TRUE); r <- setNames(seq(.02, .6, length.out = 20), lv); list(x = f, y = rbinom(n, 1, r[f])) },
  "cat/k200_signal"   = function(n) { lv <- sprintf("c%03d", 1:200); f <- sample(lv, n, TRUE, prob = rexp(200)); r <- setNames(plogis(rnorm(200, -1.5, 1)), lv); list(x = f, y = rbinom(n, 1, r[f])) },
  "cat/pure_category" = function(n) { f <- sample(c("A", "B", "C", "D"), n, TRUE, prob = c(.1, .3, .3, .3)); r <- c(A = 0, B = .05, C = .3, D = .6); list(x = f, y = rbinom(n, 1, r[f])) },
  "cat/identical_rates" = function(n) { lv <- c(sprintf("s%02d", 1:10), "h1", "h2"); f <- sample(lv, n, TRUE); r <- setNames(c(rep(.2, 10), .8, .8), lv); list(x = f, y = rbinom(n, 1, r[f])) },
  "cat/rare_levels"   = function(n) { f <- sample(c(LETTERS[1:4], "r1", "r2"), n, TRUE, prob = c(.3, .3, .2, .19, .005, .005)); r <- c(A = .1, B = .2, C = .3, D = .4, r1 = .9, r2 = 0); list(x = f, y = rbinom(n, 1, r[f])) },
  "cat/large_low_risk" = function(n) { f <- sample(c("LOW", "M1", "M2", "HI"), n, TRUE, prob = c(.5, .2, .2, .1)); r <- c(LOW = .002, M1 = .05, M2 = .1, HI = .3); list(x = f, y = rbinom(n, 1, r[f])) },
  "cat/noise_k10"     = function(n) { f <- sample(letters[1:10], n, TRUE); list(x = f, y = rbinom(n, 1, .3)) },
  "cat/with_na"       = function(n) { f <- sample(c(LETTERS[1:5], NA), n, TRUE); r <- c(A = .1, B = .2, C = .3, D = .4, E = .5); p <- ifelse(is.na(f), .7, r[f]); list(x = f, y = rbinom(n, 1, p)) }
)

#' Build the scenario corpus.
#' @param n_seeds number of independent seeds per random scenario
#' @param n sample size before the train/test split
build_corpus <- function(n_seeds = 3, n = 3000) {
  out <- list()
  for (nm in names(num_generators)) for (s in seq_len(n_seeds)) {
    sp <- make_split(num_generators[[nm]], n, seed = 1000 + s)
    out[[length(out) + 1]] <- c(list(id = sprintf("%s/s%d", nm, s), type = "num"), sp)
  }
  for (nm in names(cat_generators)) for (s in seq_len(n_seeds)) {
    sp <- make_split(cat_generators[[nm]], n, seed = 2000 + s)
    out[[length(out) + 1]] <- c(list(id = sprintf("%s/s%d", nm, s), type = "cat"), sp)
  }
  out
}

# Parameter grids applied to every scenario (the engine defaults are always
# included as "default").
param_grid <- list(
  default = list(),
  tight   = list(min_bins = 2, max_bins = 3),
  wide    = list(min_bins = 3, max_bins = 8)
)
