# Layer F (audit Section 9.3): differential fuzzing.
#
# Generates adversarial inputs (tiny n, ties, constants, NA, +-Inf, extreme
# scales, single class, min_bins > max_bins, empty strings, long names,
# high cardinality) and runs every engine on one side. Run it once with the
# baseline library and once with the candidate library (ideally a sanitizer
# build), then compare the two outcome tables with fuzz_compare.R.
#
# Usage: Rscript audit/harness/fuzz.R <library_dir> <out.rds> [cases_per_engine]

args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]; out_file <- args[2]
n_cases <- if (length(args) >= 3) as.integer(args[3]) else 300L
suppressPackageStartupMessages(library(OptimalBinningWoE, lib.loc = lib))
ns <- asNamespace("OptimalBinningWoE")
engines <- setdiff(ls(ns, pattern = "^ob_(numerical|categorical)_"),
                   c("ob_numerical_jedi_mwoe", "ob_categorical_jedi_mwoe"))

gen_num <- function(i) {
  set.seed(500000 + i)
  n <- sample(c(1, 2, 3, 5, 10, 30, 51, 200, 2000), 1)
  kind <- sample(c("normal", "ties", "constant", "two_values", "na", "inf", "tiny", "huge", "outlier", "int"), 1)
  x <- switch(kind,
    normal = rnorm(n), ties = round(rnorm(n)), constant = rep(1, n),
    two_values = sample(c(0, 1), n, TRUE),
    na = { v <- rnorm(n); v[runif(n) < .5] <- NA; v },
    inf = { v <- rnorm(n); v[sample.int(n, max(1, n %/% 10))] <- sample(c(Inf, -Inf), 1); v },
    tiny = rnorm(n) * 1e-300, huge = rnorm(n) * 1e300,
    outlier = c(rnorm(max(n - 1, 1)), 1e15)[seq_len(n)], int = sample.int(5, n, TRUE))
  ycase <- sample(c("mixed", "mixed", "mixed", "one_class", "one_event"), 1)
  y <- switch(ycase, mixed = rbinom(n, 1, .3), one_class = rep(0L, n),
              one_event = { v <- rep(0L, n); v[1] <- 1L; v })
  p <- list(min_bins = sample(c(1, 2, 3, 5), 1), max_bins = sample(c(2, 3, 5, 10), 1),
            bin_cutoff = sample(c(0.001, 0.05, 0.3), 1))
  list(x = x, y = as.integer(y), p = p, desc = paste(kind, ycase, n))
}

gen_cat <- function(i) {
  set.seed(700000 + i)
  n <- sample(c(1, 2, 5, 30, 200, 3000), 1)
  kind <- sample(c("few", "many", "single", "empty_str", "na", "long", "utf8", "sep"), 1)
  x <- switch(kind,
    few = sample(letters[1:3], n, TRUE), many = sprintf("c%05d", sample.int(5000, n, TRUE)),
    single = rep("a", n), empty_str = sample(c("", "a", "b"), n, TRUE),
    na = sample(c(NA, "a", "b", "NA"), n, TRUE), long = strrep(sample(letters, n, TRUE), 200),
    utf8 = sample(c("é", "e", "ça"), n, TRUE), sep = sample(c("a%;%b", "c", "a"), n, TRUE))
  ycase <- sample(c("mixed", "mixed", "one_class"), 1)
  y <- if (ycase == "mixed") rbinom(n, 1, .3) else rep(1L, n)
  p <- list(min_bins = sample(c(1, 2, 3), 1), max_bins = sample(c(2, 3, 6), 1),
            bin_cutoff = sample(c(0.001, 0.05, 0.4), 1))
  list(x = x, y = as.integer(y), p = p, desc = paste(kind, ycase, n))
}

check_invariants <- function(r, x) {
  if (!is.list(r) || is.null(r$count)) return("no_counts")
  bad <- character()
  if (any(r$count_pos + r$count_neg != r$count)) bad <- c(bad, "counts")
  if (!sum(r$count) %in% c(length(x), sum(!is.na(x)), sum(is.finite(suppressWarnings(as.numeric(x)))))) bad <- c(bad, "rows")
  if (any(!is.finite(r$woe))) bad <- c(bad, "woe_nonfinite")
  if (any(r$iv < -1e-12, na.rm = TRUE)) bad <- c(bad, "iv_negative")
  if (!is.null(r$cutpoints) && length(r$cutpoints) > 1 && any(diff(r$cutpoints) <= 0)) bad <- c(bad, "cuts")
  paste(bad, collapse = ",")
}

rows <- list()
for (en in engines) {
  fn <- get(en, envir = ns)
  for (i in seq_len(n_cases)) {
    d <- if (startsWith(en, "ob_numerical")) gen_num(i) else gen_cat(i)
    set.seed(42)
    r <- tryCatch(suppressWarnings(do.call(fn, c(list(feature = d$x, target = d$y), d$p))),
                  error = function(e) structure(list(msg = conditionMessage(e)), class = "err"))
    rows[[length(rows) + 1]] <- data.frame(
      engine = en, case = i, desc = d$desc, status = if (inherits(r, "err")) "error" else "ok",
      invariants = if (inherits(r, "err")) "" else check_invariants(r, d$x),
      msg = if (inherits(r, "err")) substr(r$msg, 1, 120) else "", stringsAsFactors = FALSE)
  }
}
tab <- do.call(rbind, rows)
saveRDS(tab, out_file)
cat("fuzz:", nrow(tab), "cases;", sum(tab$status == "error"), "controlled errors;",
    sum(nzchar(tab$invariants)), "invariant violations\n")
