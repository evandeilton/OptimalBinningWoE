# Layer D (audit Section 9.3): performance and scalability on one side.
# Usage: Rscript audit/harness/perf.R <library_dir> <out.rds> [reps]
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]; out_file <- args[2]; reps <- if (length(args) >= 3) as.integer(args[3]) else 3L
suppressPackageStartupMessages(library(OptimalBinningWoE, lib.loc = lib))
ns <- asNamespace("OptimalBinningWoE")
engines <- setdiff(ls(ns, pattern = "^ob_(numerical|categorical)_"),
                   c("ob_numerical_jedi_mwoe", "ob_categorical_jedi_mwoe"))
mk_num <- function(n) { set.seed(n); x <- rnorm(n); list(x = x, y = rbinom(n, 1, plogis(-1 + x))) }
mk_cat <- function(n, k) { set.seed(n + k); lv <- sprintf("c%06d", seq_len(k)); f <- sample(lv, n, TRUE)
  eff <- setNames(rnorm(k), lv); list(x = f, y = rbinom(n, 1, plogis(-1 + eff[f]))) }
sizes <- list(num = list(small = mk_num(2e4), large = mk_num(2e5)),
              cat = list(small = mk_cat(5e4, 2e3), large = mk_cat(2e5, 2e4)))
rows <- list()
for (en in engines) {
  fam <- if (startsWith(en, "ob_numerical")) "num" else "cat"
  fn <- get(en, envir = ns)
  for (sz in names(sizes[[fam]])) {
    d <- sizes[[fam]][[sz]]
    t <- vapply(seq_len(reps), function(i) { set.seed(42)
      system.time(try(suppressWarnings(fn(d$x, d$y)), silent = TRUE))[["elapsed"]] }, 0)
    rows[[length(rows) + 1]] <- data.frame(engine = en, size = sz, n = length(d$x), time = median(t))
  }
}
tab <- do.call(rbind, rows); saveRDS(tab, out_file); print(tab, row.names = FALSE)
