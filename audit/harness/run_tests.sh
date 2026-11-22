#!/usr/bin/env bash
# Run the package testthat suite against the candidate build and compare the
# failing/erroring tests with the frozen baseline list (no new failures allowed).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$(cd "$here/../.." && pwd)"
W="${OB_HARNESS_DIR:?set OB_HARNESS_DIR}"
cd "$root/tests"
R_LIBS="$W/lib_candidate" Rscript -e '
suppressPackageStartupMessages({library(testthat); library(OptimalBinningWoE)})
r <- as.data.frame(test_dir("testthat", reporter = "silent", stop_on_failure = FALSE))
bad <- r[r$failed > 0 | r$error, c("file", "test")]
base <- c(".ob_cutoff_table rejects NA score or target with a clear message",
  "obwoe() records dispatch failures per feature", "engine resolution: custom engines and missing packages",
  "obwoe_select() error rows, strict monotonicity and the base-R path", "obwoe_scorecard() argument checks and split variants",
  "obwoe_report() writes every sheet and checks its inputs", "remaining fallbacks: IV from counts, all-NA cut points, SQL skips",
  "scorecard: thin sample warning and constant WoE after transform", ".ob_bin_separator falls back for objects fitted before 1.13.2",
  "the default separator keeps working unchanged")
new <- bad[!bad$test %in% base, ]
cat(sprintf("tests: %d | expectations failed/erroring tests: %d | pre-existing (environment): %d | NEW: %d\n",
            nrow(r), nrow(bad), sum(bad$test %in% base), nrow(new)))
if (nrow(new)) { print(new); quit(status = 1) }'
