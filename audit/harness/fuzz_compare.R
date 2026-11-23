# Compare two fuzz.R outcome tables (baseline vs candidate). Fails when the
# candidate errors where the baseline did not, or shows an invariant violation
# that the baseline did not have on the same input.
#
# Usage: Rscript audit/harness/fuzz_compare.R <fuzz_base.rds> <fuzz_cand.rds> <report.md>
args <- commandArgs(trailingOnly = TRUE)
b <- readRDS(args[1]); c <- readRDS(args[2]); out <- args[3]
m <- merge(b, c, by = c("engine", "case", "desc"), suffixes = c("_B", "_C"))
new_err <- m$status_C == "error" & m$status_B == "ok"
fixed_err <- m$status_C == "ok" & m$status_B == "error"
new_inv <- m$status_C == "ok" & nzchar(m$invariants_C) &
  mapply(function(bi, ci) length(setdiff(strsplit(ci, ",")[[1]], strsplit(bi, ",")[[1]])) > 0, m$invariants_B, m$invariants_C)
fixed_inv <- nzchar(m$invariants_B) & !nzchar(m$invariants_C) & m$status_C == "ok"
verdict <- if (any(new_err | new_inv)) "FAIL" else "PASS"
L <- c("# Differential fuzzing report (layer F)", "",
       sprintf("- **Verdict:** **%s**", verdict),
       sprintf("- Cases: %d over %d engines", nrow(m), length(unique(m$engine))),
       sprintf("- Candidate errors where baseline ran: %d", sum(new_err)),
       sprintf("- New invariant violations: %d", sum(new_inv)),
       sprintf("- Baseline errors now handled (invariants OK): %d", sum(fixed_err & !nzchar(m$invariants_C))),
       sprintf("- Baseline invariant violations fixed: %d", sum(fixed_inv)),
       sprintf("- Invariant violations present in both: %d", sum(nzchar(m$invariants_B) & nzchar(m$invariants_C))), "")
show <- m[new_err | new_inv, ]
if (nrow(show)) L <- c(L, "| engine | case | input | B | C |", "|---|---|---|---|---|",
  sprintf("| %s | %d | %s | %s %s | %s %s %s |", show$engine, show$case, show$desc, show$status_B, show$invariants_B,
          show$status_C, show$invariants_C, gsub("\\|", "/", show$msg_C)))
both <- m[nzchar(m$invariants_B) & nzchar(m$invariants_C), ]
if (nrow(both)) {
  L <- c(L, "", "## Pre-existing invariant violations (baseline and candidate)", "")
  agg <- aggregate(case ~ engine + invariants_C, data = both, FUN = length)
  L <- c(L, "| engine | violation | cases |", "|---|---|---|", sprintf("| %s | %s | %d |", agg$engine, agg$invariants_C, agg$case))
}
writeLines(L, out)
cat(verdict, "\n")
quit(status = if (verdict == "PASS") 0 else 1)
