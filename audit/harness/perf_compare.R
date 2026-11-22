# Compare perf.R tables. Budget: t_C <= 1.2 * t_B + 0.05 s per (engine, size);
# scaling exponent (large vs small) of C must not exceed B's by more than 0.1.
args <- commandArgs(trailingOnly = TRUE)
b <- readRDS(args[1]); c <- readRDS(args[2]); out <- args[3]
m <- merge(b, c, by = c("engine", "size", "n"), suffixes = c("_B", "_C"))
m$ratio <- m$time_C / pmax(m$time_B, 1e-3)
m$slow <- m$time_C > 1.2 * m$time_B + 0.05
sc <- function(t) { s <- t[t$size == "small", ]; l <- t[t$size == "large", ]
  mm <- merge(s, l, by = "engine"); setNames(log(pmax(mm$time.y, 1e-3) / pmax(mm$time.x, 1e-3)) / log(mm$n.y / mm$n.x), mm$engine) }
eb <- sc(b); ec <- sc(c); worse_scaling <- names(ec)[ec > eb[names(ec)] + 0.1 & c$time[c$size == "large"][match(names(ec), c$engine[c$size == "large"])] > 0.5]
verdict <- if (any(m$slow) || length(worse_scaling)) "FAIL" else "PASS"
L <- c("# Performance report (layer D)", "", sprintf("- **Verdict:** **%s**", verdict),
       sprintf("- Over budget (t_C > 1.2 t_B + 0.05 s): %d / %d", sum(m$slow), nrow(m)),
       sprintf("- Worse scaling exponent (> +0.1, large run > 0.5 s): %s", if (length(worse_scaling)) paste(worse_scaling, collapse = ", ") else "none"), "",
       "| engine | size | n | t_B (s) | t_C (s) | ratio |", "|---|---|---|---|---|---|",
       sprintf("| %s | %s | %d | %.3f | %.3f | %.2f%s |", m$engine, m$size, m$n, m$time_B, m$time_C, m$ratio, ifelse(m$slow, " ⚠", "")))
writeLines(L, out); cat(verdict, "\n"); quit(status = if (verdict == "PASS") 0 else 1)
