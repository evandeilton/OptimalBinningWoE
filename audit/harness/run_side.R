# Run every binning engine on the scenario corpus for ONE side of the
# differential harness and save normalized outputs.
#
# Usage: Rscript audit/harness/run_side.R <library_dir> <output.rds> [n_seeds]
#
# The script is executed once against the frozen baseline library and once
# against the candidate library, always in a fresh R process, so the two builds
# never share a session (no DLL / namespace clash).

args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]; out_file <- args[2]
n_seeds <- if (length(args) >= 3) as.integer(args[3]) else 3L
.libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(OptimalBinningWoE, lib.loc = lib))
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(here, "corpus.R"))

ns <- asNamespace("OptimalBinningWoE")
engines <- ls(ns, pattern = "^ob_(numerical|categorical)_")
engines <- setdiff(engines, c("ob_numerical_jedi_mwoe", "ob_categorical_jedi_mwoe"))

# Fields that are allowed to differ between two runs of the SAME build.
volatile <- c("execution_time_ms", "metadata")

normalize <- function(res) {
  if (!is.list(res)) return(res)
  res[setdiff(names(res), volatile)]
}

corpus <- build_corpus(n_seeds = n_seeds)
results <- list()
for (sc in corpus) {
  eng <- engines[startsWith(engines, if (sc$type == "num") "ob_numerical" else "ob_categorical")]
  for (en in eng) for (pg in names(param_grid)) {
    key <- paste(en, sc$id, pg, sep = "::")
    fn <- get(en, envir = ns)
    warns <- character()
    set.seed(42)                      # engines that use R's RNG (SAB) stay reproducible
    t0 <- proc.time()[["elapsed"]]
    res <- tryCatch(
      withCallingHandlers(
        do.call(fn, c(list(feature = sc$x, target = sc$y), param_grid[[pg]])),
        warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }),
      error = function(e) structure(list(message = conditionMessage(e)), class = "engine_error"))
    elapsed <- proc.time()[["elapsed"]] - t0
    results[[key]] <- list(engine = en, scenario = sc$id, params = pg, type = sc$type,
                           res = normalize(res), time = elapsed, warnings = unique(warns))
  }
}
saveRDS(list(results = results, corpus_ids = vapply(corpus, `[[`, "", "id"),
             n_seeds = n_seeds, pkg_version = as.character(packageVersion("OptimalBinningWoE", lib.loc = lib))),
        out_file)
cat("saved", length(results), "runs to", out_file, "\n")
