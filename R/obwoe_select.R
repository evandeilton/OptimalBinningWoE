# ============================================================================ #
# Automated Variable Selection for WoE Scorecards                              #
# ============================================================================ #

#' @title Internal: Siddiqi Information Value Strength Classification
#'
#' @description
#' Vectorised classification of Information Value into the strength bands
#' popularised by Siddiqi (2006). Shared by \code{\link{summary.obwoe}} and
#' \code{\link{obwoe_select}} so both report identical labels.
#'
#' @param iv Numeric vector of total Information Values.
#'
#' @return A factor with levels \code{"Unpredictive"}, \code{"Weak"},
#'   \code{"Medium"}, \code{"Strong"}, \code{"Suspicious"}, \code{"Error"}.
#'
#' @keywords internal
.ob_iv_class <- function(iv) {
  out <- rep("Error", length(iv))
  ok <- !is.na(iv)
  out[ok & iv < 0.02] <- "Unpredictive"
  out[ok & iv >= 0.02 & iv < 0.10] <- "Weak"
  out[ok & iv >= 0.10 & iv < 0.30] <- "Medium"
  out[ok & iv >= 0.30 & iv < 0.50] <- "Strong"
  out[ok & iv >= 0.50] <- "Suspicious"
  factor(out, levels = c(
    "Unpredictive", "Weak", "Medium",
    "Strong", "Suspicious", "Error"
  ))
}


#' @title Internal: Is data.table Available?
#'
#' @description
#' Single point of truth for the fast-path check, so the base-R fallback can be
#' exercised deliberately in the test suite.
#'
#' @return \code{TRUE} when \pkg{data.table} can be loaded.
#'
#' @keywords internal
.ob_has_dt <- function() {
  requireNamespace("data.table", quietly = TRUE)
}


#' @title Internal: Coerce a data.frame to data.table When Available
#'
#' @description
#' Returns a \code{data.table} when the \pkg{data.table} package is installed
#' (the fast path preferred for wide feature bases) and the original
#' \code{data.frame} otherwise. Because \code{data.table} inherits from
#' \code{data.frame}, downstream code is unaffected either way.
#'
#' @param x A \code{data.frame}.
#'
#' @return A \code{data.table} or the input \code{data.frame}.
#'
#' @keywords internal
.ob_as_table <- function(x) {
  if (.ob_has_dt()) {
    return(data.table::as.data.table(x))
  }
  x
}


#' @title Internal: Wrap a List of Columns as a data.frame
#'
#' @description
#' Builds a \code{data.frame} from an already-validated list of equal-length
#' columns without going through \code{data.frame()}, whose name deparsing and
#' per-column coercion dominate the run time when thousands of small tables are
#' assembled.
#'
#' @param cols A named list of equal-length vectors.
#'
#' @return A \code{data.frame}.
#'
#' @keywords internal
.ob_new_df <- function(cols) {
  n <- if (length(cols) > 0L) length(cols[[1L]]) else 0L
  structure(cols, class = "data.frame", row.names = .set_row_names(n))
}


#' @title Internal: Fast Row Binding of Homogeneous Blocks
#'
#' @description
#' Uses \code{data.table::rbindlist()} when available (O(n) in the number of
#' blocks) and falls back to \code{do.call(rbind, ...)} otherwise. Blocks may
#' be \code{data.frame}s or plain named lists of equal-length columns.
#'
#' @param lst A list of blocks sharing the same columns.
#'
#' @return A single table with all rows stacked.
#'
#' @keywords internal
.ob_rbind <- function(lst) {
  lst <- lst[!vapply(lst, is.null, logical(1))]
  if (length(lst) == 0L) {
    return(NULL)
  }
  if (.ob_has_dt()) {
    return(data.table::rbindlist(lst, use.names = TRUE, fill = TRUE))
  }
  lst <- lapply(lst, function(b) if (is.data.frame(b)) b else .ob_new_df(b))
  do.call(rbind, c(lst, list(make.row.names = FALSE, stringsAsFactors = FALSE)))
}


#' @title Internal: Area Under the ROC Curve for a Binned Score
#'
#' @description
#' Exact tie-corrected AUC of a score that is constant within each bin. Bins
#' sharing the same score value are merged before accumulation, so the result
#' equals both the trapezoidal area under the binned ROC curve and the
#' normalised Mann-Whitney U statistic.
#'
#' \deqn{AUC = \frac{1}{N_1 N_0} \sum_{i} n_{1,i}
#'   \left( \sum_{j : s_j < s_i} n_{0,j} + \tfrac{1}{2} n_{0,i} \right)}
#'
#' @param pos Integer/numeric vector of event counts per bin.
#' @param neg Integer/numeric vector of non-event counts per bin.
#' @param score Numeric vector giving the score attached to each bin. Higher
#'   values must indicate higher event propensity.
#'
#' @return A list with \code{auc}, \code{gini} and \code{ks}, or \code{NA}
#'   components when the target is degenerate.
#'
#' @keywords internal
.ob_auc_ks_binned <- function(pos, neg, score) {
  # Empty bins carry no information; NaN scores cannot be ordered. Infinite
  # scores (degenerate bins) are kept: they rank first or last, as they should.
  keep <- (pos + neg) > 0 & !is.na(score)
  pos <- pos[keep]
  neg <- neg[keep]
  score <- score[keep]

  n1 <- sum(pos)
  n0 <- sum(neg)

  if (length(pos) == 0L || n1 <= 0 || n0 <= 0) {
    return(list(auc = NA_real_, gini = NA_real_, ks = NA_real_))
  }

  ord <- order(score)
  s <- score[ord]
  p <- pos[ord]
  g <- neg[ord]

  # Merge bins carrying identical scores (ties must not be split). Equality is
  # tested directly rather than through diff(), so +/-Inf is handled correctly.
  grp <- if (length(s) == 1L) {
    1L
  } else {
    cumsum(c(TRUE, s[-1L] != s[-length(s)]))
  }
  p <- as.vector(rowsum(p, grp, reorder = FALSE))
  g <- as.vector(rowsum(g, grp, reorder = FALSE))

  cum_g_below <- cumsum(g) - g
  auc <- sum(p * (cum_g_below + 0.5 * g)) / (n1 * n0)

  ks <- max(abs(cumsum(p) / n1 - cumsum(g) / n0))

  list(auc = auc, gini = 2 * auc - 1, ks = ks)
}


#' @title Internal: Monotonicity Diagnostics of a Bin Profile
#'
#' @description
#' Assesses whether a per-bin quantity is ordered across the natural bin
#' sequence. Because the Weight of Evidence is a strictly increasing function
#' of the bin odds, and the event rate is likewise strictly increasing in the
#' odds, monotonicity of the event rate and of the empirical WoE are
#' equivalent statements.
#'
#' @param v Numeric vector indexed by bin id (already in bin order).
#'
#' @return A list with \code{monotonic}, \code{strict}, \code{direction},
#'   \code{n_violations} and \code{spearman}.
#'
#' @keywords internal
.ob_monotonicity <- function(v) {
  empty <- list(
    monotonic = NA, strict = NA, direction = NA_character_,
    n_violations = NA_integer_, spearman = NA_real_
  )

  if (length(v) < 2L || anyNA(v)) {
    if (length(v) == 1L && !anyNA(v)) {
      return(list(
        monotonic = TRUE, strict = TRUE, direction = "single-bin",
        n_violations = 0L, spearman = NA_real_
      ))
    }
    return(empty)
  }

  d <- diff(v)

  non_decreasing <- all(d >= 0)
  non_increasing <- all(d <= 0)
  strict_inc <- all(d > 0)
  strict_dec <- all(d < 0)

  if (strict_inc) {
    direction <- "increasing"
  } else if (strict_dec) {
    direction <- "decreasing"
  } else if (non_decreasing && non_increasing) {
    direction <- "constant"
  } else if (non_decreasing) {
    direction <- "increasing"
  } else if (non_increasing) {
    direction <- "decreasing"
  } else {
    direction <- "non-monotonic"
  }

  # Violations are counted against the dominant trend
  dominant_up <- sum(d[d > 0]) >= abs(sum(d[d < 0]))
  n_viol <- if (dominant_up) sum(d < 0) else sum(d > 0)

  sp <- NA_real_
  if (stats::sd(v) > 0) {
    sp <- suppressWarnings(
      stats::cor(seq_along(v), v, method = "spearman")
    )
  }

  list(
    monotonic = non_decreasing || non_increasing,
    strict = strict_inc || strict_dec,
    direction = direction,
    n_violations = as.integer(n_viol),
    spearman = sp
  )
}


#' @title Internal: Per-Feature Metric Extraction
#'
#' @description
#' Builds the feature-level statistics and the bin-level gains table for a
#' single entry of \code{obj$results}.
#'
#' @param res A single feature result from an \code{"obwoe"} object.
#' @param feature Feature name.
#' @param bin_separator Separator used inside merged categorical bin labels.
#'
#' @return A list with \code{summary} (a named list of scalars) and
#'   \code{bins} (a named list of equal-length columns, or \code{NULL}).
#'
#' @keywords internal
.ob_feature_metrics <- function(res, feature, bin_separator = "%;%") {
  base_row <- list(
    feature = feature,
    type = NA_character_,
    algorithm = NA_character_,
    n_bins = NA_integer_,
    n_obs = NA_integer_,
    n_pos = NA_integer_,
    n_neg = NA_integer_,
    event_rate = NA_real_,
    total_iv = NA_real_,
    iv_class = NA_character_,
    ks = NA_real_,
    gini = NA_real_,
    auc = NA_real_,
    max_lift = NA_real_,
    min_bin_pct = NA_real_,
    min_bin_count = NA_integer_,
    n_degenerate_bins = NA_integer_,
    woe_min = NA_real_,
    woe_max = NA_real_,
    woe_range = NA_real_,
    monotonic = NA,
    monotonic_strict = NA,
    monotonic_direction = NA_character_,
    n_violations = NA_integer_,
    spearman = NA_real_,
    woe_monotonic = NA,
    converged = NA,
    iterations = NA_integer_,
    error = TRUE,
    error_msg = NA_character_
  )

  base_row$type <- if (!is.null(res$type)) res$type else NA_character_
  base_row$algorithm <- if (!is.null(res$algorithm)) res$algorithm else NA_character_

  # ---- Hard failures reported by the binning engine ---------------------- #
  if (!is.null(res$error)) {
    base_row$error_msg <- as.character(res$error)[1L]
    return(list(summary = base_row, bins = NULL))
  }

  # ---- Structural requirements for binary gains statistics --------------- #
  needed <- c("bin", "count", "count_pos", "count_neg", "woe")
  if (!all(needed %in% names(res))) {
    base_row$error_msg <- paste0(
      "Result lacks binary bin counts (", paste(setdiff(needed, names(res)), collapse = ", "),
      "); multinomial or partial results are not supported."
    )
    return(list(summary = base_row, bins = NULL))
  }

  if (is.matrix(res$woe)) {
    base_row$error_msg <- "Multinomial (matrix) WoE is not supported by obwoe_select()."
    return(list(summary = base_row, bins = NULL))
  }

  bins <- as.character(res$bin)
  cnt <- as.numeric(res$count)
  pos <- as.numeric(res$count_pos)
  neg <- as.numeric(res$count_neg)
  woe <- as.numeric(res$woe)
  k <- length(bins)

  if (k == 0L || length(cnt) != k || length(pos) != k || length(neg) != k ||
    length(woe) != k) {
    base_row$error_msg <- "Inconsistent bin vectors returned by the binning algorithm."
    return(list(summary = base_row, bins = NULL))
  }

  id <- if (!is.null(res$id)) as.numeric(res$id) else seq_len(k)
  ord <- order(id)
  bins <- bins[ord]
  cnt <- cnt[ord]
  pos <- pos[ord]
  neg <- neg[ord]
  woe <- woe[ord]
  id <- id[ord]

  n_obs <- sum(cnt)
  n_pos <- sum(pos)
  n_neg <- sum(neg)

  # ---- Bin-level gains table (C++ engine, 31 metrics) -------------------- #
  gains <- tryCatch(
    obwoe_gains_score(list(
      id = id, bin = bins, count = cnt,
      count_pos = pos, count_neg = neg
    )),
    error = function(e) NULL
  )

  if (is.null(gains)) {
    base_row$error_msg <- "Gains table computation failed for this feature."
    return(list(summary = base_row, bins = NULL))
  }

  # Empirical event rate: strictly increasing in the bin odds, hence
  # rank-equivalent to the empirical WoE while remaining finite everywhere.
  pos_rate <- ifelse(cnt > 0, pos / cnt, NA_real_)

  # Discrimination is measured on the score that is actually deployed, i.e. the
  # WoE reported by the algorithm. When that vector is unusable (missing or all
  # NaN) the empirical event rate stands in, being rank-equivalent to the
  # empirical WoE.
  score <- woe
  if (all(is.na(score))) score <- pos_rate

  disc <- .ob_auc_ks_binned(pos, neg, score)
  mono_rate <- .ob_monotonicity(pos_rate)
  mono_woe <- .ob_monotonicity(woe)

  finite_woe <- woe[is.finite(woe)]

  base_row$error <- FALSE
  base_row$n_bins <- as.integer(k)
  base_row$n_obs <- as.integer(n_obs)
  base_row$n_pos <- as.integer(n_pos)
  base_row$n_neg <- as.integer(n_neg)
  base_row$event_rate <- if (n_obs > 0) n_pos / n_obs else NA_real_
  base_row$total_iv <- sum(gains$iv, na.rm = TRUE)
  base_row$iv_class <- as.character(.ob_iv_class(base_row$total_iv))
  base_row$ks <- disc$ks
  base_row$gini <- disc$gini
  base_row$auc <- disc$auc
  base_row$max_lift <- if (all(is.na(gains$lift))) NA_real_ else max(gains$lift, na.rm = TRUE)
  base_row$min_bin_pct <- if (n_obs > 0) min(cnt) / n_obs else NA_real_
  base_row$min_bin_count <- as.integer(min(cnt))
  base_row$n_degenerate_bins <- as.integer(sum(pos == 0 | neg == 0))
  base_row$woe_min <- if (length(finite_woe)) min(finite_woe) else NA_real_
  base_row$woe_max <- if (length(finite_woe)) max(finite_woe) else NA_real_
  base_row$woe_range <- base_row$woe_max - base_row$woe_min
  base_row$monotonic <- mono_rate$monotonic
  base_row$monotonic_strict <- mono_rate$strict
  base_row$monotonic_direction <- mono_rate$direction
  base_row$n_violations <- mono_rate$n_violations
  base_row$spearman <- mono_rate$spearman
  base_row$woe_monotonic <- mono_woe$monotonic
  # Coerce to a scalar: every field must contribute exactly one value when the
  # per-feature lists are transposed into columns.
  base_row$converged <- if (length(res$converged) > 0L) {
    as.logical(res$converged)[1L]
  } else {
    NA
  }
  base_row$iterations <- if (length(res$iterations) > 0L) {
    as.integer(res$iterations)[1L]
  } else {
    NA_integer_
  }

  # ---- Bin-level detail -------------------------------------------------- #
  lower <- rep(NA_real_, k)
  upper <- rep(NA_real_, k)
  n_cat <- rep(NA_integer_, k)
  cats <- rep(NA_character_, k)

  if (identical(base_row$type, "numerical")) {
    cp <- res$cutpoints
    if (!is.null(cp) && length(cp) > 0L) {
      cp <- sort(unique(as.numeric(cp)))
      if (length(cp) + 1L == k) {
        lower <- c(-Inf, cp)
        upper <- c(cp, Inf)
      }
    } else if (k == 1L) {
      lower <- -Inf
      upper <- Inf
    }
  } else {
    # Categories are reported exactly as the algorithm merged them; no
    # trimming, so a value carrying whitespace stays recognisable.
    parts <- strsplit(bins, bin_separator, fixed = TRUE)
    n_cat <- as.integer(vapply(parts, length, integer(1)))
    cats <- vapply(parts, paste, character(1), collapse = ", ")
  }

  gains_cols <- setdiff(names(gains), c("id", "bin"))
  gains <- as.list(gains[, gains_cols, drop = FALSE])
  # 'ks' is the cumulative KS *at* the bin; the feature-level maximum is
  # attached later under the name 'ks', so disambiguate here.
  names(gains)[names(gains) == "ks"] <- "ks_bin"

  bin_tbl <- c(
    list(
      feature = rep(feature, k),
      type = rep(base_row$type, k),
      algorithm = rep(base_row$algorithm, k),
      bin_id = as.integer(seq_len(k)),
      bin = bins,
      bin_lower = lower,
      bin_upper = upper,
      n_categories = n_cat,
      categories = cats,
      woe_model = woe
    ),
    gains
  )

  list(summary = base_row, bins = bin_tbl)
}


#' @title Automated Variable Selection for Weight of Evidence Scorecards
#'
#' @description
#' Screens every binned feature of an \code{"obwoe"} model against the two
#' criteria that govern variable admission in credit risk practice:
#' \strong{predictive strength}, measured by the Information Value and graded
#' with the Siddiqi (2006) bands, and \strong{guaranteed rank ordering},
#' measured by monotonicity of the bin event rate. The function never drops
#' rows: a base with 500 candidate variables yields 500 rows, each carrying a
#' selection flag and the exact reason behind the decision, so the analyst can
#' override the automatic verdict at will.
#'
#' @param obj An object of class \code{"obwoe"} returned by \code{\link{obwoe}}.
#'   The target must be binary; multinomial models are reported as
#'   unsupported rather than silently mis-scored.
#' @param detail Character string controlling the granularity of the result:
#'   \describe{
#'     \item{\code{"summary"}}{One row per feature with its headline metrics
#'       and the selection verdict (default).}
#'     \item{\code{"full"}}{One row per feature \emph{and} optimised bin,
#'       carrying the complete gains table of every variable alongside the
#'       feature-level verdict.}
#'   }
#' @param iv_min Numeric. Minimum admissible total Information Value
#'   (inclusive). Default \code{0.02}, the \emph{Unpredictive}/\emph{Weak}
#'   boundary.
#' @param iv_max Numeric. Exclusive upper bound on the total Information
#'   Value. Default \code{0.50}, the \emph{Suspicious} threshold: variables at
#'   or above it are rejected because such strength usually signals target
#'   leakage rather than genuine signal. Use \code{Inf} to disable.
#' @param require_monotonic Character string selecting where the ordering
#'   constraint applies:
#'   \describe{
#'     \item{\code{"numeric"}}{Only numerical features must be monotonic
#'       (default). Nominal categories carry no intrinsic order, so their bins
#'       can always be sorted by WoE and are trivially rank-orderable.}
#'     \item{\code{"all"}}{Categorical features must also be monotonic in the
#'       bin order returned by the algorithm.}
#'     \item{\code{"none"}}{The ordering constraint is reported but not
#'       enforced.}
#'   }
#' @param monotonicity Character string. \code{"weak"} (default) accepts
#'   non-decreasing or non-increasing profiles; \code{"strict"} rejects ties
#'   between adjacent bins.
#' @param min_bins Integer. Minimum number of bins a feature must have to be
#'   selected. Default \code{2}.
#' @param max_bins Numeric. Maximum admissible number of bins. Default
#'   \code{Inf}.
#' @param min_bin_pct Numeric in \eqn{[0, 1)}. Minimum population share of the
#'   smallest bin. Default \code{0} (no constraint); \code{0.05} is the usual
#'   scorecard convention.
#' @param allow_degenerate Logical. If \code{FALSE} (default), features with a
#'   bin containing no events or no non-events are rejected, since their WoE
#'   is not finite and the bin cannot be scored reliably out of sample.
#' @param top_n Integer or \code{NULL} (default). When supplied, only the
#'   \code{top_n} best features among those passing every rule stay selected;
#'   the remainder are flagged \code{"NOT_IN_TOP_N"}. Ranking follows
#'   \code{sort_by}, falling back to alphabetical order when \code{sort_by}
#'   names no metric.
#' @param sort_by Character string giving the ordering of the output and the
#'   ranking used by \code{top_n}: \code{"iv"} (default), \code{"ks"},
#'   \code{"gini"}, \code{"auc"}, \code{"feature"} or \code{"none"}.
#' @param decreasing Logical. Sort the output in decreasing order? Default
#'   \code{TRUE}; ignored when \code{sort_by} is \code{"feature"} or
#'   \code{"none"}. It affects presentation only: \code{rank} and the
#'   \code{top_n} cut always keep the highest values of the ranking metric.
#' @param bin_separator Character string separating merged categories inside a
#'   categorical bin label. Default \code{"\%;\%"}, matching
#'   \code{\link{control.obwoe}}.
#'
#' @return A \code{data.table} when the \pkg{data.table} package is installed
#'   and a \code{data.frame} otherwise (\code{data.table} inherits from
#'   \code{data.frame}, so either object supports the usual accessors).
#'
#'   With \code{detail = "summary"} the table holds one row per feature:
#'
#'   \tabular{ll}{
#'     \strong{Column} \tab \strong{Meaning} \cr
#'     \code{feature} \tab Variable name \cr
#'     \code{type} \tab \code{"numerical"} or \code{"categorical"} \cr
#'     \code{algorithm} \tab Binning algorithm used \cr
#'     \code{n_bins} \tab Number of optimised bins \cr
#'     \code{n_obs}, \code{n_pos}, \code{n_neg} \tab Population and class counts \cr
#'     \code{event_rate} \tab Overall event rate \cr
#'     \code{total_iv} \tab Total Information Value \cr
#'     \code{iv_class} \tab Siddiqi strength band \cr
#'     \code{ks} \tab Kolmogorov-Smirnov statistic in \eqn{[0, 1]} \cr
#'     \code{gini}, \code{auc} \tab Discrimination of the binned score \cr
#'     \code{max_lift} \tab Largest bin lift over the base rate \cr
#'     \code{min_bin_pct}, \code{min_bin_count} \tab Smallest bin size \cr
#'     \code{n_degenerate_bins} \tab Bins with zero events or zero non-events \cr
#'     \code{woe_min}, \code{woe_max}, \code{woe_range} \tab Spread of the WoE profile \cr
#'     \code{monotonic}, \code{monotonic_strict} \tab Event-rate ordering flags \cr
#'     \code{monotonic_direction} \tab \code{"increasing"}, \code{"decreasing"},
#'       \code{"constant"} or \code{"non-monotonic"} \cr
#'     \code{n_violations} \tab Adjacent pairs breaking the dominant trend \cr
#'     \code{spearman} \tab Rank correlation between bin order and event rate \cr
#'     \code{woe_monotonic} \tab Ordering of the WoE actually applied \cr
#'     \code{converged}, \code{iterations} \tab Optimiser diagnostics \cr
#'     \code{error}, \code{error_msg} \tab Binning failure flag and message \cr
#'     \code{quality} \tab Excellence tier (see Details) \cr
#'     \code{selected} \tab \strong{Selection flag} \cr
#'     \code{reason} \tab \strong{Machine-readable reason codes} \cr
#'     \code{reason_desc} \tab \strong{Human-readable justification} \cr
#'     \code{rank} \tab Position under \code{sort_by} among selectable features
#'   }
#'
#'   With \code{detail = "full"} the table holds one row per feature \emph{and}
#'   optimised bin. Each row carries the complete gains table produced by
#'   \code{\link{obwoe_gains_score}} (\code{count}, \code{pos}, \code{neg},
#'   \code{woe}, \code{iv}, \code{total_iv}, \code{pos_rate}, \code{lift},
#'   \code{cum_pos_perc}, \code{precision}, \code{recall}, \code{f1_score},
#'   \code{kl_divergence}, \code{js_divergence}, and so on) plus:
#'
#'   \tabular{ll}{
#'     \code{bin_id}, \code{bin} \tab Bin position and label \cr
#'     \code{bin_lower}, \code{bin_upper} \tab Interval bounds of a numerical
#'       bin, on the half-open convention \eqn{(lower, upper]} \cr
#'     \code{n_categories}, \code{categories} \tab Categories merged into a
#'       categorical bin \cr
#'     \code{woe_model} \tab WoE as reported by the binning algorithm, i.e. the
#'       value actually applied by \code{\link{obwoe_apply}} \cr
#'     \code{ks_bin} \tab Cumulative KS \emph{at} this bin (the gains-table
#'       \code{ks}), renamed so that \code{ks} keeps its feature-level meaning
#'   }
#'
#'   The feature-level verdict (\code{iv_class}, \code{ks}, \code{gini},
#'   \code{auc}, the monotonicity flags, \code{quality}, \code{selected},
#'   \code{reason}, \code{reason_desc}, \code{rank}) is repeated on every bin
#'   row. Variables whose binning failed have no bins and appear as a single
#'   placeholder row, so the row count never hides a candidate variable.
#'
#' @details
#' \subsection{Predictive strength}{
#'
#' The Information Value of a binned variable aggregates the divergence
#' between the event and non-event distributions:
#'
#' \deqn{IV = \sum_{i=1}^{k} \left( \frac{n_{1,i}}{N_1} -
#'   \frac{n_{0,i}}{N_0} \right) \ln\!\left( \frac{n_{1,i}/N_1}
#'   {n_{0,i}/N_0} \right)}
#'
#' where \eqn{n_{1,i}} and \eqn{n_{0,i}} are the event and non-event counts of
#' bin \eqn{i}. This is the symmetrised Kullback-Leibler divergence (Jeffreys
#' divergence) between the two conditional distributions. Counts are taken
#' from the fitted binning, so the value is invariant to any smoothing an
#' individual algorithm may apply to its reported WoE.
#'
#' Bands follow Siddiqi (2006): \eqn{IV < 0.02} unpredictive,
#' \eqn{[0.02, 0.10)} weak, \eqn{[0.10, 0.30)} medium, \eqn{[0.30, 0.50)}
#' strong, \eqn{\ge 0.50} suspicious. The upper band is excluded by default:
#' an IV above 0.5 on a single variable is, in practice, far more often a
#' symptom of leakage than of a genuinely dominant predictor.
#' }
#'
#' \subsection{Guaranteed ordering}{
#'
#' A variable is rank-ordering when its event rate moves in one direction
#' across the bin sequence. Writing the bin odds as
#' \eqn{\theta_i = n_{1,i} / n_{0,i}}, both quantities used in scorecards are
#' strictly increasing transformations of \eqn{\theta_i}:
#'
#' \deqn{\pi_i = \frac{n_{1,i}}{n_{1,i} + n_{0,i}} = \frac{\theta_i}{1 + \theta_i},
#'   \qquad WoE_i = \ln \theta_i + \ln \frac{N_0}{N_1}}
#'
#' so monotonicity of the event rate \eqn{\pi_i} and of the empirical WoE are
#' the same statement. The check is therefore performed once, on \eqn{\pi_i},
#' which stays finite even when a bin holds a single class; the ordering of
#' the WoE actually applied by the algorithm is reported separately in
#' \code{woe_monotonic}.
#'
#' For numerical variables the bin sequence is intrinsic (ordered intervals),
#' making monotonicity a genuine constraint. For nominal categorical variables
#' the sequence is a free relabelling, so a non-monotonic profile can always
#' be repaired by sorting; the default \code{require_monotonic = "numeric"}
#' reflects this asymmetry.
#' }
#'
#' \subsection{Discrimination metrics}{
#'
#' \code{ks}, \code{auc} and \code{gini} describe the \emph{deployed} score,
#' that is the WoE the algorithm attaches to each bin — exactly what
#' \code{\link{obwoe_apply}} writes out and what a downstream logistic
#' regression consumes. Bins are ranked by that WoE and merged when tied, then
#'
#' \deqn{KS = \max_i \left| F_1(i) - F_0(i) \right|, \qquad
#'   AUC = \frac{1}{N_1 N_0} \sum_i n_{1,i}
#'   \left( \sum_{j : s_j < s_i} n_{0,j} + \tfrac{1}{2} n_{0,i} \right),
#'   \qquad Gini = 2\,AUC - 1}
#'
#' where \eqn{s_i} is the bin WoE and \eqn{F_1, F_0} are the cumulative event
#' and non-event distributions in that order. The AUC expression is the
#' tie-corrected Mann-Whitney U statistic and reproduces, to machine precision,
#' both the trapezoidal area under the binned ROC curve and the rank-based AUC
#' computed on the WoE-transformed observations. Ranking by WoE rather than by
#' bin id makes \code{ks} the true KS of the transformed variable even when
#' the binning is not monotonic.
#' }
#'
#' \subsection{Excellence tiers}{
#'
#' \code{quality} summarises how cleanly the algorithm categorised a variable:
#'
#' \tabular{ll}{
#'   \strong{Tier} \tab \strong{Definition} \cr
#'   \code{Excellent} \tab Selected, strictly monotonic, \eqn{IV \ge 0.10},
#'     at least 3 bins, no degenerate bin, smallest bin \eqn{\ge 5\%} \cr
#'   \code{Good} \tab Selected, monotonic, \eqn{IV \ge 0.10} \cr
#'   \code{Fair} \tab Selected under every active rule but failing one of the
#'     structural refinements above \cr
#'   \code{Rejected} \tab Not selected
#' }
#' }
#'
#' \subsection{Reason codes}{
#'
#' \code{reason} concatenates every violated rule with \code{";"}, so a
#' variable rejected on two counts reports both:
#'
#' \tabular{ll}{
#'   \code{OK} \tab Passed every active rule \cr
#'   \code{BINNING_ERROR} \tab The algorithm failed on this variable \cr
#'   \code{IV_BELOW_MIN} \tab \eqn{IV <} \code{iv_min} \cr
#'   \code{IV_SUSPICIOUS} \tab \eqn{IV \ge} \code{iv_max} (leakage risk) \cr
#'   \code{NOT_MONOTONIC} \tab Event rate not ordered across bins \cr
#'   \code{TOO_FEW_BINS} \tab Fewer bins than \code{min_bins} \cr
#'   \code{TOO_MANY_BINS} \tab More bins than \code{max_bins} \cr
#'   \code{SMALL_BIN} \tab Smallest bin below \code{min_bin_pct} \cr
#'   \code{DEGENERATE_BIN} \tab A bin holds no events or no non-events \cr
#'   \code{NOT_IN_TOP_N} \tab Passed the rules but ranked below \code{top_n}
#' }
#' }
#'
#' @references
#' Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
#' Intelligent Credit Scoring. \emph{John Wiley & Sons}.
#' \doi{10.1002/9781119201731}
#'
#' Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and
#' Its Applications. \emph{SIAM Monographs on Mathematical Modeling and
#' Computation}. \doi{10.1137/1.9780898718317}
#'
#' Hand, D. J., & Till, R. J. (2001). A Simple Generalisation of the Area
#' Under the ROC Curve for Multiple Class Classification Problems.
#' \emph{Machine Learning}, 45(2), 171-186. \doi{10.1023/A:1010920819831}
#'
#' Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency.
#' \emph{The Annals of Mathematical Statistics}, 22(1), 79-86.
#' \doi{10.1214/aoms/1177729694}
#'
#' @seealso
#' \code{\link{obwoe}} for fitting the binning,
#' \code{\link{obwoe_sql}} for exporting the selected variables as SQL,
#' \code{\link{obwoe_gains_score}} for the underlying bin-level engine,
#' \code{\link{summary.obwoe}} for the compact model overview.
#'
#' @examples
#' \donttest{
#' set.seed(42)
#' n <- 2000
#' df <- data.frame(
#'   age = rnorm(n, 40, 12),
#'   income = exp(rnorm(n, 10, 0.7)),
#'   noise = rnorm(n),
#'   region = sample(c("N", "S", "E", "W"), n, replace = TRUE)
#' )
#' df$target <- rbinom(n, 1, plogis(-1.2 + 0.05 * (df$age - 40)))
#'
#' model <- obwoe(df, target = "target", max_bins = 6)
#'
#' # One row per variable, every candidate kept
#' sel <- obwoe_select(model)
#' sel[, c("feature", "total_iv", "iv_class", "ks", "monotonic", "selected", "reason")]
#'
#' # Full gains detail for every optimised bin
#' det <- obwoe_select(model, detail = "full")
#' head(det[, c("feature", "bin", "count", "pos_rate", "woe", "iv", "selected")])
#'
#' # Stricter policy: monotonic everywhere, 5% minimum bin, top 10 by KS
#' strict <- obwoe_select(model,
#'   require_monotonic = "all", monotonicity = "strict",
#'   min_bin_pct = 0.05, top_n = 10, sort_by = "ks"
#' )
#' }
#'
#' @importFrom stats cor sd
#' @export
obwoe_select <- function(obj,
                         detail = c("summary", "full"),
                         iv_min = 0.02,
                         iv_max = 0.50,
                         require_monotonic = c("numeric", "all", "none"),
                         monotonicity = c("weak", "strict"),
                         min_bins = 2L,
                         max_bins = Inf,
                         min_bin_pct = 0,
                         allow_degenerate = FALSE,
                         top_n = NULL,
                         sort_by = c("iv", "ks", "gini", "auc", "feature", "none"),
                         decreasing = TRUE,
                         bin_separator = "%;%") {
  detail <- match.arg(detail)
  require_monotonic <- match.arg(require_monotonic)
  monotonicity <- match.arg(monotonicity)
  sort_by <- match.arg(sort_by)

  # ------------------------------------------------------------------------ #
  # 1. Validation
  # ------------------------------------------------------------------------ #
  if (!inherits(obj, "obwoe")) {
    stop("'obj' must be an object of class 'obwoe' produced by obwoe().")
  }

  if (!is.numeric(iv_min) || length(iv_min) != 1L || is.na(iv_min) || iv_min < 0) {
    stop("'iv_min' must be a single non-negative number.")
  }
  if (!is.numeric(iv_max) || length(iv_max) != 1L || is.na(iv_max)) {
    stop("'iv_max' must be a single number (possibly Inf).")
  }
  if (iv_max <= iv_min) {
    stop("'iv_max' must be strictly greater than 'iv_min'.")
  }
  if (!is.numeric(min_bin_pct) || length(min_bin_pct) != 1L ||
    is.na(min_bin_pct) || min_bin_pct < 0 || min_bin_pct >= 1) {
    stop("'min_bin_pct' must be a single number in [0, 1).")
  }
  if (!is.logical(allow_degenerate) || length(allow_degenerate) != 1L) {
    stop("'allow_degenerate' must be a single logical value.")
  }

  min_bins <- as.integer(min_bins)
  if (is.na(min_bins) || min_bins < 1L) {
    stop("'min_bins' must be a single integer >= 1.")
  }
  if (!is.numeric(max_bins) || length(max_bins) != 1L || is.na(max_bins) ||
    max_bins < min_bins) {
    stop("'max_bins' must be a single number >= 'min_bins'.")
  }

  if (!is.null(top_n)) {
    top_n <- as.integer(top_n)
    if (is.na(top_n) || top_n < 1L) {
      stop("'top_n' must be NULL or a single integer >= 1.")
    }
  }

  if (identical(obj$target_type, "multinomial")) {
    stop(
      "obwoe_select() supports binary targets only. The supplied model was ",
      "fitted on a multinomial target."
    )
  }

  features <- names(obj$results)
  if (length(features) == 0L) {
    stop("'obj' contains no binning results.")
  }

  # ------------------------------------------------------------------------ #
  # 2. Per-feature metrics (single pass over the results list)
  # ------------------------------------------------------------------------ #
  metrics <- lapply(features, function(f) {
    .ob_feature_metrics(obj$results[[f]], f, bin_separator = bin_separator)
  })

  # Transpose the list of per-feature scalars into columns in one pass: this is
  # markedly faster than building and stacking one data.frame per feature, which
  # matters when the base carries hundreds of candidate variables.
  field <- names(metrics[[1L]]$summary)
  summ <- .ob_new_df(stats::setNames(
    lapply(field, function(cn) {
      unlist(lapply(metrics, function(m) m$summary[[cn]]), use.names = FALSE)
    }),
    field
  ))

  # ------------------------------------------------------------------------ #
  # 3. Rule evaluation (fully vectorised across features)
  # ------------------------------------------------------------------------ #
  n <- nrow(summ)

  fail_error <- summ$error
  iv <- summ$total_iv

  fail_iv_low <- !fail_error & (is.na(iv) | iv < iv_min)
  fail_iv_high <- !fail_error & !is.na(iv) & iv >= iv_max

  mono_ok <- if (identical(monotonicity, "strict")) {
    summ$monotonic_strict
  } else {
    summ$monotonic
  }
  mono_ok[is.na(mono_ok)] <- FALSE

  mono_applies <- switch(require_monotonic,
    "all" = rep(TRUE, n),
    "numeric" = summ$type %in% "numerical",
    "none" = rep(FALSE, n)
  )
  fail_mono <- !fail_error & mono_applies & !mono_ok

  fail_few_bins <- !fail_error & (is.na(summ$n_bins) | summ$n_bins < min_bins)
  fail_many_bins <- !fail_error & !is.na(summ$n_bins) & summ$n_bins > max_bins
  fail_small_bin <- !fail_error & min_bin_pct > 0 &
    (is.na(summ$min_bin_pct) | summ$min_bin_pct < min_bin_pct)
  fail_degenerate <- !fail_error & !allow_degenerate &
    !is.na(summ$n_degenerate_bins) & summ$n_degenerate_bins > 0L

  rule_flags <- list(
    BINNING_ERROR = fail_error,
    IV_BELOW_MIN = fail_iv_low,
    IV_SUSPICIOUS = fail_iv_high,
    NOT_MONOTONIC = fail_mono,
    TOO_FEW_BINS = fail_few_bins,
    TOO_MANY_BINS = fail_many_bins,
    SMALL_BIN = fail_small_bin,
    DEGENERATE_BIN = fail_degenerate
  )

  rule_desc <- c(
    BINNING_ERROR = "binning algorithm failed for this variable",
    IV_BELOW_MIN = sprintf("Information Value below the %.4g minimum", iv_min),
    IV_SUSPICIOUS = sprintf(
      "Information Value at or above %.4g (suspicious: possible target leakage)",
      iv_max
    ),
    NOT_MONOTONIC = sprintf(
      "event rate is not %s monotonic across the optimised bins",
      if (identical(monotonicity, "strict")) "strictly" else "weakly"
    ),
    TOO_FEW_BINS = sprintf("fewer than %d optimised bins", min_bins),
    TOO_MANY_BINS = sprintf("more than %g optimised bins", max_bins),
    SMALL_BIN = sprintf(
      "smallest bin holds less than %.2f%% of the population",
      100 * min_bin_pct
    ),
    DEGENERATE_BIN = "at least one bin has no events or no non-events (WoE not finite)",
    NOT_IN_TOP_N = "passed every rule but ranked below the requested top_n cut"
  )

  fail_mat <- do.call(cbind, rule_flags)
  colnames(fail_mat) <- names(rule_flags)
  n_fail <- rowSums(fail_mat)

  passed <- n_fail == 0L

  # ------------------------------------------------------------------------ #
  # 4. Ranking and optional top_n cut
  # ------------------------------------------------------------------------ #
  rank_col <- switch(sort_by,
    "iv" = summ$total_iv,
    "ks" = summ$ks,
    "gini" = summ$gini,
    "auc" = summ$auc,
    "feature" = NULL,
    "none" = NULL
  )

  summ$rank <- NA_integer_
  if (!is.null(rank_col) && any(passed)) {
    idx <- which(passed)
    ordering <- idx[order(rank_col[idx], decreasing = TRUE, na.last = TRUE)]
    summ$rank[ordering] <- seq_along(ordering)
  } else if (any(passed)) {
    idx <- which(passed)
    ordering <- idx[order(summ$feature[idx])]
    summ$rank[ordering] <- seq_along(ordering)
  }

  fail_top_n <- rep(FALSE, n)
  if (!is.null(top_n)) {
    fail_top_n <- passed & !is.na(summ$rank) & summ$rank > top_n
  }

  selected <- passed & !fail_top_n

  # ------------------------------------------------------------------------ #
  # 5. Reason codes and descriptions
  # ------------------------------------------------------------------------ #
  all_flags <- cbind(fail_mat, NOT_IN_TOP_N = fail_top_n)
  codes <- colnames(all_flags)

  reason <- rep("OK", n)
  reason_desc <- rep("selected: passed every active screening rule", n)

  any_fail <- which(rowSums(all_flags) > 0L)
  if (length(any_fail) > 0L) {
    hit <- lapply(any_fail, function(i) codes[all_flags[i, ]])
    reason[any_fail] <- vapply(hit, paste, character(1), collapse = ";")
    reason_desc[any_fail] <- vapply(
      seq_along(any_fail),
      function(j) {
        i <- any_fail[j]
        msg <- paste(rule_desc[hit[[j]]], collapse = "; ")
        if (identical(hit[[j]], "BINNING_ERROR") && !is.na(summ$error_msg[i])) {
          msg <- paste0(msg, " (", summ$error_msg[i], ")")
        }
        paste0("rejected: ", msg)
      },
      character(1)
    )
  }

  # ------------------------------------------------------------------------ #
  # 6. Excellence tiers
  # ------------------------------------------------------------------------ #
  strict_mono <- !is.na(summ$monotonic_strict) & summ$monotonic_strict
  weak_mono <- !is.na(summ$monotonic) & summ$monotonic
  iv_material <- !is.na(iv) & iv >= 0.10
  clean_bins <- !is.na(summ$n_degenerate_bins) & summ$n_degenerate_bins == 0L
  enough_bins <- !is.na(summ$n_bins) & summ$n_bins >= 3L
  fat_bins <- !is.na(summ$min_bin_pct) & summ$min_bin_pct >= 0.05

  quality <- rep("Rejected", n)
  quality[selected] <- "Fair"
  quality[selected & weak_mono & iv_material] <- "Good"
  quality[selected & strict_mono & iv_material & clean_bins &
    enough_bins & fat_bins] <- "Excellent"

  summ$quality <- factor(
    quality,
    levels = c("Excellent", "Good", "Fair", "Rejected")
  )
  summ$selected <- selected
  summ$reason <- reason
  summ$reason_desc <- reason_desc

  summ$iv_class <- .ob_iv_class(summ$total_iv)

  # ------------------------------------------------------------------------ #
  # 7. Output ordering
  # ------------------------------------------------------------------------ #
  ord <- switch(sort_by,
    "feature" = order(summ$feature),
    "none" = seq_len(n),
    order(!summ$selected, rank_col * (if (decreasing) -1 else 1), summ$feature)
  )
  summ <- summ[ord, , drop = FALSE]
  rownames(summ) <- NULL

  criteria <- list(
    iv_min = iv_min, iv_max = iv_max,
    require_monotonic = require_monotonic, monotonicity = monotonicity,
    min_bins = min_bins, max_bins = max_bins,
    min_bin_pct = min_bin_pct, allow_degenerate = allow_degenerate,
    top_n = top_n, sort_by = sort_by,
    n_features = n, n_selected = sum(selected)
  )

  # ------------------------------------------------------------------------ #
  # 8. Optional bin-level expansion
  # ------------------------------------------------------------------------ #
  if (identical(detail, "summary")) {
    out <- .ob_as_table(summ)
    attr(out, "obwoe_criteria") <- criteria
    return(out)
  }

  bins <- .ob_rbind(lapply(metrics, `[[`, "bins"))

  if (is.null(bins)) {
    # No feature produced a usable gains table: return the verdicts alone so
    # the caller still sees one row per variable.
    out <- .ob_as_table(summ)
    attr(out, "obwoe_criteria") <- criteria
    return(out)
  }

  bins <- .ob_new_df(as.list(bins))

  # Feature-level verdict columns attached to every bin. 'total_iv' is not
  # repeated here: the gains table already carries it with the same value.
  verdict_cols <- c(
    "iv_class", "ks", "gini", "auc",
    "monotonic", "monotonic_strict", "monotonic_direction",
    "quality", "selected", "reason", "reason_desc", "rank"
  )

  # Features whose binning failed contribute no bins. Add a placeholder row for
  # each so the promise of "every candidate variable appears in the output"
  # still holds at bin level.
  missing_feats <- setdiff(summ$feature, bins$feature)
  if (length(missing_feats) > 0L) {
    stub <- bins[rep(NA_integer_, length(missing_feats)), , drop = FALSE]
    rownames(stub) <- NULL
    m <- match(missing_feats, summ$feature)
    stub$feature <- missing_feats
    stub$type <- summ$type[m]
    stub$algorithm <- summ$algorithm[m]
    bins <- .ob_new_df(as.list(.ob_rbind(list(bins, stub))))
  }

  key <- match(bins$feature, summ$feature)
  for (v in verdict_cols) {
    bins[[v]] <- summ[[v]][key]
  }

  bins <- bins[order(key, bins$bin_id, na.last = TRUE), , drop = FALSE]
  rownames(bins) <- NULL

  out <- .ob_as_table(bins)
  attr(out, "obwoe_criteria") <- criteria
  out
}
