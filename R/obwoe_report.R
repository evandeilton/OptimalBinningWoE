# ============================================================================ #
# obwoe_report(): the scorecard workbook                                       #
# ============================================================================ #

#' @keywords internal
.ob_has_openxlsx <- function() {
  requireNamespace("openxlsx", quietly = TRUE)
}


#' @title Internal: Write One Styled Sheet
#'
#' @param wb An \pkg{openxlsx} workbook.
#' @param name Sheet name; Excel caps these at 31 characters.
#' @param df The table to write.
#' @param digits Rounding applied to double columns.
#' @param widths Column widths, or \code{"auto"}.
#'
#' @keywords internal
.ob_xlsx_sheet <- function(wb, name, df, digits = 6L, widths = "auto") {
  if (is.null(df) || nrow(df) == 0L) {
    df <- data.frame(note = "No rows for this section.", stringsAsFactors = FALSE)
  }
  df <- as.data.frame(df, stringsAsFactors = FALSE)

  num <- vapply(df, is.numeric, logical(1))
  df[num] <- lapply(df[num], function(v) {
    if (is.integer(v)) v else round(v, digits)
  })
  # Excel has no Inf: keep the information as text rather than write a number
  # that is not the one that was computed.
  df[] <- lapply(df, function(v) {
    if (is.numeric(v) && any(is.infinite(v))) {
      v <- as.character(v)
    }
    v
  })

  openxlsx::addWorksheet(wb, name)
  openxlsx::writeData(wb, name, df,
    headerStyle = openxlsx::createStyle(
      textDecoration = "bold", fgFill = "#1A5276",
      fontColour = "white", halign = "left", border = "bottom"
    )
  )
  openxlsx::freezePane(wb, name, firstActiveRow = 2L)
  if (ncol(df) > 1L) {
    openxlsx::addFilter(wb, name, rows = 1L, cols = seq_len(ncol(df)))
  }
  openxlsx::setColWidths(wb, name, cols = seq_len(ncol(df)), widths = widths)
  invisible(NULL)
}


#' @title Internal: Deployment SQL That Returns Points
#'
#' @description
#' Reuses the \code{\link{obwoe_sql}} machinery — the same interval convention,
#' the same escaping, the same explicit \code{IS NULL} branch — but emits the
#' integer points of each bin instead of its Weight of Evidence, and sums them
#' into the score. This is what a deployment engineer actually runs: the card,
#' not the model.
#'
#' @param x An \code{"obwoe_scorecard"} object.
#' @param table Source table name.
#' @param dialect SQL dialect.
#' @param keep_columns Columns carried through unchanged.
#'
#' @return A character scalar holding the statement.
#'
#' @keywords internal
.ob_points_sql <- function(x, table, dialect, keep_columns = NULL) {
  d <- .ob_sql_dialect(dialect)
  specs <- .ob_sql_spec(x$binning)
  bin_sep <- .ob_bin_separator(x$binning)

  items <- character(0)
  if (!is.null(keep_columns)) {
    items <- .ob_sql_ident(keep_columns, d, "auto")
  }

  # [C-08/A-08] Reuses the SAME "bins consistent with cutpoints" guard
  # obwoe_sql() applies (R/obwoe_sql.R, around the "Numerical features must
  # have a cut-point vector consistent with the bins" comment). Without it,
  # a duplicated cutpoint (floating-point ties collapsed by unique()) makes
  # length(cp) + 1L != k, and .ob_sql_case() indexes its per-bin literal
  # vector one past its length for the affected WHEN branch: R silently
  # returns NA for the out-of-bounds element, sprintf("%s", NA) renders the
  # literal text "NA" into the generated SQL, and the branch becomes
  # `<col> <= NA` -- broken, uncaught SQL instead of a warning. Kept as a
  # for loop (not vapply) so an inconsistent feature can be skipped, the
  # same way obwoe_sql() skips it, instead of corrupting the whole
  # statement.
  ok_final <- character(0)
  exprs <- character(0)
  for (feat in x$final) {
    spec <- specs[[feat]]
    tbl <- x$points[x$points$variable == feat, , drop = FALSE]
    tbl <- tbl[order(tbl$bin_id), , drop = FALSE]
    k <- nrow(tbl)

    if (identical(spec$type, "numerical")) {
      cp <- spec$cutpoints
      cp <- if (is.null(cp)) numeric(0) else sort(unique(as.numeric(cp)))
      if (length(cp) + 1L != k && !(k == 1L && length(cp) == 0L)) {
        warning(sprintf(
          paste0(
            "Feature '%s': %d bins are inconsistent with %d distinct cut ",
            "points, so an exact points SQL mapping cannot be derived. Skipped."
          ),
          feat, k, length(cp)
        ))
        next
      }
      spec$cutpoints <- cp
    }

    # A value in no fitted bin scores this variable's na_woe fallback, read
    # from the points table itself so that SQL and R cannot drift apart.
    neutral <- unname(attr(x$points, "points_na")[[feat]])

    expr <- .ob_sql_case(
      spec = spec, feature = feat,
      col = .ob_sql_ident(feat, d, "auto"), d = d,
      values = .ob_sql_num(tbl$points),
      else_value = .ob_sql_num(neutral),
      null_value = .ob_sql_num(neutral),
      explicit_bounds = TRUE, indent = "    ",
      # NOTE: "%;%" is hardcoded here, same as obwoe_sql()'s own default
      # (R/obwoe_sql.R) that R/obwoe_report.R's "10_SQL_WoE" sheet inherits a
      # [B-02] The model's real separator, now that [B-01] makes obwoe()
      # record the control it was fitted with. This was hard-coded to "%;%"
      # before: a model fitted with a custom separator had the points SQL
      # split its bin labels on the wrong string, silently corrupting the
      # categorical mapping in the deployed scorecard.
      bin_separator = bin_sep,
      trim_categories = FALSE
    )
    ok_final <- c(ok_final, feat)
    exprs <- c(exprs, expr)
  }

  if (length(ok_final) == 0L) {
    stop("No variable produced a valid points SQL expression.")
  }

  aliases <- .ob_sql_ident(paste0(ok_final, "_points"), d, "auto")
  inner <- c(items, sprintf("%s AS %s", exprs, aliases))

  # The total is computed in an outer SELECT. Referring to a select-list alias
  # from another expression in the same SELECT is a MySQL extension; ANSI SQL,
  # SQLite, PostgreSQL, SQL Server and Oracle all reject it.
  keep_out <- if (is.null(keep_columns)) {
    character(0)
  } else {
    .ob_sql_ident(keep_columns, d, "auto")
  }
  outer <- c(
    keep_out, aliases,
    sprintf(
      "%s AS %s", paste(aliases, collapse = " + "),
      .ob_sql_ident("score", d, "auto")
    )
  )

  header <- c(
    "-- ---------------------------------------------------------------",
    "-- Scorecard points",
    sprintf("-- Generated by OptimalBinningWoE %s", x$version$package),
    sprintf("-- Engine: %s   Variables: %d", x$engine$name, length(x$final)),
    sprintf(
      "-- PDO %g, %g points at %g:1 good:bad",
      x$scaling$pdo, x$scaling$score_ref, x$scaling$odds_ref
    ),
    "-- The score is the sum of the integer points below.",
    "-- ---------------------------------------------------------------"
  )

  paste(c(
    header,
    "SELECT",
    paste(outer, collapse = ",\n"),
    "FROM (",
    "SELECT",
    paste(inner, collapse = ",\n"),
    sprintf("FROM %s", .ob_sql_ident(table, d, "auto")),
    ") AS t;"
  ), collapse = "\n")
}


#' @title Internal: Cutoff Strategy Table
#' @keywords internal
.ob_cutoff_table <- function(score, y, n_points = 20L,
                             direction = "higher_is_safer") {
  probs <- seq(0.05, 0.95, length.out = n_points)
  cuts <- unique(round(stats::quantile(score, probs = probs, na.rm = TRUE)))

  # Which side of the cut is approved depends on which way the scale runs.
  # Under "higher_is_riskier" a >= rule would approve the worst applicants and
  # report a bad rate above the rejected one.
  safer_above <- identical(direction, "higher_is_safer")

  rows <- lapply(cuts, function(cut) {
    approve <- if (safer_above) score >= cut else score <= cut
    n_app <- sum(approve)
    n_rej <- sum(!approve)
    data.frame(
      cutoff_score = cut,
      approval_rate = n_app / length(score),
      n_approved = n_app,
      n_rejected = n_rej,
      bad_rate_approved = if (n_app > 0L) mean(y[approve]) else NA_real_,
      bad_rate_rejected = if (n_rej > 0L) mean(y[!approve]) else NA_real_,
      bads_avoided = sum(y[!approve]),
      goods_lost = sum(1L - y[!approve]),
      swap_ratio = if (sum(1L - y[!approve]) > 0L) {
        sum(y[!approve]) / sum(1L - y[!approve])
      } else {
        NA_real_
      },
      stringsAsFactors = FALSE
    )
  })
  as.data.frame(.ob_rbind(rows), stringsAsFactors = FALSE)
}


#' @title Write a Scorecard Workbook
#'
#' @description
#' Turns a fitted \code{\link{obwoe_scorecard}} into a multi-sheet
#' \code{.xlsx} file: the points table a branch officer reads, the evidence a
#' validation team asks for, and the SQL a deployment engineer runs.
#'
#' @param x An \code{"obwoe_scorecard"} object.
#' @param file Path to the \code{.xlsx} file to write.
#' @param control An \code{"obwoe_scorecard_control"} object; only
#'   \code{digits} and \code{overwrite} are used. Defaults to the settings
#'   stored on \code{x} where available.
#' @param table Character string naming the source table used in the generated
#'   SQL. Default \code{"your_table"}.
#' @param dialect SQL dialect for the deployment sheets, passed to
#'   \code{\link{obwoe_sql}}. Default \code{"ansi"}.
#' @param keep_columns Columns the generated SQL carries through unchanged.
#'
#' @return The path written, invisibly.
#'
#' @details
#' The workbook has one sheet per stage, in the order a reviewer reads them:
#'
#' \tabular{ll}{
#'   \code{01_Model_Summary} \tab provenance, the funnel, headline metrics \cr
#'   \code{02_Scorecard} \tab \strong{the deliverable}: one row per variable and
#'     bin, with the integer points \cr
#'   \code{03_Coefficients} \tab the fit, with the sign check and standard errors \cr
#'   \code{04_Bin_Statistics} \tab the gains table of every binned variable used
#'     in training \cr
#'   \code{05_Screening} \tab every candidate and why it lived or died \cr
#'   \code{06_Correlations} \tab redundancy in the WoE space and what was pruned \cr
#'   \code{07_Score_Gains} \tab rank ordering per sample, with observed versus
#'     predicted \cr
#'   \code{08_Stability_PSI} \tab the score and every variable, per sample \cr
#'   \code{09_Cutoff_Strategy} \tab approval rate, bad rate and swap set by cutoff \cr
#'   \code{10_SQL_WoE} \tab \code{\link{obwoe_sql}} output: the WoE transform \cr
#'   \code{11_SQL_Points} \tab the same bins returning integer points, summed
#'     into the score \cr
#'   \code{12_Reproducibility} \tab the call, versions, and every warning raised
#' }
#'
#' Splitting the deployment SQL in two is deliberate. The WoE sheet reproduces
#' the model exactly and is what a data scientist re-scores with; the points
#' sheet reproduces the \emph{card}, which is what the business signed and what
#' the decision engine should run.
#'
#' @seealso \code{\link{obwoe_scorecard}}, \code{\link{obwoe_sql}}
#'
#' @examples
#' \donttest{
#' german <- read.csv(
#'   gzfile(system.file("extdata", "germancredit.csv.gz",
#'     package = "OptimalBinningWoE"
#'   )),
#'   stringsAsFactors = FALSE
#' )
#' german$default <- 1L - german$credit_risk
#' german$credit_risk <- NULL
#'
#' sc <- obwoe_scorecard(german, target = "default", seed = 1)
#' obwoe_report(sc, file = file.path(tempdir(), "scorecard.xlsx"))
#' }
#'
#' @export
obwoe_report <- function(x,
                         file,
                         control = NULL,
                         table = "your_table",
                         dialect = "ansi",
                         keep_columns = NULL) {
  if (!inherits(x, "obwoe_scorecard")) {
    stop("'x' must be an 'obwoe_scorecard' object from obwoe_scorecard().")
  }
  if (!.ob_has_openxlsx()) {
    stop(paste(
      "Writing the workbook needs the 'openxlsx' package.",
      "Install it with install.packages(\"openxlsx\"), or keep file = NULL",
      "and use the returned object directly."
    ))
  }
  if (!is.character(file) || length(file) != 1L) {
    stop("'file' must be a single path.")
  }
  if (is.null(control)) control <- control.obwoe_scorecard()
  digits <- control$digits

  wb <- openxlsx::createWorkbook()

  # -- 01 Model summary ---------------------------------------------------- #
  perf <- do.call(rbind, lapply(names(x$samples), function(nm) {
    m <- x$samples[[nm]]$metrics
    data.frame(
      item = sprintf("%s: n / events / KS / Gini / AUC", nm),
      value = sprintf(
        "%d / %d / %.4f / %.4f / %.4f", m$n, m$events, m$ks, m$gini, m$auc
      ),
      stringsAsFactors = FALSE
    )
  }))

  drift <- if (is.null(x$points)) {
    NA_real_
  } else {
    max(vapply(x$samples, function(s) s$max_points_drift, numeric(1)), na.rm = TRUE)
  }

  summary_tbl <- rbind(
    data.frame(
      item = c(
        "Package", "R", "Built on", "Seed", "Target", "Event level", "Split",
        "Candidates", "Screened in", "In model", "Engine", "Additive",
        "PDO", "Reference score", "Reference odds", "Factor", "Offset"
      ),
      value = c(
        x$version$package, x$version$R, format(x$built_on),
        if (is.null(x$seed)) "not set" else as.character(x$seed),
        x$target, x$event_level, x$split,
        as.character(length(x$candidates)), as.character(length(x$selected)),
        as.character(length(x$final)), x$engine$name,
        as.character(x$engine$additive),
        as.character(x$scaling$pdo), as.character(x$scaling$score_ref),
        as.character(x$scaling$odds_ref),
        sprintf("%.6f", x$scaling$factor), sprintf("%.6f", x$scaling$offset)
      ),
      stringsAsFactors = FALSE
    ),
    perf,
    data.frame(
      item = c("Score range (train)", "Max card-vs-model drift", "Drift bound (k/2)"),
      value = c(
        paste(round(range(x$samples[[1L]]$score)), collapse = " to "),
        if (is.na(drift)) "n/a (no points table)" else sprintf("%.2f", drift),
        sprintf("%.1f", length(x$final) / 2)
      ),
      stringsAsFactors = FALSE
    )
  )
  .ob_xlsx_sheet(wb, "01_Model_Summary", summary_tbl, digits)

  # -- 02 Scorecard -------------------------------------------------------- #
  if (!is.null(x$points)) {
    card <- x$points
    card$points_delta <- card$points - round(
      x$scaling$offset / length(x$final) +
        (if (identical(x$scaling$direction, "higher_is_safer")) -1 else 1) *
          x$scaling$factor * x$coefficients[["(Intercept)"]] / length(x$final)
    )
    .ob_xlsx_sheet(wb, "02_Scorecard", card, digits)
  } else {
    .ob_xlsx_sheet(wb, "02_Scorecard", data.frame(
      note = paste(
        "The engine is not additive in the Weight of Evidence, so the model",
        "cannot be decomposed into per-bin points. Scores, gains and stability",
        "are in the other sheets."
      ), stringsAsFactors = FALSE
    ), digits)
  }

  # -- 03 Coefficients ----------------------------------------------------- #
  if (!is.null(x$coefficients)) {
    cf <- x$coefficients
    dg <- x$diagnostics
    grab <- function(nm) {
      v <- dg[[nm]]
      if (is.null(v)) rep(NA_real_, length(cf)) else as.numeric(v[names(cf)])
    }
    rng <- if (is.null(x$points)) NULL else {
      stats::aggregate(points ~ variable, data = x$points, FUN = function(p) {
        max(p) - min(p)
      })
    }
    coef_tbl <- data.frame(
      variable = names(cf),
      beta = as.numeric(cf),
      se = grab("se"), z = grab("z"), p_value = grab("p_value"),
      expected_sign = ifelse(names(cf) == "(Intercept)", "", "positive"),
      sign_ok = ifelse(names(cf) == "(Intercept)", NA, as.numeric(cf) >= 0),
      points_range = if (is.null(rng)) NA_real_ else {
        rng$points[match(names(cf), rng$variable)]
      },
      stringsAsFactors = FALSE
    )
    .ob_xlsx_sheet(wb, "03_Coefficients", coef_tbl, digits)
  }

  # -- 04 Bin statistics of the variables used in training ----------------- #
  bins <- as.data.frame(x$screening_bins, stringsAsFactors = FALSE)
  bins$in_model <- bins$feature %in% x$final
  bins <- bins[order(!bins$in_model, bins$feature, bins$bin_id), , drop = FALSE]
  .ob_xlsx_sheet(wb, "04_Bin_Statistics", bins, digits)

  # -- 05 Screening -------------------------------------------------------- #
  .ob_xlsx_sheet(wb, "05_Screening",
    as.data.frame(x$screening, stringsAsFactors = FALSE), digits)

  # -- 06 Correlations ----------------------------------------------------- #
  corr <- x$correlation$pairs
  if (!is.null(corr)) {
    corr <- as.data.frame(corr, stringsAsFactors = FALSE)
    corr$above_cutoff <- corr$abs_corr >= x$correlation$cutoff
    corr <- corr[order(-corr$abs_corr), , drop = FALSE]
  }
  .ob_xlsx_sheet(wb, "06_Correlations", corr, digits)
  .ob_xlsx_sheet(wb, "06b_Correlations_Pruned",
    x$correlation$dropped, digits)

  # -- 07 Score gains, all samples stacked --------------------------------- #
  gains <- lapply(names(x$samples), function(nm) {
    s <- x$samples[[nm]]
    g <- s$gains
    g$sample <- nm
    g$mean_score <- NA_real_
    g[, c("sample", setdiff(names(g), "sample")), drop = FALSE]
  })
  .ob_xlsx_sheet(wb, "07_Score_Gains",
    as.data.frame(.ob_rbind(gains), stringsAsFactors = FALSE), digits)

  # -- 08 Stability -------------------------------------------------------- #
  .ob_xlsx_sheet(wb, "08_Stability_PSI",
    if (is.null(x$stability)) NULL else {
      as.data.frame(x$stability, stringsAsFactors = FALSE)
    }, digits)

  # -- 09 Cutoff strategy -------------------------------------------------- #
  cutoffs <- lapply(names(x$samples), function(nm) {
    s <- x$samples[[nm]]
    tb <- .ob_cutoff_table(s$score, s$y, direction = x$scaling$direction)
    tb$sample <- nm
    tb[, c("sample", setdiff(names(tb), "sample")), drop = FALSE]
  })
  .ob_xlsx_sheet(wb, "09_Cutoff_Strategy",
    as.data.frame(.ob_rbind(cutoffs), stringsAsFactors = FALSE), digits)

  # -- 10/11 SQL ----------------------------------------------------------- #
  # [C-02/A-01] Use the na_woe the scorecard itself was built with (stored in
  # x$control as of 1.13.1), not obwoe_sql()'s own default of 0, so the SQL
  # emitted here returns the same value predict(x, ..., type = "score") or
  # type = "card" does for an unseen category. Falls back to this call's own
  # 'control' (or its default) for a scorecard object saved before 1.13.1,
  # which carries no $control.
  na_value <- if (!is.null(x$control) && !is.null(x$control$na_woe)) {
    x$control$na_woe
  } else {
    control$na_woe
  }
  # [B-02] Same separator the model carries, so the WoE-form SQL splits bin
  # labels exactly as obwoe_apply() does rather than assuming the default.
  woe_sql <- obwoe_sql(x$binning,
    table = table, features = x$final,
    keep_columns = keep_columns, dialect = dialect, na_value = na_value,
    bin_separator = .ob_bin_separator(x$binning)
  )
  .ob_xlsx_sheet(wb, "10_SQL_WoE", .ob_sql_lines(as.character(woe_sql)),
    digits, widths = 120)

  if (!is.null(x$points)) {
    .ob_xlsx_sheet(wb, "11_SQL_Points",
      .ob_sql_lines(.ob_points_sql(x, table, dialect, keep_columns)),
      digits, widths = 120)
  }

  # -- 12 Reproducibility -------------------------------------------------- #
  repro <- rbind(
    data.frame(
      item = "call", value = paste(deparse(x$call), collapse = " "),
      stringsAsFactors = FALSE
    ),
    data.frame(
      item = c("package", "R", "built_on", "seed", "engine", "split"),
      value = c(
        x$version$package, x$version$R, format(x$built_on),
        if (is.null(x$seed)) "not set" else as.character(x$seed),
        sprintf("%s (requested %s)", x$engine$used, x$engine$requested),
        x$split
      ),
      stringsAsFactors = FALSE
    ),
    if (length(x$warnings) > 0L) {
      data.frame(
        item = sprintf("warning %d", seq_along(x$warnings)),
        value = x$warnings, stringsAsFactors = FALSE
      )
    } else {
      data.frame(item = "warnings", value = "none", stringsAsFactors = FALSE)
    }
  )
  .ob_xlsx_sheet(wb, "12_Reproducibility", repro, digits, widths = c(24, 120))

  if (file.exists(file) && !isTRUE(control$overwrite)) {
    stop(sprintf("'%s' exists and overwrite is FALSE.", file))
  }
  openxlsx::saveWorkbook(wb, file, overwrite = TRUE)
  invisible(file)
}


#' @keywords internal
.ob_sql_lines <- function(sql) {
  data.frame(
    line_no = seq_along(strsplit(sql, "\n", fixed = TRUE)[[1L]]),
    sql = strsplit(sql, "\n", fixed = TRUE)[[1L]],
    stringsAsFactors = FALSE
  )
}
