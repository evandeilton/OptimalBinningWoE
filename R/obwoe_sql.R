# ============================================================================ #
# SQL Code Generation for Fitted WoE Binnings                                  #
# ============================================================================ #

#' @title Internal: SQL Dialect Registry
#'
#' @description
#' Per-dialect syntax facts needed to emit portable, exact SQL: the identifier
#' quoting characters, whether backslashes act as escapes inside string
#' literals, and the statement used to (re)create a view.
#'
#' @param dialect Character string naming the dialect.
#'
#' @return A list with \code{open}, \code{close}, \code{escape_backslash} and
#'   \code{create_view}.
#'
#' @keywords internal
.ob_sql_dialect <- function(dialect) {
  switch(dialect,
    "mysql" = ,
    "mariadb" = list(
      open = "`", close = "`", escape_backslash = TRUE,
      create_view = "CREATE OR REPLACE VIEW"
    ),
    "spark" = ,
    "hive" = ,
    "databricks" = list(
      open = "`", close = "`", escape_backslash = TRUE,
      create_view = "CREATE OR REPLACE VIEW"
    ),
    "bigquery" = list(
      open = "`", close = "`", escape_backslash = TRUE,
      create_view = "CREATE OR REPLACE VIEW"
    ),
    "sqlserver" = list(
      open = "[", close = "]", escape_backslash = FALSE,
      create_view = "CREATE VIEW"
    ),
    # ANSI default: double-quoted identifiers, backslash is an ordinary
    # character inside string literals.
    list(
      open = "\"", close = "\"", escape_backslash = FALSE,
      create_view = "CREATE OR REPLACE VIEW"
    )
  )
}


#' @title Internal: Reserved Words Requiring Identifier Quoting
#' @keywords internal
.ob_sql_reserved <- function() {
  c(
    "all", "alter", "and", "any", "as", "asc", "between", "by", "case", "cast",
    "check", "column", "create", "cross", "current", "current_date",
    "current_time", "current_timestamp", "default", "delete", "desc",
    "distinct", "drop", "else", "end", "except", "exists", "false", "for",
    "foreign", "from", "full", "grant", "group", "having", "in", "index",
    "inner", "insert", "intersect", "into", "is", "join", "left", "like",
    "limit", "not", "null", "on", "or", "order", "outer", "primary",
    "references", "right", "select", "set", "some", "table", "then", "to",
    "true", "union", "unique", "update", "user", "using", "values", "view",
    "when", "where", "with"
  )
}


#' @title Internal: Quote a SQL Identifier
#'
#' @param x Character vector of identifiers. Dotted names such as
#'   \code{"schema.table"} are quoted part by part.
#' @param d Dialect list from \code{\link{.ob_sql_dialect}}.
#' @param mode One of \code{"auto"}, \code{"always"}, \code{"never"}.
#'
#' @return Character vector of quoted identifiers.
#'
#' @keywords internal
.ob_sql_ident <- function(x, d, mode = "auto") {
  reserved <- .ob_sql_reserved()

  quote_one <- function(part) {
    needs <- switch(mode,
      "always" = TRUE,
      "never" = FALSE,
      !grepl("^[A-Za-z_][A-Za-z0-9_]*$", part) || tolower(part) %in% reserved
    )
    if (!needs) {
      return(part)
    }
    # Doubling the closing delimiter is the portable way to embed it.
    part <- gsub(d$close, paste0(d$close, d$close), part, fixed = TRUE)
    paste0(d$open, part, d$close)
  }

  vapply(x, function(nm) {
    parts <- strsplit(nm, ".", fixed = TRUE)[[1L]]
    if (length(parts) == 0L) parts <- nm
    paste(vapply(parts, quote_one, character(1)), collapse = ".")
  }, character(1), USE.NAMES = FALSE)
}


#' @title Internal: SQL String Literal
#'
#' @description
#' Wraps a value in single quotes, doubling embedded single quotes as ANSI
#' SQL requires. On dialects where a backslash also escapes inside string
#' literals (MySQL/MariaDB and the Hive family by default) backslashes are
#' doubled as well, so a category such as \code{"a\\b"} survives verbatim.
#'
#' @param x Character vector.
#' @param d Dialect list from \code{\link{.ob_sql_dialect}}.
#'
#' @return Character vector of quoted literals.
#'
#' @keywords internal
.ob_sql_str <- function(x, d) {
  x <- as.character(x)
  x <- gsub("'", "''", x, fixed = TRUE)
  if (isTRUE(d$escape_backslash)) {
    x <- gsub("\\", "\\\\", x, fixed = TRUE)
  }
  paste0("'", x, "'")
}


#' @title Internal: Round-Trip-Safe SQL Numeric Literal
#'
#' @description
#' Renders a double as the shortest fixed-notation decimal string that parses
#' back to the identical binary value. Cut points and WoE values must survive
#' the round trip exactly, otherwise an observation sitting on a boundary
#' could fall into the wrong bin.
#'
#' @param x Numeric vector.
#' @param digits Optional integer. When supplied, values are rounded to that
#'   many significant digits instead of being written exactly.
#'
#' @return Character vector of SQL numeric literals.
#'
#' @keywords internal
.ob_sql_num <- function(x, digits = NULL) {
  vapply(as.numeric(x), function(v) {
    if (is.na(v)) {
      return("NULL")
    }
    if (is.infinite(v)) {
      return(if (v > 0) "1e308" else "-1e308")
    }
    if (!is.null(digits)) {
      return(format(round(v, digits), scientific = FALSE, trim = TRUE))
    }
    for (dg in 1:17) {
      s <- format(v, digits = dg, scientific = FALSE, trim = TRUE)
      if (!is.na(suppressWarnings(as.numeric(s))) && as.numeric(s) == v) {
        return(s)
      }
    }
    format(v, digits = 22, scientific = FALSE, trim = TRUE)
  }, character(1))
}


#' @title Internal: Normalise Supported Inputs to a Binning Specification
#'
#' @description
#' Accepts an \code{"obwoe"} object, a prepped \code{"step_obwoe"} step, or a
#' prepped \pkg{recipes} recipe containing one, and returns a uniform list of
#' per-feature binning specifications.
#'
#' @param obj The input object.
#'
#' @return A named list; each element holds \code{type}, \code{bin},
#'   \code{woe}, \code{cutpoints} and \code{error}.
#'
#' @keywords internal
.ob_sql_spec <- function(obj) {
  if (inherits(obj, "obwoe")) {
    out <- lapply(obj$results, function(r) {
      list(
        type = r$type,
        bin = r$bin,
        woe = r$woe,
        iv = r$iv,
        cutpoints = r$cutpoints,
        error = if (!is.null(r$error)) as.character(r$error)[1L] else NULL
      )
    })
    names(out) <- names(obj$results)
    return(out)
  }

  if (inherits(obj, "recipe")) {
    steps <- Filter(function(s) inherits(s, "step_obwoe"), obj$steps)
    if (length(steps) == 0L) {
      stop("The supplied recipe contains no step_obwoe() step.")
    }
    specs <- lapply(steps, .ob_sql_spec)
    out <- do.call(c, specs)
    if (anyDuplicated(names(out))) {
      stop("The supplied recipe bins the same variable in more than one step.")
    }
    return(out)
  }

  if (inherits(obj, "step_obwoe")) {
    if (!isTRUE(obj$trained)) {
      stop("The step_obwoe() step must be prepped before SQL can be generated.")
    }
    out <- lapply(obj$binning_results, function(r) {
      list(
        type = r$type, bin = r$bin, woe = r$woe, iv = r$iv,
        cutpoints = r$cutpoints, error = NULL
      )
    })
    names(out) <- names(obj$binning_results)
    return(out)
  }

  stop(
    "'obj' must be an 'obwoe' object, a prepped 'step_obwoe' step, or a ",
    "prepped recipe containing one."
  )
}


#' @title Internal: CASE Expression for a Single Feature
#'
#' @param spec One element of \code{\link{.ob_sql_spec}}.
#' @param feature Feature name (unquoted).
#' @param col Quoted column reference used inside the predicates.
#' @param d Dialect list.
#' @param values Character vector of THEN literals, one per bin.
#' @param else_value Character scalar used by the ELSE branch.
#' @param null_value Character scalar returned when the column is NULL.
#' @param explicit_bounds Logical; emit fully qualified interval predicates.
#' @param indent Character string prefixed to each WHEN line.
#' @param bin_separator Separator inside merged categorical bin labels.
#' @param trim_categories Logical; strip surrounding whitespace from category
#'   names after splitting a merged bin label.
#'
#' @return A character scalar holding the CASE expression.
#'
#' @keywords internal
.ob_sql_case <- function(spec, feature, col, d, values, else_value, null_value,
                         explicit_bounds, indent, bin_separator,
                         trim_categories = FALSE) {
  lines <- c("CASE")
  lines <- c(lines, sprintf("%sWHEN %s IS NULL THEN %s", indent, col, null_value))

  if (identical(spec$type, "numerical")) {
    cp <- spec$cutpoints
    cp <- if (is.null(cp)) numeric(0) else sort(unique(as.numeric(cp)))
    k <- length(values)

    if (k == 1L) {
      return(paste(c(lines, sprintf("%sELSE %s", indent, values[1L]), "END"),
        collapse = "\n"
      ))
    }

    lit <- .ob_sql_num(cp)

    for (i in seq_len(k - 1L)) {
      pred <- if (explicit_bounds && i > 1L) {
        sprintf("%s > %s AND %s <= %s", col, lit[i - 1L], col, lit[i])
      } else {
        sprintf("%s <= %s", col, lit[i])
      }
      lines <- c(lines, sprintf("%sWHEN %s THEN %s", indent, pred, values[i]))
    }

    if (explicit_bounds) {
      lines <- c(lines, sprintf(
        "%sWHEN %s > %s THEN %s",
        indent, col, lit[k - 1L], values[k]
      ))
      lines <- c(lines, sprintf("%sELSE %s", indent, else_value))
    } else {
      lines <- c(lines, sprintf("%sELSE %s", indent, values[k]))
    }

    return(paste(c(lines, "END"), collapse = "\n"))
  }

  # ---- Categorical -------------------------------------------------------- #
  # Bin labels are the original category strings joined by bin_separator, so the
  # split pieces reproduce the categories byte for byte. They are NOT trimmed by
  # default: a category whose value carries leading or trailing whitespace must
  # still be matched exactly.
  parts <- strsplit(as.character(spec$bin), bin_separator, fixed = TRUE)
  if (isTRUE(trim_categories)) parts <- lapply(parts, trimws)

  for (i in seq_along(parts)) {
    cats <- parts[[i]]
    if (length(cats) == 0L) next
    lit <- .ob_sql_str(cats, d)
    pred <- if (length(lit) == 1L) {
      sprintf("%s = %s", col, lit)
    } else {
      sprintf("%s IN (%s)", col, paste(lit, collapse = ", "))
    }
    lines <- c(lines, sprintf("%sWHEN %s THEN %s", indent, pred, values[i]))
  }

  lines <- c(lines, sprintf("%sELSE %s", indent, else_value))
  paste(c(lines, "END"), collapse = "\n")
}


#' @title Generate SQL for a Fitted Optimal Binning
#'
#' @description
#' Translates a fitted binning into executable SQL so the Weight of Evidence
#' transformation can be applied directly inside a database, with no round
#' trip through R. Every optimised bin becomes one \code{WHEN} branch of a
#' \code{CASE} expression, reproducing the interval and category assignments
#' of \code{\link{obwoe_apply}} exactly.
#'
#' @param obj An object of class \code{"obwoe"} from \code{\link{obwoe}}, a
#'   prepped \code{\link{step_obwoe}} step, or a prepped \pkg{recipes} recipe
#'   containing one.
#' @param table Character string naming the source table. May be qualified
#'   (\code{"schema.table"}); each part is quoted independently. Required for
#'   every \code{style} except \code{"case"}.
#' @param features Character vector restricting which variables are exported.
#'   \code{NULL} (default) exports every successfully binned variable. This is
#'   the natural place to feed the output of \code{\link{obwoe_select}}, e.g.
#'   \code{features = sel$feature[sel$selected]}.
#' @param output Character string choosing what the \code{CASE} returns:
#'   \describe{
#'     \item{\code{"woe"}}{The Weight of Evidence value (default).}
#'     \item{\code{"bin"}}{The bin label as a string literal.}
#'     \item{\code{"index"}}{The 1-based bin id.}
#'     \item{\code{"both"}}{Two expressions per feature: bin label and WoE.}
#'   }
#' @param style Character string choosing how the expressions are assembled:
#'   \describe{
#'     \item{\code{"select"}}{A complete \code{SELECT ... FROM table} (default).}
#'     \item{\code{"case"}}{A named character vector of bare \code{CASE}
#'       expressions, one per generated column, with no aliases.}
#'     \item{\code{"cte"}}{The \code{SELECT} wrapped in a
#'       \code{WITH ... AS (...)} common table expression.}
#'     \item{\code{"view"}}{A \code{CREATE VIEW} statement built on the
#'       \code{SELECT}.}
#'   }
#' @param dialect Character string naming the target SQL dialect. One of
#'   \code{"ansi"} (default), \code{"postgres"}, \code{"mysql"},
#'   \code{"mariadb"}, \code{"sqlserver"}, \code{"oracle"}, \code{"spark"},
#'   \code{"hive"}, \code{"databricks"}, \code{"bigquery"},
#'   \code{"snowflake"}, \code{"redshift"}, \code{"duckdb"}, \code{"sqlite"}.
#' @param view_name Character string naming the view or CTE. Defaults to
#'   \code{"woe_transform"}.
#' @param keep_columns Character vector of extra columns to carry through
#'   unchanged (identifiers, the target, partition keys). Default \code{NULL}.
#' @param suffix_woe,suffix_bin Character strings appended to the feature name
#'   to build the output column aliases. Defaults \code{"_woe"} and
#'   \code{"_bin"}.
#' @param na_value Numeric value returned for \code{NULL} inputs and for
#'   categories unseen during training. Default \code{0}, matching the
#'   \code{na_woe} default of \code{\link{obwoe_apply}}.
#' @param null_to_na_bin Logical. When \code{TRUE} (default) and the training
#'   data contained missing values that the binner folded into a bin under one
#'   of \code{na_categories}, \code{NULL} inputs are routed to that bin's value
#'   instead of \code{na_value}.
#' @param na_categories Character vector of tokens the binner uses to
#'   represent missing categories. Default \code{c("NA", "Missing", "")},
#'   matching \code{\link{ob_apply_woe_cat}}.
#' @param explicit_bounds Logical. When \code{TRUE} (default) each numerical
#'   branch states both of its bounds, e.g.
#'   \code{WHEN x > 7 AND x <= 10 THEN ...}, so every branch is correct in
#'   isolation and survives reordering or copy-and-paste. When \code{FALSE} the
#'   branches cascade with upper bounds only, which is shorter and equally
#'   exact given the top-down evaluation order of \code{CASE}.
#' @param digits Integer or \code{NULL} (default). \code{NULL} writes cut
#'   points and WoE values at full precision, as the shortest decimal string
#'   that parses back to the identical double. Supplying a value rounds the
#'   literals for readability, at the cost of exactness on bin boundaries.
#' @param quote_identifiers Character string: \code{"auto"} (default) quotes
#'   only identifiers that need it, \code{"always"} quotes everything,
#'   \code{"never"} emits bare names.
#' @param indent Integer giving the number of spaces prefixed to each
#'   \code{WHEN} line. Default \code{4}.
#' @param comment Logical. Prepend an audit header with the package version,
#'   the algorithm and the per-variable Information Value? Default \code{TRUE}.
#' @param class_index Integer or \code{NULL} (default). For multinomial models
#'   the WoE is a matrix; this selects which class column to export.
#' @param bin_separator Character string separating merged categories inside a
#'   bin label. Default \code{"\%;\%"}, matching \code{\link{control.obwoe}}.
#' @param trim_categories Logical. When \code{FALSE} (default) category names
#'   are matched byte for byte, including any leading or trailing whitespace
#'   they carry in the data. Set to \code{TRUE} to reproduce the whitespace
#'   trimming that \code{\link{obwoe_apply}} applies when it rebuilds its
#'   category map; do so only if your pipeline depends on that behaviour, since
#'   trimming makes a padded category unmatchable.
#' @param file Optional path. When supplied, the generated SQL is also written
#'   there with \code{\link{writeLines}}.
#'
#' @return An object of class \code{"obwoe_sql"}, which is a character vector
#'   with a \code{print} method that echoes the statement verbatim. For
#'   \code{style = "case"} the vector is named by output column and holds one
#'   \code{CASE} expression per element; for every other style it is a single
#'   string containing the complete statement. Use \code{as.character()} to
#'   strip the class, or \code{cat()} to display it.
#'
#' @details
#' \subsection{Interval semantics}{
#'
#' Numerical bins are half-open on the right, the convention used throughout
#' the package. For cut points \eqn{c_1 < c_2 < \cdots < c_k} the \eqn{k+1}
#' bins are
#'
#' \deqn{(-\infty, c_1],\; (c_1, c_2],\; \ldots,\; (c_{k-1}, c_k],\;
#'   (c_k, +\infty)}
#'
#' which the generated SQL renders as \code{x <= c1},
#' \code{x > c1 AND x <= c2}, and so on. This mirrors
#' \code{cut(x, breaks = c(-Inf, cutpoints, Inf), right = TRUE)}, the exact
#' call made by \code{\link{obwoe_apply}}, so an observation sitting on a cut
#' point lands in the \emph{lower} bin in R and in SQL alike.
#'
#' Cut points are read from the fitted \code{cutpoints} vector, never parsed
#' back from bin labels: label formatting varies between algorithms whereas
#' \code{cutpoints} is the authoritative numeric boundary. Duplicated cut
#' points are removed exactly as \code{\link{obwoe_apply}} removes them, and a
#' variable whose de-duplicated cut points no longer match its bin count is
#' skipped with a warning rather than exported with a wrong mapping.
#' }
#'
#' \subsection{Numeric literals}{
#'
#' With \code{digits = NULL} each cut point is written as the shortest
#' fixed-notation decimal that parses back to the identical IEEE 754 double.
#' Rounding a boundary such as \code{4049.5} to fewer digits would silently
#' move observations between bins, so exactness is the default; scientific
#' notation is avoided because dialects differ on how they type such literals.
#' }
#'
#' \subsection{NULL handling}{
#'
#' In SQL, \code{NULL <= 5} evaluates to \code{NULL}, not to \code{FALSE}, so
#' a missing value matches no comparison branch and would silently fall
#' through to \code{ELSE}. Every generated expression therefore opens with an
#' explicit \code{WHEN <col> IS NULL} branch. Categorical binners represent
#' training missings as the literal category \code{"NA"}; when
#' \code{null_to_na_bin = TRUE} the \code{IS NULL} branch returns that bin's
#' value, so database \code{NULL}s are scored the way missings were scored
#' during fitting. The \code{ELSE} branch catches categories never seen in
#' training and returns \code{na_value}.
#' }
#'
#' \subsection{Escaping}{
#'
#' Bin labels are the original category strings joined by
#' \code{bin_separator}, so splitting a label recovers the categories byte for
#' byte. They are matched exactly, whitespace included, unless
#' \code{trim_categories = TRUE}.
#'
#' Category labels are emitted as quoted literals with embedded single quotes
#' doubled per ANSI SQL. On MySQL, MariaDB and the Hive family — where a
#' backslash also escapes inside string literals under default settings —
#' backslashes are doubled too. Identifiers are quoted with the dialect's own
#' delimiters, and by default only when the name is not a plain
#' \code{[A-Za-z_][A-Za-z0-9_]*} token or collides with a reserved word;
#' this keeps generated code readable on case-folding engines such as Oracle
#' and Snowflake, where blanket quoting would force case sensitivity.
#' }
#'
#' @references
#' Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
#' Intelligent Credit Scoring. \emph{John Wiley & Sons}.
#' \doi{10.1002/9781119201731}
#'
#' International Organization for Standardization (2016). \emph{ISO/IEC
#' 9075-2:2016 Information technology — Database languages — SQL — Part 2:
#' Foundation (SQL/Foundation)}.
#'
#' @seealso
#' \code{\link{obwoe}} for fitting the binning,
#' \code{\link{obwoe_select}} for choosing which variables to export,
#' \code{\link{obwoe_apply}} for the equivalent transformation in R.
#'
#' @importFrom utils packageVersion
#'
#' @examples
#' \donttest{
#' set.seed(42)
#' n <- 1000
#' df <- data.frame(
#'   age = rnorm(n, 40, 12),
#'   region = sample(c("North", "South", "O'Hare"), n, replace = TRUE)
#' )
#' df$target <- rbinom(n, 1, plogis(-1 + 0.04 * (df$age - 40)))
#'
#' model <- obwoe(df, target = "target", max_bins = 5)
#'
#' # Complete SELECT statement
#' sql <- obwoe_sql(model, table = "credit.applications", keep_columns = "target")
#' sql
#'
#' # Only the variables that survived automatic screening
#' sel <- obwoe_select(model)
#' obwoe_sql(model,
#'   table = "credit.applications",
#'   features = sel$feature[sel$selected]
#' )
#'
#' # Bin labels and WoE side by side, as a Spark view
#' obwoe_sql(model,
#'   table = "credit.applications", output = "both",
#'   style = "view", dialect = "spark", view_name = "v_woe"
#' )
#'
#' # Bare CASE expressions for embedding in an existing query
#' obwoe_sql(model, style = "case")
#' }
#'
#' @export
obwoe_sql <- function(obj,
                      table = "your_table",
                      features = NULL,
                      output = c("woe", "bin", "index", "both"),
                      style = c("select", "case", "cte", "view"),
                      dialect = c(
                        "ansi", "postgres", "mysql", "mariadb", "sqlserver",
                        "oracle", "spark", "hive", "databricks", "bigquery",
                        "snowflake", "redshift", "duckdb", "sqlite"
                      ),
                      view_name = "woe_transform",
                      keep_columns = NULL,
                      suffix_woe = "_woe",
                      suffix_bin = "_bin",
                      na_value = 0,
                      null_to_na_bin = TRUE,
                      na_categories = c("NA", "Missing", ""),
                      explicit_bounds = TRUE,
                      digits = NULL,
                      quote_identifiers = c("auto", "always", "never"),
                      indent = 4L,
                      comment = TRUE,
                      class_index = NULL,
                      bin_separator = "%;%",
                      trim_categories = FALSE,
                      file = NULL) {
  output <- match.arg(output)
  style <- match.arg(style)
  dialect <- match.arg(dialect)
  quote_identifiers <- match.arg(quote_identifiers)

  # ------------------------------------------------------------------------ #
  # 1. Validation
  # ------------------------------------------------------------------------ #
  if (!is.character(table) || length(table) != 1L || is.na(table)) {
    stop("'table' must be a single character string.")
  }
  if (!is.character(view_name) || length(view_name) != 1L || is.na(view_name)) {
    stop("'view_name' must be a single character string.")
  }
  if (!is.numeric(na_value) || length(na_value) != 1L || is.na(na_value)) {
    stop("'na_value' must be a single non-missing number.")
  }
  if (!is.logical(explicit_bounds) || length(explicit_bounds) != 1L) {
    stop("'explicit_bounds' must be a single logical value.")
  }
  if (!is.logical(null_to_na_bin) || length(null_to_na_bin) != 1L) {
    stop("'null_to_na_bin' must be a single logical value.")
  }
  if (!is.logical(trim_categories) || length(trim_categories) != 1L) {
    stop("'trim_categories' must be a single logical value.")
  }
  if (!is.null(digits)) {
    digits <- as.integer(digits)
    if (is.na(digits) || digits < 0L) {
      stop("'digits' must be NULL or a non-negative integer.")
    }
  }
  indent_str <- strrep(" ", max(0L, as.integer(indent)))

  d <- .ob_sql_dialect(dialect)
  specs <- .ob_sql_spec(obj)

  if (length(specs) == 0L) {
    stop("'obj' contains no binning results.")
  }

  if (!is.null(features)) {
    if (!is.character(features)) {
      stop("'features' must be NULL or a character vector.")
    }
    unknown <- setdiff(features, names(specs))
    if (length(unknown) > 0L) {
      stop(sprintf(
        "Feature(s) not present in the fitted binning: %s",
        paste(unknown, collapse = ", ")
      ))
    }
    specs <- specs[features]
  }

  # ------------------------------------------------------------------------ #
  # 2. Per-feature CASE expressions
  # ------------------------------------------------------------------------ #
  exprs <- list()
  meta <- list()

  for (feat in names(specs)) {
    spec <- specs[[feat]]

    if (!is.null(spec$error)) {
      warning(sprintf(
        "Feature '%s' has no valid binning (%s); skipped.", feat, spec$error
      ))
      next
    }

    bins <- as.character(spec$bin)
    woe <- spec$woe
    k <- length(bins)

    if (k == 0L) {
      warning(sprintf("Feature '%s' has no bins; skipped.", feat))
      next
    }

    # Multinomial models store WoE as a bins x classes matrix.
    if (is.matrix(woe)) {
      if (output %in% c("woe", "both")) {
        if (is.null(class_index)) {
          warning(sprintf(
            paste0(
              "Feature '%s' comes from a multinomial model: supply ",
              "'class_index' to choose a WoE column, or use output = \"bin\" ",
              "/ \"index\". Skipped."
            ),
            feat
          ))
          next
        }
        ci <- as.integer(class_index)
        if (is.na(ci) || ci < 1L || ci > ncol(woe)) {
          stop(sprintf(
            "'class_index' must be between 1 and %d for feature '%s'.",
            ncol(woe), feat
          ))
        }
        woe <- woe[, ci]
      } else {
        woe <- rep(NA_real_, k)
      }
    }

    woe <- as.numeric(woe)

    # Numerical features must have a cut-point vector consistent with the bins.
    if (identical(spec$type, "numerical")) {
      cp <- spec$cutpoints
      cp <- if (is.null(cp)) numeric(0) else sort(unique(as.numeric(cp)))
      if (length(cp) + 1L != k) {
        if (!(k == 1L && length(cp) == 0L)) {
          warning(sprintf(
            paste0(
              "Feature '%s': %d bins are inconsistent with %d distinct cut ",
              "points, so an exact SQL mapping cannot be derived. Skipped."
            ),
            feat, k, length(cp)
          ))
          next
        }
      }
      spec$cutpoints <- cp
    }

    col <- .ob_sql_ident(feat, d, quote_identifiers)

    # Value returned for NULL input: the missing bin when the binner created
    # one, the fallback otherwise.
    na_bin_idx <- NA_integer_
    if (null_to_na_bin && !identical(spec$type, "numerical")) {
      parts <- strsplit(bins, bin_separator, fixed = TRUE)
      if (isTRUE(trim_categories)) parts <- lapply(parts, trimws)
      hit <- which(vapply(parts, function(p) any(p %in% na_categories), logical(1)))
      if (length(hit) > 0L) na_bin_idx <- hit[1L]
    }

    make_expr <- function(kind) {
      vals <- switch(kind,
        "woe" = .ob_sql_num(woe, digits),
        "bin" = .ob_sql_str(bins, d),
        "index" = .ob_sql_num(seq_len(k))
      )
      fallback <- switch(kind,
        "woe" = .ob_sql_num(na_value, digits),
        "bin" = "NULL",
        "index" = "NULL"
      )
      null_val <- if (!is.na(na_bin_idx)) vals[na_bin_idx] else fallback

      .ob_sql_case(
        spec = spec, feature = feat, col = col, d = d,
        values = vals, else_value = fallback, null_value = null_val,
        explicit_bounds = explicit_bounds, indent = indent_str,
        bin_separator = bin_separator, trim_categories = trim_categories
      )
    }

    kinds <- if (identical(output, "both")) c("bin", "woe") else output
    for (kind in kinds) {
      alias <- switch(kind,
        "woe" = paste0(feat, suffix_woe),
        "bin" = paste0(feat, suffix_bin),
        "index" = paste0(feat, "_idx")
      )
      exprs[[alias]] <- make_expr(kind)
    }

    meta[[feat]] <- list(
      type = spec$type, n_bins = k,
      iv = if (!is.null(spec$iv)) sum(as.numeric(spec$iv), na.rm = TRUE) else NA_real_
    )
  }

  if (length(exprs) == 0L) {
    stop("No feature produced a valid SQL expression.")
  }

  # ------------------------------------------------------------------------ #
  # 3. Audit header
  # ------------------------------------------------------------------------ #
  header <- character(0)
  if (isTRUE(comment)) {
    alg <- if (inherits(obj, "obwoe")) {
      a <- obj$summary$algorithm
      paste(unique(a[!is.na(a)]), collapse = ", ")
    } else if (!is.null(obj$algorithm)) {
      obj$algorithm
    } else {
      "unknown"
    }

    header <- c(
      "-- ---------------------------------------------------------------",
      "-- Weight of Evidence transformation",
      sprintf(
        "-- Generated by OptimalBinningWoE %s",
        as.character(utils::packageVersion("OptimalBinningWoE"))
      ),
      sprintf("-- Algorithm(s): %s", alg),
      sprintf("-- Dialect: %s", dialect),
      "-- Interval convention: (lower, upper]  -- upper bound inclusive",
      sprintf("-- Variables: %d", length(meta)),
      "--",
      "-- Variable                 Type          Bins        IV",
      vapply(names(meta), function(nm) {
        m <- meta[[nm]]
        sprintf(
          "--   %-22s %-12s %5d %9s", substr(nm, 1L, 22L),
          if (is.null(m$type)) "?" else m$type, m$n_bins,
          if (is.na(m$iv)) "-" else formatC(m$iv, format = "f", digits = 5)
        )
      }, character(1), USE.NAMES = FALSE),
      "-- ---------------------------------------------------------------"
    )
  }

  # ------------------------------------------------------------------------ #
  # 4. Assembly
  # ------------------------------------------------------------------------ #
  if (identical(style, "case")) {
    out <- unlist(exprs, use.names = TRUE)
    if (isTRUE(comment)) attr(out, "header") <- paste(header, collapse = "\n")
    class(out) <- c("obwoe_sql", "character")
    if (!is.null(file)) writeLines(out, file)
    return(out)
  }

  select_items <- character(0)
  if (!is.null(keep_columns)) {
    if (!is.character(keep_columns)) {
      stop("'keep_columns' must be NULL or a character vector.")
    }
    select_items <- .ob_sql_ident(keep_columns, d, quote_identifiers)
  }

  for (alias in names(exprs)) {
    select_items <- c(select_items, sprintf(
      "%s AS %s", exprs[[alias]],
      .ob_sql_ident(alias, d, quote_identifiers)
    ))
  }

  body <- paste(
    "SELECT",
    paste(select_items, collapse = ",\n"),
    sprintf("FROM %s", .ob_sql_ident(table, d, quote_identifiers)),
    sep = "\n"
  )

  sql <- switch(style,
    "select" = body,
    "cte" = paste0(
      sprintf("WITH %s AS (\n", .ob_sql_ident(view_name, d, quote_identifiers)),
      body, "\n)\nSELECT * FROM ",
      .ob_sql_ident(view_name, d, quote_identifiers)
    ),
    "view" = paste0(
      sprintf(
        "%s %s AS\n", d$create_view,
        .ob_sql_ident(view_name, d, quote_identifiers)
      ),
      body
    )
  )

  if (isTRUE(comment)) {
    sql <- paste(c(header, sql), collapse = "\n")
  }

  sql <- paste0(sql, ";")
  class(sql) <- c("obwoe_sql", "character")

  if (!is.null(file)) writeLines(as.character(sql), file)

  sql
}


#' @title Print Method for Generated SQL
#'
#' @description
#' Echoes the generated SQL verbatim instead of the quoted, escaped form base
#' R would print for a character vector.
#'
#' @param x An object of class \code{"obwoe_sql"}.
#' @param ... Additional arguments (currently ignored).
#'
#' @return Invisibly returns \code{x}.
#'
#' @seealso \code{\link{obwoe_sql}}
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' df <- data.frame(x = rnorm(500))
#' df$target <- rbinom(500, 1, plogis(df$x))
#' print(obwoe_sql(obwoe(df, target = "target"), table = "t"))
#' }
#'
#' @method print obwoe_sql
#' @export
print.obwoe_sql <- function(x, ...) {
  hdr <- attr(x, "header")
  if (!is.null(hdr)) {
    cat(hdr, "\n", sep = "")
  }
  nms <- names(x)
  if (!is.null(nms)) {
    for (i in seq_along(x)) {
      cat("-- ", nms[i], "\n", sep = "")
      cat(x[[i]], "\n\n", sep = "")
    }
  } else {
    cat(as.character(x), "\n", sep = "")
  }
  invisible(x)
}
