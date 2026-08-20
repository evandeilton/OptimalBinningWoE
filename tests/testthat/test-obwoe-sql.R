# ===========================================================================#
# obwoe_sql(): SQL code generation
#
# The generated text is not compared against the code that produced it. It is
# parsed and executed by the small CASE interpreter in
# helper-germancredit.R, then the resulting bin assignment is checked against
# the counts the binning algorithm itself reported. A wrong boundary, a
# mis-escaped literal or a mishandled NULL therefore surfaces as a wrong bin
# assignment, not as a string diff.
# ===========================================================================#

test_that("generated SQL reproduces the fitted bins on German Credit", {
  skip_if_no_german()
  df <- german_credit()
  feats <- setdiff(names(df), "target")

  numeric_feats <- feats[vapply(feats, function(f) is.numeric(df[[f]]), logical(1))]

  # "ir" is numeric-only; the rest handle both feature types.
  cases <- list(
    jedi = feats, mob = feats, dp = feats, cm = feats, udt = feats,
    ir = numeric_feats
  )

  for (alg in names(cases)) {
    use <- cases[[alg]]
    model <- suppressWarnings(obwoe(df,
      target = "target", feature = use,
      algorithm = alg, max_bins = 6
    ))
    problems <- sql_reproduces_bin_counts(model, df, use)
    expect_equal(problems, character(0), info = alg)
  }
})


test_that("cascading and fully-bounded predicates agree exactly", {
  skip_if_no_german()
  df <- german_credit()
  feats <- setdiff(names(df), "target")
  model <- obwoe(df, target = "target", feature = feats, max_bins = 7)

  a <- obwoe_sql(model,
    output = "woe", style = "case", comment = FALSE,
    quote_identifiers = "never", explicit_bounds = TRUE
  )
  b <- obwoe_sql(model,
    output = "woe", style = "case", comment = FALSE,
    quote_identifiers = "never", explicit_bounds = FALSE
  )
  expect_setequal(names(a), names(b))

  for (f in feats) {
    alias <- paste0(f, "_woe")
    if (!alias %in% names(a)) next
    va <- sql_eval_case_num(a[[alias]], df[[f]], f)
    vb <- sql_eval_case_num(b[[alias]], df[[f]], f)
    expect_equal(va, vb, info = f)
  }

  # Fully-bounded form states both limits; the cascading form does not
  num <- feats[vapply(feats, function(f) is.numeric(df[[f]]), logical(1))][1]
  expect_match(a[[paste0(num, "_woe")]], " AND ")
  expect_false(grepl(" AND ", b[[paste0(num, "_woe")]]))
})


test_that("the WoE emitted by SQL equals the WoE applied in R", {
  skip_if_no_german()
  df <- german_credit()
  feats <- setdiff(names(df), "target")
  model <- obwoe(df, target = "target", feature = feats, max_bins = 6)

  scored <- obwoe_apply(df, model)
  cases <- obwoe_sql(model,
    output = "woe", style = "case",
    comment = FALSE, quote_identifiers = "never"
  )

  for (f in feats) {
    alias <- paste0(f, "_woe")
    if (!alias %in% names(cases)) next
    expect_equal(
      sql_eval_case_num(cases[[alias]], df[[f]], f),
      scored[[alias]],
      tolerance = 0, info = f
    )
  }
})


test_that("values sitting exactly on a cut point fall in the lower bin", {
  # (a, b] means the upper bound belongs to the bin: an observation equal to a
  # cut point must be scored by the bin that ends there.
  set.seed(13)
  n <- 3000
  x <- rep(c(1, 2, 3, 4, 5), length.out = n)
  df <- data.frame(
    x = x,
    target = rbinom(n, 1, c(0.1, 0.2, 0.35, 0.5, 0.7)[x])
  )
  model <- obwoe(df, target = "target", min_bins = 3, max_bins = 5)
  cp <- sort(unique(model$results$x$cutpoints))
  skip_if(length(cp) == 0, "no cut points produced")

  case <- obwoe_sql(model,
    output = "index", style = "case",
    comment = FALSE, quote_identifiers = "never"
  )[["x_idx"]]

  probe <- c(cp, cp - .Machine$double.eps^0.5, cp + .Machine$double.eps^0.5)
  got <- sql_eval_case_num(case, probe, "x")
  ref <- cut(probe,
    breaks = c(-Inf, cp, Inf), labels = FALSE,
    right = TRUE, include.lowest = TRUE
  )
  expect_equal(got, as.numeric(ref))

  # explicitly: a value equal to cp[1] belongs to bin 1, not bin 2
  expect_equal(sql_eval_case_num(case, cp[1], "x"), 1)
})


test_that("cut points survive the decimal round trip exactly", {
  f <- OptimalBinningWoE:::.ob_sql_num

  tricky <- c(
    0.1, 0.3, 1 / 3, 2 / 3, 4049.5, 1e-9, 1e9, -1e-7,
    pi, exp(1), 0.1 + 0.2, 1234567.891, -0.0000001234,
    .Machine$double.eps, 1 - .Machine$double.eps
  )
  lit <- f(tricky)
  expect_equal(as.numeric(lit), tricky, tolerance = 0)

  # No scientific notation: dialects disagree on how such literals are typed
  expect_false(any(grepl("e", lit, ignore.case = TRUE)))

  # NA and infinities degrade safely
  expect_equal(f(NA_real_), "NULL")
  expect_equal(f(c(-Inf, Inf)), c("-1e308", "1e308"))

  # Rounding is opt-in
  expect_equal(f(pi, digits = 3), "3.142")
})


test_that("string literals are escaped for the target dialect", {
  esc <- OptimalBinningWoE:::.ob_sql_str
  ansi <- OptimalBinningWoE:::.ob_sql_dialect("ansi")
  mysql <- OptimalBinningWoE:::.ob_sql_dialect("mysql")

  expect_equal(esc("O'Brien", ansi), "'O''Brien'")
  expect_equal(esc("a''b", ansi), "'a''''b'")
  expect_equal(esc('say "hi"', ansi), "'say \"hi\"'")

  # Backslash is literal under ANSI and an escape character under MySQL
  expect_equal(esc("a\\b", ansi), "'a\\b'")
  expect_equal(esc("a\\b", mysql), "'a\\\\b'")

  # Round trip through the reader
  expect_equal(sql_parse_strings(esc("O'Brien", ansi)), "O'Brien")
  expect_equal(sql_parse_strings(esc("a, b", ansi)), "a, b")
})


test_that("awkward category names are matched exactly", {
  set.seed(17)
  n <- 3000
  cats <- c(
    "O'Brien", "a\\b", "50% off", "say \"hi\"", "acentuacao",
    "  padded  ", "x;y", "tab\there", "UPPER", "upper"
  )
  g <- sample(cats, n, TRUE)
  df <- data.frame(
    g = g,
    target = rbinom(n, 1, plogis(-0.4 + 0.8 * (g %in% cats[1:3]))),
    stringsAsFactors = FALSE
  )

  model <- obwoe(df, target = "target", max_bins = 10)
  skip_if(!is.null(model$results$g$error), "binning failed on the constructed case")

  expect_equal(sql_reproduces_bin_counts(model, df, "g"), character(0))

  # Whitespace is significant: a padded category must not be trimmed away
  case <- obwoe_sql(model,
    output = "bin", style = "case", comment = FALSE,
    quote_identifiers = "never"
  )[["g_bin"]]
  expect_true(grepl("'  padded  '", case, fixed = TRUE))
  expect_equal(sql_eval_case_str(case, "  padded  ", "g"),
    grep("padded", model$results$g$bin, value = TRUE)[1],
    ignore_attr = TRUE
  )
  # Trimming is opt-in, for a database column that holds the unpadded form
  trimmed <- obwoe_sql(model,
    output = "bin", style = "case", comment = FALSE,
    quote_identifiers = "never", trim_categories = TRUE
  )[["g_bin"]]
  expect_true(grepl("'padded'", trimmed, fixed = TRUE))

  # Case is significant too
  expect_false(identical(
    sql_eval_case_str(case, "UPPER", "g"),
    sql_eval_case_str(case, "upper", "g")
  ))
})


test_that("NULL never leaks into a comparison branch", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df,
    target = "target", feature = c("duration", "purpose"),
    max_bins = 5
  )

  cases <- obwoe_sql(model,
    output = "woe", style = "case", comment = FALSE,
    quote_identifiers = "never", na_value = -99
  )

  # The IS NULL branch must be the first WHEN of every expression
  for (nm in names(cases)) {
    first_when <- grep("^\\s*WHEN ", strsplit(cases[[nm]], "\n")[[1L]],
      value = TRUE
    )[1L]
    expect_match(first_when, "IS NULL")
  }

  expect_equal(sql_eval_case_num(cases[["duration_woe"]], NA_real_, "duration"), -99)
  expect_equal(sql_eval_case_num(cases[["purpose_woe"]], NA_character_, "purpose"), -99)

  # Unseen categories fall through to ELSE, not to a neighbouring bin
  expect_equal(
    sql_eval_case_num(cases[["purpose_woe"]], "never-seen", "purpose"), -99
  )
})


test_that("NULL is routed to the missing bin when the binner created one", {
  set.seed(23)
  n <- 2000
  g <- sample(c("lo", "mid", "hi"), n, TRUE)
  df <- data.frame(
    g = g,
    target = rbinom(n, 1, plogis(-0.4 + 0.8 * (g == "hi"))),
    stringsAsFactors = FALSE
  )
  df$g[1:300] <- NA

  model <- obwoe(df, target = "target", max_bins = 5)
  bins <- model$results$g$bin
  skip_if(!any(grepl("NA", bins, fixed = TRUE)), "no missing bin was created")

  na_woe <- model$results$g$woe[grep("NA", bins, fixed = TRUE)[1L]]

  on <- obwoe_sql(model,
    output = "woe", style = "case", comment = FALSE,
    quote_identifiers = "never", null_to_na_bin = TRUE
  )[["g_woe"]]
  expect_equal(sql_eval_case_num(on, NA_character_, "g"), na_woe)

  off <- obwoe_sql(model,
    output = "woe", style = "case", comment = FALSE,
    quote_identifiers = "never", null_to_na_bin = FALSE, na_value = 0
  )[["g_woe"]]
  expect_equal(sql_eval_case_num(off, NA_character_, "g"), 0)

  # With NULLs mapped to the missing bin, the SQL reproduces the fitted counts
  ref <- df
  ref$g[is.na(ref$g)] <- "NA"
  expect_equal(sql_reproduces_bin_counts(model, ref, "g"), character(0))
})


test_that("identifier quoting follows the dialect and the mode", {
  ident <- OptimalBinningWoE:::.ob_sql_ident
  ansi <- OptimalBinningWoE:::.ob_sql_dialect("ansi")
  ms <- OptimalBinningWoE:::.ob_sql_dialect("sqlserver")
  my <- OptimalBinningWoE:::.ob_sql_dialect("mysql")

  expect_equal(ident("plain_col", ansi, "auto"), "plain_col")
  expect_equal(ident("plain_col", ansi, "always"), "\"plain_col\"")
  expect_equal(ident("plain_col", ms, "always"), "[plain_col]")
  expect_equal(ident("plain_col", my, "always"), "`plain_col`")

  # Reserved words and awkward names are quoted even in auto mode
  expect_equal(ident("select", ansi, "auto"), "\"select\"")
  expect_equal(ident("ORDER", ansi, "auto"), "\"ORDER\"")
  expect_equal(ident("my col", ansi, "auto"), "\"my col\"")
  expect_equal(ident("2fast", ansi, "auto"), "\"2fast\"")
  expect_equal(ident("never quoted", ansi, "never"), "never quoted")

  # Qualified names are quoted part by part
  expect_equal(ident("schema.table", ansi, "always"), "\"schema\".\"table\"")
  expect_equal(ident("db.select", ansi, "auto"), "db.\"select\"")

  # An embedded delimiter is doubled
  expect_equal(ident("we\"ird", ansi, "auto"), "\"we\"\"ird\"")
})


test_that("reserved-word columns produce runnable predicates", {
  set.seed(29)
  n <- 1500
  df <- data.frame(order = rnorm(n))
  df$target <- rbinom(n, 1, plogis(df$order))

  model <- obwoe(df, target = "target", feature = "order", max_bins = 4)
  case <- obwoe_sql(model,
    output = "index", style = "case", comment = FALSE
  )[["order_idx"]]

  expect_true(grepl("\"order\"", case, fixed = TRUE))
  cp <- sort(unique(model$results$order$cutpoints))
  ref <- cut(df$order,
    breaks = c(-Inf, cp, Inf), labels = FALSE,
    right = TRUE, include.lowest = TRUE
  )
  expect_equal(sql_eval_case_num(case, df$order, "order"), as.numeric(ref))
})


test_that("assembly styles produce the expected statement shapes", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", feature = c("duration", "purpose"))

  sel <- obwoe_sql(model, table = "credit.app", keep_columns = "target")
  expect_s3_class(sel, "obwoe_sql")
  expect_length(sel, 1L)
  expect_match(as.character(sel), "^--")
  expect_match(as.character(sel), "SELECT")
  expect_match(as.character(sel), "FROM credit\\.app")
  expect_match(as.character(sel), ";$")

  bare <- obwoe_sql(model, table = "t", comment = FALSE)
  expect_match(as.character(bare), "^SELECT")

  cte <- obwoe_sql(model, table = "t", style = "cte", view_name = "w", comment = FALSE)
  expect_match(as.character(cte), "^WITH w AS \\(")
  expect_match(as.character(cte), "SELECT \\* FROM w;$")

  vw <- obwoe_sql(model, table = "t", style = "view", view_name = "v", comment = FALSE)
  expect_match(as.character(vw), "^CREATE OR REPLACE VIEW v AS")

  # SQL Server has no CREATE OR REPLACE VIEW
  vsrv <- obwoe_sql(model,
    table = "t", style = "view", view_name = "v",
    dialect = "sqlserver", comment = FALSE
  )
  expect_match(as.character(vsrv), "^CREATE VIEW v AS")
  expect_false(grepl("OR REPLACE", as.character(vsrv)))

  cs <- obwoe_sql(model, style = "case", comment = FALSE)
  expect_length(cs, 2L)
  expect_setequal(names(cs), c("duration_woe", "purpose_woe"))
  expect_false(any(grepl(" AS ", cs)))
  expect_false(any(grepl("SELECT", cs)))
})


test_that("output modes emit the requested columns", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", feature = "duration", max_bins = 5)

  expect_equal(names(obwoe_sql(model, style = "case")), "duration_woe")
  expect_equal(
    names(obwoe_sql(model, style = "case", output = "bin")), "duration_bin"
  )
  expect_equal(
    names(obwoe_sql(model, style = "case", output = "index")), "duration_idx"
  )
  expect_equal(
    names(obwoe_sql(model, style = "case", output = "both")),
    c("duration_bin", "duration_woe")
  )

  # Custom suffixes
  expect_equal(
    names(obwoe_sql(model, style = "case", suffix_woe = "__W")), "duration__W"
  )

  # index output enumerates the bins in fitted order
  idx <- obwoe_sql(model,
    style = "case", output = "index", comment = FALSE,
    quote_identifiers = "never"
  )[["duration_idx"]]
  got <- sql_eval_case_num(idx, df$duration, "duration")
  expect_setequal(unique(got), seq_along(model$results$duration$bin))
})


test_that("feature filtering integrates with obwoe_select", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", max_bins = 6)
  sel <- as.data.frame(obwoe_select(model))
  keep <- sel$feature[sel$selected]
  skip_if(length(keep) == 0, "nothing was selected")

  cases <- obwoe_sql(model, features = keep, style = "case", comment = FALSE)
  expect_setequal(names(cases), paste0(keep, "_woe"))

  expect_error(
    obwoe_sql(model, features = "not_a_variable"),
    "not present in the fitted binning"
  )
})


test_that("digits rounds the emitted literals", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", feature = "duration", max_bins = 5)

  exact <- obwoe_sql(model, style = "case", comment = FALSE)[["duration_woe"]]
  short <- obwoe_sql(model,
    style = "case", comment = FALSE, digits = 4
  )[["duration_woe"]]

  expect_gt(nchar(exact), nchar(short))
  decimals <- regmatches(short, gregexpr("\\.[0-9]+", short))[[1L]]
  expect_true(all(nchar(decimals) - 1L <= 4L))
})


test_that("the audit header records the provenance of the code", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", feature = "age", algorithm = "mob")

  sql <- as.character(obwoe_sql(model, table = "t"))
  expect_match(sql, "OptimalBinningWoE")
  expect_match(sql, as.character(utils::packageVersion("OptimalBinningWoE")),
    fixed = TRUE
  )
  expect_match(sql, "mob")
  expect_match(sql, "(lower, upper]", fixed = TRUE)

  expect_false(grepl("^--", as.character(
    obwoe_sql(model, table = "t", comment = FALSE)
  )))
})


test_that("SQL can be written to a file", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", feature = "age")
  path <- tempfile(fileext = ".sql")
  on.exit(unlink(path), add = TRUE)

  sql <- obwoe_sql(model, table = "t", file = path)
  expect_true(file.exists(path))
  expect_equal(
    paste(readLines(path), collapse = "\n"),
    as.character(sql)
  )
})


test_that("the print method echoes the statement verbatim", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", feature = "age")

  sql <- obwoe_sql(model, table = "t", comment = FALSE)
  expect_output(print(sql), "SELECT")
  expect_output(res <- print(sql))
  expect_identical(res, sql)

  cases <- obwoe_sql(model, style = "case", comment = FALSE)
  expect_output(print(cases), "age_woe")
  expect_output(print(cases), "CASE")
})


test_that("input validation rejects malformed arguments", {
  df <- data.frame(x = rnorm(400))
  df$target <- rbinom(400, 1, plogis(df$x))
  model <- obwoe(df, target = "target")

  expect_error(obwoe_sql(list()), "must be an 'obwoe' object")
  expect_error(obwoe_sql(model, table = c("a", "b")), "single character")
  expect_error(obwoe_sql(model, na_value = NA), "non-missing number")
  expect_error(obwoe_sql(model, digits = -1), "non-negative integer")
  expect_error(obwoe_sql(model, explicit_bounds = "yes"), "logical")
  expect_error(obwoe_sql(model, null_to_na_bin = 1), "logical")
  expect_error(obwoe_sql(model, trim_categories = "no"), "logical")
  expect_error(obwoe_sql(model, keep_columns = 1), "character vector")
  expect_error(obwoe_sql(model, features = 1), "character vector")
  expect_error(obwoe_sql(model, dialect = "nosuchdb"))
  expect_error(obwoe_sql(model, style = "nope"))
})


test_that("features whose binning failed are skipped with a warning", {
  df <- data.frame(
    good = rnorm(400),
    broken = c(NA_real_, rnorm(399)),
    target = rbinom(400, 1, 0.4)
  )
  model <- suppressWarnings(obwoe(df, target = "target"))
  skip_if(!any(model$summary$error), "no feature failed in this fit")

  expect_warning(
    cases <- obwoe_sql(model, style = "case", comment = FALSE),
    "no valid binning"
  )
  expect_equal(names(cases), "good_woe")

  # If nothing is left, that is an error rather than an empty statement
  broken_only <- model
  broken_only$results <- model$results["broken"]
  expect_error(
    suppressWarnings(obwoe_sql(broken_only, style = "case")),
    "No feature produced a valid SQL expression"
  )
})


test_that("a single-bin numerical feature emits a constant expression", {
  set.seed(31)
  n <- 400
  df <- data.frame(flat = rep(2.5, n), target = rbinom(n, 1, 0.3))
  model <- suppressWarnings(obwoe(df, target = "target"))
  skip_if(!is.null(model$results$flat$error), "binning failed")
  skip_if(length(model$results$flat$bin) != 1L, "more than one bin produced")

  case <- obwoe_sql(model,
    style = "case", comment = FALSE,
    quote_identifiers = "never"
  )[["flat_woe"]]
  expect_equal(
    sql_eval_case_num(case, c(-1e6, 2.5, 1e6), "flat"),
    rep(model$results$flat$woe[1L], 3L)
  )
  expect_equal(sql_eval_case_num(case, NA_real_, "flat"), 0)
})


test_that("multinomial models require an explicit class", {
  set.seed(37)
  n <- 900
  df <- data.frame(x = rnorm(n), y = sample(0:2, n, TRUE))
  model <- obwoe(df, target = "y")
  expect_equal(model$target_type, "multinomial")

  expect_warning(
    expect_error(obwoe_sql(model, style = "case"), "No feature produced"),
    "class_index"
  )

  cases <- obwoe_sql(model, style = "case", comment = FALSE, class_index = 2L)
  expect_equal(names(cases), "x_woe")
  expect_equal(
    sql_eval_case_num(cases[["x_woe"]], df$x, "x"),
    model$results$x$woe[
      cut(df$x,
        breaks = c(-Inf, sort(unique(model$results$x$cutpoints)), Inf),
        labels = FALSE, right = TRUE, include.lowest = TRUE
      ), 2L
    ]
  )

  # Bin labels do not depend on the class, so they need no class_index
  expect_equal(
    names(obwoe_sql(model, style = "case", output = "bin")), "x_bin"
  )
})


test_that("a prepped step_obwoe recipe can be exported", {
  skip_if_not_installed("recipes")
  skip_if_no_german()
  df <- german_credit()
  df$target <- factor(df$target, levels = c(0, 1))

  rec <- recipes::recipe(target ~ duration + purpose + age, data = df)
  rec <- step_obwoe(rec, recipes::all_predictors(),
    outcome = "target", max_bins = 5
  )
  prepped <- recipes::prep(rec, training = df)

  cases <- obwoe_sql(prepped, style = "case", comment = FALSE)
  expect_true(all(c("duration_woe", "purpose_woe", "age_woe") %in% names(cases)))

  # The step and the model agree on the mapping
  model <- obwoe(df, target = "target", feature = "duration", max_bins = 5)
  direct <- obwoe_sql(model,
    style = "case", comment = FALSE,
    quote_identifiers = "never"
  )[["duration_woe"]]
  from_rec <- obwoe_sql(prepped,
    style = "case", comment = FALSE,
    quote_identifiers = "never"
  )[["duration_woe"]]
  expect_equal(
    sql_eval_case_num(from_rec, df$duration, "duration"),
    sql_eval_case_num(direct, df$duration, "duration")
  )

  expect_error(obwoe_sql(rec), "must be prepped")
})


test_that("all dialects emit syntactically consistent code", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df,
    target = "target", feature = c("duration", "purpose"),
    max_bins = 5
  )

  dialects <- c(
    "ansi", "postgres", "mysql", "mariadb", "sqlserver", "oracle", "spark",
    "hive", "databricks", "bigquery", "snowflake", "redshift", "duckdb",
    "sqlite"
  )
  for (dl in dialects) {
    sql <- as.character(obwoe_sql(model,
      table = "credit.app", dialect = dl,
      keep_columns = "target", comment = FALSE
    ))
    expect_match(sql, "^SELECT", info = dl)
    expect_match(sql, "FROM credit\\.app", info = dl)
    expect_equal(
      lengths(regmatches(sql, gregexpr("\\bCASE\\b", sql))),
      lengths(regmatches(sql, gregexpr("\\bEND\\b", sql))),
      info = dl
    )
    expect_match(sql, ";$", info = dl)
  }
})
