# Generate SQL for a Fitted Optimal Binning

Translates a fitted binning into executable SQL so the Weight of
Evidence transformation can be applied directly inside a database, with
no round trip through R. Every optimised bin becomes one `WHEN` branch
of a `CASE` expression, reproducing the interval and category
assignments of
[`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
exactly.

## Usage

``` r
obwoe_sql(
  obj,
  table = "your_table",
  features = NULL,
  output = c("woe", "bin", "index", "both"),
  style = c("select", "case", "cte", "view"),
  dialect = c("ansi", "postgres", "mysql", "mariadb", "sqlserver", "oracle", "spark",
    "hive", "databricks", "bigquery", "snowflake", "redshift", "duckdb", "sqlite"),
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
  file = NULL
)
```

## Arguments

- obj:

  An object of class `"obwoe"` from
  [`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md),
  a prepped
  [`step_obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md)
  step, or a prepped recipes recipe containing one.

- table:

  Character string naming the source table. May be qualified
  (`"schema.table"`); each part is quoted independently. Required for
  every `style` except `"case"`.

- features:

  Character vector restricting which variables are exported. `NULL`
  (default) exports every successfully binned variable. This is the
  natural place to feed the output of
  [`obwoe_select`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_select.md),
  e.g. `features = sel$feature[sel$selected]`.

- output:

  Character string choosing what the `CASE` returns:

  `"woe"`

  :   The Weight of Evidence value (default).

  `"bin"`

  :   The bin label as a string literal.

  `"index"`

  :   The 1-based bin id.

  `"both"`

  :   Two expressions per feature: bin label and WoE.

- style:

  Character string choosing how the expressions are assembled:

  `"select"`

  :   A complete `SELECT ... FROM table` (default).

  `"case"`

  :   A named character vector of bare `CASE` expressions, one per
      generated column, with no aliases.

  `"cte"`

  :   The `SELECT` wrapped in a `WITH ... AS (...)` common table
      expression.

  `"view"`

  :   A `CREATE VIEW` statement built on the `SELECT`.

- dialect:

  Character string naming the target SQL dialect. One of `"ansi"`
  (default), `"postgres"`, `"mysql"`, `"mariadb"`, `"sqlserver"`,
  `"oracle"`, `"spark"`, `"hive"`, `"databricks"`, `"bigquery"`,
  `"snowflake"`, `"redshift"`, `"duckdb"`, `"sqlite"`.

- view_name:

  Character string naming the view or CTE. Defaults to
  `"woe_transform"`.

- keep_columns:

  Character vector of extra columns to carry through unchanged
  (identifiers, the target, partition keys). Default `NULL`.

- suffix_woe, suffix_bin:

  Character strings appended to the feature name to build the output
  column aliases. Defaults `"_woe"` and `"_bin"`.

- na_value:

  Numeric value returned for `NULL` inputs and for categories unseen
  during training. Default `0`, matching the `na_woe` default of
  [`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md).

- null_to_na_bin:

  Logical. When `TRUE` (default) and the training data contained missing
  values that the binner folded into a bin under one of `na_categories`,
  `NULL` inputs are routed to that bin's value instead of `na_value`.
  [`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
  applies the same rule to `NA` for categorical features as of 1.13.1,
  so the two stay in agreement; keep this `TRUE` unless you have a
  specific reason to diverge from the R-side behavior.

- na_categories:

  Character vector of tokens the binner uses to represent missing
  categories. Default `c("NA", "Missing", "")`, matching
  [`ob_apply_woe_cat`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_cat.md).

- explicit_bounds:

  Logical. When `TRUE` (default) each numerical branch states both of
  its bounds, e.g. `WHEN x > 7 AND x <= 10 THEN ...`, so every branch is
  correct in isolation and survives reordering or copy-and-paste. When
  `FALSE` the branches cascade with upper bounds only, which is shorter
  and equally exact given the top-down evaluation order of `CASE`.

- digits:

  Integer or `NULL` (default). `NULL` writes cut points and WoE values
  at full precision, as the shortest decimal string that parses back to
  the identical double. Supplying a value rounds the literals for
  readability, at the cost of exactness on bin boundaries.

- quote_identifiers:

  Character string: `"auto"` (default) quotes only identifiers that need
  it, `"always"` quotes everything, `"never"` emits bare names.

- indent:

  Integer giving the number of spaces prefixed to each `WHEN` line.
  Default `4`.

- comment:

  Logical. Prepend an audit header with the package version, the
  algorithm and the per-variable Information Value? Default `TRUE`.

- class_index:

  Integer or `NULL` (default). For multinomial models the WoE is a
  matrix; this selects which class column to export.

- bin_separator:

  Character string separating merged categories inside a bin label.
  Default `"%;%"`, matching
  [`control.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/control.obwoe.md).

- trim_categories:

  Logical. When `FALSE` (default) category names are matched byte for
  byte, including any leading or trailing whitespace they carry, which
  is what
  [`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
  does in R. Set it to `TRUE` only when the database column holds the
  unpadded form of a category that reached R padded — a `CHAR(n)` column
  read through a driver that strips the padding, for instance. Trimming
  otherwise makes a genuinely padded category unmatchable.

- file:

  Optional path. When supplied, the generated SQL is also written there
  with [`writeLines`](https://rdrr.io/r/base/writeLines.html).

## Value

An object of class `"obwoe_sql"`, which is a character vector with a
`print` method that echoes the statement verbatim. For `style = "case"`
the vector is named by output column and holds one `CASE` expression per
element; for every other style it is a single string containing the
complete statement. Use
[`as.character()`](https://rdrr.io/r/base/character.html) to strip the
class, or [`cat()`](https://rdrr.io/r/base/cat.html) to display it.

## Details

### Interval semantics

Numerical bins are half-open on the right, the convention used
throughout the package. For cut points \\c_1 \< c_2 \< \cdots \< c_k\\
the \\k+1\\ bins are

\$\$(-\infty, c_1\],\\ (c_1, c_2\],\\ \ldots,\\ (c\_{k-1}, c_k\],\\
(c_k, +\infty)\$\$

which the generated SQL renders as `x <= c1`, `x > c1 AND x <= c2`, and
so on. This mirrors
`cut(x, breaks = c(-Inf, cutpoints, Inf), right = TRUE)`, the exact call
made by
[`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md),
so an observation sitting on a cut point lands in the *lower* bin in R
and in SQL alike.

Cut points are read from the fitted `cutpoints` vector, never parsed
back from bin labels: label formatting varies between algorithms whereas
`cutpoints` is the authoritative numeric boundary. Duplicated cut points
are removed exactly as
[`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
removes them, and a variable whose de-duplicated cut points no longer
match its bin count is skipped with a warning rather than exported with
a wrong mapping.

### Numeric literals

With `digits = NULL` each cut point is written as the shortest
fixed-notation decimal that parses back to the identical IEEE 754
double. Rounding a boundary such as `4049.5` to fewer digits would
silently move observations between bins, so exactness is the default;
scientific notation is avoided because dialects differ on how they type
such literals.

### NULL handling

In SQL, `NULL <= 5` evaluates to `NULL`, not to `FALSE`, so a missing
value matches no comparison branch and would silently fall through to
`ELSE`. Every generated expression therefore opens with an explicit
`WHEN <col> IS NULL` branch. Categorical binners represent training
missings as the literal category `"NA"`; when `null_to_na_bin = TRUE`
the `IS NULL` branch returns that bin's value, so database `NULL`s are
scored the way missings were scored during fitting. The `ELSE` branch
catches categories never seen in training and returns `na_value`.

### Escaping

Bin labels are the original category strings joined by `bin_separator`,
so splitting a label recovers the categories byte for byte. They are
matched exactly, whitespace included, unless `trim_categories = TRUE`.

Category labels are emitted as quoted literals with embedded single
quotes doubled per ANSI SQL. On MySQL, MariaDB and the Hive family —
where a backslash also escapes inside string literals under default
settings — backslashes are doubled too. Identifiers are quoted with the
dialect's own delimiters, and by default only when the name is not a
plain `[A-Za-z_][A-Za-z0-9_]*` token or collides with a reserved word;
this keeps generated code readable on case-folding engines such as
Oracle and Snowflake, where blanket quoting would force case
sensitivity.

## References

Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
Intelligent Credit Scoring. *John Wiley & Sons*.
[doi:10.1002/9781119201731](https://doi.org/10.1002/9781119201731)

International Organization for Standardization (2016). *ISO/IEC
9075-2:2016 Information technology — Database languages — SQL — Part 2:
Foundation (SQL/Foundation)*.

## See also

[`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
for fitting the binning,
[`obwoe_select`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_select.md)
for choosing which variables to export,
[`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
for the equivalent transformation in R.

## Examples

``` r
# \donttest{
set.seed(42)
n <- 1000
df <- data.frame(
  age = rnorm(n, 40, 12),
  region = sample(c("North", "South", "O'Hare"), n, replace = TRUE)
)
df$target <- rbinom(n, 1, plogis(-1 + 0.04 * (df$age - 40)))

model <- obwoe(df, target = "target", max_bins = 5)

# Complete SELECT statement
sql <- obwoe_sql(model, table = "credit.applications", keep_columns = "target")
sql
#> -- ---------------------------------------------------------------
#> -- Weight of Evidence transformation
#> -- Generated by OptimalBinningWoE 1.13.1
#> -- Algorithm(s): jedi
#> -- Dialect: ansi
#> -- Interval convention: (lower, upper]  -- upper bound inclusive
#> -- Variables: 2
#> --
#> -- Variable                 Type          Bins        IV
#> --   age                    numerical        5   0.18377
#> --   region                 categorical      3   0.00243
#> -- ---------------------------------------------------------------
#> SELECT
#> target,
#> CASE
#>     WHEN age IS NULL THEN 0
#>     WHEN age <= 20.066811041022255 THEN -1.3355998243282543
#>     WHEN age > 20.066811041022255 AND age <= 24.523799880116762 THEN -0.654722736360123
#>     WHEN age > 24.523799880116762 AND age <= 55.38971035747166 THEN -0.03375929404687619
#>     WHEN age > 55.38971035747166 AND age <= 58.399365905947526 THEN 0.6204626961910772
#>     WHEN age > 58.399365905947526 THEN 1.1843981452710164
#>     ELSE 0
#> END AS age_woe,
#> CASE
#>     WHEN region IS NULL THEN 0
#>     WHEN region = 'South' THEN -0.07314379909961256
#>     WHEN region = 'North' THEN 0.024697485123917937
#>     WHEN region = 'O''Hare' THEN 0.03923833154382199
#>     ELSE 0
#> END AS region_woe
#> FROM credit.applications;

# Only the variables that survived automatic screening
sel <- obwoe_select(model)
obwoe_sql(model,
  table = "credit.applications",
  features = sel$feature[sel$selected]
)
#> -- ---------------------------------------------------------------
#> -- Weight of Evidence transformation
#> -- Generated by OptimalBinningWoE 1.13.1
#> -- Algorithm(s): jedi
#> -- Dialect: ansi
#> -- Interval convention: (lower, upper]  -- upper bound inclusive
#> -- Variables: 1
#> --
#> -- Variable                 Type          Bins        IV
#> --   age                    numerical        5   0.18377
#> -- ---------------------------------------------------------------
#> SELECT
#> CASE
#>     WHEN age IS NULL THEN 0
#>     WHEN age <= 20.066811041022255 THEN -1.3355998243282543
#>     WHEN age > 20.066811041022255 AND age <= 24.523799880116762 THEN -0.654722736360123
#>     WHEN age > 24.523799880116762 AND age <= 55.38971035747166 THEN -0.03375929404687619
#>     WHEN age > 55.38971035747166 AND age <= 58.399365905947526 THEN 0.6204626961910772
#>     WHEN age > 58.399365905947526 THEN 1.1843981452710164
#>     ELSE 0
#> END AS age_woe
#> FROM credit.applications;

# Bin labels and WoE side by side, as a Spark view
obwoe_sql(model,
  table = "credit.applications", output = "both",
  style = "view", dialect = "spark", view_name = "v_woe"
)
#> -- ---------------------------------------------------------------
#> -- Weight of Evidence transformation
#> -- Generated by OptimalBinningWoE 1.13.1
#> -- Algorithm(s): jedi
#> -- Dialect: spark
#> -- Interval convention: (lower, upper]  -- upper bound inclusive
#> -- Variables: 2
#> --
#> -- Variable                 Type          Bins        IV
#> --   age                    numerical        5   0.18377
#> --   region                 categorical      3   0.00243
#> -- ---------------------------------------------------------------
#> CREATE OR REPLACE VIEW v_woe AS
#> SELECT
#> CASE
#>     WHEN age IS NULL THEN NULL
#>     WHEN age <= 20.066811041022255 THEN '(-Inf;20.066811]'
#>     WHEN age > 20.066811041022255 AND age <= 24.523799880116762 THEN '(20.066811;24.523800]'
#>     WHEN age > 24.523799880116762 AND age <= 55.38971035747166 THEN '(24.523800;55.389710]'
#>     WHEN age > 55.38971035747166 AND age <= 58.399365905947526 THEN '(55.389710;58.399366]'
#>     WHEN age > 58.399365905947526 THEN '(58.399366;+Inf]'
#>     ELSE NULL
#> END AS age_bin,
#> CASE
#>     WHEN age IS NULL THEN 0
#>     WHEN age <= 20.066811041022255 THEN -1.3355998243282543
#>     WHEN age > 20.066811041022255 AND age <= 24.523799880116762 THEN -0.654722736360123
#>     WHEN age > 24.523799880116762 AND age <= 55.38971035747166 THEN -0.03375929404687619
#>     WHEN age > 55.38971035747166 AND age <= 58.399365905947526 THEN 0.6204626961910772
#>     WHEN age > 58.399365905947526 THEN 1.1843981452710164
#>     ELSE 0
#> END AS age_woe,
#> CASE
#>     WHEN region IS NULL THEN NULL
#>     WHEN region = 'South' THEN 'South'
#>     WHEN region = 'North' THEN 'North'
#>     WHEN region = 'O''Hare' THEN 'O''Hare'
#>     ELSE NULL
#> END AS region_bin,
#> CASE
#>     WHEN region IS NULL THEN 0
#>     WHEN region = 'South' THEN -0.07314379909961256
#>     WHEN region = 'North' THEN 0.024697485123917937
#>     WHEN region = 'O''Hare' THEN 0.03923833154382199
#>     ELSE 0
#> END AS region_woe
#> FROM credit.applications;

# Bare CASE expressions for embedding in an existing query
obwoe_sql(model, style = "case")
#> -- ---------------------------------------------------------------
#> -- Weight of Evidence transformation
#> -- Generated by OptimalBinningWoE 1.13.1
#> -- Algorithm(s): jedi
#> -- Dialect: ansi
#> -- Interval convention: (lower, upper]  -- upper bound inclusive
#> -- Variables: 2
#> --
#> -- Variable                 Type          Bins        IV
#> --   age                    numerical        5   0.18377
#> --   region                 categorical      3   0.00243
#> -- ---------------------------------------------------------------
#> -- age_woe
#> CASE
#>     WHEN age IS NULL THEN 0
#>     WHEN age <= 20.066811041022255 THEN -1.3355998243282543
#>     WHEN age > 20.066811041022255 AND age <= 24.523799880116762 THEN -0.654722736360123
#>     WHEN age > 24.523799880116762 AND age <= 55.38971035747166 THEN -0.03375929404687619
#>     WHEN age > 55.38971035747166 AND age <= 58.399365905947526 THEN 0.6204626961910772
#>     WHEN age > 58.399365905947526 THEN 1.1843981452710164
#>     ELSE 0
#> END
#> 
#> -- region_woe
#> CASE
#>     WHEN region IS NULL THEN 0
#>     WHEN region = 'South' THEN -0.07314379909961256
#>     WHEN region = 'North' THEN 0.024697485123917937
#>     WHEN region = 'O''Hare' THEN 0.03923833154382199
#>     ELSE 0
#> END
#> 
# }
```
