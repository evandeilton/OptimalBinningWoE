# Internal: CASE Expression for a Single Feature

Internal: CASE Expression for a Single Feature

## Usage

``` r
.ob_sql_case(
  spec,
  feature,
  col,
  d,
  values,
  else_value,
  null_value,
  explicit_bounds,
  indent,
  bin_separator,
  trim_categories = FALSE
)
```

## Arguments

- spec:

  One element of
  [`.ob_sql_spec`](https://evandeilton.github.io/OptimalBinningWoE/reference/dot-ob_sql_spec.md).

- feature:

  Feature name (unquoted).

- col:

  Quoted column reference used inside the predicates.

- d:

  Dialect list.

- values:

  Character vector of THEN literals, one per bin.

- else_value:

  Character scalar used by the ELSE branch.

- null_value:

  Character scalar returned when the column is NULL.

- explicit_bounds:

  Logical; emit fully qualified interval predicates.

- indent:

  Character string prefixed to each WHEN line.

- bin_separator:

  Separator inside merged categorical bin labels.

- trim_categories:

  Logical; strip surrounding whitespace from category names after
  splitting a merged bin label.

## Value

A character scalar holding the CASE expression.
