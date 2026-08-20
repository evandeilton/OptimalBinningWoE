# Internal: Deployment SQL That Returns Points

Reuses the
[`obwoe_sql`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_sql.md)
machinery — the same interval convention, the same escaping, the same
explicit `IS NULL` branch — but emits the integer points of each bin
instead of its Weight of Evidence, and sums them into the score. This is
what a deployment engineer actually runs: the card, not the model.

## Usage

``` r
.ob_points_sql(x, table, dialect, keep_columns = NULL)
```

## Arguments

- x:

  An `"obwoe_scorecard"` object.

- table:

  Source table name.

- dialect:

  SQL dialect.

- keep_columns:

  Columns carried through unchanged.

## Value

A character scalar holding the statement.
