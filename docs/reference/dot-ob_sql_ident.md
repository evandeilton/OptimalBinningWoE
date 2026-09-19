# Internal: Quote a SQL Identifier

Internal: Quote a SQL Identifier

## Usage

``` r
.ob_sql_ident(x, d, mode = "auto")
```

## Arguments

- x:

  Character vector of identifiers. Dotted names such as `"schema.table"`
  are quoted part by part.

- d:

  Dialect list from
  [`.ob_sql_dialect`](https://evandeilton.github.io/OptimalBinningWoE/reference/dot-ob_sql_dialect.md).

- mode:

  One of `"auto"`, `"always"`, `"never"`.

## Value

Character vector of quoted identifiers.
