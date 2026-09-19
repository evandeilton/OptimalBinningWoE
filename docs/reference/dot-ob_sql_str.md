# Internal: SQL String Literal

Wraps a value in single quotes, doubling embedded single quotes as ANSI
SQL requires. On dialects where a backslash also escapes inside string
literals (MySQL/MariaDB and the Hive family by default) backslashes are
doubled as well, so a category such as `"a\b"` survives verbatim.

## Usage

``` r
.ob_sql_str(x, d)
```

## Arguments

- x:

  Character vector.

- d:

  Dialect list from
  [`.ob_sql_dialect`](https://evandeilton.github.io/OptimalBinningWoE/reference/dot-ob_sql_dialect.md).

## Value

Character vector of quoted literals.
