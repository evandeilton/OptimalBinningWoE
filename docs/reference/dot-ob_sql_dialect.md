# Internal: SQL Dialect Registry

Per-dialect syntax facts needed to emit portable, exact SQL: the
identifier quoting characters, whether backslashes act as escapes inside
string literals, and the statement used to (re)create a view.

## Usage

``` r
.ob_sql_dialect(dialect)
```

## Arguments

- dialect:

  Character string naming the dialect.

## Value

A list with `open`, `close`, `escape_backslash` and `create_view`.
