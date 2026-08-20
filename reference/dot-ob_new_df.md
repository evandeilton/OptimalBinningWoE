# Internal: Wrap a List of Columns as a data.frame

Builds a `data.frame` from an already-validated list of equal-length
columns without going through
[`data.frame()`](https://rdrr.io/r/base/data.frame.html), whose name
deparsing and per-column coercion dominate the run time when thousands
of small tables are assembled.

## Usage

``` r
.ob_new_df(cols)
```

## Arguments

- cols:

  A named list of equal-length vectors.

## Value

A `data.frame`.
