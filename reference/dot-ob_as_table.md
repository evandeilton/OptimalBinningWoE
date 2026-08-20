# Internal: Coerce a data.frame to data.table When Available

Returns a `data.table` when the data.table package is installed (the
fast path preferred for wide feature bases) and the original
`data.frame` otherwise. Because `data.table` inherits from `data.frame`,
downstream code is unaffected either way.

## Usage

``` r
.ob_as_table(x)
```

## Arguments

- x:

  A `data.frame`.

## Value

A `data.table` or the input `data.frame`.
