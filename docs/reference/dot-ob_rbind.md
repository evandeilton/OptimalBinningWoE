# Internal: Fast Row Binding of Homogeneous Blocks

Uses
[`data.table::rbindlist()`](https://rdrr.io/pkg/data.table/man/rbindlist.html)
when available (O(n) in the number of blocks) and falls back to
`do.call(rbind, ...)` otherwise. Blocks may be `data.frame`s or plain
named lists of equal-length columns.

## Usage

``` r
.ob_rbind(lst)
```

## Arguments

- lst:

  A list of blocks sharing the same columns.

## Value

A single table with all rows stacked.
