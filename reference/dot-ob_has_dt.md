# Internal: Is data.table Available?

Single point of truth for the fast-path check, so the base-R fallback
can be exercised deliberately in the test suite.

## Usage

``` r
.ob_has_dt()
```

## Value

`TRUE` when data.table can be loaded.
