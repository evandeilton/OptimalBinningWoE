# Internal: Normalise Supported Inputs to a Binning Specification

Accepts an `"obwoe"` object, a prepped `"step_obwoe"` step, or a prepped
recipes recipe containing one, and returns a uniform list of per-feature
binning specifications.

## Usage

``` r
.ob_sql_spec(obj)
```

## Arguments

- obj:

  The input object.

## Value

A named list; each element holds `type`, `bin`, `woe`, `cutpoints` and
`error`.
