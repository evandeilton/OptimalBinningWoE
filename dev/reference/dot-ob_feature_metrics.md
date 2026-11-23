# Internal: Per-Feature Metric Extraction

Builds the feature-level statistics and the bin-level gains table for a
single entry of `obj$results`.

## Usage

``` r
.ob_feature_metrics(res, feature, bin_separator = "%;%")
```

## Arguments

- res:

  A single feature result from an `"obwoe"` object.

- feature:

  Feature name.

- bin_separator:

  Separator used inside merged categorical bin labels.

## Value

A list with `summary` (a named list of scalars) and `bins` (a named list
of equal-length columns, or `NULL`).
