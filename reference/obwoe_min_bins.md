# Minimum Bins Parameter

A quantitative tuning parameter for the minimum number of bins in
[`step_obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md).

## Usage

``` r
obwoe_min_bins(range = c(2L, 5L), trans = NULL)
```

## Arguments

- range:

  A two-element integer vector specifying the minimum and maximum values
  for the parameter. Default is `c(2L, 5L)`.

- trans:

  A transformation object from the `scales` package, or `NULL` for no
  transformation. Default is `NULL`.

## Value

A `dials` quantitative parameter object.

## Details

The minimum number of bins constrains the algorithm to create at least
this many bins. Setting `min_bins = 2` allows maximum flexibility, while
higher values ensure more granular discretization.

For credit scoring applications, `min_bins` is typically set between 2
and 4 to avoid forcing artificial splits on weakly predictive variables.

## See also

[`step_obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md),
[`obwoe_max_bins`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_max_bins.md)

## Examples

``` r
obwoe_min_bins()
#> Minimum Bins (quantitative)
#> Range: [2, 5]
obwoe_min_bins(range = c(3L, 7L))
#> Minimum Bins (quantitative)
#> Range: [3, 7]
```
