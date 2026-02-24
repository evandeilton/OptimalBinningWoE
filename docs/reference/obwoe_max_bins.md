# Maximum Bins Parameter

A quantitative tuning parameter for the maximum number of bins in
[`step_obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md).

## Usage

``` r
obwoe_max_bins(range = c(5L, 20L), trans = NULL)
```

## Arguments

- range:

  A two-element integer vector specifying the minimum and maximum values
  for the parameter. Default is `c(5L, 20L)`.

- trans:

  A transformation object from the `scales` package, or `NULL` for no
  transformation. Default is `NULL`.

## Value

A `dials` quantitative parameter object.

## Details

The maximum number of bins limits algorithm complexity and helps prevent
overfitting. Higher values allow more granular discretization but may
capture noise rather than signal.

For credit scoring applications, `max_bins` is typically set between 5
and 10 to balance predictive power with interpretability. Values above
15 are rarely necessary and may indicate overfitting.

## See also

[`step_obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md),
[`obwoe_min_bins`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_min_bins.md)

## Examples

``` r
obwoe_max_bins()
#> Maximum Bins (quantitative)
#> Range: [5, 20]
obwoe_max_bins(range = c(4L, 12L))
#> Maximum Bins (quantitative)
#> Range: [4, 12]
```
