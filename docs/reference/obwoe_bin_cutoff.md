# Bin Cutoff Parameter

A quantitative tuning parameter for the minimum bin support (proportion
of observations per bin) in
[`step_obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md).

## Usage

``` r
obwoe_bin_cutoff(range = c(0.01, 0.1), trans = NULL)
```

## Arguments

- range:

  A two-element numeric vector specifying the minimum and maximum values
  for the parameter. Default is `c(0.01, 0.10)`.

- trans:

  A transformation object from the `scales` package, or `NULL` for no
  transformation. Default is `NULL`.

## Value

A `dials` quantitative parameter object.

## Details

The bin cutoff specifies the minimum proportion of observations that
each bin must contain. Bins with fewer observations are merged with
adjacent bins. This serves as a regularization mechanism:

- Lower values (e.g., 0.01) allow smaller bins, capturing subtle
  patterns but risking unstable WoE estimates.

- Higher values (e.g., 0.10) enforce larger bins, producing more stable
  estimates but potentially missing important patterns.

For credit scoring, values between 0.02 and 0.05 are typical. Regulatory
guidelines often require minimum bin sizes for model stability.

## See also

[`step_obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md),
[`control.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/control.obwoe.md)

## Examples

``` r
obwoe_bin_cutoff()
#> Bin Support Cutoff (quantitative)
#> Range: [0.01, 0.1]
obwoe_bin_cutoff(range = c(0.02, 0.08))
#> Bin Support Cutoff (quantitative)
#> Range: [0.02, 0.08]
```
