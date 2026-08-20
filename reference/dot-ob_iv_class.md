# Internal: Siddiqi Information Value Strength Classification

Vectorised classification of Information Value into the strength bands
popularised by Siddiqi (2006). Shared by
[`summary.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/summary.obwoe.md)
and
[`obwoe_select`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_select.md)
so both report identical labels.

## Usage

``` r
.ob_iv_class(iv)
```

## Arguments

- iv:

  Numeric vector of total Information Values.

## Value

A factor with levels `"Unpredictive"`, `"Weak"`, `"Medium"`, `"Strong"`,
`"Suspicious"`, `"Error"`.
