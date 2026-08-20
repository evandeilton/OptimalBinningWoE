# Turn Log-Odds into Scorecard Points

Applies a
[`obwoe_scale`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scale.md)
scaling to the linear predictor of a model that predicts the log-odds of
the event.

## Usage

``` r
obwoe_score(link, scaling, round = FALSE)
```

## Arguments

- link:

  Numeric vector of log-odds of the event, as returned by
  `predict(fit, type = "link")`.

- scaling:

  An `"obwoe_scaling"` object from
  [`obwoe_scale`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scale.md).

- round:

  Logical. Round to whole points? Default `FALSE`; the unrounded score
  is what the additivity identity holds for.

## Value

Numeric vector of scores, of the same length as `link`.

## See also

[`obwoe_scale`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scale.md)

## Examples

``` r
scaling <- obwoe_scale()
obwoe_score(c(-log(50), -log(100), 0), scaling)
#> [1] 600.0000 620.0000 487.1229
```
