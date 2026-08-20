# Score New Data With a Fitted Scorecard

Applies a fitted scorecard to new data. Reads only the binning, the
coefficients and the scaling — never the raw fitted model — so a
scorecard saved with [`saveRDS()`](https://rdrr.io/r/base/readRDS.html)
keeps scoring after the engine's package is gone.

## Usage

``` r
# S3 method for class 'obwoe_scorecard'
predict(
  object,
  new_data,
  type = c("score", "card", "link", "prob", "woe"),
  ...
)
```

## Arguments

- object:

  An `"obwoe_scorecard"` object.

- new_data:

  A `data.frame` carrying the model's variables.

- type:

  One of `"score"` (default), `"card"` (the sum of the integer points,
  i.e. what the deployed table gives), `"link"` (log-odds of the event),
  `"prob"` or `"woe"`.

- ...:

  Ignored.

## Value

A numeric vector, or a `data.frame` when `type = "woe"`.

## See also

[`obwoe_scorecard`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scorecard.md)

## Examples

``` r
# \donttest{
german <- read.csv(
  gzfile(system.file("extdata", "germancredit.csv.gz",
    package = "OptimalBinningWoE"
  )),
  stringsAsFactors = FALSE
)
german$default <- 1L - german$credit_risk
german$credit_risk <- NULL

sc <- obwoe_scorecard(german, target = "default", seed = 1)
#> Warning: 210 events for 13 variables (16.2 per variable): below the 20-events-per-variable rule of thumb, so the coefficients are unstable.
summary(predict(sc, german))
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#>   420.4   494.4   517.1   517.5   538.7   629.8 
# }
```
