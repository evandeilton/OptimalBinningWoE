# Plot Gains Table

Visualizes gains table metrics including cumulative capture curves, KS
plot, and lift chart.

## Usage

``` r
# S3 method for class 'obwoe_gains'
plot(x, type = c("cumulative", "ks", "lift", "woe_iv"), ...)
```

## Arguments

- x:

  An object of class `"obwoe_gains"`.

- type:

  Character string: `"cumulative"` (default), `"ks"`, `"lift"`, or
  `"woe_iv"`.

- ...:

  Additional arguments passed to plotting functions.

## Value

Invisibly returns `NULL`.
