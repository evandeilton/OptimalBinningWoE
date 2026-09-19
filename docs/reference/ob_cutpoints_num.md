# Binning Numerical Variables using Custom Cutpoints

This function applies user-defined binning to a numerical variable by
using specified cutpoints to create intervals and calculates Weight of
Evidence (WoE) and Information Value (IV) for each interval bin.

## Usage

``` r
ob_cutpoints_num(feature, target, cutpoints)
```

## Arguments

- feature:

  A numeric vector representing the continuous predictor variable.

- target:

  An integer vector containing binary outcome values (0 or 1). Must be
  the same length as `feature`.

- cutpoints:

  A numeric vector of cutpoints that define bin boundaries. These will
  be automatically sorted in ascending order.

## Value

A list containing:

- `woefeature`:

  Numeric vector of WoE values corresponding to each observation in the
  input `feature`

- `woebin`:

  Data frame with one row per bin containing:

  - `id`: Sequential bin identifier

  - `bin`: The bin interval notation (e.g., "(10.00;20.00\]")

  - `count`: Total number of observations in the bin

  - `count_pos`: Number of positive outcomes (target=1) in the bin

  - `count_neg`: Number of negative outcomes (target=0) in the bin

  - `woe`: Weight of Evidence for the bin

  - `iv`: Information Value contribution of the bin

- `cutpoints`:

  The sorted numeric cutpoints, exposed at the top level so this result
  can be reassembled into
  [`ob_apply_woe_num`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_num.md)'s
  `obresults` argument (see Details)

- `id`:

  Sequential bin identifier, same as `woebin$id`; duplicated at the top
  level for the same reason as `cutpoints`

## Details

The function takes a numeric vector of cutpoints that define the
boundaries between bins. For `n` cutpoints, `n+1` bins are created, each
right-closed – \\(a, b\]\\ – matching
[`ob_apply_woe_num`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_num.md)'s
default (`include_upper_bound = TRUE`) and the convention
[`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)/[`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
use throughout the package:

- Bin 1: \\(-\infty, cutpoint_1\]\\

- Bin 2: \\(cutpoint_1, cutpoint_2\]\\

- ...

- Bin n+1: \\(cutpoint_n, +\infty)\\

`result` does not have the exact shape
[`ob_apply_woe_num`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_num.md)
expects as `obresults` (which needs top-level `cutpoints`, `woe` and
`id`), because `woebin` also carries `count`/ `iv` columns. To
round-trip, reassemble it:
`list(cutpoints = result$cutpoints, woe = result$woebin$woe, id = result$id, bin = result$woebin$bin)`.
Before 1.13.1, `result` carried no `id` or top-level `cutpoints` at all
(so a round trip through
[`ob_apply_woe_num`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_num.md)
was not even possible without reconstructing them by hand), and the bin
boundaries were left-closed \\\[a, b)\\, the opposite of
[`ob_apply_woe_num`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_num.md)'s
`include_upper_bound = TRUE` default – a value that landed exactly on a
cutpoint got a different, often sign-flipped, WoE depending on whether
it went through the fit or the apply side.

## Note

- Target variable must contain only 0 and 1 values.

- Cutpoints are sorted automatically in ascending order.

- Interval notation uses "(" for exclusive and "\]" for inclusive bounds
  – a value exactly equal to a cutpoint falls in the bin that *ends* at
  that cutpoint.

- Infinite values in feature are handled appropriately.

## Examples

``` r
# Sample data
feature <- c(5, 15, 25, 35, 45, 55, 65, 75)
target <- c(0, 0, 1, 1, 1, 1, 0, 0)

# Define custom cutpoints
cutpoints <- c(30, 60)

# Apply binning
result <- ob_cutpoints_num(feature, target, cutpoints)

# View bin statistics
print(result$woebin)
#>   id           bin count count_pos count_neg        woe        iv
#> 1  1  (-Inf;30.00]     3         1         2 -0.6931472 0.1732868
#> 2  2 (30.00;60.00]     3         3         0  8.9226583 6.6911015
#> 3  3  (60.00;+Inf]     2         0         2 -8.5171932 4.2577449

# View WoE-transformed feature
print(result$woefeature)
#> [1] -0.6931472 -0.6931472 -0.6931472  8.9226583  8.9226583  8.9226583 -8.5171932
#> [8] -8.5171932

# Round-trip through ob_apply_woe_num() with its defaults
apply_obj <- list(
  cutpoints = result$cutpoints, woe = result$woebin$woe,
  id = result$id, bin = result$woebin$bin
)
woe_new <- ob_apply_woe_num(apply_obj, feature)
stopifnot(isTRUE(all.equal(woe_new$woe, result$woefeature)))
```
