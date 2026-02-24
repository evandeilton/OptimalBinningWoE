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

  - `bin`: The bin interval notation (e.g., "\[10.00;20.00)")

  - `count`: Total number of observations in the bin

  - `count_pos`: Number of positive outcomes (target=1) in the bin

  - `count_neg`: Number of negative outcomes (target=0) in the bin

  - `woe`: Weight of Evidence for the bin

  - `iv`: Information Value contribution of the bin

## Details

The function takes a numeric vector of cutpoints that define the
boundaries between bins. For `n` cutpoints, `n+1` bins are created:

- Bin 1: \\(-\infty, cutpoint_1)\\

- Bin 2: \\\[cutpoint_1, cutpoint_2)\\

- ...

- Bin n+1: \\\[cutpoint_n, +\infty)\\

## Note

- Target variable must contain only 0 and 1 values.

- Cutpoints are sorted automatically in ascending order.

- Interval notation uses "\[" for inclusive and ")" for exclusive
  bounds.

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
#>             bin count count_pos count_neg        woe        iv
#> 1  [-Inf;30.00)     3         1         2 -0.6931472 0.1732868
#> 2 [30.00;60.00)     3         3         0  8.9226583 6.6911015
#> 3  [60.00;+Inf]     2         0         2 -8.5171932 4.2577449

# View WoE-transformed feature
print(result$woefeature)
#> [1] -0.6931472 -0.6931472 -0.6931472  8.9226583  8.9226583  8.9226583 -8.5171932
#> [8] -8.5171932
```
