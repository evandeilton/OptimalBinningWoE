# Summary Method for obwoe Objects

Generates comprehensive summary statistics for optimal binning results,
including predictive power classification based on established IV
thresholds (Siddiqi, 2006), aggregate metrics, and feature-level
diagnostics.

## Usage

``` r
# S3 method for class 'obwoe'
summary(object, sort_by = "iv", decreasing = TRUE, ...)
```

## Arguments

- object:

  An object of class `"obwoe"`.

- sort_by:

  Character string specifying the column to sort by. Options: `"iv"`
  (default), `"n_bins"`, `"feature"`.

- decreasing:

  Logical. Sort in decreasing order? Default is `TRUE` for IV, `FALSE`
  for feature names.

- ...:

  Additional arguments (currently ignored).

## Value

An S3 object of class `"summary.obwoe"` containing:

- `feature_summary`:

  Data frame with per-feature statistics including IV classification
  (Unpredictive/Weak/Medium/Strong/Suspicious)

- `aggregate`:

  Named list of aggregate statistics:

  `n_features`

  :   Total features processed

  `n_successful`

  :   Features without errors

  `n_errors`

  :   Features with errors

  `total_iv_sum`

  :   Sum of all feature IVs

  `mean_iv`

  :   Mean IV across features

  `median_iv`

  :   Median IV across features

  `mean_bins`

  :   Mean number of bins

  `iv_range`

  :   Min and max IV values

- `iv_distribution`:

  Table of IV classification counts

- `target`:

  Target column name

- `target_type`:

  Target type (binary/multinomial)

## Details

### IV Classification Thresholds

Following Siddiqi (2006), features are classified by predictive power:

|                    |              |
|--------------------|--------------|
| **Classification** | **IV Range** |
| Unpredictive       | \< 0.02      |
| Weak               | 0.02 - 0.10  |
| Medium             | 0.10 - 0.30  |
| Strong             | 0.30 - 0.50  |
| Suspicious         | \> 0.50      |

Features with IV \> 0.50 should be examined for data leakage or
overfitting, as such high values are rarely observed in practice.

## References

Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
Intelligent Credit Scoring. *John Wiley & Sons*.
[doi:10.1002/9781119201731](https://doi.org/10.1002/9781119201731)

## See also

[`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
for the main binning function,
[`print.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/print.obwoe.md),
[`plot.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/plot.obwoe.md).

## Examples

``` r
# \donttest{
set.seed(42)
df <- data.frame(
  x1 = rnorm(500), x2 = rnorm(500), x3 = rnorm(500),
  target = rbinom(500, 1, 0.2)
)
result <- obwoe(df, target = "target")
summary(result)
#> Summary: Optimal Binning Weight of Evidence
#> ============================================
#> 
#> Target: target ( binary )
#> 
#> Aggregate Statistics:
#>   Features: 3 total, 3 successful, 0 errors
#>   Total IV: 0.0930
#>   Mean IV: 0.0310 (SD: 0.0423)
#>   Median IV: 0.0109
#>   IV Range: [0.0025, 0.0796]
#>   Mean Bins: 5.0
#> 
#> IV Classification (Siddiqi, 2006):
#>   Unpredictive: 2 features
#>   Weak        : 1 features
#> 
#> Feature Details:
#>  feature      type n_bins    total_iv     iv_class
#>       x3 numerical      5 0.079623601         Weak
#>       x1 numerical      5 0.010895822 Unpredictive
#>       x2 numerical      5 0.002463832 Unpredictive
# }
```
