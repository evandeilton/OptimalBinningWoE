# Compute Gains Table for a Binned Feature Vector

Calculates a full gains table by aggregating a raw binned dataframe
against a binary target. Unlike
[`ob_gains_table`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_gains_table.md)
which expects pre-aggregated counts, this function takes
observation-level data, aggregates it by the specified group variable
(bin, WoE, or ID), and then computes all statistical metrics.

## Usage

``` r
ob_gains_table_feature(binned_df, target, group_var = "bin")
```

## Arguments

- binned_df:

  A `data.frame` resulting from a binning transformation (e.g., via
  `obwoe_apply`), containing at least the following columns:

  `feature`

  :   Original feature values (optional, for reference).

  `bin`

  :   Character vector of bin labels.

  `woe`

  :   Numeric vector of Weight of Evidence values.

  `idbin`

  :   Numeric vector of bin IDs (required for correct sorting).

- target:

  A numeric vector of binary outcomes (0 for non-event, 1 for event).
  Must have the same length as `binned_df`. Missing values are not
  allowed.

- group_var:

  Character string specifying the aggregation key. Options:

  - `"bin"`: Group by bin label (default).

  - `"woe"`: Group by WoE value.

  - `"idbin"`: Group by bin ID.

## Value

A `data.frame` containing the same extensive set of metrics as
[`ob_gains_table`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_gains_table.md),
aggregated by `group_var` and sorted by `idbin`.

## Details

### Aggregation and Sorting

The function first aggregates the binary target by the specified
`group_var`. Crucially, it uses the `idbin` column to sort the resulting
groups. This ensures that cumulative metrics (like KS and Gini) are
calculated based on the logical order of the bins (e.g., low score to
high score), not alphabetical order.

### Advanced Metrics

In addition to standard credit scoring metrics, this function computes:

- **Jensen-Shannon Divergence**: A symmetrized and smoothed version of
  KL divergence, useful for measuring stability between the bin
  distribution and the population distribution.

- **F1-Score, Precision, Recall**: Treating each bin as a potential
  classification threshold.

## References

Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing
Intelligent Credit Scoring*. Wiley.

Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency.
*The Annals of Mathematical Statistics*.

## Examples

``` r
# \donttest{
# Mock data representing a binned feature
df_binned <- data.frame(
  feature = c(10, 20, 30, 10, 20, 50),
  bin = c("Low", "Mid", "High", "Low", "Mid", "High"),
  woe = c(-0.5, 0.2, 1.1, -0.5, 0.2, 1.1),
  idbin = c(1, 2, 3, 1, 2, 3)
)
target <- c(0, 0, 1, 1, 0, 1)

# Calculate gains table grouped by bin ID
gt <- ob_gains_table_feature(df_binned, target, group_var = "idbin")

# Inspect key metrics
print(gt[, c("id", "count", "pos_rate", "lift", "js_divergence")])
#>   id count pos_rate lift js_divergence
#> 1  1     2      0.5    1     0.0000000
#> 2  2     2      0.0    0     0.2157616
#> 3  3     2      1.0    2     0.2157616
# }
```
