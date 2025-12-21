# Optimal Binning using Fisher's Exact Test

Performs supervised discretization of continuous numerical variables
using Fisher's Exact Test. This method iteratively merges adjacent bins
that are statistically similar (highest p-value) while strictly
enforcing a monotonic Weight of Evidence (WoE) trend.

## Usage

``` r
ob_numerical_fetb(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  max_n_prebins = 20,
  convergence_threshold = 1e-06,
  max_iterations = 1000
)
```

## Arguments

- feature:

  A numeric vector representing the continuous predictor variable.
  Missing values (NA) should be handled prior to binning.

- target:

  An integer vector of binary outcomes (0/1) corresponding to each
  observation in `feature`. Must have the same length as `feature`.

- min_bins:

  Integer. The minimum number of bins to produce. Must be \\\ge\\ 2.
  Defaults to 3.

- max_bins:

  Integer. The maximum number of bins to produce. Must be \\\ge\\
  `min_bins`. Defaults to 5.

- max_n_prebins:

  Integer. The number of initial quantiles to generate during the
  pre-binning phase. Defaults to 20.

- convergence_threshold:

  Numeric. The threshold for the change in Information Value (IV) to
  determine convergence during the iterative merging process. Defaults
  to 1e-6.

- max_iterations:

  Integer. Safety limit for the maximum number of merging iterations.
  Defaults to 1000.

## Value

A list containing the binning results:

- `id`: Integer vector of bin identifiers.

- `bin`: Character vector of bin labels in interval notation.

- `woe`: Numeric vector of Weight of Evidence for each bin.

- `iv`: Numeric vector of Information Value contribution per bin.

- `count`: Integer vector of total observations per bin.

- `count_pos`: Integer vector of positive cases.

- `count_neg`: Integer vector of negative cases.

- `cutpoints`: Numeric vector of upper boundaries (excluding Inf).

- `converged`: Logical indicating if the algorithm converged.

- `iterations`: Integer count of iterations performed.

## Details

The **Fisher's Exact Test Binning (FETB)** algorithm provides a robust
statistical alternative to ChiMerge.

**Key Differences from ChiMerge:**

- **Exact Probability:** Instead of relying on the Chi-Square asymptotic
  approximation (which can be unreliable for small bin counts), FETB
  calculates the exact hypergeometric probability of independence
  between the bin index and the target.

- **Merge Criterion:** In each step, the algorithm identifies the pair
  of adjacent bins with the *highest* p-value (indicating they are the
  most statistically indistinguishable) and merges them.

- **Monotonicity:** The algorithm incorporates a check after every merge
  to ensure the WoE trend remains monotonic, merging strictly violating
  bins immediately.

This method is particularly recommended when working with smaller
datasets or highly imbalanced target classes, where the assumptions of
the Chi-Square test might be violated.

## See also

[`ob_numerical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_cm.md)

## Examples

``` r
# Example: Binning a small dataset where Fisher's Exact Test excels
set.seed(123)
feature <- rnorm(100)
target <- rbinom(100, 1, 0.2)

result <- ob_numerical_fetb(feature, target,
  min_bins = 2,
  max_bins = 4,
  max_n_prebins = 10
)

print(result$bin)
#> [1] "(-inf; -1.06782]"    "(-1.06782; 1.36065]" "(1.36065; inf]"     
print(result$woe)
#> [1] -1.6521490  0.1035344  0.1685980
```
