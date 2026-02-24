# Optimal Binning for Numerical Variables using Branch and Bound Algorithm

Performs supervised discretization of continuous numerical variables
using a Branch and Bound-style approach. This algorithm optimally
creates bins based on the relationship with a binary target variable,
maximizing Information Value (IV) while optionally enforcing
monotonicity in Weight of Evidence (WoE).

## Usage

``` r
ob_numerical_bb(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  is_monotonic = TRUE,
  convergence_threshold = 1e-06,
  max_iterations = 1000
)
```

## Arguments

- feature:

  A numeric vector representing the continuous predictor variable to be
  binned. NA values are handled by exclusion during the pre-binning
  phase.

- target:

  An integer vector of binary outcomes (0/1) corresponding to each
  observation in `feature`. Must have the same length as `feature`.

- min_bins:

  Integer. The minimum number of bins to produce. Must be \\\ge\\ 2.
  Defaults to 3.

- max_bins:

  Integer. The maximum number of bins to produce. Must be \\\ge\\
  `min_bins`. Defaults to 5.

- bin_cutoff:

  Numeric. The minimum fraction of total observations required for a bin
  to be considered valid. Bins with frequency \< `bin_cutoff` will be
  merged with neighbors. Value must be in (0, 1). Defaults to 0.05.

- max_n_prebins:

  Integer. The number of initial quantiles to generate during the
  pre-binning phase. Higher values provide more granular starting points
  but increase computation time. Must be \\\ge\\ `min_bins`. Defaults to
  20.

- is_monotonic:

  Logical. If `TRUE`, the algorithm enforces a strict monotonic
  relationship (increasing or decreasing) between the bin indices and
  their WoE values. This makes the variable more interpretable for
  linear models. Defaults to `TRUE`.

- convergence_threshold:

  Numeric. The threshold for the change in total IV to determine
  convergence during the iterative merging process. Defaults to 1e-6.

- max_iterations:

  Integer. Safety limit for the maximum number of merging iterations.
  Defaults to 1000.

## Value

A list containing the binning results:

- `id`: Integer vector of bin identifiers (1 to k).

- `bin`: Character vector of bin labels in interval notation (e.g.,
  `"(0.5;1.2]"`).

- `woe`: Numeric vector of Weight of Evidence for each bin.

- `iv`: Numeric vector of Information Value contribution per bin.

- `count`: Integer vector of total observations per bin.

- `count_pos`: Integer vector of positive cases (target=1) per bin.

- `count_neg`: Integer vector of negative cases (target=0) per bin.

- `cutpoints`: Numeric vector of upper boundaries for the bins
  (excluding Inf).

- `converged`: Logical indicating if the algorithm converged properly.

- `iterations`: Integer count of iterations performed.

- `total_iv`: The total Information Value of the binned variable.

## Details

The algorithm proceeds in several distinct phases to ensure stability
and optimality:

1.  **Pre-binning:** The numerical feature is initially discretized into
    `max_n_prebins` using quantiles. This handles outliers and provides
    a granular starting point.

2.  **Rare Bin Management:** Bins containing fewer observations than the
    threshold defined by `bin_cutoff` are iteratively merged with their
    nearest neighbors to ensure statistical robustness.

3.  **Monotonicity Enforcement (Optional):** If `is_monotonic = TRUE`,
    the algorithm checks if the WoE trend is strictly increasing or
    decreasing. If not, it simulates merges in both directions to find
    the path that preserves the maximum possible Information Value while
    satisfying the monotonicity constraint.

4.  **Optimization Phase:** The algorithm iteratively merges adjacent
    bins that have the lowest contribution to the total Information
    Value (IV). This process continues until the number of bins is
    reduced to `max_bins` or the change in IV falls below
    `convergence_threshold`.

**Information Value (IV) Interpretation:**

- \\\< 0.02\\: Not predictive

- \\0.02 \text{ to } 0.1\\: Weak predictive power

- \\0.1 \text{ to } 0.3\\: Medium predictive power

- \\0.3 \text{ to } 0.5\\: Strong predictive power

- \\\> 0.5\\: Suspiciously high (check for leakage)

## Examples

``` r
# Example: Binning a variable with a sigmoid relationship to target
set.seed(123)
n <- 1000
# Generate feature
feature <- rnorm(n)

# Generate target based on logistic probability
prob <- 1 / (1 + exp(-2 * feature))
target <- rbinom(n, 1, prob)

# Perform Optimal Binning
result <- ob_numerical_bb(feature, target,
  min_bins = 3,
  max_bins = 5,
  is_monotonic = TRUE
)

# Check results
print(data.frame(
  Bin = result$bin,
  Count = result$count,
  WoE = round(result$woe, 4),
  IV = round(result$iv, 4)
))
#>                    Bin Count     WoE     IV
#> 1     (-Inf;-1.622584]    50 -3.4487 0.3213
#> 2 (-1.622584;0.841413]   750 -0.3461 0.0882
#> 3  (0.841413;1.254752]   100  1.9167 0.2916
#> 4  (1.254752;1.676134]    50  3.5443 0.3473
#> 5      (1.676134;+Inf]    50  3.5443 0.3473

cat("Total IV:", result$total_iv, "\n")
#> Total IV: 1.395824 
```
