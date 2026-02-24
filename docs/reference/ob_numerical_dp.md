# Optimal Binning for Numerical Variables using Dynamic Programming

Performs supervised discretization of continuous numerical variables
using a greedy heuristic approach that resembles Dynamic Programming.
This method is particularly effective at strictly enforcing monotonic
trends (ascending or descending) in the Weight of Evidence (WoE), which
is critical for the interpretability of logistic regression models in
credit scoring.

## Usage

``` r
ob_numerical_dp(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  convergence_threshold = 1e-06,
  max_iterations = 1000,
  monotonic_trend = c("auto", "ascending", "descending", "none")
)
```

## Arguments

- feature:

  A numeric vector representing the continuous predictor variable.
  Missing values (NA) should be handled prior to binning, as they are
  not supported by this algorithm.

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
  merged. Value must be in (0, 1). Defaults to 0.05.

- max_n_prebins:

  Integer. The number of initial quantiles to generate during the
  pre-binning phase. Defaults to 20.

- convergence_threshold:

  Numeric. The threshold for the change in metrics to determine
  convergence during the iterative merging process. Defaults to 1e-6.

- max_iterations:

  Integer. Safety limit for the maximum number of merging iterations.
  Defaults to 1000.

- monotonic_trend:

  Character string specifying the desired direction of the Weight of
  Evidence (WoE) trend.

  - `"auto"`: Automatically determines the most likely trend (ascending
    or descending) based on the correlation between the feature and the
    target.

  - `"ascending"`: Forces the WoE to increase as the feature value
    increases.

  - `"descending"`: Forces the WoE to decrease as the feature value
    increases.

  - `"none"`: Does not enforce any monotonic constraint (allows peaks
    and valleys).

  Defaults to `"auto"`.

## Value

A list containing the binning results:

- `id`: Integer vector of bin identifiers.

- `bin`: Character vector of bin labels in interval notation.

- `woe`: Numeric vector of Weight of Evidence for each bin.

- `iv`: Numeric vector of Information Value contribution per bin.

- `count`: Integer vector of total observations per bin.

- `count_pos`: Integer vector of positive cases.

- `count_neg`: Integer vector of negative cases.

- `event_rate`: Numeric vector of the target event rate in each bin.

- `cutpoints`: Numeric vector of upper boundaries (excluding Inf).

- `total_iv`: The total Information Value of the binned variable.

- `monotonic_trend`: The actual trend enforced ("ascending",
  "descending", or "none").

- `execution_time_ms`: Execution time in milliseconds.

## Details

Although named "DP" (Dynamic Programming) in some contexts, this
implementation primarily uses a **greedy heuristic** to optimize the
Information Value (IV) while satisfying constraints.

**Algorithm Steps:**

1.  **Pre-binning:** Generates initial granular bins based on quantiles.

2.  **Trend Determination:** If `monotonic_trend = "auto"`, calculates
    the Pearson correlation between the feature and target to decide if
    the WoE should increase or decrease.

3.  **Monotonicity Enforcement:** Iteratively merges adjacent bins that
    violate the determined or requested trend.

4.  **Constraint Satisfaction:** Merges rare bins (below `bin_cutoff`)
    and ensures the number of bins is within `[min_bins, max_bins]`.

5.  **Optimization:** Greedily merges similar bins (based on WoE
    difference) to reduce complexity while attempting to preserve
    information.

This method is often preferred when strict business logic dictates a
specific relationship direction (e.g., "higher income must imply lower
risk").

## See also

[`ob_numerical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_cm.md),
[`ob_numerical_bb`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_bb.md)

## Examples

``` r
# Example: forcing a descending trend
set.seed(123)
feature <- runif(1000, 0, 100)
# Target has a complex relationship, but we want to force a linear view
target <- rbinom(1000, 1, 0.5 + 0.003 * feature) # slightly positive trend

# Force "descending" (even if data suggests ascending) to see enforcement
result <- ob_numerical_dp(feature, target,
  min_bins = 3,
  max_bins = 5,
  monotonic_trend = "descending"
)

print(result$bin)
#> [1] "(-Inf;89.189612]"      "(89.189612;95.310121]" "(95.310121;+Inf]"     
print(result$woe) # Should be strictly decreasing
#> [1] -0.05783551  0.64413073  0.53267052
```
