# Optimal Binning using Isotonic Regression (PAVA)

Performs supervised discretization of continuous numerical variables
using Isotonic Regression (specifically the Pool Adjacent Violators
Algorithm - PAVA). This method ensures a strictly monotonic relationship
between bin indices and the empirical event rate, making it ideal for
applications requiring shape constraints like credit scoring.

## Usage

``` r
ob_numerical_ir(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  auto_monotonicity = TRUE,
  convergence_threshold = 1e-06,
  max_iterations = 1000
)
```

## Arguments

- feature:

  A numeric vector representing the continuous predictor variable.
  Missing values (NA) are excluded from the binning process.

- target:

  An integer vector of binary outcomes (0/1) corresponding to each
  observation in `feature`. Must have the same length as `feature`.

- min_bins:

  Integer. The target minimum number of bins. Must be \\\ge\\ 2.
  Defaults to 3. Enforcing monotonicity can require pooling bins
  together, so the result may hold fewer bins than requested; see
  Details.

- max_bins:

  Integer. The maximum number of bins to produce. Must be \\\ge\\
  `min_bins`. Defaults to 5.

- bin_cutoff:

  Numeric. The minimum fraction of total observations required for a bin
  to be considered valid. Bins with frequency \< `bin_cutoff` will be
  merged with neighbors. Value must be in (0, 1). Defaults to 0.05.

- max_n_prebins:

  Integer. The number of initial quantiles to generate during the
  pre-binning phase. Defaults to 20.

- auto_monotonicity:

  Logical. If `TRUE`, the algorithm automatically determines the optimal
  monotonicity direction (increasing or decreasing) based on the Pearson
  correlation between feature values and target. If `FALSE`, defaults to
  increasing monotonicity. Defaults to `TRUE`.

- convergence_threshold:

  Numeric. Reserved for future use. Currently not actively used by the
  PAVA algorithm, which has guaranteed convergence. Defaults to 1e-6.

- max_iterations:

  Integer. Safety limit for iterative merging operations during
  pre-processing steps (e.g., rare bin merging). Defaults to 1000.

## Value

A list containing the binning results:

- `id`: Integer vector of bin identifiers.

- `bin`: Character vector of bin labels in interval notation.

- `woe`: Numeric vector of Weight of Evidence for each bin.

- `iv`: Numeric vector of Information Value contribution per bin.

- `count`: Integer vector of total observations per bin.

- `count_pos`: Integer vector of positive cases observed in the bin.

- `count_neg`: Integer vector of negative cases observed in the bin.

- `cutpoints`: Numeric vector of upper boundaries (excluding Inf).

- `total_iv`: The total Information Value of the binned variable.

- `monotone_increasing`: Logical indicating if the final WoE trend is
  increasing.

- `converged`: Logical indicating successful completion.

## Details

This function implements a **shape-constrained** binning approach using
**Isotonic Regression**. Unlike heuristic merging strategies (ChiMerge,
DP), this method finds the optimal monotonic fit in a single pass.

**Core Algorithm (PAVA):** The Pool Adjacent Violators Algorithm (Best &
Chakravarti, 1990) is used to transform the empirical event rates of
initial bins into a sequence that is either monotonically increasing or
decreasing. It works by scanning the sequence and merging ("pooling")
any adjacent pairs that violate the desired trend until a perfect fit is
achieved. This guarantees an optimal solution in \\O(n)\\ time.

**Process Flow:**

1.  **Pre-binning:** Creates initial bins using quantiles.

2.  **Stabilization:** Merges bins below `bin_cutoff`.

3.  **Trend Detection:** If `auto_monotonicity = TRUE`, calculates the
    correlation between feature midpoints and bin event rates to
    determine if the relationship should be increasing or decreasing.

4.  **Shape Enforcement:** Applies PAVA to the sequence of bin event
    rates and *merges* the bins that the algorithm pools into each
    block, so the surviving bins are monotonic in their own observed
    event rate.

5.  **Metric Calculation:** Derives WoE and IV from the observed counts
    of those bins.

**Pooling merges bins:** with the bin counts as weights, the fitted rate
PAVA assigns to a block is exactly \\\sum count\\pos / \sum count\\ over
the block's members. Merging them therefore reproduces the isotonic fit
exactly while `count`, `count_pos` and `count_neg` remain the counts
actually observed between the reported `cutpoints`. A consequence is
that `min_bins` is a *target* rather than a guarantee for this
algorithm: when the empirical event rate violates the trend,
monotonicity can only be restored by pooling, and the result may hold
fewer bins than requested. Use a non-shape-constrained method such as
[`ob_numerical_mdlp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mdlp.md)
when the bin count matters more than the shape constraint.

**Advantages:**

- **Global Optimality:** PAVA finds the best fit under the monotonicity
  constraint.

- **No Hyperparameters:** Unlike ChiMerge's p-value threshold, PAVA
  requires no significance level tuning for the core regression step.

- **Robustness:** Less sensitive to arbitrary thresholds compared to
  greedy merging.

## References

Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D.
(1972). *Statistical inference under order restrictions*. John Wiley &
Sons.

Best, M. J., & Chakravarti, N. (1990). Active set algorithms for
isotonic regression; A unifying framework. *Mathematical Programming*,
47(1-3), 425-439.

## See also

[`ob_numerical_dp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_dp.md)
for greedy dynamic programming approaches.

## Examples

``` r
# Example: Forcing a monotonic WoE trend
set.seed(123)
feature <- rnorm(500)
# Create a slightly noisy but generally increasing relationship
prob <- plogis(0.5 * feature + rnorm(500, 0, 0.3))
target <- rbinom(500, 1, prob)

result <- ob_numerical_ir(feature, target,
  min_bins = 4,
  max_bins = 6,
  auto_monotonicity = TRUE
)

print(result$bin)
#> [1] "(-Inf;-0.945409]"     "(-0.945409;0.418982]" "(0.418982;0.976973]" 
#> [4] "(0.976973;+Inf]"     
print(round(result$woe, 3))
#> [1] -0.339 -0.088  0.071  0.532
print(paste("Monotonic Increasing:", result$monotone_increasing))
#> [1] "Monotonic Increasing: TRUE"
```
