# Optimal Binning using Joint Entropy-Driven Interval Discretization (JEDI)

Performs supervised discretization of continuous numerical variables
using a holistic approach that balances entropy reduction (information
gain) with statistical stability. The JEDI algorithm combines
quantile-based initialization with an iterative optimization process
that enforces monotonicity and minimizes Information Value (IV) loss.

## Usage

``` r
ob_numerical_jedi(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
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

- bin_cutoff:

  Numeric. The minimum fraction of total observations required for a bin
  to be considered valid. Bins smaller than this threshold are merged.
  Value must be in (0, 1). Defaults to 0.05.

- max_n_prebins:

  Integer. The number of initial quantiles to generate during the
  initialization phase. Defaults to 20.

- convergence_threshold:

  Numeric. The threshold for the change in total IV to determine
  convergence during the iterative optimization. Defaults to 1e-6.

- max_iterations:

  Integer. Safety limit for the maximum number of iterations. Defaults
  to 1000.

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

The JEDI algorithm is designed to be a robust "all-rounder" for credit
scoring and risk modeling. Its methodology proceeds in four distinct
stages:

1.  **Initialization (Quantile Pre-binning):** The feature space is
    divided into `max_n_prebins` segments containing approximately equal
    numbers of observations. This ensures the algorithm starts with a
    statistically balanced view of the data.

2.  **Stabilization (Rare Bin Merging):** Adjacent bins with frequencies
    below `bin_cutoff` are merged. The merge direction is chosen to
    minimize the distortion of the event rate (similar to ChiMerge).

3.  **Monotonicity Enforcement:** The algorithm heuristically determines
    the dominant trend (increasing or decreasing) of the Weight of
    Evidence (WoE) and iteratively merges adjacent bins that violate
    this trend. This step effectively reduces the conditional entropy of
    the binning sequence with respect to the target.

4.  **IV Optimization:** If the number of bins exceeds `max_bins`, the
    algorithm merges the pair of adjacent bins that results in the
    smallest decrease in total Information Value. This greedy approach
    ensures that the final discretization retains the maximum possible
    predictive power given the constraints.

This joint approach (Entropy/IV + Stability constraints) makes JEDI
particularly effective for datasets with noise or non-monotonic initial
distributions that require smoothing.

## See also

[`ob_numerical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_cm.md),
[`ob_numerical_ir`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_ir.md)

## Examples

``` r
# Example: Binning a variable with a complex relationship
set.seed(123)
feature <- rnorm(1000)
# Target probability has a quadratic component (non-monotonic)
# JEDI will try to force a monotonic approximation that maximizes IV
target <- rbinom(1000, 1, plogis(0.5 * feature + 0.1 * feature^2))

result <- ob_numerical_jedi(feature, target,
  min_bins = 3,
  max_bins = 6,
  max_n_prebins = 20
)

print(result$bin)
#> [1] "(-Inf;-1.052513]"      "(-1.052513;-0.097412]" "(-0.097412;0.840540]" 
#> [4] "(0.840540;1.253815]"   "(1.253815;1.675697]"   "(1.675697;+Inf]"      
```
