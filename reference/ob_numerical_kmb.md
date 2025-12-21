# Optimal Binning using K-means Inspired Initialization (KMB)

Performs supervised discretization of continuous numerical variables
using a K-means inspired binning strategy. Initial bin boundaries are
determined by placing centroids uniformly across the feature range and
defining cuts at midpoints. The algorithm then optimizes these bins
using statistical constraints.

## Usage

``` r
ob_numerical_kmb(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  enforce_monotonic = TRUE,
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

  Integer. The number of initial centroids/bins to generate during the
  initialization phase. Defaults to 20.

- enforce_monotonic:

  Logical. If `TRUE`, the algorithm enforces a monotonic relationship in
  the Weight of Evidence (WoE) across bins. Defaults to `TRUE`.

- convergence_threshold:

  Numeric. The threshold for determining convergence during the
  iterative optimization process. Defaults to 1e-6.

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

- `centroids`: Numeric vector of bin centroids (mean feature value per
  bin).

- `cutpoints`: Numeric vector of upper boundaries (excluding Inf).

- `total_iv`: The total Information Value of the binned variable.

- `converged`: Logical indicating if the algorithm converged.

## Details

The KMB algorithm offers a unique initialization strategy compared to
standard binning methods:

1.  **Initialization (K-means Style):** Instead of using quantiles,
    `max_n_prebins` centroids are placed uniformly across the range
    \\\[min(x), max(x)\]\\. Bin boundaries are then defined as the
    midpoints between adjacent centroids. This can lead to more evenly
    distributed initial bin widths in terms of the feature's scale.

2.  **Optimization:** The initialized bins undergo standard
    post-processing:

    - **Rare Bin Merging:** Bins below `bin_cutoff` are merged with
      their most similar neighbor (by event rate).

    - **Monotonicity:** If `enforce_monotonic = TRUE`, adjacent bins
      violating the dominant WoE trend are merged.

    - **Bin Count Adjustment:** If the number of bins exceeds
      `max_bins`, the algorithm greedily merges adjacent bins with the
      smallest absolute difference in Information Value.

This method can be advantageous when the underlying distribution of the
feature is relatively uniform, as it avoids creating overly granular
bins in dense regions from the start.

## See also

[`ob_numerical_ewb`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_ewb.md),
[`ob_numerical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_cm.md)

## Examples

``` r
# Example: Comparing KMB with EWB on uniform data
set.seed(123)
feature <- runif(1000, 0, 100)
target <- rbinom(1000, 1, plogis(0.02 * feature))

result_kmb <- ob_numerical_kmb(feature, target, max_bins = 5)
print(result_kmb$bin)
#> [1] "(-Inf;20.025318]"      "(20.025318;40.004102]" "(40.004102;59.982886]"
#> [4] "(59.982886;79.961669]" "(79.961669;+Inf]"     
print(paste("KMB Total IV:", round(result_kmb$total_iv, 4)))
#> [1] "KMB Total IV: 0.4792"
```
