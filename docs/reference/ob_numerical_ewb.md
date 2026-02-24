# Hybrid Optimal Binning using Equal-Width Initialization and IV Optimization

Performs supervised discretization of continuous numerical variables
using a hybrid approach. The algorithm initializes with an Equal-Width
Binning (EWB) strategy to capture the scale of the variable, followed by
an iterative, supervised optimization phase that merges bins to maximize
Information Value (IV) and enforce monotonicity.

## Usage

``` r
ob_numerical_ewb(
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

  A numeric vector representing the continuous predictor variable.
  Missing values (NA) are excluded during the pre-binning phase but
  should ideally be handled prior to binning.

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
  to be considered valid. Bins with frequency \< `bin_cutoff` are merged
  with their most similar neighbor (based on event rate). Value must be
  in (0, 1). Defaults to 0.05.

- max_n_prebins:

  Integer. The number of initial equal-width intervals to generate
  during the pre-binning phase. This parameter defines the initial
  granularity/search space. Defaults to 20.

- is_monotonic:

  Logical. If `TRUE`, the algorithm enforces a strict monotonic
  relationship (increasing or decreasing) between the bin indices and
  their Weight of Evidence (WoE). Defaults to `TRUE`.

- convergence_threshold:

  Numeric. The threshold for determining convergence during the
  iterative merging process. Defaults to 1e-6.

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

- `total_iv`: The total Information Value of the binned variable.

- `converged`: Logical indicating if the algorithm converged.

## Details

Unlike standard Equal-Width binning which is purely unsupervised, this
function implements a **Hybrid Discretization Pipeline**:

1.  **Phase 1: Unsupervised Initialization (Scale Preservation)** The
    range of the feature \\\[min(x), max(x)\]\\ is divided into
    `max_n_prebins` intervals of equal width \\w = (max(x) - min(x)) /
    N\\. This step preserves the cardinal magnitude of the data but is
    sensitive to outliers.

2.  **Phase 2: Statistical Stabilization** Bins falling below the
    `bin_cutoff` threshold are merged. Unlike naive approaches, this
    implementation merges rare bins with the neighbor that has the most
    similar class distribution (event rate), minimizing the distortion
    of the predictive relationship.

3.  **Phase 3: Monotonicity Enforcement** If `is_monotonic = TRUE`, the
    algorithm checks for non-monotonic trends in the Weight of Evidence
    (WoE). Violating adjacent bins are iteratively merged to ensure a
    strictly increasing or decreasing relationship, which is a key
    requirement for interpretable Logistic Regression scorecards.

4.  **Phase 4: IV-Based Optimization** If the number of bins exceeds
    `max_bins`, the algorithm applies a hierarchical bottom-up merging
    strategy. It calculates the *Information Value Loss* for every
    possible pair of adjacent bins: \$\$\Delta IV = (IV_i + IV\_{i+1}) -
    IV\_{merged}\$\$ The pair minimizing this loss is merged, ensuring
    that the final coarse classes retain the maximum possible predictive
    power of the original variable.

**Technical Note on Outliers:** Because the initialization is based on
the range, extreme outliers can compress the majority of the data into a
single initial bin. If your data is highly skewed or contains outliers,
consider using
[`ob_numerical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_cm.md)
(Quantile/ChiMerge) or winsorizing the data before using this function.

## References

Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and
unsupervised discretization of continuous features. *Machine Learning
Proceedings*, 194-202.

Siddiqi, N. (2012). *Credit Risk Scorecards: Developing and Implementing
Intelligent Credit Scoring*. John Wiley & Sons.

Catlett, J. (1991). On changing continuous attributes into ordered
discrete attributes. *Proceedings of the European Working Session on
Learning on Machine Learning*, 164-178.

## See also

[`ob_numerical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_cm.md)
for Quantile/Chi-Square binning,
[`ob_numerical_dp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_dp.md)
for Dynamic Programming approaches.

## Examples

``` r
# Example 1: Uniform distribution (Ideal for Equal-Width)
set.seed(123)
feature <- runif(1000, 0, 100)
target <- rbinom(1000, 1, plogis(0.05 * feature - 2))

res_ewb <- ob_numerical_ewb(feature, target, max_bins = 5)
print(res_ewb$bin)
#> [1] "(-Inf;5.041231]"       "(5.041231;30.014710]"  "(30.014710;54.988190]"
#> [4] "(54.988190;79.961669]" "(79.961669;+Inf]"     
print(paste("Total IV:", round(res_ewb$total_iv, 4)))
#> [1] "Total IV: 1.6104"

# Example 2: Effect of Outliers (The weakness of Equal-Width)
feature_outlier <- c(feature, 10000) # One extreme outlier
target_outlier <- c(target, 0)

# Note: The algorithm tries to recover, but the initial split is distorted
res_outlier <- ob_numerical_ewb(feature_outlier, target_outlier, max_bins = 5)
print(res_outlier$bin)
#> [1] "(-Inf;500.044208]"        "(500.044208;9500.002327]"
#> [3] "(9500.002327;+Inf]"      
```
