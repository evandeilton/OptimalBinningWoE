# Optimal Binning using Local Polynomial Density Binning (LPDB)

Performs supervised discretization of continuous numerical variables
using a novel approach that combines non-parametric density estimation
with information-theoretic optimization. The algorithm first identifies
natural clusters and boundaries in the feature distribution using local
polynomial density estimation, then refines the bins to maximize
predictive power.

## Usage

``` r
ob_numerical_lpdb(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  polynomial_degree = 3,
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

  Integer. The maximum number of initial candidate cut points to
  generate during the density estimation phase. Defaults to 20.

- polynomial_degree:

  Integer. The degree of the local polynomial used for density
  estimation (note: currently approximated via KDE). Defaults to 3.

- enforce_monotonic:

  Logical. If `TRUE`, the algorithm forces the Weight of Evidence (WoE)
  trend to be strictly monotonic. Defaults to `TRUE`.

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

- `event_rate`: Numeric vector of the target event rate in each bin.

- `centroids`: Numeric vector of the geometric centroids of the final
  bins.

- `cutpoints`: Numeric vector of upper boundaries (excluding Inf).

- `total_iv`: The total Information Value of the binned variable.

- `monotonicity`: Character string indicating the final WoE trend
  ("increasing", "decreasing", or "none").

## Details

The **Local Polynomial Density Binning (LPDB)** algorithm is a two-stage
process:

1.  **Density-Based Initialization:**

    - Estimates the probability density function \\f(x)\\ of the
      `feature` using Kernel Density Estimation (KDE), which
      approximates local polynomial regression.

    - Identifies *critical points* on the density curve, such as local
      minima and inflection points. These points often correspond to
      natural boundaries between clusters or modes in the data.

    - Uses these critical points as initial candidate cut points to form
      pre-bins.

2.  **Supervised Refinement:**

    - Calculates WoE and IV for each pre-bin.

    - Enforces monotonicity by merging bins that violate the trend
      (determined by the correlation between bin centroids and WoE
      values).

    - Merges bins with frequencies below `bin_cutoff`.

    - Iteratively merges bins to meet the `max_bins` constraint,
      choosing merges that minimize the loss of total Information Value.

This method is particularly powerful for complex, multi-modal
distributions where standard quantile or equal-width binning might
obscure important structural breaks.

## See also

[`ob_numerical_kmb`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_kmb.md),
[`ob_numerical_jedi`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_jedi.md)

## Examples

``` r
# Example: Binning a tri-modal distribution
set.seed(123)
# Feature with three distinct clusters
feature <- c(rnorm(300, mean = -3), rnorm(400, mean = 0), rnorm(300, mean = 3))
# Target depends on these clusters
target <- rbinom(1000, 1, plogis(feature))

result <- ob_numerical_lpdb(feature, target,
  min_bins = 3,
  max_bins = 5
)

print(result$bin) # Should ideally find cuts near -1.5 and 1.5
#> [1] "(-Inf; -1.851192]"     "(-1.851192; 0.120719]" "(0.120719; 1.985886]" 
#> [4] "(1.985886; 3.046734]"  "(3.046734; +Inf]"     
print(result$monotonicity)
#> [1] "increasing"
```
