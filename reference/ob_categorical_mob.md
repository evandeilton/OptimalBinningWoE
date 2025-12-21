# Optimal Binning for Categorical Variables using Monotonic Optimal Binning (MOB)

This function performs optimal binning for categorical variables using
the Monotonic Optimal Binning (MOB) algorithm. It creates bins that
maintain monotonic Weight of Evidence (WoE) trends while maximizing
Information Value.

## Usage

``` r
ob_categorical_mob(
  feature,
  target,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L,
  bin_separator = "%;%",
  convergence_threshold = 1e-06,
  max_iterations = 1000L
)
```

## Arguments

- feature:

  A character vector or factor representing the categorical predictor
  variable. Missing values (NA) will be converted to the string "NA" and
  treated as a separate category.

- target:

  An integer vector containing binary outcome values (0 or 1). Must be
  the same length as `feature`. Cannot contain missing values.

- min_bins:

  Integer. Minimum number of bins to create. Must be at least 1. Default
  is 3.

- max_bins:

  Integer. Maximum number of bins to create. Must be greater than or
  equal to `min_bins`. Default is 5.

- bin_cutoff:

  Numeric. Minimum relative frequency threshold for individual
  categories. Categories with frequency below this proportion will be
  merged with others. Value must be between 0 and 1. Default is 0.05
  (5%).

- max_n_prebins:

  Integer. Maximum number of initial bins before optimization. Used to
  control computational complexity when dealing with high-cardinality
  categorical variables. Default is 20.

- bin_separator:

  Character string used to separate category names when multiple
  categories are merged into a single bin. Default is "%;%".

- convergence_threshold:

  Numeric. Threshold for determining algorithm convergence based on
  changes in total Information Value. Must be positive. Default is 1e-6.

- max_iterations:

  Integer. Maximum number of iterations for the optimization process.
  Must be positive. Default is 1000.

## Value

A list containing the results of the optimal binning procedure:

- `id`:

  Numeric vector of bin identifiers (1 to n_bins)

- `bin`:

  Character vector of bin labels, which are combinations of original
  categories separated by `bin_separator`

- `woe`:

  Numeric vector of Weight of Evidence values for each bin

- `iv`:

  Numeric vector of Information Values for each bin

- `count`:

  Integer vector of total observations in each bin

- `count_pos`:

  Integer vector of positive outcomes in each bin

- `count_neg`:

  Integer vector of negative outcomes in each bin

- `total_iv`:

  Numeric scalar. Total Information Value across all bins

- `converged`:

  Logical. Whether the algorithm converged within the specified
  tolerance

- `iterations`:

  Integer. Number of iterations performed

## Details

The MOB algorithm follows these steps:

1.  Initial sorting: Categories are ordered by their individual WoE
    values

2.  Rare category handling: Categories below `bin_cutoff` frequency are
    merged with similar ones

3.  Pre-binning limitation: Reduces initial bins to `max_n_prebins`
    using similarity-based merging

4.  Monotonicity enforcement: Ensures WoE is either consistently
    increasing or decreasing across bins

5.  Bin count optimization: Adjusts to meet `min_bins`/`max_bins`
    constraints

Key features include:

- Automatic sorting of categories by WoE for initial structure

- Bayesian smoothing to stabilize WoE estimates for sparse categories

- Guaranteed monotonic WoE trend across final bins

- Configurable minimum and maximum bin counts

- Similarity-based merging for optimal bin combinations

Mathematical definitions: \$\$WoE_i =
\ln\left(\frac{p_i^{(1)}}{p_i^{(0)}}\right)\$\$ where \\p_i^{(1)}\\ and
\\p_i^{(0)}\\ are the proportions of positive and negative cases in bin
\\i\\, respectively, adjusted using Bayesian smoothing.

\$\$IV = \sum\_{i=1}^{n} (p_i^{(1)} - p_i^{(0)}) \times WoE_i\$\$

## Note

- Target variable must contain both 0 and 1 values.

- Empty strings in the feature vector are not allowed and will cause an
  error.

- For datasets with very few observations in either class (\<5),
  warnings will be issued as results may be unstable.

- The algorithm guarantees monotonic WoE across bins.

- When the number of unique categories is less than `max_bins`, each
  category will form its own bin.

## Examples

``` r
# Generate sample data
set.seed(123)
n <- 1000
feature <- sample(letters[1:8], n, replace = TRUE)
target <- rbinom(n, 1, prob = ifelse(feature %in% c("a", "b"), 0.7, 0.3))

# Perform optimal binning
result <- ob_categorical_mob(feature, target)
print(result[c("bin", "woe", "iv", "count")])
#> $bin
#> [1] "c%;%f" "e%;%h" "d%;%g" "a%;%b"
#> 
#> $woe
#> [1] -0.6160508 -0.4979681 -0.2902418  1.2813219
#> 
#> $iv
#> [1] 0.08435798 0.05880863 0.02029562 0.41586980
#> 
#> $count
#> [1] 242 253 249 256
#> 

# With custom parameters
result2 <- ob_categorical_mob(
  feature = feature,
  target = target,
  min_bins = 2,
  max_bins = 4,
  bin_cutoff = 0.03
)

# Handling missing values
feature_with_na <- feature
feature_with_na[sample(length(feature_with_na), 50)] <- NA
result3 <- ob_categorical_mob(feature_with_na, target)
```
