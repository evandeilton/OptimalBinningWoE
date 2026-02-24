# Optimal Binning for Categorical Variables using SBLP

This function performs optimal binning for categorical variables using
the Similarity-Based Logistic Partitioning (SBLP) algorithm. This
approach combines logistic properties (sorting categories by event rate)
with dynamic programming to find the optimal partition that maximizes
Information Value (IV).

## Usage

``` r
ob_categorical_sblp(
  feature,
  target,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L,
  convergence_threshold = 1e-06,
  max_iterations = 1000L,
  bin_separator = "%;%",
  alpha = 0.5
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

  Integer. Minimum number of bins to create. Must be at least 2. Default
  is 3.

- max_bins:

  Integer. Maximum number of bins to create. Must be greater than or
  equal to `min_bins`. Default is 5.

- bin_cutoff:

  Numeric. Minimum relative frequency threshold for individual
  categories. Categories with frequency below this proportion will be
  merged with similar categories before the main optimization. Value
  must be between 0 and 1. Default is 0.05 (5%).

- max_n_prebins:

  Integer. Maximum number of initial bins/groups allowed before the
  dynamic programming optimization. If the number of unique categories
  exceeds this, similar adjacent categories are pre-merged. Default is
  20.

- convergence_threshold:

  Numeric. Threshold for determining algorithm convergence based on
  changes in total Information Value. Default is 1e-6.

- max_iterations:

  Integer. Maximum number of iterations for the optimization process.
  Default is 1000.

- bin_separator:

  Character string used to separate category names when multiple
  categories are merged into a single bin. Default is "%;%".

- alpha:

  Numeric. Laplace smoothing parameter added to counts to avoid division
  by zero and stabilize WoE calculations for sparse data. Must be
  non-negative. Default is 0.5.

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

- `rate`:

  Numeric vector of the observed event rate in each bin

- `total_iv`:

  Numeric scalar. Total Information Value across all bins

- `converged`:

  Logical. Whether the algorithm converged

- `iterations`:

  Integer. Number of iterations performed

## Details

The SBLP algorithm follows these steps:

1.  **Preprocessing**: Handling of missing values and calculation of
    initial statistics.

2.  **Rare Category Consolidation**: Categories with frequency below
    `bin_cutoff` are merged with statistically similar categories based
    on their target rates.

3.  **Sorting**: Unique categories (or merged groups) are sorted by
    their empirical event rate (probability of target=1).

4.  **Dynamic Programming**: An optimal partitioning algorithm (similar
    to Jenks Natural Breaks but optimizing IV) is applied to the sorted
    sequence to determine the cutpoints that maximize the total IV.

5.  **Refinement**: Post-processing ensures constraints like
    monotonicity and minimum bin size are met.

A key feature of this implementation is the use of **Laplace Smoothing**
(controlled by the `alpha` parameter) to prevent infinite WoE values and
stabilize estimates for categories with small counts.

Mathematical definitions with smoothing:

The smoothed event rate \\p_i\\ for a bin is calculated as: \$\$p_i =
\frac{n\_{pos} + \alpha}{n\_{total} + 2\alpha}\$\$

The Weight of Evidence (WoE) is computed using smoothed proportions:
\$\$WoE_i = \ln\left(\frac{p_i^{(1)}}{p_i^{(0)}}\right)\$\$

where \\p_i^{(1)}\\ and \\p_i^{(0)}\\ are the smoothed distributions of
positive and negative classes across bins.

## Note

- Target variable must contain both 0 and 1 values.

- Unlike heuristic methods, this algorithm uses Dynamic Programming
  which guarantees an optimal partition given the sorted order of
  categories.

- Monotonicity is generally enforced by the sorting step, but strictly
  checked and corrected in the final output.

## Examples

``` r
# Generate sample data
set.seed(123)
n <- 1000
feature <- sample(letters[1:8], n, replace = TRUE)
# Create a relationship where 'a' and 'b' have high probability
target <- rbinom(n, 1, prob = ifelse(feature %in% c("a", "b"), 0.8, 0.2))

# Perform optimal binning
result <- ob_categorical_sblp(feature, target)
print(result[c("bin", "woe", "iv", "count")])
#> $bin
#> [1] "f"         "d%;%e"     "c%;%g%;%h" "a"         "b"        
#> 
#> $woe
#> [1] -1.1094145 -0.8567210 -0.6674771  1.8184450  2.2121155
#> 
#> $iv
#> [1] 0.1173476 0.1453568 0.1523956 0.4263670 0.5956975
#> 
#> $count
#> [1] 120 235 389 128 128
#> 

# Using a higher smoothing parameter (alpha)
result_smooth <- ob_categorical_sblp(
  feature = feature,
  target = target,
  alpha = 1.0
)

# Handling missing values
feature_with_na <- feature
feature_with_na[sample(length(feature_with_na), 50)] <- NA
result_na <- ob_categorical_sblp(feature_with_na, target)
```
