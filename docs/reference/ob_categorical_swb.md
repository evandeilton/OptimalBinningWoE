# Optimal Binning for Categorical Variables using Sliding Window Binning (SWB)

This function performs optimal binning for categorical variables using
the Sliding Window Binning (SWB) algorithm. This approach combines
initial grouping based on frequency thresholds with iterative
optimization to achieve monotonic Weight of Evidence (WoE) while
maximizing Information Value (IV).

## Usage

``` r
ob_categorical_swb(
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
  grouped together into a single "rare" bin. Value must be between 0
  and 1. Default is 0.05 (5%).

- max_n_prebins:

  Integer. Maximum number of initial bins created after the
  frequency-based grouping step. Used to control early-stage complexity.
  Default is 20.

- bin_separator:

  Character string used to separate category names when multiple
  categories are merged into a single bin. Default is "%;%".

- convergence_threshold:

  Numeric. Threshold for determining algorithm convergence based on
  changes in total Information Value between iterations. Default is
  1e-6.

- max_iterations:

  Integer. Maximum number of iterations for the optimization process.
  Default is 1000.

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

- `event_rate`:

  Numeric vector of the observed event rate in each bin

- `total_iv`:

  Numeric scalar. Total Information Value across all bins

- `converged`:

  Logical. Whether the algorithm converged within specified tolerances

- `iterations`:

  Integer. Number of iterations performed

## Details

The SWB algorithm follows these steps:

1.  **Initialization**: Categories are initially grouped based on
    frequency thresholds (`bin_cutoff`), separating frequent categories
    from rare ones.

2.  **Preprocessing**: Initial bins are sorted by their WoE values to
    establish a baseline ordering.

3.  **Sliding Window Optimization**: An iterative process evaluates
    adjacent bin pairs and merges those that contribute least to the
    overall Information Value or violate monotonicity constraints.

4.  **Constraint Enforcement**: The final binning respects the specified
    `min_bins` and `max_bins` limits while maintaining WoE monotonicity.

Key features of this implementation:

- **Frequency-based Pre-grouping**: Automatically identifies and groups
  rare categories to reduce dimensionality.

- **Statistical Similarity Measures**: Utilizes Jensen-Shannon
  divergence to determine optimal merge candidates.

- **Monotonicity Preservation**: Ensures final bins exhibit consistent
  WoE trends (either increasing or decreasing).

- **Laplace Smoothing**: Employs additive smoothing to prevent numerical
  instabilities in WoE/IV calculations.

Mathematical concepts:

Weight of Evidence (WoE) with Laplace smoothing: \$\$WoE =
\ln\left(\frac{(p\_{pos} + \alpha)/(N\_{pos} + 2\alpha)}{(p\_{neg} +
\alpha)/(N\_{neg} + 2\alpha)}\right)\$\$

Information Value (IV): \$\$IV = \left(\frac{p\_{pos} +
\alpha}{N\_{pos} + 2\alpha} - \frac{p\_{neg} + \alpha}{N\_{neg} +
2\alpha}\right) \times WoE\$\$

where \\p\_{pos}\\ and \\p\_{neg}\\ are bin-level counts, \\N\_{pos}\\
and \\N\_{neg}\\ are dataset-level totals, and \\\alpha\\ is the
smoothing parameter (default 0.5).

Jensen-Shannon Divergence between two bins: \$\$JSD(P\|\|Q) =
\frac{1}{2}\left\[KL(P\|\|M) + KL(Q\|\|M)\right\]\$\$ where \\M =
\frac{1}{2}(P+Q)\\ and \\KL\\ represents Kullback-Leibler divergence.

## Note

- Target variable must contain both 0 and 1 values.

- The algorithm prioritizes monotonicity over strict adherence to bin
  count limits when conflicts arise.

- For datasets with very few unique categories (\< 3), each category
  forms its own bin without optimization.

- Rare category grouping helps stabilize WoE estimates for infrequent
  values.

## Examples

``` r
# Generate sample data with varying category frequencies
set.seed(456)
n <- 5000
# Create categories with power-law frequency distribution
categories <- c(
  rep("A", 1500), rep("B", 1000), rep("C", 800),
  rep("D", 500), rep("E", 300), rep("F", 200),
  sample(letters[7:26], 700, replace = TRUE)
)
feature <- sample(categories, n, replace = TRUE)
# Create target with dependency on top categories
target_probs <- ifelse(feature %in% c("A", "B"), 0.7,
  ifelse(feature %in% c("C", "D"), 0.5, 0.3)
)
target <- rbinom(n, 1, prob = target_probs)

# Perform sliding window binning
result <- ob_categorical_swb(feature, target)
print(result[c("bin", "woe", "iv", "count")])
#> $bin
#> [1] "E"                                                                                
#> [2] "i%;%g%;%y%;%r%;%q%;%l%;%t%;%z%;%w%;%n%;%F%;%k%;%j%;%x%;%m%;%p%;%h%;%u%;%s%;%v%;%o"
#> [3] "C%;%D"                                                                            
#> [4] "A"                                                                                
#> [5] "B"                                                                                
#> 
#> $woe
#> [1] -1.1552050 -1.0541862 -0.1830370  0.6332108  0.6365736
#> 
#> $iv
#> [1] 0.073747445 0.195525930 0.008783081 0.112135969 0.076546047
#> 
#> $count
#> [1]  291  916 1303 1486 1004
#> 

# With stricter bin limits
result_strict <- ob_categorical_swb(
  feature = feature,
  target = target,
  min_bins = 4,
  max_bins = 6
)

# Handling missing values
feature_with_na <- feature
feature_with_na[sample(length(feature_with_na), 100)] <- NA
result_na <- ob_categorical_swb(feature_with_na, target)
```
