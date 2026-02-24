# Optimal Binning for Categorical Variables using Heuristic Algorithm

This function performs optimal binning for categorical variables using a
heuristic merging approach to maximize Information Value (IV) while
maintaining monotonic Weight of Evidence (WoE). Despite its name
containing "MILP", it does NOT use Mixed Integer Linear Programming but
rather a greedy optimization algorithm.

## Usage

``` r
ob_categorical_milp(
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

  Integer. Minimum number of bins to create. Must be at least 2. Default
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

  Integer vector of bin identifiers (1 to n_bins)

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

The algorithm follows these steps:

1.  Pre-binning: Each unique category becomes an initial bin

2.  Rare category handling: Categories below `bin_cutoff` frequency are
    merged with similar ones

3.  Bin reduction: Greedily merge bins to satisfy `min_bins` and
    `max_bins` constraints

4.  Monotonicity enforcement: Ensures WoE is either consistently
    increasing or decreasing across bins

5.  Optimization: Iteratively improves Information Value

Key features include:

- Bayesian smoothing to stabilize WoE estimates for sparse categories

- Automatic handling of missing values (converted to "NA" category)

- Monotonicity constraint enforcement

- Configurable minimum and maximum bin counts

- Rare category pooling based on relative frequency thresholds

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

- The algorithm uses a greedy heuristic approach, not true MILP
  optimization. For exact solutions, external solvers like Gurobi or
  CPLEX would be required.

## Examples

``` r
# Generate sample data
set.seed(123)
n <- 1000
feature <- sample(letters[1:8], n, replace = TRUE)
target <- rbinom(n, 1, prob = ifelse(feature %in% c("a", "b"), 0.7, 0.3))

# Perform optimal binning
result <- ob_categorical_milp(feature, target)
print(result[c("bin", "woe", "iv", "count")])
#> $bin
#> [1] "h"             "b%;%f%;%c%;%g" "a%;%d%;%e"    
#> 
#> $woe
#> [1] -0.55634710  0.02805836  0.14604873
#> 
#> $iv
#> [1] 0.0379899544 0.0003986034 0.0078405377
#> 
#> $count
#> [1] 132 505 363
#> 

# With custom parameters
result2 <- ob_categorical_milp(
  feature = feature,
  target = target,
  min_bins = 2,
  max_bins = 4,
  bin_cutoff = 0.03
)

# Handling missing values
feature_with_na <- feature
feature_with_na[sample(length(feature_with_na), 50)] <- NA
result3 <- ob_categorical_milp(feature_with_na, target)
```
