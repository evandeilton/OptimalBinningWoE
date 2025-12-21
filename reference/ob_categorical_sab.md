# Optimal Binning for Categorical Variables using Simulated Annealing

This function performs optimal binning for categorical variables using a
Simulated Annealing (SA) optimization algorithm. It maximizes
Information Value (IV) while maintaining monotonic Weight of Evidence
(WoE) trends.

## Usage

``` r
ob_categorical_sab(
  feature,
  target,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L,
  bin_separator = "%;%",
  initial_temperature = 1,
  cooling_rate = 0.995,
  max_iterations = 1000L,
  convergence_threshold = 1e-06,
  adaptive_cooling = TRUE
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

  Numeric. Minimum relative frequency threshold for individual bins.
  Bins with frequency below this proportion will be penalized. Value
  must be between 0 and 1. Default is 0.05 (5%).

- max_n_prebins:

  Integer. Maximum number of initial categories before optimization (not
  directly used in current implementation). Must be greater than or
  equal to `max_bins`. Default is 20.

- bin_separator:

  Character string used to separate category names when multiple
  categories are merged into a single bin. Default is "%;%".

- initial_temperature:

  Numeric. Starting temperature for the simulated annealing algorithm.
  Higher values allow more exploration. Must be positive. Default is
  1.0.

- cooling_rate:

  Numeric. Rate at which temperature decreases during optimization.
  Value must be between 0 and 1. Lower values lead to faster cooling.
  Default is 0.995.

- max_iterations:

  Integer. Maximum number of iterations for the optimization process.
  Must be positive. Default is 1000.

- convergence_threshold:

  Numeric. Threshold for determining algorithm convergence based on
  changes in Information Value. Must be positive. Default is 1e-6.

- adaptive_cooling:

  Logical. Whether to use adaptive cooling that modifies the cooling
  rate based on search progress. Default is TRUE.

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

The SAB (Simulated Annealing Binning) algorithm follows these steps:

1.  Initialization: Categories are initially assigned to bins using a
    k-means-like strategy based on event rates

2.  Optimization: Simulated annealing explores different bin assignments
    to maximize IV

3.  Neighborhood generation: Multiple strategies are employed to
    generate neighboring solutions (swaps, reassignments, event-rate
    based moves)

4.  Acceptance criteria: New solutions are accepted based on the
    Metropolis criterion with adaptive temperature control

5.  Monotonicity enforcement: Final solutions are adjusted to ensure
    monotonic WoE trends

Key features include:

- Global optimization approach using simulated annealing

- Adaptive cooling schedule to balance exploration and exploitation

- Multiple neighborhood generation strategies for better search

- Bayesian smoothing to stabilize WoE estimates for sparse categories

- Guaranteed monotonic WoE trend across final bins

- Configurable optimization parameters for fine-tuning

Mathematical definitions: \$\$WoE_i =
\ln\left(\frac{p_i^{(1)}}{p_i^{(0)}}\right)\$\$ where \\p_i^{(1)}\\ and
\\p_i^{(0)}\\ are the proportions of positive and negative cases in bin
\\i\\, respectively, adjusted using Bayesian smoothing.

\$\$IV = \sum\_{i=1}^{n} (p_i^{(1)} - p_i^{(0)}) \times WoE_i\$\$

The acceptance probability in simulated annealing is: \$\$P(accept) =
\exp\left(\frac{IV\_{new} - IV\_{current}}{T}\right)\$\$ where \\T\\ is
the current temperature.

## Note

- Target variable must contain both 0 and 1 values.

- Empty strings in the feature vector are not allowed and will cause an
  error.

- For datasets with very few observations in either class (\<5),
  warnings will be issued as results may be unstable.

- The algorithm uses global optimization which may require more
  computational time compared to heuristic approaches.

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
result <- ob_categorical_sab(feature, target)
print(result[c("bin", "woe", "iv", "count")])
#> $bin
#> [1] "f"     "e%;%h" "c%;%d" "a%;%g" "b"    
#> 
#> $woe
#> [1] -0.6671948 -0.4979681 -0.4781139  0.4367024  1.4348044
#> 
#> $iv
#> [1] 0.04870896 0.05880863 0.05074278 0.05152071 0.25663321
#> 
#> $count
#> [1] 120 253 236 263 128
#> 

# With custom parameters
result2 <- ob_categorical_sab(
  feature = feature,
  target = target,
  min_bins = 2,
  max_bins = 4,
  initial_temperature = 2.0,
  cooling_rate = 0.99
)

# Handling missing values
feature_with_na <- feature
feature_with_na[sample(length(feature_with_na), 50)] <- NA
result3 <- ob_categorical_sab(feature_with_na, target)
```
