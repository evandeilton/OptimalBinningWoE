# Optimal Binning for Categorical Variables using a User-Defined Technique (UDT)

This function performs optimal binning for categorical variables using a
User-Defined Technique (UDT) that combines frequency-based grouping with
statistical similarity measures to create meaningful bins for predictive
modeling.

## Usage

``` r
ob_categorical_udt(
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
  merged into a collective "rare" bin before optimization. Value must be
  between 0 and 1. Default is 0.05 (5%).

- max_n_prebins:

  Integer. Upper limit on initial bins after frequency filtering.
  Controls computational complexity in early stages. Default is 20.

- bin_separator:

  Character string used to separate category names when multiple
  categories are combined into a single bin. Default is "%;%".

- convergence_threshold:

  Numeric. Threshold for determining algorithm convergence based on
  relative changes in total Information Value. Default is 1e-6.

- max_iterations:

  Integer. Maximum number of iterations permitted for the optimization
  routine. Default is 1000.

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

  Logical. Whether the algorithm converged

- `iterations`:

  Integer. Number of iterations executed

## Details

The UDT algorithm follows these steps:

1.  **Initialization**: Each unique category is initially placed in its
    own bin.

2.  **Frequency Filtering**: Categories below the `bin_cutoff` frequency
    threshold are grouped into a single "rare" bin.

3.  **Iterative Optimization**: Bins are progressively merged based on
    statistical similarity (measured by Jensen-Shannon divergence) until
    the desired number of bins (`max_bins`) is achieved.

4.  **Monotonicity Enforcement**: Final bins are sorted by Weight of
    Evidence to ensure consistent trends.

Key characteristics of this implementation:

- **Flexible Framework**: Designed as a customizable foundation for
  categorical binning approaches.

- **Statistical Rigor**: Uses information-theoretic measures to guide
  bin combination decisions.

- **Robust Estimation**: Implements Laplace smoothing to ensure stable
  WoE/IV calculations even with sparse data.

- **Efficiency Focus**: Employs targeted merging strategies to minimize
  computational overhead.

Mathematical foundations:

Laplace-smoothed probability estimates: \$\$p\_{smoothed} =
\frac{count + \alpha}{total + 2\alpha}\$\$

Weight of Evidence calculation: \$\$WoE =
\ln\left(\frac{p\_{pos,smoothed}}{p\_{neg,smoothed}}\right)\$\$

Information Value computation: \$\$IV = (p\_{pos,smoothed} -
p\_{neg,smoothed}) \times WoE\$\$

Jensen-Shannon divergence between bins: \$\$JSD(P\|\|Q) =
\frac{1}{2}\[KL(P\|\|M) + KL(Q\|\|M)\]\$\$ where \\M =
\frac{1}{2}(P+Q)\\ and \\KL\\ denotes Kullback-Leibler divergence.

## Note

- Target variable must contain both 0 and 1 values.

- For datasets with 1 or 2 unique categories, no optimization occurs
  beyond basic WoE/IV calculation.

- The algorithm does not perform bin splitting; it only merges existing
  bins to respect `max_bins`.

- Rare category pooling improves stability of WoE estimates for
  infrequent values.

## Examples

``` r
# Generate sample data with skewed category distribution
set.seed(789)
n <- 3000
# Power-law distributed categories
categories <- c(
  rep("X1", 1200), rep("X2", 800), rep("X3", 400),
  sample(LETTERS[4:20], 600, replace = TRUE)
)
feature <- sample(categories, n, replace = TRUE)
# Target probabilities based on category importance
probs <- ifelse(grepl("X", feature), 0.7,
  ifelse(grepl("[A-C]", feature), 0.5, 0.3)
)
target <- rbinom(n, 1, prob = probs)

# Perform user-defined technique binning
result <- ob_categorical_udt(feature, target)
print(result[c("bin", "woe", "iv", "count")])
#> $bin
#> [1] "K%;%S%;%H%;%F%;%O%;%T%;%I%;%G%;%M%;%P%;%L%;%E%;%Q%;%R"
#> [2] "N%;%J%;%D"                                            
#> [3] "X3"                                                   
#> [4] "X2"                                                   
#> [5] "X1"                                                   
#> 
#> $woe
#> [1] -1.3704280 -1.2269348  0.2610244  0.3379009  0.3845215
#> 
#> $iv
#> [1] 0.322320167 0.037475936 0.009463915 0.029143017 0.055138785
#> 
#> $count
#> [1]  513   73  431  802 1181
#> 

# Adjust parameters for finer control
result_custom <- ob_categorical_udt(
  feature = feature,
  target = target,
  min_bins = 2,
  max_bins = 7,
  bin_cutoff = 0.03
)

# Handling missing values
feature_with_na <- feature
feature_with_na[sample(length(feature_with_na), 150)] <- NA
result_na <- ob_categorical_udt(feature_with_na, target)
```
