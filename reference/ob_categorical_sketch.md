# Optimal Binning for Categorical Variables using Sketch-based Algorithm

This function performs optimal binning for categorical variables using a
Sketch-based algorithm designed for large-scale data processing. It
employs probabilistic data structures (Count-Min Sketch) to efficiently
estimate category frequencies and event rates, enabling near real-time
binning on massive datasets.

## Usage

``` r
ob_categorical_sketch(
  feature,
  target,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L,
  bin_separator = "%;%",
  convergence_threshold = 1e-06,
  max_iterations = 1000L,
  sketch_width = 2000L,
  sketch_depth = 5L
)
```

## Arguments

- feature:

  A character vector or factor representing the categorical predictor
  variable. Missing values (NA) will be converted to the string "N/A"
  and treated as a separate category.

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

  Numeric. Minimum relative frequency threshold for categories to be
  considered "heavy hitters". Categories below this proportion will be
  grouped together. Value must be between 0 and 1. Default is 0.05 (5%).

- max_n_prebins:

  Integer. Maximum number of initial bins created during pre-binning
  phase. Controls early-stage complexity. Default is 20.

- bin_separator:

  Character string used to separate category names when multiple
  categories are merged into a single bin. Default is "%;%".

- convergence_threshold:

  Numeric. Threshold for determining algorithm convergence based on
  changes in total Information Value. Default is 1e-6.

- max_iterations:

  Integer. Maximum number of iterations for the optimization process.
  Default is 1000.

- sketch_width:

  Integer. Width of the Count-Min Sketch (number of counters per hash
  function). Larger values reduce estimation error but increase memory
  usage. Must be \>= 100. Default is 2000.

- sketch_depth:

  Integer. Depth of the Count-Min Sketch (number of hash functions).
  Larger values reduce collision probability but increase computational
  overhead. Must be \>= 3. Default is 5.

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

  Integer. Number of iterations performed

## Details

The Sketch-based algorithm follows these steps:

1.  **Frequency Estimation**: Uses Count-Min Sketch to approximate the
    frequency of each category in a single data pass.

2.  **Heavy Hitter Detection**: Identifies frequently occurring
    categories (above a threshold defined by `bin_cutoff`) using sketch
    estimates.

3.  **Pre-binning**: Creates initial bins from detected heavy
    categories, grouping rare categories separately.

4.  **Optimization**: Applies iterative merging based on statistical
    divergence measures to optimize Information Value (IV) while
    respecting bin count constraints (`min_bins`, `max_bins`).

5.  **Monotonicity Enforcement**: Ensures the final binning has
    monotonic Weight of Evidence (WoE).

Key advantages of this approach:

- **Memory Efficiency**: Uses sub-linear space complexity, independent
  of dataset size.

- **Speed**: Single-pass algorithm with constant-time updates.

- **Scalability**: Suitable for streaming data or datasets too large to
  fit in memory.

- **Approximation**: Trades perfect accuracy for significant gains in
  speed and memory usage.

Mathematical concepts:

The Count-Min Sketch uses multiple hash functions to map items to
counters: \$\$CMS\[i\]\[h_i(x)\] += 1 \quad \forall i \in
\\1,\ldots,d\\\$\$ where \\d\\ is the sketch depth and \\w\\ is the
sketch width.

Frequency estimates are obtained by taking the minimum across all
counters: \$\$\hat{f}(x) = \min\_{i} CMS\[i\]\[h_i(x)\]\$\$

Statistical divergence between bins is measured using Jensen-Shannon
divergence: \$\$JSD(P\|\|Q) = \frac{1}{2} \left\[ KL(P\|\|M) +
KL(Q\|\|M) \right\]\$\$ where \\M = \frac{1}{2}(P+Q)\\ and \\KL\\ is the
Kullback-Leibler divergence.

Laplace smoothing is applied to WoE and IV calculations:
\$\$p\_{smoothed} = \frac{count + \alpha}{total + 2\alpha}\$\$

## Note

- Target variable must contain both 0 and 1 values.

- Due to the probabilistic nature of sketches, results may vary slightly
  between runs. For deterministic results, consider setting fixed random
  seeds in the underlying C++ code.

- Accuracy of frequency estimates depends on `sketch_width` and
  `sketch_depth`. Increase these parameters for higher precision at the
  cost of memory/computation.

- This algorithm is particularly beneficial when dealing with
  high-cardinality categorical features or streaming data scenarios.

- For small to medium datasets, deterministic algorithms like SBLP or
  MOB may provide more accurate results.

## References

Cormode, G., & Muthukrishnan, S. (2005). An improved data stream
summary: the count-min sketch and its applications. Journal of
Algorithms, 55(1), 58-75.

Lin, J., & Keogh, E., Wei, L., & Lonardi, S. (2007). Experiencing SAX: a
novel symbolic representation of time series. Data Mining and Knowledge
Discovery, 15(2), 107-144.

## Examples

``` r
# Generate sample data
set.seed(123)
n <- 10000
feature <- sample(letters, n, replace = TRUE, prob = c(rep(0.04, 13), rep(0.02, 13)))
# Create a relationship where early letters have higher probability
target_probs <- ifelse(as.numeric(factor(feature)) <= 10, 0.7, 0.3)
target <- rbinom(n, 1, prob = target_probs)

# Perform sketch-based optimal binning
result <- ob_categorical_sketch(feature, target)
print(result[c("bin", "woe", "iv", "count")])
#> $bin
#> [1] "k%;%z%;%o%;%t"                 "u%;%v%;%y%;%l%;%m"            
#> [3] "n%;%p%;%x%;%w%;%r%;%s%;%q"     "h%;%j%;%f%;%g%;%b%;%d%;%c%;%i"
#> [5] "a%;%e"                        
#> 
#> $woe
#> [1] -1.0042752 -0.8308164 -0.8519467  0.8438619  0.8290098
#> 
#> $iv
#> [1] 0.12617874 0.12118256 0.11522899 0.27409018 0.06727459
#> 
#> $count
#> [1] 1354 1855 1682 4075 1034
#> 

# With custom sketch parameters for higher accuracy
result_high_acc <- ob_categorical_sketch(
  feature = feature,
  target = target,
  min_bins = 3,
  max_bins = 7,
  sketch_width = 4000,
  sketch_depth = 7
)

# Handling missing values
feature_with_na <- feature
feature_with_na[sample(length(feature_with_na), 200)] <- NA
result_na <- ob_categorical_sketch(feature_with_na, target)
```
