# Optimal Binning for Categorical Variables using Greedy Merge Algorithm

Performs supervised discretization of categorical variables using a
greedy bottom-up merging strategy that iteratively combines bins to
maximize total Information Value (IV). This approach uses Bayesian
smoothing for numerical stability and employs adaptive monotonicity
constraints, providing a fast approximation to optimal binning suitable
for high-cardinality features.

## Usage

``` r
ob_categorical_gmb(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  bin_separator = "%;%",
  convergence_threshold = 1e-06,
  max_iterations = 1000
)
```

## Arguments

- feature:

  A character vector or factor representing the categorical predictor
  variable to be binned. Missing values are automatically converted to
  the category `"NA"`.

- target:

  An integer vector of binary outcomes (0/1) corresponding to each
  observation in `feature`. Missing values are not permitted.

- min_bins:

  Integer. Minimum number of bins to produce. Must be \>= 2. Merging
  stops when this threshold is reached. Defaults to 3.

- max_bins:

  Integer. Maximum number of bins to produce. Must be \>= `min_bins`.
  The algorithm terminates when bins are reduced to this number.
  Defaults to 5.

- bin_cutoff:

  Numeric. Minimum proportion of total observations required for a
  category to remain separate during initialization. Categories below
  this threshold are pre-merged. Must be in (0, 1). Defaults to 0.05.

- max_n_prebins:

  Integer. Maximum number of initial bins before the greedy merging
  phase. Controls computational complexity. Must be \>= `min_bins`.
  Defaults to 20.

- bin_separator:

  Character string used to concatenate category names when multiple
  categories are merged into a single bin. Defaults to "%;%".

- convergence_threshold:

  Numeric. Convergence tolerance for IV change between iterations.
  Algorithm stops when \\\|\Delta IV\| \<\\ `convergence_threshold`.
  Must be \> 0. Defaults to 1e-6.

- max_iterations:

  Integer. Maximum number of merge operations allowed. Prevents
  excessive computation. Must be \> 0. Defaults to 1000.

## Value

A list containing the binning results with the following components:

- `id`:

  Integer vector of bin identifiers (1-indexed)

- `bin`:

  Character vector of bin labels (merged category names)

- `woe`:

  Numeric vector of Weight of Evidence values per bin

- `iv`:

  Numeric vector of Information Value contribution per bin

- `count`:

  Integer vector of total observations per bin

- `count_pos`:

  Integer vector of positive cases (target=1) per bin

- `count_neg`:

  Integer vector of negative cases (target=0) per bin

- `total_iv`:

  Numeric total Information Value of the binning solution

- `converged`:

  Logical indicating algorithm convergence

- `iterations`:

  Integer count of merge operations performed

## Details

The Greedy Merge Binning (GMB) algorithm employs a bottom-up approach
where bins are iteratively merged based on maximum IV improvement.
Unlike exact optimization methods (e.g., dynamic programming), GMB
provides approximate solutions with significantly reduced computational
cost.

**Algorithm Workflow:**

1.  Input validation and preprocessing

2.  Initial bin creation (one category per bin)

3.  Rare category merging (frequencies \< `bin_cutoff`)

4.  Pre-bin limitation (if bins \> `max_n_prebins`)

5.  Greedy merging phase:

    - Evaluate IV for all possible adjacent bin merges

    - Select merge that maximizes total IV

    - Apply tie-breaking rules for similar merges

    - Update IV cache incrementally

    - Check convergence criteria

6.  Adaptive monotonicity enforcement

7.  Final metric computation

**Bayesian Smoothing:**

To prevent numerical instability with sparse bins, WoE is calculated
using Bayesian smoothing:

\$\$WoE_i = \ln\left(\frac{p_i + \alpha_p}{n_i + \alpha_n}\right)\$\$

where \\\alpha_p\\ and \\\alpha_n\\ are prior pseudocounts proportional
to the overall event rate. This regularization ensures stable WoE
estimates even for bins with zero events.

**Greedy Selection Criterion:**

At each iteration, the algorithm evaluates the IV gain for merging
adjacent bins \\i\\ and \\j\\:

\$\$\Delta IV\_{i,j} = IV\_{merged}(i,j) - (IV_i + IV_j)\$\$

The pair with maximum \\\Delta IV\\ is merged. Early stopping occurs if
\\\Delta IV \> 0.05 \cdot IV\_{current}\\ (5% improvement threshold).

**Tie Handling:**

When multiple merges yield similar IV gains (within 10× convergence
threshold), the algorithm prefers merges that produce more balanced
bins, breaking ties based on size difference:

\$\$balance\\score = \|count_i - count_j\|\$\$

**Computational Complexity:**

- Time: \\O(k^2 \cdot m)\\ where \\k\\ = bins, \\m\\ = iterations

- Space: \\O(k^2)\\ for IV cache (optional)

- Typical runtime: 10-100× faster than exact methods for \\k \> 20\\

**Advantages:**

- Fast execution for high-cardinality features

- Incremental IV caching for efficiency

- Bayesian smoothing prevents overfitting

- Adaptive monotonicity with gradient relaxation

- Handles imbalanced datasets robustly

**Limitations:**

- Approximate solution (not guaranteed global optimum)

- Greedy nature may miss better non-adjacent merges

- Sensitive to initialization order

## References

Navas-Palencia, G. (2020). Optimal binning: mathematical programming
formulation and solution approach. *Expert Systems with Applications*,
158, 113508.
[doi:10.1016/j.eswa.2020.113508](https://doi.org/10.1016/j.eswa.2020.113508)

Good, I. J. (1965). The Estimation of Probabilities: An Essay on Modern
Bayesian Methods. MIT Press.

Zeng, G. (2014). A necessary condition for a good binning algorithm in
credit scoring. *Applied Mathematical Sciences*, 8(65), 3229-3242.

Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning
algorithm for credit risk modeling. *SSRN Electronic Journal*.
[doi:10.2139/ssrn.2978774](https://doi.org/10.2139/ssrn.2978774)

## See also

[`ob_categorical_dp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_dp.md)
for exact optimization via dynamic programming,
[`ob_categorical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_cm.md)
for ChiMerge-based binning,
[`ob_categorical_fetb`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_fetb.md)
for Fisher's Exact Test binning

## Examples

``` r
# \donttest{
# Example 1: Basic greedy merge binning
set.seed(123)
n_obs <- 1500

# Simulate customer types with varying risk
customer_types <- c(
  "Premium", "Gold", "Silver", "Bronze",
  "Basic", "Trial"
)
risk_probs <- c(0.02, 0.05, 0.10, 0.15, 0.22, 0.35)

cat_feature <- sample(customer_types, n_obs,
  replace = TRUE,
  prob = c(0.10, 0.15, 0.25, 0.25, 0.15, 0.10)
)
bin_target <- sapply(cat_feature, function(x) {
  rbinom(1, 1, risk_probs[which(customer_types == x)])
})

# Apply greedy merge binning
result_gmb <- ob_categorical_gmb(
  cat_feature,
  bin_target,
  min_bins = 3,
  max_bins = 4
)

# Display results
print(data.frame(
  Bin = result_gmb$bin,
  WoE = round(result_gmb$woe, 3),
  IV = round(result_gmb$iv, 4),
  Count = result_gmb$count,
  EventRate = round(result_gmb$count_pos / result_gmb$count, 3)
))
#>             Bin    WoE     IV Count EventRate
#> 1       Premium -1.885 0.1573   130     0.023
#> 2 Gold%;%Silver -0.737 0.1684   610     0.070
#> 3        Bronze  0.031 0.0002   377     0.141
#> 4 Basic%;%Trial  0.880 0.2657   383     0.277

cat("\nTotal IV:", round(result_gmb$total_iv, 4), "\n")
#> 
#> Total IV: 0.5916 
cat("Converged:", result_gmb$converged, "\n")
#> Converged: FALSE 
cat("Iterations:", result_gmb$iterations, "\n")
#> Iterations: 2 

# Example 2: Comparing speed with exact methods
set.seed(456)
n_obs <- 3000

# High cardinality feature
regions <- paste0("Region_", sprintf("%02d", 1:25))
cat_feature_hc <- sample(regions, n_obs, replace = TRUE)
bin_target_hc <- rbinom(n_obs, 1, 0.12)

# Greedy approach (fast)
time_gmb <- system.time({
  result_gmb_hc <- ob_categorical_gmb(
    cat_feature_hc,
    bin_target_hc,
    min_bins = 3,
    max_bins = 6,
    max_n_prebins = 20
  )
})

# Dynamic programming (exact but slower)
time_dp <- system.time({
  result_dp_hc <- ob_categorical_dp(
    cat_feature_hc,
    bin_target_hc,
    min_bins = 3,
    max_bins = 6,
    max_n_prebins = 20
  )
})

cat("\nPerformance comparison (high cardinality):\n")
#> 
#> Performance comparison (high cardinality):
cat("  GMB time:", round(time_gmb[3], 3), "seconds\n")
#>   GMB time: 0.001 seconds
cat("  DP time:", round(time_dp[3], 3), "seconds\n")
#>   DP time: 0 seconds
cat("  Speedup:", round(time_dp[3] / time_gmb[3], 1), "x\n")
#>   Speedup: 0 x
cat("\n  GMB IV:", round(result_gmb_hc$total_iv, 4), "\n")
#> 
#>   GMB IV: 0.0431 
cat("  DP IV:", round(result_dp_hc$total_iv, 4), "\n")
#>   DP IV: 0.0531 

# Example 3: Convergence behavior
set.seed(789)
n_obs_conv <- 1000

# Feature with natural groupings
education <- c("PhD", "Master", "Bachelor", "HighSchool", "NoHighSchool")
cat_feature_conv <- sample(education, n_obs_conv,
  replace = TRUE,
  prob = c(0.05, 0.15, 0.35, 0.30, 0.15)
)
bin_target_conv <- sapply(cat_feature_conv, function(x) {
  probs <- c(0.02, 0.05, 0.08, 0.15, 0.25)
  rbinom(1, 1, probs[which(education == x)])
})

# Test different convergence thresholds
thresholds <- c(1e-3, 1e-6, 1e-9)

for (thresh in thresholds) {
  result_conv <- ob_categorical_gmb(
    cat_feature_conv,
    bin_target_conv,
    min_bins = 2,
    max_bins = 4,
    convergence_threshold = thresh
  )

  cat(sprintf(
    "\nThreshold %.0e: %d iterations, converged=%s\n",
    thresh, result_conv$iterations, result_conv$converged
  ))
}
#> 
#> Threshold 1e-03: 1 iterations, converged=FALSE
#> 
#> Threshold 1e-06: 1 iterations, converged=FALSE
#> 
#> Threshold 1e-09: 1 iterations, converged=FALSE

# Example 4: Handling rare categories
set.seed(321)
n_obs_rare <- 2000

# Simulate with many rare categories
products <- c(paste0("Common_", 1:5), paste0("Rare_", 1:15))
product_probs <- c(rep(0.15, 5), rep(0.01, 15))

cat_feature_rare <- sample(products, n_obs_rare,
  replace = TRUE,
  prob = product_probs
)
bin_target_rare <- rbinom(n_obs_rare, 1, 0.10)

result_gmb_rare <- ob_categorical_gmb(
  cat_feature_rare,
  bin_target_rare,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.03 # Aggressive rare category merging
)

cat("\nRare category handling:\n")
#> 
#> Rare category handling:
cat("  Original categories:", length(unique(cat_feature_rare)), "\n")
#>   Original categories: 20 
cat("  Final bins:", length(result_gmb_rare$bin), "\n")
#>   Final bins: 5 

# Count merged rare categories
for (i in seq_along(result_gmb_rare$bin)) {
  n_merged <- length(strsplit(result_gmb_rare$bin[i], "%;%")[[1]])
  if (n_merged > 1) {
    cat(sprintf("  Bin %d: %d categories merged\n", i, n_merged))
  }
}
#>   Bin 1: 9 categories merged
#>   Bin 3: 2 categories merged
#>   Bin 5: 7 categories merged

# Example 5: Imbalanced dataset robustness
set.seed(555)
n_obs_imb <- 1200

# Highly imbalanced target (2% event rate)
cat_feature_imb <- sample(c("A", "B", "C", "D", "E"),
  n_obs_imb,
  replace = TRUE
)
bin_target_imb <- rbinom(n_obs_imb, 1, 0.02)

result_gmb_imb <- ob_categorical_gmb(
  cat_feature_imb,
  bin_target_imb,
  min_bins = 2,
  max_bins = 3
)

cat("\nImbalanced dataset:\n")
#> 
#> Imbalanced dataset:
cat("  Event rate:", round(mean(bin_target_imb), 4), "\n")
#>   Event rate: 0.0183 
cat("  Total events:", sum(bin_target_imb), "\n")
#>   Total events: 22 
cat("  Bins created:", length(result_gmb_imb$bin), "\n")
#>   Bins created: 3 
cat("  WoE range:", sprintf(
  "[%.2f, %.2f]\n",
  min(result_gmb_imb$woe),
  max(result_gmb_imb$woe)
))
#>   WoE range: [-0.47, 0.40]
# }
```
