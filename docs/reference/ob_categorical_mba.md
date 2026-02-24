# Optimal Binning for Categorical Variables using Monotonic Binning Algorithm

Performs supervised discretization of categorical variables using the
Monotonic Binning Algorithm (MBA), which enforces strict Weight of
Evidence monotonicity while optimizing Information Value through
intelligent bin merging strategies. This implementation includes
Bayesian smoothing for numerical stability and adaptive thresholding for
robust monotonicity enforcement.

## Usage

``` r
ob_categorical_mba(
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

  Integer. Minimum number of bins to produce. Must be \>= 2. The
  algorithm will not merge below this threshold. Defaults to 3.

- max_bins:

  Integer. Maximum number of bins to produce. Must be \>= `min_bins`.
  The algorithm reduces bins until this constraint is met. Defaults to
  5.

- bin_cutoff:

  Numeric. Minimum proportion of total observations required for a
  category to remain separate. Categories below this threshold are
  pre-merged with similar categories. Must be in (0, 1). Defaults to
  0.05.

- max_n_prebins:

  Integer. Maximum number of initial bins before the main optimization
  phase. Controls computational complexity. Must be \>= `max_bins`.
  Defaults to 20.

- bin_separator:

  Character string used to concatenate category names when multiple
  categories are merged into a single bin. Defaults to "%;%".

- convergence_threshold:

  Numeric. Convergence tolerance based on Information Value change
  between iterations. Algorithm stops when \\\|\Delta IV\| \<\\
  `convergence_threshold`. Must be \> 0. Defaults to 1e-6.

- max_iterations:

  Integer. Maximum number of optimization iterations. Prevents infinite
  loops. Must be \> 0. Defaults to 1000.

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

  Integer count of optimization iterations performed

## Details

The Monotonic Binning Algorithm (MBA) implements a sophisticated
approach to categorical binning that guarantees strict Weight of
Evidence monotonicity through intelligent violation detection and repair
mechanisms.

**Algorithm Workflow:**

1.  Input validation and preprocessing

2.  Initial bin creation (one category per bin)

3.  Pre-binning limitation to `max_n_prebins`

4.  Rare category merging (frequencies \< `bin_cutoff`)

5.  Bayesian-smoothed WoE calculation

6.  Strict monotonicity enforcement with adaptive thresholds

7.  IV-optimized bin merging to meet `max_bins` constraint

8.  Final consistency verification

**Monotonicity Enforcement:**

MBA enforces strict monotonicity through an iterative repair process:

1.  Sort bins by current WoE values

2.  Calculate adaptive threshold: \\\tau = \min(\epsilon,
    0.01\bar{\Delta})\\

3.  Identify violations: \\sign(WoE_i - WoE\_{i-1}) \neq
    sign(WoE\_{i+1} - WoE_i)\\

4.  Rank violations by severity: \\s_i = \|WoE_i - WoE\_{i-1}\| +
    \|WoE\_{i+1} - WoE_i\|\\

5.  Repair most severe violations by merging adjacent bins

6.  Repeat until no violations remain or `min_bins` reached

**Bayesian Smoothing:**

To ensure numerical stability and prevent overfitting, MBA applies
Bayesian smoothing to WoE and IV calculations:

\$\$p'\_i = \frac{n\_{i,pos} + \alpha_p}{N\_{pos} + \alpha\_{total}}\$\$
\$\$n'\_i = \frac{n\_{i,neg} + \alpha_n}{N\_{neg} + \alpha\_{total}}\$\$

where priors are proportional to overall prevalence:

\$\$\alpha_p = \alpha\_{total} \times \frac{N\_{pos}}{N\_{pos} +
N\_{neg}}\$\$ \$\$\alpha_n = \alpha\_{total} - \alpha_p\$\$

with \\\alpha\_{total} = 1.0\\ as the prior strength parameter.

**Intelligent Bin Merging:**

When reducing bins to meet the `max_bins` constraint, MBA employs an
IV-loss minimization strategy:

\$\$\Delta IV\_{i,j} = IV_i + IV_j - IV\_{merged}(i,j)\$\$

The pair with minimum \\\Delta IV\\ is merged to preserve maximum
predictive information.

**Computational Complexity:**

- Time: \\O(k^2 \cdot m)\\ where \\k\\ = bins, \\m\\ = iterations

- Space: \\O(k^2)\\ for IV loss cache

- Cache hit rate typically \> 75% for \\k \> 10\\

**Key Features:**

- **Guaranteed monotonicity**: Strict enforcement with adaptive
  thresholds

- **Bayesian regularization**: Robust to sparse bins and class imbalance

- **Intelligent merging**: Preserves maximum information during
  reduction

- **Adaptive thresholds**: Context-aware violation detection

- **Consistency verification**: Final integrity checks

## References

Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning
algorithm for credit risk modeling. *SSRN Electronic Journal*.
[doi:10.2139/ssrn.2978774](https://doi.org/10.2139/ssrn.2978774)

Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and
Implementing Better Credit Risk Scorecards* (2nd ed.). Wiley.

Good, I. J. (1965). *The Estimation of Probabilities: An Essay on Modern
Bayesian Methods*. MIT Press.

Zeng, G. (2014). A necessary condition for a good binning algorithm in
credit scoring. *Applied Mathematical Sciences*, 8(65), 3229-3242.

## See also

[`ob_categorical_jedi`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_jedi.md)
for joint entropy-driven optimization,
[`ob_categorical_dp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_dp.md)
for dynamic programming approach,
[`ob_categorical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_cm.md)
for ChiMerge-based binning

## Examples

``` r
# \donttest{
# Example 1: Basic monotonic binning with guaranteed WoE ordering
set.seed(42)
n_obs <- 1500

# Simulate risk ratings with natural monotonic relationship
ratings <- c("AAA", "AA", "A", "BBB", "BB", "B", "CCC")
default_probs <- c(0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)

cat_feature <- sample(ratings, n_obs,
  replace = TRUE,
  prob = c(0.05, 0.10, 0.20, 0.25, 0.20, 0.15, 0.05)
)
bin_target <- sapply(cat_feature, function(x) {
  rbinom(1, 1, default_probs[which(ratings == x)])
})

# Apply MBA algorithm
result_mba <- ob_categorical_mba(
  cat_feature,
  bin_target,
  min_bins = 3,
  max_bins = 5
)

# Display results with guaranteed monotonic WoE
print(data.frame(
  Bin = result_mba$bin,
  WoE = round(result_mba$woe, 3),
  IV = round(result_mba$iv, 4),
  Count = result_mba$count,
  EventRate = round(result_mba$count_pos / result_mba$count, 3)
))
#>            Bin    WoE     IV Count EventRate
#> 1 A%;%AA%;%AAA -1.305 0.3529   486     0.053
#> 2          BBB -0.700 0.1015   394     0.094
#> 3           BB  0.211 0.0098   307     0.205
#> 4            B  1.045 0.2117   217     0.373
#> 5          CCC  1.725 0.2845    96     0.542

cat("\nMonotonicity check (WoE differences):\n")
#> 
#> Monotonicity check (WoE differences):
woe_diffs <- diff(result_mba$woe)
cat("  Differences:", paste(round(woe_diffs, 4), collapse = ", "), "\n")
#>   Differences: 0.6051, 0.9113, 0.8342, 0.6795 
cat("  All positive (increasing):", all(woe_diffs >= -1e-10), "\n")
#>   All positive (increasing): TRUE 
cat("  Total IV:", round(result_mba$total_iv, 4), "\n")
#>   Total IV: 0.9604 
cat("  Converged:", result_mba$converged, "\n")
#>   Converged: TRUE 

# Example 2: Comparison with non-monotonic methods
set.seed(123)
n_obs_comp <- 2000

sectors <- c("Tech", "Health", "Finance", "Manufacturing", "Retail")
cat_feature_comp <- sample(sectors, n_obs_comp, replace = TRUE)
bin_target_comp <- rbinom(n_obs_comp, 1, 0.15)

# MBA (strictly monotonic)
result_mba_comp <- ob_categorical_mba(
  cat_feature_comp, bin_target_comp,
  min_bins = 3, max_bins = 4
)

# Standard binning (may not be monotonic)
result_std_comp <- ob_categorical_cm(
  cat_feature_comp, bin_target_comp,
  min_bins = 3, max_bins = 4
)

cat("\nMonotonicity comparison:\n")
#> 
#> Monotonicity comparison:
cat(
  "  MBA WoE differences:",
  paste(round(diff(result_mba_comp$woe), 4), collapse = ", "), "\n"
)
#>   MBA WoE differences: 0.1479, 0.0667, 0.0646 
cat("  MBA monotonic:", all(diff(result_mba_comp$woe) >= -1e-10), "\n")
#>   MBA monotonic: TRUE 
cat(
  "  Std WoE differences:",
  paste(round(diff(result_std_comp$woe), 4), collapse = ", "), "\n"
)
#>   Std WoE differences: 0.1431, 0.0695, 0.0645 
cat("  Std monotonic:", all(diff(result_std_comp$woe) >= -1e-10), "\n")
#>   Std monotonic: TRUE 

# Example 3: Bayesian smoothing with sparse data
set.seed(789)
n_obs_sparse <- 400

# Small sample with rare categories
categories <- c("A", "B", "C", "D", "E", "F")
cat_probs <- c(0.30, 0.25, 0.20, 0.15, 0.07, 0.03)

cat_feature_sparse <- sample(categories, n_obs_sparse,
  replace = TRUE,
  prob = cat_probs
)
bin_target_sparse <- rbinom(n_obs_sparse, 1, 0.08) # 8% event rate

result_mba_sparse <- ob_categorical_mba(
  cat_feature_sparse,
  bin_target_sparse,
  min_bins = 2,
  max_bins = 4,
  bin_cutoff = 0.02
)

cat("\nBayesian smoothing (sparse data):\n")
#> 
#> Bayesian smoothing (sparse data):
cat("  Sample size:", n_obs_sparse, "\n")
#>   Sample size: 400 
cat("  Events:", sum(bin_target_sparse), "\n")
#>   Events: 29 
cat("  Final bins:", length(result_mba_sparse$bin), "\n\n")
#>   Final bins: 4 
#> 

# Show how smoothing prevents extreme WoE values
for (i in seq_along(result_mba_sparse$bin)) {
  cat(sprintf(
    "  Bin %d: events=%d/%d, WoE=%.3f (smoothed)\n",
    i,
    result_mba_sparse$count_pos[i],
    result_mba_sparse$count[i],
    result_mba_sparse$woe[i]
  ))
}
#>   Bin 1: events=4/81, WoE=-0.421 (smoothed)
#>   Bin 2: events=9/122, WoE=0.003 (smoothed)
#>   Bin 3: events=13/168, WoE=0.054 (smoothed)
#>   Bin 4: events=3/29, WoE=0.368 (smoothed)

# Example 4: High cardinality with pre-binning
set.seed(456)
n_obs_hc <- 3000

# Simulate ZIP codes (high cardinality)
zips <- paste0("ZIP_", sprintf("%04d", 1:50))

cat_feature_hc <- sample(zips, n_obs_hc, replace = TRUE)
bin_target_hc <- rbinom(n_obs_hc, 1, 0.12)

result_mba_hc <- ob_categorical_mba(
  cat_feature_hc,
  bin_target_hc,
  min_bins = 4,
  max_bins = 6,
  max_n_prebins = 20,
  bin_cutoff = 0.01
)

cat("\nHigh cardinality performance:\n")
#> 
#> High cardinality performance:
cat("  Original categories:", length(unique(cat_feature_hc)), "\n")
#>   Original categories: 50 
cat("  Final bins:", length(result_mba_hc$bin), "\n")
#>   Final bins: 6 
cat(
  "  Largest merged bin contains:",
  max(sapply(strsplit(result_mba_hc$bin, "%;%"), length)), "categories\n"
)
#>   Largest merged bin contains: 19 categories

# Verify monotonicity in high-cardinality case
woe_monotonic <- all(diff(result_mba_hc$woe) >= -1e-10)
cat("  WoE monotonic:", woe_monotonic, "\n")
#>   WoE monotonic: TRUE 

# Example 5: Convergence behavior
set.seed(321)
n_obs_conv <- 1000

business_sizes <- c("Micro", "Small", "Medium", "Large", "Enterprise")
cat_feature_conv <- sample(business_sizes, n_obs_conv, replace = TRUE)
bin_target_conv <- rbinom(n_obs_conv, 1, 0.18)

# Test different convergence thresholds
thresholds <- c(1e-3, 1e-6, 1e-9)

for (thresh in thresholds) {
  result_conv <- ob_categorical_mba(
    cat_feature_conv,
    bin_target_conv,
    min_bins = 2,
    max_bins = 4,
    convergence_threshold = thresh,
    max_iterations = 50
  )

  cat(sprintf("\nThreshold %.0e:\n", thresh))
  cat("  Final bins:", length(result_conv$bin), "\n")
  cat("  Total IV:", round(result_conv$total_iv, 4), "\n")
  cat("  Converged:", result_conv$converged, "\n")
  cat("  Iterations:", result_conv$iterations, "\n")

  # Check monotonicity preservation
  monotonic <- all(diff(result_conv$woe) >= -1e-10)
  cat("  Monotonic:", monotonic, "\n")
}
#> 
#> Threshold 1e-03:
#>   Final bins: 4 
#>   Total IV: 0.0404 
#>   Converged: TRUE 
#>   Iterations: 1 
#>   Monotonic: TRUE 
#> 
#> Threshold 1e-06:
#>   Final bins: 4 
#>   Total IV: 0.0404 
#>   Converged: FALSE 
#>   Iterations: 1 
#>   Monotonic: TRUE 
#> 
#> Threshold 1e-09:
#>   Final bins: 4 
#>   Total IV: 0.0404 
#>   Converged: FALSE 
#>   Iterations: 1 
#>   Monotonic: TRUE 

# Example 6: Missing value handling
set.seed(555)
cat_feature_na <- cat_feature
na_indices <- sample(n_obs, 75) # 5% missing
cat_feature_na[na_indices] <- NA

result_mba_na <- ob_categorical_mba(
  cat_feature_na,
  bin_target,
  min_bins = 3,
  max_bins = 5
)

# Locate NA bin
na_bin_idx <- grep("NA", result_mba_na$bin)
if (length(na_bin_idx) > 0) {
  cat("\nMissing value treatment:\n")
  cat("  NA bin:", result_mba_na$bin[na_bin_idx], "\n")
  cat("  NA count:", result_mba_na$count[na_bin_idx], "\n")
  cat(
    "  NA event rate:",
    round(result_mba_na$count_pos[na_bin_idx] /
      result_mba_na$count[na_bin_idx], 3), "\n"
  )
  cat("  NA WoE:", round(result_mba_na$woe[na_bin_idx], 3), "\n")
  cat(
    "  Monotonicity preserved:",
    all(diff(result_mba_na$woe) >= -1e-10), "\n"
  )
}
#> 
#> Missing value treatment:
#>   NA bin: NA%;%BB 
#>   NA count: 367 
#>   NA event rate: 0.202 
#>   NA WoE: 0.189 
#>   Monotonicity preserved: TRUE 
# }
```
