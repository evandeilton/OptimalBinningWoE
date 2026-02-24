# Optimal Binning for Categorical Variables using Information Value Dynamic Programming

Performs supervised discretization of categorical variables using a
dynamic programming algorithm specifically designed to maximize total
Information Value (IV). This implementation employs Bayesian smoothing
for numerical stability, maintains monotonic Weight of Evidence
constraints, and uses efficient caching strategies for optimal
performance with high-cardinality features.

## Usage

``` r
ob_categorical_ivb(
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
  algorithm searches for solutions within \[`min_bins`, `max_bins`\]
  that maximize total IV. Defaults to 3.

- max_bins:

  Integer. Maximum number of bins to produce. Must be \>= `min_bins`.
  Defines the upper bound of the search space. Defaults to 5.

- bin_cutoff:

  Numeric. Minimum proportion of total observations required for a
  category to remain separate. Categories below this threshold are
  pre-merged before the optimization phase. Must be in (0, 1). Defaults
  to 0.05.

- max_n_prebins:

  Integer. Maximum number of initial bins before dynamic programming
  optimization. Controls computational complexity for high-cardinality
  features. Must be \>= 2. Defaults to 20.

- bin_separator:

  Character string used to concatenate category names when multiple
  categories are merged into a single bin. Defaults to "%;%".

- convergence_threshold:

  Numeric. Convergence tolerance for the iterative optimization process
  based on IV change. Algorithm stops when \\\|\Delta IV\| \<\\
  `convergence_threshold`. Must be \> 0. Defaults to 1e-6.

- max_iterations:

  Integer. Maximum number of optimization iterations. Prevents excessive
  computation. Must be \> 0. Defaults to 1000.

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

The Information Value Binning (IVB) algorithm uses dynamic programming
to find the globally optimal binning solution that maximizes total IV
subject to constraints on bin count and monotonicity.

**Algorithm Workflow:**

1.  Input validation and preprocessing

2.  Single-pass category counting and statistics computation

3.  Rare category pre-merging (frequencies \< `bin_cutoff`)

4.  Pre-bin limitation (if categories \> `max_n_prebins`)

5.  Category sorting by event rate

6.  Cumulative statistics cache initialization

7.  Dynamic programming table computation:

    - State: \\DP\[i\]\[k\]\\ = max IV using first \\i\\ categories in
      \\k\\ bins

    - Transition: \\DP\[i\]\[k\] = \max_j \\DP\[j\]\[k-1\] + IV(j+1,
      i)\\\\

    - Banded optimization to skip infeasible splits

8.  Backtracking to reconstruct optimal bins

9.  Adaptive monotonicity enforcement

10. Final metric computation with Bayesian smoothing

**Dynamic Programming Formulation:**

Let \\DP\[i\]\[k\]\\ represent the maximum total IV achievable using the
first \\i\\ categories (sorted by event rate) partitioned into \\k\\
bins.

**Recurrence relation:** \$\$DP\[i\]\[k\] = \max\_{k-1 \leq j \< i}
\\DP\[j\]\[k-1\] + IV(j+1, i)\\\$\$

**Base case:** \$\$DP\[i\]\[1\] = IV(1, i) \quad \forall i\$\$

where \\IV(j+1, i)\\ is the Information Value of a bin containing
categories from \\j+1\\ to \\i\\.

**Bayesian Smoothing:**

To prevent numerical instability and overfitting with sparse bins, WoE
and IV are calculated using Bayesian smoothing with pseudocounts:

\$\$p'\_i = \frac{n\_{i,pos} + \alpha_p}{N\_{pos} + \alpha\_{total}}\$\$
\$\$n'\_i = \frac{n\_{i,neg} + \alpha_n}{N\_{neg} + \alpha\_{total}}\$\$

where \\\alpha_p\\ and \\\alpha_n\\ are prior pseudocounts proportional
to the overall event rate, and \\\alpha\_{total} = 1.0\\ (prior
strength).

\$\$WoE_i = \ln\left(\frac{p'\_i}{n'\_i}\right)\$\$ \$\$IV_i = (p'\_i -
n'\_i) \times WoE_i\$\$

**Adaptive Monotonicity Enforcement:**

After finding the optimal bins, the algorithm enforces WoE monotonicity
by:

1.  Computing average WoE gap: \\\bar{\Delta} =
    \frac{1}{k-1}\sum\_{i=1}^{k-1}\|WoE\_{i+1} - WoE_i\|\\

2.  Setting adaptive threshold: \\\tau = \min(\epsilon,
    0.01\bar{\Delta})\\

3.  Identifying worst violation: \\i^\* = \arg\max_i \\WoE_i -
    WoE\_{i+1}\\\\

4.  Evaluating forward and backward merges by IV retention

5.  Selecting merge direction that maximizes total IV

**Computational Complexity:**

- Time: \\O(k^2 \cdot n)\\ where \\k\\ = max_bins, \\n\\ = categories

- Space: \\O(k \cdot n)\\ for DP tables and cumulative caches

- IV calculations are \\O(1)\\ due to cumulative statistics caching

**Advantages over Alternative Methods:**

- **Global optimality**: Guaranteed maximum IV (within constraint space)

- **Bayesian regularization**: Robust to sparse bins and class imbalance

- **Efficient caching**: Cumulative stats and IV memoization

- **Banded optimization**: Reduced search space via feasibility pruning

- **Adaptive monotonicity**: Context-aware threshold for enforcement

**Comparison with Related Methods:**

- **vs DP (general)**: IVB specifically optimizes IV; general DP more
  flexible

- **vs GMB**: IVB guarantees optimality; GMB is faster but approximate

- **vs ChiMerge**: IVB uses IV criterion; ChiMerge uses chi-square

## References

Navas-Palencia, G. (2020). Optimal binning: mathematical programming
formulation and solution approach. *Expert Systems with Applications*,
158, 113508.
[doi:10.1016/j.eswa.2020.113508](https://doi.org/10.1016/j.eswa.2020.113508)

Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.

Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and
Implementing Better Credit Risk Scorecards* (2nd ed.). Wiley.

Good, I. J. (1965). *The Estimation of Probabilities: An Essay on Modern
Bayesian Methods*. MIT Press.

Anderson, R. (2007). *The Credit Scoring Toolkit: Theory and Practice
for Retail Credit Risk Management and Decision Automation*. Oxford
University Press.

## See also

[`ob_categorical_dp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_dp.md)
for general dynamic programming binning,
[`ob_categorical_gmb`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_gmb.md)
for greedy merge approximation,
[`ob_categorical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_cm.md)
for ChiMerge-based binning

## Examples

``` r
# \donttest{
# Example 1: Basic IV optimization with Bayesian smoothing
set.seed(42)
n_obs <- 1200

# Simulate industry sectors with varying default risk
industries <- c(
  "Technology", "Healthcare", "Finance", "Manufacturing",
  "Retail", "Energy"
)
default_rates <- c(0.03, 0.05, 0.08, 0.12, 0.18, 0.25)

cat_feature <- sample(industries, n_obs,
  replace = TRUE,
  prob = c(0.20, 0.18, 0.22, 0.18, 0.12, 0.10)
)
bin_target <- sapply(cat_feature, function(x) {
  rbinom(1, 1, default_rates[which(industries == x)])
})

# Apply IVB optimization
result_ivb <- ob_categorical_ivb(
  cat_feature,
  bin_target,
  min_bins = 3,
  max_bins = 4
)

# Display results
print(data.frame(
  Bin = result_ivb$bin,
  WoE = round(result_ivb$woe, 3),
  IV = round(result_ivb$iv, 4),
  Count = result_ivb$count,
  EventRate = round(result_ivb$count_pos / result_ivb$count, 3)
))
#>                    Bin    WoE     IV Count EventRate
#> 1           Technology -1.723 0.3003   231     0.022
#> 2 Finance%;%Healthcare -0.542 0.0965   487     0.068
#> 3        Manufacturing  0.397 0.0342   223     0.157
#> 4      Retail%;%Energy  0.879 0.2311   259     0.232

cat("\nTotal IV (maximized):", round(result_ivb$total_iv, 4), "\n")
#> 
#> Total IV (maximized): 0.6621 
cat("Converged:", result_ivb$converged, "\n")
#> Converged: TRUE 
cat("Iterations:", result_ivb$iterations, "\n")
#> Iterations: 1 

# Example 2: Comparing IV optimization with other methods
set.seed(123)
n_obs_comp <- 1500

regions <- c("North", "South", "East", "West", "Central")
cat_feature_comp <- sample(regions, n_obs_comp, replace = TRUE)
bin_target_comp <- rbinom(n_obs_comp, 1, 0.15)

# IVB (IV-optimized)
result_ivb_comp <- ob_categorical_ivb(
  cat_feature_comp, bin_target_comp,
  min_bins = 2, max_bins = 3
)

# GMB (greedy approximation)
result_gmb_comp <- ob_categorical_gmb(
  cat_feature_comp, bin_target_comp,
  min_bins = 2, max_bins = 3
)

# DP (general optimization)
result_dp_comp <- ob_categorical_dp(
  cat_feature_comp, bin_target_comp,
  min_bins = 2, max_bins = 3
)

cat("\nMethod comparison:\n")
#> 
#> Method comparison:
cat("  IVB total IV:", round(result_ivb_comp$total_iv, 4), "\n")
#>   IVB total IV: 0.053 
cat("  GMB total IV:", round(result_gmb_comp$total_iv, 4), "\n")
#>   GMB total IV: 0.053 
cat("  DP total IV:", round(result_dp_comp$total_iv, 4), "\n")
#>   DP total IV: 0.0531 
cat("\nIVB typically achieves highest IV due to explicit optimization\n")
#> 
#> IVB typically achieves highest IV due to explicit optimization
# }
```
