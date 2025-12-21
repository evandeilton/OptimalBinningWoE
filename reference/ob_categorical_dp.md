# Optimal Binning for Categorical Variables using Dynamic Programming

Performs supervised discretization of categorical variables using a
dynamic programming algorithm with optional monotonicity constraints.
This method maximizes the total Information Value (IV) while ensuring
optimal bin formation that respects user-defined constraints on bin
count and frequency. The algorithm guarantees global optimality through
dynamic programming.

## Usage

``` r
ob_categorical_dp(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  convergence_threshold = 1e-06,
  max_iterations = 1000,
  bin_separator = "%;%",
  monotonic_trend = "auto"
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
  Defines the upper bound of the search space for the optimal solution.
  Defaults to 5.

- bin_cutoff:

  Numeric. Minimum proportion of total observations required for a
  category to remain separate. Categories below this threshold are
  merged with similar categories. Must be in (0, 1). Defaults to 0.05.

- max_n_prebins:

  Integer. Maximum number of initial bins before dynamic programming
  optimization. Controls computational complexity. Must be \>= 2.
  Defaults to 20.

- convergence_threshold:

  Numeric. Convergence tolerance for the iterative dynamic programming
  updates. Smaller values require stricter convergence. Must be \> 0.
  Defaults to 1e-6.

- max_iterations:

  Integer. Maximum number of dynamic programming iterations. Prevents
  excessive computation in edge cases. Must be \> 0. Defaults to 1000.

- bin_separator:

  Character string used to concatenate category names when multiple
  categories are merged into a single bin. Defaults to "%;%".

- monotonic_trend:

  Character string specifying monotonicity constraint for Weight of
  Evidence. Must be one of:

  `"auto"`

  :   Automatically determine trend direction (default)

  `"ascending"`

  :   Enforce increasing WoE across bins

  `"descending"`

  :   Enforce decreasing WoE across bins

  `"none"`

  :   No monotonicity constraint

  Monotonicity constraints are enforced during the DP optimization
  phase. Defaults to `"auto"`.

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

- `event_rate`:

  Numeric vector of event rates per bin

- `total_iv`:

  Numeric total Information Value of the binning solution

- `converged`:

  Logical indicating if the DP algorithm converged

- `iterations`:

  Integer count of DP iterations performed

- `execution_time_ms`:

  Numeric execution time in milliseconds

## Details

This implementation uses dynamic programming to find the globally
optimal binning solution that maximizes total Information Value subject
to constraints.

**Algorithm Workflow:**

1.  Input validation and data preprocessing

2.  Rare category merging (frequencies below `bin_cutoff`)

3.  Pre-binning limitation (if categories exceed `max_n_prebins`)

4.  Category sorting by event rate

5.  Dynamic programming table initialization

6.  Iterative DP optimization with optional monotonicity constraints

7.  Backtracking to construct optimal bins

8.  Final metric computation

**Dynamic Programming Formulation:**

Let \\DP\[i\]\[k\]\\ represent the maximum total IV achievable using the
first \\i\\ categories partitioned into \\k\\ bins. The recurrence
relation is:

\$\$DP\[i\]\[k\] = \max\_{j\<i} \\DP\[j\]\[k-1\] + IV(j+1, i)\\\$\$

where \\IV(j+1, i)\\ is the Information Value of a bin containing
categories from \\j+1\\ to \\i\\. Monotonicity constraints are enforced
by restricting transitions that violate WoE ordering.

**Computational Complexity:**

- Time: \\O(n^2 \cdot k \cdot m)\\ where \\n\\ = categories, \\k\\ =
  max_bins, \\m\\ = iterations

- Space: \\O(n \cdot k)\\ for DP tables

**Advantages over Heuristic Methods:**

- Guarantees global optimality (within constraint space)

- Explicit monotonicity enforcement

- Deterministic and reproducible results

- Efficient caching mechanism for bin statistics

## References

Navas-Palencia, G. (2022). Optimal Binning: Mathematical Programming
Formulation. *arXiv preprint arXiv:2001.08025*.

Bellman, R. (1954). The theory of dynamic programming. *Bulletin of the
American Mathematical Society*, 60(6), 503-515.

Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and
Implementing Better Credit Risk Scorecards* (2nd ed.). Wiley.

Thomas, L. C., Edelman, D. B., & Crook, J. N. (2017). *Credit Scoring
and Its Applications* (2nd ed.). SIAM.

## See also

[`ob_categorical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_cm.md)
for ChiMerge-based binning,
[`ob_categorical_dmiv`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_dmiv.md)
for divergence measure-based binning

## Examples

``` r
# \donttest{
# Example 1: Basic usage with monotonic WoE enforcement
set.seed(123)
n_obs <- 1000

# Simulate education levels with increasing default risk
education <- c("High School", "Associate", "Bachelor", "Master", "PhD")
default_probs <- c(0.20, 0.15, 0.10, 0.06, 0.03)

cat_feature <- sample(education, n_obs,
  replace = TRUE,
  prob = c(0.30, 0.25, 0.25, 0.15, 0.05)
)
bin_target <- sapply(cat_feature, function(x) {
  rbinom(1, 1, default_probs[which(education == x)])
})

# Apply DP binning with ascending monotonicity
result_dp <- ob_categorical_dp(
  cat_feature,
  bin_target,
  min_bins = 2,
  max_bins = 4,
  monotonic_trend = "ascending"
)

# Display results
print(data.frame(
  Bin = result_dp$bin,
  WoE = round(result_dp$woe, 3),
  IV = round(result_dp$iv, 4),
  Count = result_dp$count,
  EventRate = round(result_dp$event_rate, 3)
))
#>            Bin    WoE     IV Count EventRate
#> 1 PhD%;%Master -1.080 0.1538   198     0.045
#> 2     Bachelor -0.372 0.0314   261     0.088
#> 3    Associate -0.167 0.0064   245     0.106
#> 4  High School  0.696 0.1846   296     0.220

cat("Total IV:", round(result_dp$total_iv, 4), "\n")
#> Total IV: 0.3761 
cat("Converged:", result_dp$converged, "\n")
#> Converged: TRUE 

# Example 2: Comparing monotonicity constraints
result_dp_asc <- ob_categorical_dp(
  cat_feature, bin_target,
  max_bins = 3,
  monotonic_trend = "ascending"
)

result_dp_none <- ob_categorical_dp(
  cat_feature, bin_target,
  max_bins = 3,
  monotonic_trend = "none"
)

cat("\nWith monotonicity:\n")
#> 
#> With monotonicity:
cat("  Bins:", length(result_dp_asc$bin), "\n")
#>   Bins: 3 
cat("  Total IV:", round(result_dp_asc$total_iv, 4), "\n")
#>   Total IV: 0.3713 

cat("\nWithout monotonicity:\n")
#> 
#> Without monotonicity:
cat("  Bins:", length(result_dp_none$bin), "\n")
#>   Bins: 3 
cat("  Total IV:", round(result_dp_none$total_iv, 4), "\n")
#>   Total IV: 0.3713 

# Example 3: High cardinality with pre-binning
set.seed(456)
n_obs_large <- 5000

# Simulate customer segments (high cardinality)
segments <- paste0("Segment_", LETTERS[1:20])
segment_probs <- runif(20, 0.01, 0.20)

cat_feature_hc <- sample(segments, n_obs_large, replace = TRUE)
bin_target_hc <- rbinom(
  n_obs_large, 1,
  segment_probs[match(cat_feature_hc, segments)]
)

result_dp_hc <- ob_categorical_dp(
  cat_feature_hc,
  bin_target_hc,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.03,
  max_n_prebins = 10
)

cat("\nHigh cardinality example:\n")
#> 
#> High cardinality example:
cat("  Original categories:", length(unique(cat_feature_hc)), "\n")
#>   Original categories: 20 
cat("  Final bins:", length(result_dp_hc$bin), "\n")
#>   Final bins: 5 
cat("  Execution time:", result_dp_hc$execution_time_ms, "ms\n")
#>   Execution time: 0 ms

# Example 4: Handling missing values
set.seed(789)
cat_feature_na <- cat_feature
cat_feature_na[sample(n_obs, 50)] <- NA # Introduce 5% missing

result_dp_na <- ob_categorical_dp(
  cat_feature_na,
  bin_target,
  min_bins = 2,
  max_bins = 4
)

# Check if NA was treated as a category
na_bin <- grep("NA", result_dp_na$bin, value = TRUE)
if (length(na_bin) > 0) {
  cat("\nNA handling:\n")
  cat("  Bin containing NA:", na_bin, "\n")
}
#> 
#> NA handling:
#>   Bin containing NA: PhD%;%Master%;%NA 
# }
```
