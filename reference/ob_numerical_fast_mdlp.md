# Optimal Binning using MDLP with Monotonicity Constraints

Performs supervised discretization of continuous numerical variables
using the Minimum Description Length Principle (MDLP) algorithm,
enhanced with optional monotonicity constraints on the Weight of
Evidence (WoE). This method is particularly suitable for creating
interpretable bins for logistic regression models in domains like credit
scoring.

## Usage

``` r
ob_numerical_fast_mdlp(
  feature,
  target,
  min_bins = 2L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 100L,
  convergence_threshold = 1e-06,
  max_iterations = 1000L,
  force_monotonicity = TRUE
)
```

## Arguments

- feature:

  A numeric vector representing the continuous predictor variable.
  Missing values (NA) are excluded during the binning process.

- target:

  An integer vector of binary outcomes (0/1) corresponding to each
  observation in `feature`. Must have the same length as `feature`.

- min_bins:

  Integer. The minimum number of bins to produce. Must be \\\ge\\ 2.
  Defaults to 2.

- max_bins:

  Integer. The maximum number of bins to produce. Must be \\\ge\\
  `min_bins`. Defaults to 5.

- bin_cutoff:

  Numeric. Currently unused in this implementation (reserved for future
  versions). Defaults to 0.05.

- max_n_prebins:

  Integer. Currently unused in this implementation (reserved for future
  versions). Defaults to 100.

- convergence_threshold:

  Numeric. The threshold for determining convergence during the
  iterative monotonicity enforcement process. Defaults to 1e-6.

- max_iterations:

  Integer. Safety limit for the maximum number of iterations in the
  monotonicity enforcement phase. Defaults to 1000.

- force_monotonicity:

  Logical. If `TRUE`, the algorithm enforces a strict monotonic
  relationship (increasing or decreasing) between the bin indices and
  their Weight of Evidence (WoE) values. Defaults to `TRUE`.

## Value

A list containing the binning results:

- `id`: Integer vector of bin identifiers.

- `bin`: Character vector of bin labels in interval notation.

- `woe`: Numeric vector of Weight of Evidence for each bin.

- `iv`: Numeric vector of Information Value contribution per bin.

- `count`: Integer vector of total observations per bin.

- `count_pos`: Integer vector of positive cases.

- `count_neg`: Integer vector of negative cases.

- `cutpoints`: Numeric vector of upper boundaries (excluding Inf).

- `converged`: Logical indicating if the monotonicity enforcement
  converged.

- `iterations`: Integer count of iterations in monotonicity phase.

## Details

This function implements a sophisticated hybrid approach combining the
classic MDLP algorithm with modern monotonicity constraints.

**Algorithm Pipeline:**

1.  **Data Preparation:** Removes NA values and sorts the data by
    feature value.

2.  **MDLP Discretization (Fayyad & Irani, 1993):**

    - Recursively evaluates all possible binary splits of the sorted
      data.

    - For each potential split, calculates the Information Gain (IG).

    - Applies the MDLP stopping criterion: \$\$IG \> \frac{\log_2(N-1) +
      \Delta}{N}\$\$ where \\N\\ is the total number of samples and
      \\\Delta = \log_2(3^k - 2) - k \cdot E(S)\\ (for binary
      classification, \\k=2\\).

    - Only accepts splits that significantly reduce entropy beyond what
      would be expected by chance, balancing model fit with complexity.

3.  **Constraint Enforcement:**

    - **Min/Max Bins:** Adjusts the number of bins to meet
      `[min_bins, max_bins]` requirements through intelligent splitting
      or merging.

    - **Monotonicity (if enabled):** Iteratively merges adjacent bins
      with the most similar WoE values until a strictly increasing or
      decreasing trend is achieved across all bins.

**Technical Notes:**

- The algorithm uses Laplace smoothing (\\\alpha = 0.5\\) when
  calculating WoE to prevent \\\log(0)\\ errors for bins with pure class
  distributions.

- When all feature values are identical, the algorithm creates
  artificial bins.

- The monotonicity enforcement phase is iterative and uses the
  `convergence_threshold` to determine when changes in WoE become
  negligible.

## References

Fayyad, U. M., & Irani, K. B. (1993). Multi-interval discretization of
continuous-valued attributes for classification learning. *Proceedings
of the 13th International Joint Conference on Artificial Intelligence*,
1022-1029.

Kurgan, L. A., & Musilek, P. (2006). A survey of techniques. *IEEE
Transactions on Knowledge and Data Engineering*, 18(5), 673-689.

Garcia, S., Luengo, J., & Herrera, F. (2013). Data preprocessing in data
mining. *Springer Science & Business Media*.

## See also

[`ob_numerical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_cm.md)
for ChiMerge-based approaches,
[`ob_numerical_dp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_dp.md)
for dynamic programming methods.

## Examples

``` r
# Example: Standard usage with monotonicity
set.seed(123)
feature <- rnorm(1000)
target <- rbinom(1000, 1, plogis(2 * feature)) # Positive relationship

result <- ob_numerical_fast_mdlp(feature, target,
  min_bins = 3,
  max_bins = 6,
  force_monotonicity = TRUE
)

print(result$bin)
#> [1] "(-Inf;-1.185289]"      "(-1.185289;-0.506334]" "(-0.506334;0.299594]" 
#> [4] "(0.299594;1.214589]"   "(1.214589;+Inf]"      
print(result$woe) # Should show a monotonic trend
#> [1] -3.4659526 -1.6251167 -0.2908404  1.3817919  3.8181822

# Example: Disabling monotonicity for exploratory analysis
result_no_mono <- ob_numerical_fast_mdlp(feature, target,
  min_bins = 3,
  max_bins = 6,
  force_monotonicity = FALSE
)

print(result_no_mono$woe) # May show non-monotonic patterns
#> [1] -3.4659526 -1.6251167 -0.2908404  1.3817919  3.8181822
```
