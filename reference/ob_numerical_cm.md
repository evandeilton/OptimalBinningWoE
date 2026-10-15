# Optimal Binning for Numerical Variables using Enhanced ChiMerge Algorithm

Performs supervised discretization of continuous numerical variables
using the ChiMerge algorithm (Kerber, 1992) or the Chi2 algorithm (Liu &
Setiono, 1995). This function merges adjacent bins based on Chi-square
statistics to maximize the discrimination of the binary target variable
while ensuring monotonicity and statistical robustness.

## Usage

``` r
ob_numerical_cm(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  convergence_threshold = 1e-06,
  max_iterations = 1000,
  init_method = "equal_frequency",
  chi_merge_threshold = 0.05,
  use_chi2_algorithm = FALSE
)
```

## Arguments

- feature:

  A numeric vector representing the continuous predictor variable.
  Missing values (`NA`/`NaN`) are excluded from the fit; `-Inf`/`Inf`
  fall in the first/last bin.

- target:

  An integer vector of binary outcomes (0/1) corresponding to each
  observation in `feature`. Must have the same length as `feature`.

- min_bins:

  Integer. The minimum number of bins to produce. Must be \\\ge\\ 2.
  Defaults to 3.

- max_bins:

  Integer. The maximum number of bins to produce. Must be \\\ge\\
  `min_bins`. Defaults to 5.

- bin_cutoff:

  Numeric. The minimum fraction of total observations required for a bin
  to be considered valid. Bins with frequency \< `bin_cutoff` will be
  merged. Value must be in (0, 1). Defaults to 0.05.

- max_n_prebins:

  Integer. The number of initial bins created during the pre-binning
  phase before the merging process begins. Higher values provide more
  granular starting points. Must be \\\ge\\ `max_bins`. Defaults to 20.

- convergence_threshold:

  Numeric. The threshold for the change in total IV to determine
  convergence during the iterative merging process. Defaults to 1e-6.

- max_iterations:

  Integer. Safety limit for the maximum number of merging iterations.
  Defaults to 1000.

- init_method:

  Character string specifying the initialization method. Options are
  `"equal_frequency"` (quantile-based) or `"equal_width"`. Defaults to
  `"equal_frequency"`.

- chi_merge_threshold:

  Numeric in (0, 1). The significance level (\\\alpha\\) of the
  Chi-square test; the critical value is
  `qchisq(1 - chi_merge_threshold, 1)`. **It does not change the result
  of this function**: the initial bins are merged, smallest chi-square
  first, until at most `max_bins` remain, before the significance test
  is ever reached, so the test never has a merge left to decide (see
  Details). Control the number of bins with `max_bins`. The categorical
  [`ob_categorical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_cm.md)
  does use the level as its stopping rule. Defaults to 0.05.

- use_chi2_algorithm:

  Logical. If `TRUE`, runs the Chi2 variant, a schedule of decreasing
  significance levels (0.5, 0.1, 0.05, 0.01, 0.005, 0.001). For the
  reason given for `chi_merge_threshold` the schedule merges nothing
  beyond `max_bins`; the bins match the default path except that the
  `bin_cutoff` rare-bin pass is skipped. Defaults to `FALSE`.

## Value

A list containing the binning results:

- `id`: Integer vector of bin identifiers (1 to k).

- `bin`: Character vector of bin labels in interval notation.

- `woe`: Numeric vector of Weight of Evidence for each bin.

- `iv`: Numeric vector of Information Value contribution per bin.

- `count`: Integer vector of total observations per bin.

- `count_pos`: Integer vector of positive cases (target=1).

- `count_neg`: Integer vector of negative cases (target=0).

- `cutpoints`: Numeric vector of upper boundaries (excluding Inf).

- `converged`: Logical indicating if the algorithm converged.

- `iterations`: Integer count of iterations performed.

- `total_iv`: The total Information Value of the binned variable.

- `algorithm`: String identifying the algorithm used ("ChiMerge" or
  "Chi2").

- `monotonic`: Logical indicating if the final WoE trend is monotonic.

## Details

The function implements two major discretization strategies:

1.  **Standard ChiMerge:**

    - Initializes bins using `init_method`.

    - Iteratively merges the adjacent pair with the lowest \\\chi^2\\
      statistic (2x2 table, Yates continuity correction clamped at
      zero).

    - Merging continues until the number of bins reaches `max_bins`. The
      significance test of Kerber (1992) is never reached: merging stops
      at `max_bins` first (see below).

2.  **Chi2 Algorithm:**

    - Activated when `use_chi2_algorithm = TRUE`.

    - Loops over a schedule of decreasing significance levels (0.5
      \\\to\\ 0.001) and checks the inconsistency rate of the data.

**The significance level does not decide the bin count.** Before either
method starts, the initial bins are merged, smallest \\\chi^2\\ first,
until at most `max_bins` remain. The significance tests never run,
because no merge is left to decide once `max_bins` is met, so the number
of bins is set by `max_bins` (and by `min_bins`, `bin_cutoff` and
monotonicity), not by `chi_merge_threshold`. The categorical
[`ob_categorical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_cm.md)
differs: it merges down to `max_bins` and then keeps merging while pairs
are not significantly different at the chosen level.

Post-processing steps enforce:

- **Minimum Bin Size:** Merging rare bins smaller than `bin_cutoff`
  (standard ChiMerge path only).

- **Monotonicity:** Ensuring WoE trend is strictly increasing or
  decreasing to improve model interpretability.

Monotonicity merging stops at `min_bins`; the rare-bin pass does not
check it again between merges, so a large `bin_cutoff` (for example 0.3)
can leave fewer than `min_bins` bins.

## References

Kerber, R. (1992). ChiMerge: Discretization of numeric attributes.
*Proceedings of the Tenth National Conference on Artificial
Intelligence*, 123-128.

Liu, H., & Setiono, R. (1995). Chi2: Feature selection and
discretization of numeric attributes. *Tools with Artificial
Intelligence*, 388-391.

## Examples

``` r
# Example 1: Standard ChiMerge
set.seed(123)
feature <- rnorm(1000)
# Create a target with a relationship to the feature
target <- rbinom(1000, 1, plogis(2 * feature))

res_cm <- ob_numerical_cm(feature, target,
  min_bins = 3,
  max_bins = 6,
  init_method = "equal_frequency"
)

print(res_cm$bin)
#> [1] "(-Inf;-1.284716]"      "(-1.284716;-0.489870]" "(-0.489870;-0.097412]"
#> [4] "(-0.097412;0.373241]"  "(0.373241;1.253815]"   "(1.253815;+Inf]"      
print(res_cm$iv)
#> [1] 0.599764630 0.451713110 0.059311155 0.005623405 0.458128572 0.731526553

# Example 2: Using the Chi2 Algorithm variant
res_chi2 <- ob_numerical_cm(feature, target,
  min_bins = 3,
  max_bins = 6,
  use_chi2_algorithm = TRUE
)

cat("Total IV (ChiMerge):", res_cm$total_iv, "\n")
#> Total IV (ChiMerge): 2.306067 
cat("Total IV (Chi2):", res_chi2$total_iv, "\n")
#> Total IV (Chi2): 2.306067 
```
