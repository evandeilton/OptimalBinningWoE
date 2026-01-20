# Optimal Binning for Numerical Features using Monotonic Risk Binning with Likelihood Ratio Pre-binning

Implements a greedy binning algorithm with monotonicity enforcement and
**majority-vote direction detection**. *Important Note*: Despite the
"Likelihood Ratio Pre-binning" designation in the name, the current
implementation uses **equal-frequency pre-binning** without likelihood
ratio statistics. The algorithm is functionally a variant of Monotonic
Optimal Binning (MOB) with minor differences in merge strategies.

This method is suitable for credit scoring applications requiring
monotonic WoE patterns, but users should be aware that it does not
employ the statistical rigor implied by "Likelihood Ratio" in the name.

## Usage

``` r
ob_numerical_mrblp(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  convergence_threshold = 1e-06,
  max_iterations = 1000,
  laplace_smoothing = 0.5
)
```

## Arguments

- feature:

  Numeric vector of feature values to be binned. Missing values (NA) and
  infinite values are **not permitted** and will trigger an error
  (unlike other binning methods that issue warnings).

- target:

  Integer vector of binary target values (must contain only 0 and 1).
  Must have the same length as `feature`.

- min_bins:

  Minimum number of bins to generate (default: 3). Must be at least 1.
  Acts as a hard constraint during monotonicity enforcement.

- max_bins:

  Maximum number of bins to generate (default: 5). Must be greater than
  or equal to `min_bins`.

- bin_cutoff:

  Minimum fraction of total observations required in each bin (default:
  0.05). Bins with frequency below this threshold are merged. Must be in
  the range (0, 1).

- max_n_prebins:

  Maximum number of pre-bins before optimization (default: 20). Must be
  at least equal to `min_bins`.

- convergence_threshold:

  Convergence threshold (default: 1e-6). Currently used to check if WoE
  range is below threshold; primary stopping criterion is
  `max_iterations`.

- max_iterations:

  Maximum number of iterations for bin merging and monotonicity
  enforcement (default: 1000). Prevents infinite loops.

- laplace_smoothing:

  Laplace smoothing parameter for WoE calculation (default: 0.5). Must
  be non-negative.

## Value

A list containing:

- id:

  Integer vector of bin identifiers (1-based indexing).

- bin:

  Character vector of bin intervals in the format `"[lower;upper)"`.

- woe:

  Numeric vector of Weight of Evidence values. Guaranteed to be
  monotonic.

- iv:

  Numeric vector of Information Value contributions per bin.

- count:

  Integer vector of total observations per bin.

- count_pos:

  Integer vector of positive class counts per bin.

- count_neg:

  Integer vector of negative class counts per bin.

- event_rate:

  Numeric vector of event rates per bin.

- cutpoints:

  Numeric vector of bin boundaries (excluding -Inf and +Inf).

- total_iv:

  Total Information Value (sum of bin IVs).

- converged:

  Logical flag indicating convergence within `max_iterations`.

- iterations:

  Integer count of iterations performed.

## Details

**Algorithm Overview**

The MRBLP algorithm executes in five phases:

**Phase 1: Equal-Frequency Pre-binning**

Initial bins are created by dividing sorted data into approximately
equal-sized groups:

\$\$n\_{\text{bin}} = \max\left(1, \left\lfloor
\frac{N}{\text{max\\n\\prebins}} \right\rfloor\right)\$\$

**Note**: Despite "Likelihood Ratio Pre-binning" in the name, no
likelihood ratio statistics are computed. A true likelihood ratio
approach would compute:

\$\$\text{LR}(c) = \prod\_{x \le c} \frac{P(x\|y=1)}{P(x\|y=0)} \times
\prod\_{x \> c} \frac{P(x\|y=1)}{P(x\|y=0)}\$\$

and select cutpoints \\c\\ that maximize \\\|\log \text{LR}(c)\|\\. This
is **not implemented** in the current version.

**Phase 2: Rare Bin Merging**

Bins with total count below `bin_cutoff` \\\times N\\ are merged. The
merge direction (left or right) is chosen to minimize IV loss:

\$\$\text{direction} = \arg\min\_{d \in \\\text{left}, \text{right}\\}
\left( \text{IV}\_i + \text{IV}\_{i+d} - \text{IV}\_{\text{merged}}
\right)\$\$

**Phase 3: Initial WoE/IV Calculation**

Weight of Evidence for bin \\i\\:

\$\$\text{WoE}\_i = \ln\left(\frac{n_i^{+} + \alpha}{n^{+} + k\alpha}
\bigg/ \frac{n_i^{-} + \alpha}{n^{-} + k\alpha}\right)\$\$

where \\\alpha = \text{laplace\\smoothing}\\ and \\k\\ is the number of
bins.

**Phase 4: Monotonicity Enforcement**

The algorithm determines the desired monotonicity direction via
**majority vote**:

\$\$\text{increasing} = \begin{cases} \text{TRUE} & \text{if }
\\\\\text{WoE}\_i \> \text{WoE}\_{i-1}\\ \ge \\\\\text{WoE}\_i \<
\text{WoE}\_{i-1}\\ \\ \text{FALSE} & \text{otherwise} \end{cases}\$\$

This differs from:

- **MOB**: Uses first two bins only (`WoE[1] >= WoE[0]`)

- **MBLP**: Uses Pearson correlation between bin indices and WoE

Violations are detected as: \$\$\text{violation} = \begin{cases}
\text{WoE}\_i \< \text{WoE}\_{i-1} & \text{if increasing} \\
\text{WoE}\_i \> \text{WoE}\_{i-1} & \text{if decreasing}
\end{cases}\$\$

Violating bins are merged iteratively until monotonicity is achieved or
`min_bins` is reached.

**Phase 5: Bin Count Reduction**

If the number of bins exceeds `max_bins`, the algorithm merges bins with
the **smallest absolute IV difference**:

\$\$\text{merge\\idx} = \arg\min\_{i=0}^{k-2} \|\text{IV}\_i -
\text{IV}\_{i+1}\|\$\$

**Critique**: This criterion assumes bins with similar IVs are
redundant, which is not theoretically justified. A more rigorous
approach (used in MBLP) minimizes IV loss **after merge**:

\$\$\Delta \text{IV} = \text{IV}\_i + \text{IV}\_{i+1} -
\text{IV}\_{\text{merged}}\$\$

**Theoretical Foundations**

- **Monotonicity Enforcement**: Based on Zeng (2014), ensuring stability
  under data distribution shifts.

- **Likelihood Ratio (Theoretical)**: Neyman-Pearson lemma establishes
  likelihood ratio as the optimal test statistic for hypothesis testing.
  For binning, cutpoints maximizing LR would theoretically yield optimal
  class separation. **However, this is not implemented**.

- **Practical Equivalence**: The algorithm is functionally equivalent to
  MOB with minor differences in direction detection and merge
  strategies.

**Comparison with Related Methods**

|            |                 |                         |                     |
|------------|-----------------|-------------------------|---------------------|
| **Method** | **Pre-binning** | **Direction Detection** | **Merge Criterion** |
| MRBLP      | Equal-frequency | Majority vote           | Min IV difference   |
| MOB        | Equal-frequency | First two bins          | Min IV loss         |
| MBLP       | Quantile-based  | Pearson correlation     | Min IV loss         |
| MDLP       | Equal-frequency | N/A (optional)          | MDL cost            |

**Computational Complexity**

Identical to MOB: \\O(N \log N + k^2 \times \text{max\\iterations})\\

**When to Use MRBLP vs Alternatives**

- **Use MRBLP**: If you specifically need majority-vote direction
  detection and can tolerate the non-standard merge criterion.

- **Use MOB**: For simplicity and slightly faster direction detection.

- **Use MBLP**: For more robust direction detection via correlation.

- **Use MDLP**: For information-theoretic optimality without mandatory
  monotonicity.

## References

- Neyman, J., & Pearson, E. S. (1933). "On the Problem of the Most
  Efficient Tests of Statistical Hypotheses". *Philosophical
  Transactions of the Royal Society A*, 231(694-706), 289-337.
  \[Theoretical foundation for likelihood ratio, not implemented in
  code\]

- Mironchyk, P., & Tchistiakov, V. (2017). "Monotone optimal binning
  algorithm for credit risk modeling". *Frontiers in Applied Mathematics
  and Statistics*, 3, 2.

- Zeng, G. (2014). "A Necessary Condition for a Good Binning Algorithm
  in Credit Scoring". *Applied Mathematical Sciences*, 8(65), 3229-3242.

- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and
  Implementing Intelligent Credit Scoring*. Wiley.

- Anderson, R. (2007). *The Credit Scoring Toolkit: Theory and Practice
  for Retail Credit Risk Management and Decision Automation*. Oxford
  University Press.

- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied
  Logistic Regression* (3rd ed.). Wiley.

## See also

[`ob_numerical_mob`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mob.md)
for the base monotonic binning algorithm,
[`ob_numerical_mblp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mblp.md)
for correlation-based direction detection,
[`ob_numerical_mdlp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mdlp.md)
for information-theoretic binning.

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
# Simulate credit scoring data
set.seed(2024)
n <- 10000
feature <- c(
  rnorm(4000, mean = 620, sd = 50),
  rnorm(4000, mean = 690, sd = 45),
  rnorm(2000, mean = 740, sd = 35)
)
target <- c(
  rbinom(4000, 1, 0.20),
  rbinom(4000, 1, 0.10),
  rbinom(2000, 1, 0.04)
)

# Apply MRBLP
result <- ob_numerical_mrblp(
  feature = feature,
  target = target,
  min_bins = 3,
  max_bins = 5
)

# Compare with MOB (should be very similar)
result_mob <- ob_numerical_mob(
  feature = feature,
  target = target,
  min_bins = 3,
  max_bins = 5
)

# Compare results
data.frame(
  Method = c("MRBLP", "MOB"),
  N_Bins = c(length(result$woe), length(result_mob$woe)),
  Total_IV = c(result$total_iv, result_mob$total_iv),
  Iterations = c(result$iterations, result_mob$iterations)
)
#>   Method N_Bins  Total_IV Iterations
#> 1  MRBLP      5 0.1654134         16
#> 2    MOB      5 0.1744501         15
# }
```
