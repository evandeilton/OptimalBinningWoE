# Optimal Binning for Numerical Variables using Unsupervised Binning with Standard Deviation

Implements a **hybrid binning algorithm** that initializes bins using
**unsupervised statistical properties** (mean and standard deviation of
the feature) and refines them through **supervised optimization** using
Weight of Evidence (WoE) and Information Value (IV).

**Important Clarification**: Despite "Unsupervised" in the name, this
method is **predominantly supervised**. The unsupervised component is
limited to the initial bin creation step (~1% of the algorithm). All
subsequent refinement (merge, monotonicity enforcement, bin count
adjustment) uses the target variable extensively.

The statistical initialization via \\\mu \pm k\sigma\\ provides a
data-driven starting point that may be advantageous for approximately
normal distributions, but offers no guarantees for skewed or multimodal
data.

## Usage

``` r
ob_numerical_ubsd(
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

  Numeric vector of feature values. Missing values (NA) and infinite
  values are **not permitted** and will trigger an error.

- target:

  Integer or numeric vector of binary target values (must contain only 0
  and 1). Must have the same length as `feature`.

- min_bins:

  Minimum number of bins (default: 3). Must be at least 2.

- max_bins:

  Maximum number of bins (default: 5). Must be \\\ge\\ `min_bins`.

- bin_cutoff:

  Minimum fraction of total observations per bin (default: 0.05). Must
  be in (0, 1).

- max_n_prebins:

  Maximum number of pre-bins before optimization (default: 20). Must be
  at least equal to `min_bins`.

- convergence_threshold:

  Convergence threshold for IV change (default: 1e-6).

- max_iterations:

  Maximum iterations for optimization (default: 1000).

- laplace_smoothing:

  Laplace smoothing parameter (default: 0.5). Must be non-negative.

## Value

A list containing:

- id:

  Integer bin identifiers (1-based).

- bin:

  Character bin intervals `"[lower;upper)"`.

- woe:

  Numeric WoE values (monotonic after enforcement).

- iv:

  Numeric IV contributions per bin.

- count:

  Integer total observations per bin.

- count_pos:

  Integer positive class counts.

- count_neg:

  Integer negative class counts.

- event_rate:

  Numeric event rates per bin.

- cutpoints:

  Numeric bin boundaries (excluding \\\pm\infty\\).

- total_iv:

  Total Information Value.

- converged:

  Logical convergence flag.

- iterations:

  Integer iteration count.

## Details

**Algorithm Overview**

UBSD executes in six phases:

**Phase 1: Statistical Initialization (UNSUPERVISED)**

Initial bin edges are created by combining two approaches:

1.  **Standard deviation-based cutpoints**: \$\$\\\mu - 2\sigma, \mu -
    \sigma, \mu, \mu + \sigma, \mu + 2\sigma\\\$\$ where \\\mu\\ is the
    sample mean and \\\sigma\\ is the sample standard deviation (with
    Bessel correction: \\N-1\\ divisor).

2.  **Equal-width cutpoints**: \$\$\left\\x\_{\min} + i \times
    \frac{x\_{\max} -
    x\_{\min}}{\text{max\\n\\prebins}}\right\\\_{i=1}^{\text{max\\n\\prebins}-1}\$\$

The union of these two sets is taken, sorted, and limited to
`max_n_prebins` edges (plus \\-\infty\\ and \\+\infty\\ boundaries).

**Rationale**: For approximately normal distributions, \\\mu \pm
k\sigma\\ cutpoints align with natural quantiles:

- \\\mu - 2\sigma\\ to \\\mu + 2\sigma\\ captures ~95% of data
  (68-95-99.7 rule)

- Equal-width ensures coverage of entire range

**Limitation**: For skewed distributions (e.g., log-normal), \\\mu -
2\sigma\\ may fall outside the data range, creating empty bins.

**Special Case**: If \\\sigma \< \epsilon\\ (feature is nearly
constant), fallback to pure equal-width binning.

**Phase 2: Observation Assignment**

Each observation is assigned to a bin via linear search:
\$\$\text{bin}(x_i) = \min\\j : x_i \> \text{lower}\_j \land x_i \le
\text{upper}\_j\\\$\$

Counts are accumulated: `count`, `count_pos`, `count_neg`.

**Phase 3: Rare Bin Merging (SUPERVISED)**

Bins with \\\text{count} \< \text{bin\\cutoff} \times N\\ are merged
with adjacent bins. Merge direction is chosen to minimize IV loss:

\$\$\text{direction} = \arg\min\_{d \in \\\text{left}, \text{right}\\}
\left( \text{IV}\_i + \text{IV}\_{i+d} \right)\$\$

This is a **supervised** step (uses IV computed from target).

**Phase 4: WoE/IV Calculation (SUPERVISED)**

Weight of Evidence with Laplace smoothing: \$\$\text{WoE}\_i =
\ln\left(\frac{n_i^{+} + \alpha}{n^{+} + k\alpha} \bigg/ \frac{n_i^{-} +
\alpha}{n^{-} + k\alpha}\right)\$\$

Information Value: \$\$\text{IV}\_i = \left(\frac{n_i^{+} +
\alpha}{n^{+} + k\alpha} - \frac{n_i^{-} + \alpha}{n^{-} +
k\alpha}\right) \times \text{WoE}\_i\$\$

**Phase 5: Monotonicity Enforcement (SUPERVISED)**

Direction is auto-detected via majority vote: \$\$\text{increasing} =
\begin{cases} \text{TRUE} & \text{if } \sum_i
\mathbb{1}\_{\\\text{WoE}\_i \> \text{WoE}\_{i-1}\\} \ge \sum_i
\mathbb{1}\_{\\\text{WoE}\_i \< \text{WoE}\_{i-1}\\} \\ \text{FALSE} &
\text{otherwise} \end{cases}\$\$

Violations are resolved via PAVA (Pool Adjacent Violators Algorithm).

**Phase 6: Bin Count Adjustment (SUPERVISED)**

If \\k \> \text{max\\bins}\\, bins are merged to minimize IV loss:
\$\$\text{merge\\idx} = \arg\min\_{i=0}^{k-2} \left( \text{IV}\_i +
\text{IV}\_{i+1} \right)\$\$

**Convergence Criterion**: \$\$\|\text{IV}\_{\text{total}}^{(t)} -
\text{IV}\_{\text{total}}^{(t-1)}\| \< \text{convergence\\threshold}\$\$

**Comparison with Related Methods**

|            |                                   |                         |                      |
|------------|-----------------------------------|-------------------------|----------------------|
| **Method** | **Initialization**                | **Truly Unsupervised?** | **Best For**         |
| UBSD       | \\\mu \pm k\sigma\\ + equal-width | No (1 pct unsup)        | Normal distributions |
| MOB/MRBLP  | Equal-frequency                   | No (0 pct unsup)        | General use          |
| MDLP       | Equal-frequency                   | No (0 pct unsup)        | Information theory   |
| Sketch     | KLL Sketch quantiles              | No (0 pct unsup)        | Streaming data       |

**When to Use UBSD**

- **Use UBSD**: If you have prior knowledge that the feature is
  approximately normally distributed and want bins aligned with standard
  deviations (e.g., for interpretability: "2 standard deviations below
  mean").

- **Avoid UBSD**: For skewed distributions (use MDLP or MOB), for
  multimodal distributions (use LDB), or when you need provable
  optimality (use Sketch for quantile guarantees).

- **Alternative**: For true unsupervised binning (no target), use
  [`cut()`](https://rdrr.io/r/base/cut.html) with `breaks = "Sturges"`
  or `"FD"` (Freedman-Diaconis).

**Computational Complexity**

Identical to MOB/MRBLP: \\O(N + k^2 \times \text{max\\iterations})\\

## References

- Sturges, H. A. (1926). "The Choice of a Class Interval". *Journal of
  the American Statistical Association*, 21(153), 65-66.

- Scott, D. W. (1979). "On optimal and data-based histograms".
  *Biometrika*, 66(3), 605-610.

- Freedman, D., & Diaconis, P. (1981). "On the histogram as a density
  estimator: L2 theory". *Zeitschrift fuer Wahrscheinlichkeitstheorie*,
  57(4), 453-476.

- Thomas, L. C. (2009). *Consumer Credit Models: Pricing, Profit, and
  Portfolios*. Oxford University Press.

- Zeng, G. (2014). "A Necessary Condition for a Good Binning Algorithm
  in Credit Scoring". *Applied Mathematical Sciences*, 8(65), 3229-3242.

- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.

## See also

[`ob_numerical_mdlp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mdlp.md)
for information-theoretic binning,
[`ob_numerical_mob`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mob.md)
for pure supervised binning, [`cut`](https://rdrr.io/r/base/cut.html)
for true unsupervised binning.

## Author

Lopes, J. E.

## Examples

``` r
if (FALSE) { # \dontrun{
# Simulate normally distributed credit scores
set.seed(123)
n <- 5000

# Feature: Normally distributed FICO scores
feature <- rnorm(n, mean = 680, sd = 60)

# Target: Logistic relationship with score
prob_default <- 1 / (1 + exp((feature - 680) / 30))
target <- rbinom(n, 1, prob_default)

# Apply UBSD
result <- ob_numerical_ubsd(
  feature = feature,
  target = target,
  min_bins = 3,
  max_bins = 5
)

# Compare with MDLP (should be similar for normal data)
result_mdlp <- ob_numerical_mdlp(feature, target)

data.frame(
  Method = c("UBSD", "MDLP"),
  N_Bins = c(length(result$woe), length(result_mdlp$woe)),
  Total_IV = c(result$total_iv, result_mdlp$total_iv)
)
} # }
```
