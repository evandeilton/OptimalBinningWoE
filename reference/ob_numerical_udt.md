# Optimal Binning for Numerical Variables using Entropy-Based Partitioning

Implements a supervised binning algorithm that uses Information Gain
(Entropy) to identify the most informative initial split points,
followed by a bottom-up merging process to satisfy constraints (minimum
frequency, monotonicity, max bins).

Although historically referred to as "Unsupervised Decision Trees" in
some contexts, this method is strictly \*\*supervised\*\* (uses target
variable) and operates \*\*bottom-up\*\* after an initial entropy-based
selection of cutpoints. It is particularly effective when the
relationship between feature and target is non-linear but highly
informative in specific regions.

## Usage

``` r
ob_numerical_udt(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  laplace_smoothing = 0.5,
  monotonicity_direction = "none",
  convergence_threshold = 1e-06,
  max_iterations = 1000
)
```

## Arguments

- feature:

  Numeric vector of feature values. Missing values (NA) are handled by
  placing them in a separate bin. Infinite values are treated as valid
  numeric extremes or placed in the missing bin if they represent
  errors.

- target:

  Integer vector of binary target values (must contain only 0 and 1).
  Must have the same length as `feature`.

- min_bins:

  Minimum number of bins (default: 3). Must be at least 2.

- max_bins:

  Maximum number of bins (default: 5). Must be greater than or equal to
  `min_bins`.

- bin_cutoff:

  Minimum fraction of total observations per bin (default: 0.05). Bins
  below this threshold are merged based on Event Rate similarity.

- max_n_prebins:

  Maximum number of pre-bins (default: 20). The algorithm will select
  the top `max_n_prebins` cutpoints with highest Information Gain.
  **Performance Note**: High values (\>50) may significantly slow down
  processing for large datasets due to the O(N^2) nature of cutpoint
  selection.

- laplace_smoothing:

  Laplace smoothing parameter (default: 0.5) for WoE calculation.

- monotonicity_direction:

  String specifying monotonicity constraint:

  - `"none"` (default): No monotonicity enforcement.

  - `"increasing"`: WoE must be non-decreasing.

  - `"decreasing"`: WoE must be non-increasing.

  - `"auto"`: Automatically determined by Pearson correlation.

- convergence_threshold:

  Convergence threshold for IV optimization (default: 1e-6).

- max_iterations:

  Maximum iterations for optimization loop (default: 1000).

## Value

A list containing:

- id:

  Integer bin identifiers (1-based).

- bin:

  Character bin intervals `"(lower;upper]"`.

- woe:

  Numeric WoE values.

- iv:

  Numeric IV contributions.

- event_rate:

  Numeric event rates.

- count:

  Integer total observations.

- count_pos:

  Integer positive class counts.

- count_neg:

  Integer negative class counts.

- cutpoints:

  Numeric bin boundaries.

- total_iv:

  Total Information Value.

- gini:

  Gini index (2\*AUC - 1) calculated on the binned data.

- ks:

  Kolmogorov-Smirnov statistic calculated on the binned data.

- converged:

  Logical convergence flag.

- iterations:

  Integer iteration count.

## Details

**Algorithm Overview**

The UDT algorithm executes in four phases:

**Phase 1: Entropy-Based Pre-binning**

The algorithm evaluates every possible cutpoint \\c\\ (midpoints between
sorted unique values) using Information Gain (IG): \$\$IG(S, c) = H(S) -
\left( \frac{\|S_L\|}{\|S\|} H(S_L) + \frac{\|S_R\|}{\|S\|} H(S_R)
\right)\$\$

The top `max_n_prebins` cutpoints with the highest IG are selected to
form the initial bins. This ensures that the starting bins capture the
most discriminative regions of the feature space.

**Phase 2: Rare Bin Merging**

Bins with frequency \\\< \text{bin\\cutoff}\\ are merged. The merge
partner is chosen to minimize the difference in Event Rates:
\$\$\text{merge\\idx} = \arg\min\_{j \in \\i-1, i+1\\} \|ER_i -
ER_j\|\$\$ This differs from IV-based methods and aims to preserve local
risk probability smoothness.

**Phase 3: Monotonicity Enforcement**

If requested, monotonicity is enforced by iteratively merging bins that
violate the specified direction (`"increasing"`, `"decreasing"`, or
`"auto"`). Auto-direction is determined by the sign of the Pearson
correlation between feature and target.

**Phase 4: Constraint Satisfaction**

If \\k \> \text{max\\bins}\\, bins are merged minimizing IV loss until
the constraint is met.

**Warning on Complexity**

The pre-binning phase evaluates Information Gain for *all* unique
values. For continuous features with many unique values (e.g., \\N \>
10,000\\), this step can be computationally intensive (\\O(N^2)\\).
Consider rounding or using
[`ob_numerical_sketch`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_sketch.md)
for very large datasets.

## References

- Quinlan, J. R. (1986). "Induction of Decision Trees". *Machine
  Learning*, 1(1), 81-106.

- Fayyad, U. M., & Irani, K. B. (1992). "On the Handling of
  Continuous-Valued Attributes in Decision Tree Generation". *Machine
  Learning*, 8, 87-102.

- Liu, H., et al. (2002). "Discretization: An Enabling Technique". *Data
  Mining and Knowledge Discovery*, 6(4), 393-423.

## See also

[`ob_numerical_mdlp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mdlp.md)
for a pure MDL-based approach,
[`ob_numerical_sketch`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_sketch.md)
for fast approximation on large data.

## Author

Lopes, J. E.
