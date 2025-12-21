# Optimal Binning for Multiclass Targets using JEDI M-WOE

Performs supervised discretization of continuous numerical variables for
**multiclass** target variables (e.g., 0, 1, 2). It extends the Joint
Entropy-Driven Interval (JEDI) discretization framework to calculate and
optimize the Multinomial Weight of Evidence (M-WOE) for each class
simultaneously.

## Usage

``` r
ob_numerical_jedi_mwoe(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  convergence_threshold = 1e-06,
  max_iterations = 1000
)
```

## Arguments

- feature:

  A numeric vector representing the continuous predictor variable.
  Missing values (NA) should be excluded prior to execution.

- target:

  An integer vector of multiclass outcomes (0, 1, ..., K-1)
  corresponding to each observation in `feature`. Must have at least 2
  distinct classes.

- min_bins:

  Integer. The minimum number of bins to produce. Must be \\\ge\\ 2.
  Defaults to 3.

- max_bins:

  Integer. The maximum number of bins to produce. Must be \\\ge\\
  `min_bins`. Defaults to 5.

- bin_cutoff:

  Numeric. The minimum fraction of total observations required for a bin
  to be considered valid. Bins smaller than this threshold are merged.
  Defaults to 0.05.

- max_n_prebins:

  Integer. The number of initial quantiles to generate during the
  pre-binning phase. Defaults to 20.

- convergence_threshold:

  Numeric. The threshold for the change in total Multinomial IV to
  determine convergence. Defaults to 1e-6.

- max_iterations:

  Integer. Safety limit for the maximum number of iterations. Defaults
  to 1000.

## Value

A list containing the binning results:

- `id`: Integer vector of bin identifiers.

- `bin`: Character vector of bin labels in interval notation.

- `woe`: A numeric matrix where each column represents the WoE for a
  specific class (One-vs-Rest).

- `iv`: A numeric matrix where each column represents the IV
  contribution for a specific class.

- `count`: Integer vector of total observations per bin.

- `class_counts`: A matrix of observation counts per class per bin.

- `cutpoints`: Numeric vector of upper boundaries (excluding Inf).

- `n_classes`: The number of distinct target classes found.

## Details

**Multinomial Weight of Evidence (M-WOE):** For a target with \\K\\
classes, the WoE for class \\k\\ in bin \\i\\ is defined using a
"One-vs-Rest" approach: \$\$WOE\_{i,k} = \ln\left(\frac{P(X \in bin_i \|
Y=k)}{P(X \in bin_i \| Y \neq k)}\right)\$\$

**Algorithm Workflow:**

1.  **Multiclass Initialization:** The algorithm starts with
    quantile-based bins and computes the initial event rates for all
    \\K\\ classes.

2.  **Joint Monotonicity:** The algorithm attempts to enforce
    monotonicity for *all* classes. If bin \\i\\ violates the trend for
    Class 1 OR Class 2, it may be merged. This ensures the variable is
    predictive across the entire spectrum of outcomes.

3.  **Global IV Optimization:** When reducing the number of bins to
    `max_bins`, the algorithm merges the pair of bins that minimizes the
    loss of the *Sum of IVs* across all classes: \$\$Loss =
    \sum\_{k=0}^{K-1} \Delta IV_k\$\$

This method is ideal for use cases like:

- predicting loan status (Current, Late, Default)

- customer churn levels (Active, Dormant, Churned)

- ordinal survey responses.

## See also

[`ob_numerical_jedi`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_jedi.md)
for the binary version.

## Examples

``` r
# Example: Multiclass target (0, 1, 2)
set.seed(123)
feature <- rnorm(1000)
# Class 0: low feature, Class 1: medium, Class 2: high
target <- cut(feature + rnorm(1000, 0, 0.5),
  breaks = c(-Inf, -0.5, 0.5, Inf),
  labels = FALSE
) - 1

result <- ob_numerical_jedi_mwoe(feature, target,
  min_bins = 3,
  max_bins = 5
)

# Check WoE for Class 2 (High values)
print(result$woe[, 3]) # Column 3 corresponds to Class 2
#> [1] -20.4501899 -20.4501899 -20.4501899 -20.4501899   0.3630006
```
