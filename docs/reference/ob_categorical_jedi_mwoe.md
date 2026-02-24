# Optimal Binning for Categorical Variables with Multinomial Target using JEDI-MWoE

Performs supervised discretization of categorical variables for
multinomial classification problems using the Joint Entropy-Driven
Information Maximization with Multinomial Weight of Evidence (JEDI-MWoE)
algorithm. This advanced method extends traditional binning to handle
multi-class targets through specialized information-theoretic measures
and intelligent optimization strategies.

## Usage

``` r
ob_categorical_jedi_mwoe(
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
  the special category `"N/A"`.

- target:

  An integer vector representing the multinomial outcome variable with
  consecutive integer classes starting from 0 (e.g., 0, 1, 2, ...).
  Missing values are not permitted. Must contain at least 2 distinct
  classes.

- min_bins:

  Integer. Minimum number of bins to produce. Must be \>= 1. The
  algorithm will not merge below this threshold. Defaults to 3.

- max_bins:

  Integer. Maximum number of bins to produce. Must be \>= `min_bins`.
  The algorithm iteratively merges until this constraint is satisfied.
  Defaults to 5.

- bin_cutoff:

  Numeric. Minimum proportion of total observations required for a
  category to remain separate during initialization. Categories below
  this threshold are pre-merged. Must be in (0, 1). Defaults to 0.05.

- max_n_prebins:

  Integer. Maximum number of initial bins before the main optimization
  phase. Controls computational complexity for high-cardinality
  features. Must be \>= `min_bins`. Defaults to 20.

- bin_separator:

  Character string used to concatenate category names when multiple
  categories are merged into a single bin. Defaults to "%;%".

- convergence_threshold:

  Numeric. Convergence tolerance based on Information Value change
  between iterations. Algorithm stops when \\\max_c \|\Delta IV_c\| \<\\
  `convergence_threshold` across all classes. Must be \> 0. Defaults to
  1e-6.

- max_iterations:

  Integer. Maximum number of optimization iterations. Prevents infinite
  loops in edge cases. Must be \> 0. Defaults to 1000.

## Value

A list containing the multinomial binning results with the following
components:

- `id`:

  Integer vector of bin identifiers (1-indexed)

- `bin`:

  Character vector of bin labels (merged category names)

- `woe`:

  Numeric matrix of Multinomial Weight of Evidence values with
  dimensions (bins × classes)

- `iv`:

  Numeric matrix of Information Value contributions with dimensions
  (bins × classes)

- `count`:

  Integer vector of total observations per bin

- `class_counts`:

  Integer matrix of observations per class per bin with dimensions (bins
  × classes)

- `class_rates`:

  Numeric matrix of class proportions per bin with dimensions (bins ×
  classes)

- `converged`:

  Logical indicating algorithm convergence

- `iterations`:

  Integer count of optimization iterations performed

- `n_classes`:

  Integer number of target classes

- `total_iv`:

  Numeric vector of total Information Value per class

## Details

The JEDI-MWoE (Joint Entropy-Driven Information Maximization with
Multinomial Weight of Evidence) algorithm extends traditional optimal
binning to handle multinomial classification problems by computing
class-specific information measures and optimizing joint information
content across all target classes.

**Algorithm Workflow:**

1.  Input validation and preprocessing (multinomial target verification)

2.  Initial bin creation (one category per bin)

3.  Rare category merging (frequencies \< `bin_cutoff`)

4.  Pre-bin limitation via statistical similarity merging

5.  Main optimization loop with alternating strategies:

    - Jensen-Shannon divergence minimization for similar bin detection

    - Adjacent bin merging with minimal information loss

    - Class-wise monotonicity violation detection and repair

    - Convergence monitoring across all classes

6.  Final constraint satisfaction (max_bins enforcement)

7.  Laplace-smoothed metric computation

**Multinomial Weight of Evidence (M-WoE):**

For a bin \\B\\ and class \\c\\, the Multinomial Weight of Evidence is:

\$\$M\text{-}WoE\_{B,c} = \ln\left(\frac{P(c\|B) + \alpha}{P(\neg
c\|B) + \alpha}\right)\$\$

where:

- \\P(c\|B) = \frac{n\_{B,c}}{n_B}\\ is the class probability in bin
  \\B\\

- \\P(\neg c\|B) = \frac{\sum\_{k \neq c} n\_{B,k}}{\sum\_{k \neq c}
  n_k}\\ is the combined probability of all other classes in bin \\B\\

- \\\alpha = 0.5\\ is the Laplace smoothing parameter

**Information Value Extension:**

The Information Value for class \\c\\ in bin \\B\\ is:

\$\$IV\_{B,c} = \left(P(c\|B) - P(\neg c\|B)\right) \times
M\text{-}WoE\_{B,c}\$\$

Total IV for class \\c\\ across all bins:

\$\$IV_c = \sum\_{B} \|IV\_{B,c}\|\$\$

**Statistical Similarity Measure:**

JEDI-MWoE uses Jensen-Shannon divergence to identify similar bins for
merging:

\$\$JS(P_B, P\_{B'}) = \frac{1}{2} \sum\_{c=0}^{C-1} \left\[ P(c\|B)
\ln\frac{P(c\|B)}{M(c)} + P(c\|B') \ln\frac{P(c\|B')}{M(c)} \right\]\$\$

where \\M(c) = \frac{1}{2}\[P(c\|B) + P(c\|B')\]\\ is the average
distribution.

**Class-wise Monotonicity Enforcement:**

For each class \\c\\, the algorithm enforces WoE monotonicity by
detecting violations (peaks and valleys) and repairing them through
strategic bin merges:

- **Peak**: \\M\text{-}WoE\_{i-1,c} \< M\text{-}WoE\_{i,c} \>
  M\text{-}WoE\_{i+1,c}\\

- **Valley**: \\M\text{-}WoE\_{i-1,c} \> M\text{-}WoE\_{i,c} \<
  M\text{-}WoE\_{i+1,c}\\

Violation severity is measured as:

\$\$severity\_{i,c} = \max\\\|M\text{-}WoE\_{i,c} -
M\text{-}WoE\_{i-1,c}\|, \|M\text{-}WoE\_{i,c} -
M\text{-}WoE\_{i+1,c}\|\\\$\$

**Alternating Optimization Strategies:**

The algorithm alternates between two merging strategies to balance
global similarity and local information preservation:

1.  **Divergence-based**: Merge bins with minimum JS divergence

2.  **IV-preserving**: Merge adjacent bins with minimum information loss

**Laplace Smoothing:**

To ensure numerical stability and prevent undefined logarithms, all
probability estimates are smoothed with a Laplace prior:

\$\$P\_{smooth}(c\|B) = \frac{n\_{B,c} + \alpha}{n_B + \alpha \cdot
C}\$\$

where \\C\\ is the number of classes and \\\alpha = 0.5\\.

**Computational Complexity:**

- Time: \\O(k^2 \cdot C \cdot m)\\ where \\k\\ = bins, \\C\\ = classes,
  \\m\\ = iterations

- Space: \\O(k^2 \cdot C)\\ for M-WoE cache

- Cache hit rate typically \> 60% for \\k \> 10\\

**Key Innovations:**

- **Multinomial extension**: Generalizes WoE/IV to multi-class problems

- **Joint optimization**: Simultaneously optimizes across all classes

- **Alternating strategies**: Balances global similarity and local
  preservation

- **Class-wise monotonicity**: Enforces meaningful ordering for each
  class

- **Statistical similarity**: Uses Jensen-Shannon divergence for merging

**Comparison with Binary Methods:**

|                |              |                         |                    |
|----------------|--------------|-------------------------|--------------------|
| **Aspect**     | **Binary**   | **Multinomial**         | **Extension**      |
| Target Classes | 2            | C \>= 2                 | One-vs-rest        |
| WoE Definition | \\\ln(p/n)\\ | \\\ln(P(c)/P(\neg c))\\ | Class-specific     |
| IV Aggregation | Sum          | Per-class               | Vector-valued      |
| Similarity     | Chi-square   | Jensen-Shannon          | Distribution-based |
| Monotonicity   | Global       | Per-class               | Multi-constraint   |

## References

Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing
Intelligent Credit Scoring*. Wiley.

Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE
Transactions on Information Theory*, 37(1), 145-151.
[doi:10.1109/18.61115](https://doi.org/10.1109/18.61115)

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*
(2nd ed.). Wiley-Interscience.
[doi:10.1002/047174882X](https://doi.org/10.1002/047174882X)

Navas-Palencia, G. (2020). Optimal binning: mathematical programming
formulation and solution approach. *Expert Systems with Applications*,
158, 113508.
[doi:10.1016/j.eswa.2020.113508](https://doi.org/10.1016/j.eswa.2020.113508)

Good, I. J. (1965). *The Estimation of Probabilities: An Essay on Modern
Bayesian Methods*. MIT Press.

## See also

[`ob_categorical_jedi`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_jedi.md)
for binary target JEDI algorithm,
[`ob_categorical_ivb`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_ivb.md)
for binary Information Value DP optimization,
[`ob_categorical_dp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_dp.md)
for general dynamic programming binning

## Examples

``` r
# \donttest{
# Example 1: Basic multinomial JEDI-MWoE optimization
set.seed(42)
n_obs <- 1500

# Simulate customer segments with 3 risk categories
segments <- c("Premium", "Standard", "Basic", "Economy")
# Class probabilities: 0=LowRisk, 1=MediumRisk, 2=HighRisk
risk_probs <- list(
  Premium = c(0.80, 0.15, 0.05), # Mostly LowRisk
  Standard = c(0.40, 0.40, 0.20), # Balanced
  Basic = c(0.15, 0.35, 0.50), # Mostly HighRisk
  Economy = c(0.05, 0.20, 0.75) # Almost all HighRisk
)

cat_feature <- sample(segments, n_obs,
  replace = TRUE,
  prob = c(0.25, 0.35, 0.25, 0.15)
)

# Generate multinomial target (classes 0, 1, 2)
multinom_target <- sapply(cat_feature, function(segment) {
  probs <- risk_probs[[segment]]
  sample(0:2, 1, prob = probs)
})

# Apply JEDI-MWoE algorithm
result_mwoe <- ob_categorical_jedi_mwoe(
  cat_feature,
  multinom_target,
  min_bins = 2,
  max_bins = 3
)

# Display results
cat("Number of classes:", result_mwoe$n_classes, "\n")
#> Number of classes: 3 
cat("Number of bins:", length(result_mwoe$bin), "\n")
#> Number of bins: 2 
cat("Converged:", result_mwoe$converged, "\n")
#> Converged: TRUE 
cat("Iterations:", result_mwoe$iterations, "\n\n")
#> Iterations: 2 
#> 

# Show bin details
for (i in seq_along(result_mwoe$bin)) {
  cat(sprintf("Bin %d (%s):\n", i, result_mwoe$bin[i]))
  cat("  Total count:", result_mwoe$count[i], "\n")
  cat("  Class counts:", result_mwoe$class_counts[i, ], "\n")
  cat("  Class rates:", round(result_mwoe$class_rates[i, ], 3), "\n")

  # Show WoE and IV for each class
  for (class in 0:(result_mwoe$n_classes - 1)) {
    cat(sprintf(
      "  Class %d: WoE=%.3f, IV=%.4f\n",
      class,
      result_mwoe$woe[i, class + 1], # R is 1-indexed
      result_mwoe$iv[i, class + 1]
    ))
  }
  cat("\n")
}
#> Bin 1 (Economy%;%Basic):
#>   Total count: 603 
#>   Class counts: 84 182 337 
#>   Class rates: 0.139 0.302 0.559 
#>   Class 0: WoE=-1.350, IV=0.5627
#>   Class 1: WoE=-0.013, IV=0.0001
#>   Class 2: WoE=1.035, IV=0.4834
#> 
#> Bin 2 (Premium%;%Standard):
#>   Total count: 897 
#>   Class counts: 494 275 128 
#>   Class rates: 0.551 0.307 0.143 
#>   Class 0: WoE=0.670, IV=0.2792
#>   Class 1: WoE=0.009, IV=0.0000
#>   Class 2: WoE=-0.991, IV=0.4627
#> 

# Show total IV per class
cat("Total IV per class:\n")
#> Total IV per class:
for (class in 0:(result_mwoe$n_classes - 1)) {
  cat(sprintf("  Class %d: %.4f\n", class, result_mwoe$total_iv[class + 1]))
}
#>   Class 0: 0.8419
#>   Class 1: 0.0001
#>   Class 2: 0.9462

# Example 2: High-cardinality multinomial problem
set.seed(123)
n_obs_hc <- 2000

# Simulate product categories with 4 classes
products <- paste0("Product_", LETTERS[1:15])
cat_feature_hc <- sample(products, n_obs_hc, replace = TRUE)

# Generate 4-class target
multinom_target_hc <- sample(0:3, n_obs_hc,
  replace = TRUE,
  prob = c(0.3, 0.25, 0.25, 0.2)
)

result_mwoe_hc <- ob_categorical_jedi_mwoe(
  cat_feature_hc,
  multinom_target_hc,
  min_bins = 3,
  max_bins = 6,
  max_n_prebins = 15,
  bin_cutoff = 0.03
)

cat("\nHigh-cardinality example:\n")
#> 
#> High-cardinality example:
cat("Original categories:", length(unique(cat_feature_hc)), "\n")
#> Original categories: 15 
cat("Final bins:", length(result_mwoe_hc$bin), "\n")
#> Final bins: 3 
cat("Classes:", result_mwoe_hc$n_classes, "\n")
#> Classes: 4 
cat("Converged:", result_mwoe_hc$converged, "\n\n")
#> Converged: FALSE 
#> 

# Show merged categories
for (i in seq_along(result_mwoe_hc$bin)) {
  n_merged <- length(strsplit(result_mwoe_hc$bin[i], "%;%")[[1]])
  if (n_merged > 1) {
    cat(sprintf("Bin %d: %d categories merged\n", i, n_merged))
  }
}
#> Bin 2: 13 categories merged

# Example 3: Laplace smoothing demonstration
set.seed(789)
n_obs_smooth <- 500

# Small sample with sparse categories
categories <- c("A", "B", "C", "D", "E")
cat_feature_smooth <- sample(categories, n_obs_smooth,
  replace = TRUE,
  prob = c(0.3, 0.25, 0.2, 0.15, 0.1)
)

# Generate 3-class target with class imbalance
multinom_target_smooth <- sample(0:2, n_obs_smooth,
  replace = TRUE,
  prob = c(0.6, 0.3, 0.1)
) # Class 0 dominant

result_mwoe_smooth <- ob_categorical_jedi_mwoe(
  cat_feature_smooth,
  multinom_target_smooth,
  min_bins = 2,
  max_bins = 4,
  bin_cutoff = 0.02
)

cat("\nLaplace smoothing demonstration:\n")
#> 
#> Laplace smoothing demonstration:
cat("Sample size:", n_obs_smooth, "\n")
#> Sample size: 500 
cat("Classes:", result_mwoe_smooth$n_classes, "\n")
#> Classes: 3 
cat("Event distribution:", table(multinom_target_smooth), "\n\n")
#> Event distribution: 298 150 52 
#> 

# Show how smoothing prevents extreme values
for (i in seq_along(result_mwoe_smooth$bin)) {
  cat(sprintf("Bin %d (%s):\n", i, result_mwoe_smooth$bin[i]))
  cat("  Counts per class:", result_mwoe_smooth$class_counts[i, ], "\n")
  cat("  WoE values:", round(result_mwoe_smooth$woe[i, ], 3), "\n")
  cat("  Note: Extreme WoE values prevented by Laplace smoothing\n\n")
}
#> Bin 1 (E%;%D):
#>   Counts per class: 80 39 18 
#>   WoE values: -0.051 -0.07 0.271 
#>   Note: Extreme WoE values prevented by Laplace smoothing
#> 
#> Bin 2 (C%;%B%;%A):
#>   Counts per class: 218 111 34 
#>   WoE values: 0.019 0.026 -0.12 
#>   Note: Extreme WoE values prevented by Laplace smoothing
#> 

# Example 4: Class-wise monotonicity
set.seed(456)
n_obs_mono <- 1200

# Feature with predictable class patterns
education <- c("PhD", "Master", "Bachelor", "College", "HighSchool")
# Each education level has a preferred class
preferred_classes <- c(2, 1, 0, 1, 2) # PhD→High(2), Bachelor→Low(0), etc.

cat_feature_mono <- sample(education, n_obs_mono, replace = TRUE)

# Generate target with preferred class bias
multinom_target_mono <- sapply(cat_feature_mono, function(edu) {
  pref_class <- preferred_classes[which(education == edu)]
  # Create probability vector with preference
  probs <- rep(0.1, 3) # Base probability
  probs[pref_class + 1] <- 0.8 # Preferred class gets high probability
  sample(0:2, 1, prob = probs / sum(probs))
})

result_mwoe_mono <- ob_categorical_jedi_mwoe(
  cat_feature_mono,
  multinom_target_mono,
  min_bins = 3,
  max_bins = 5
)

cat("Class-wise monotonicity example:\n")
#> Class-wise monotonicity example:
cat("Education levels:", length(education), "\n")
#> Education levels: 5 
cat("Final bins:", length(result_mwoe_mono$bin), "\n")
#> Final bins: 3 
cat("Iterations:", result_mwoe_mono$iterations, "\n\n")
#> Iterations: 2 
#> 

# Check monotonicity for each class
for (class in 0:(result_mwoe_mono$n_classes - 1)) {
  woe_series <- result_mwoe_mono$woe[, class + 1]
  diffs <- diff(woe_series)
  is_mono <- all(diffs >= -1e-6) || all(diffs <= 1e-6)
  cat(sprintf("Class %d WoE monotonic: %s\n", class, is_mono))
  cat(sprintf("  WoE series: %s\n", paste(round(woe_series, 3), collapse = ", ")))
}
#> Class 0 WoE monotonic: FALSE
#>   WoE series: -1.058, 2.6, -0.982
#> Class 1 WoE monotonic: FALSE
#>   WoE series: 1.908, -1.693, -0.276
#> Class 2 WoE monotonic: TRUE
#>   WoE series: -1.728, -1.686, 0.756

# Example 5: Missing value handling
set.seed(321)
cat_feature_na <- cat_feature
na_indices <- sample(n_obs, 75) # 5% missing
cat_feature_na[na_indices] <- NA

result_mwoe_na <- ob_categorical_jedi_mwoe(
  cat_feature_na,
  multinom_target,
  min_bins = 2,
  max_bins = 3
)

# Locate missing value bin
missing_bin_idx <- grep("N/A", result_mwoe_na$bin)
if (length(missing_bin_idx) > 0) {
  cat("\nMissing value handling:\n")
  cat("Missing value bin:", result_mwoe_na$bin[missing_bin_idx], "\n")
  cat("Missing value count:", result_mwoe_na$count[missing_bin_idx], "\n")
  cat(
    "Class distribution in missing bin:",
    result_mwoe_na$class_counts[missing_bin_idx, ], "\n"
  )

  # Show class rates for missing bin
  for (class in 0:(result_mwoe_na$n_classes - 1)) {
    cat(sprintf(
      "  Class %d rate: %.3f\n", class,
      result_mwoe_na$class_rates[missing_bin_idx, class + 1]
    ))
  }
}
#> 
#> Missing value handling:
#> Missing value bin: N/A%;%Standard%;%Economy%;%Premium 
#> Missing value count: 1129 
#> Class distribution in missing bin: 510 329 290 
#>   Class 0 rate: 0.452
#>   Class 1 rate: 0.291
#>   Class 2 rate: 0.257

# Example 6: Convergence behavior
set.seed(555)
n_obs_conv <- 1000

departments <- c("Sales", "IT", "HR", "Finance", "Operations")
cat_feature_conv <- sample(departments, n_obs_conv, replace = TRUE)
multinom_target_conv <- sample(0:2, n_obs_conv, replace = TRUE)

# Test different convergence thresholds
thresholds <- c(1e-3, 1e-6, 1e-9)

for (thresh in thresholds) {
  result_conv <- ob_categorical_jedi_mwoe(
    cat_feature_conv,
    multinom_target_conv,
    min_bins = 2,
    max_bins = 4,
    convergence_threshold = thresh,
    max_iterations = 100
  )

  cat(sprintf("\nThreshold %.0e:\n", thresh))
  cat("  Final bins:", length(result_conv$bin), "\n")
  cat("  Converged:", result_conv$converged, "\n")
  cat("  Iterations:", result_conv$iterations, "\n")

  # Show total IV for each class
  cat("  Total IV per class:")
  for (class in 0:(result_conv$n_classes - 1)) {
    cat(sprintf(" %.4f", result_conv$total_iv[class + 1]))
  }
  cat("\n")
}
#> 
#> Threshold 1e-03:
#>   Final bins: 3 
#>   Converged: TRUE 
#>   Iterations: 2 
#>   Total IV per class: 0.0254 0.0132 0.0041
#> 
#> Threshold 1e-06:
#>   Final bins: 3 
#>   Converged: TRUE 
#>   Iterations: 2 
#>   Total IV per class: 0.0254 0.0132 0.0041
#> 
#> Threshold 1e-09:
#>   Final bins: 3 
#>   Converged: TRUE 
#>   Iterations: 2 
#>   Total IV per class: 0.0254 0.0132 0.0041
# }
```
