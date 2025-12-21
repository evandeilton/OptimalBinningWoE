# Optimal Binning for Categorical Variables using JEDI Algorithm

Performs supervised discretization of categorical variables using the
Joint Entropy-Driven Information Maximization (JEDI) algorithm. This
advanced method combines information-theoretic optimization with
intelligent bin merging strategies, employing Bayesian smoothing for
numerical stability and adaptive monotonicity enforcement to produce
robust, interpretable binning solutions.

## Usage

``` r
ob_categorical_jedi(
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
  The algorithm iteratively merges until this constraint is satisfied.
  Defaults to 5.

- bin_cutoff:

  Numeric. Minimum proportion of total observations required for a
  category to remain separate during initialization. Categories below
  this threshold are pre-merged into an "Others" bin. Must be in (0, 1).
  Defaults to 0.05.

- max_n_prebins:

  Integer. Maximum number of initial bins before the main optimization
  phase. Controls computational complexity for high-cardinality
  features. Must be \>= `min_bins`. Defaults to 20.

- bin_separator:

  Character string used to concatenate category names when multiple
  categories are merged into a single bin. Defaults to "%;%".

- convergence_threshold:

  Numeric. Convergence tolerance based on Information Value change
  between iterations. Algorithm stops when \\\|\Delta IV\| \<\\
  `convergence_threshold`. Must be \> 0. Defaults to 1e-6.

- max_iterations:

  Integer. Maximum number of optimization iterations. Prevents infinite
  loops in edge cases. Must be \> 0. Defaults to 1000.

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

The JEDI (Joint Entropy-Driven Information Maximization) algorithm
represents a sophisticated approach to categorical binning that jointly
optimizes Information Value while maintaining monotonic Weight of
Evidence constraints through intelligent violation detection and repair
strategies.

**Algorithm Workflow:**

1.  Input validation and preprocessing

2.  Initial bin creation (one category per bin)

3.  Rare category merging (frequencies \< `bin_cutoff`)

4.  WoE-based monotonic sorting

5.  Pre-bin limitation via minimal IV-loss merging

6.  Main optimization loop:

    - Monotonicity violation detection (peaks and valleys)

    - Violation severity quantification

    - Intelligent merge selection (minimize IV loss)

    - Convergence monitoring

    - Best solution tracking

7.  Final constraint satisfaction (max_bins enforcement)

8.  Bayesian-smoothed metric computation

**Joint Entropy-Driven Optimization:**

Unlike greedy algorithms that optimize locally, JEDI considers the
global impact of each merge on total Information Value:

\$\$IV\_{total} = \sum\_{i=1}^{k} (p_i - n_i) \times
\ln\left(\frac{p_i}{n_i}\right)\$\$

For each potential merge of bins \\j\\ and \\j+1\\, JEDI evaluates:

\$\$\Delta IV\_{j,j+1} = IV\_{current} - IV\_{merged}(j,j+1)\$\$

The pair with minimum \\\Delta IV\\ (least information loss) is
selected.

**Violation Detection and Repair:**

JEDI identifies two types of monotonicity violations:

- **Peaks**: \\WoE_i \> WoE\_{i-1}\\ and \\WoE_i \> WoE\_{i+1}\\

- **Valleys**: \\WoE_i \< WoE\_{i-1}\\ and \\WoE_i \< WoE\_{i+1}\\

For each violation, severity is quantified as:

\$\$severity_i = \max\\\|WoE_i - WoE\_{i-1}\|, \|WoE_i -
WoE\_{i+1}\|\\\$\$

The algorithm prioritizes fixing the most severe violation first,
evaluating both forward merge \\(i, i+1)\\ and backward merge \\(i-1,
i)\\ to select the option that minimizes information loss.

**Bayesian Smoothing:**

To ensure numerical stability with sparse bins, JEDI applies Bayesian
smoothing:

\$\$p'\_i = \frac{n\_{i,pos} + \alpha_p}{N\_{pos} + \alpha\_{total}}\$\$
\$\$n'\_i = \frac{n\_{i,neg} + \alpha_n}{N\_{neg} + \alpha\_{total}}\$\$

where prior pseudocounts are proportional to overall prevalence:

\$\$\alpha_p = \alpha\_{total} \times \frac{N\_{pos}}{N\_{pos} +
N\_{neg}}\$\$ \$\$\alpha_n = \alpha\_{total} - \alpha_p\$\$

with \\\alpha\_{total} = 1.0\\ as the prior strength parameter.

**Adaptive Monotonicity Threshold:**

Rather than using a fixed threshold, JEDI computes a context-aware
tolerance:

\$\$\bar{\Delta} = \frac{1}{k-1}\sum\_{i=1}^{k-1}\|WoE\_{i+1} -
WoE_i\|\$\$ \$\$\tau = \min(\epsilon, 0.01\bar{\Delta})\$\$

This adaptive approach prevents over-merging when natural WoE gaps are
small.

**Computational Complexity:**

- Time: \\O(k^2 \cdot m)\\ where \\k\\ = bins, \\m\\ = iterations

- Space: \\O(k^2)\\ for IV cache

- Cache hit rate typically \> 70% for \\k \> 10\\

**Key Innovations:**

- **Joint optimization**: Global IV consideration (vs. local greedy)

- **Smart violation repair**: Severity-based prioritization

- **Bidirectional merge evaluation**: Forward vs. backward analysis

- **Best solution tracking**: Retains optimal intermediate states

- **Adaptive thresholds**: Context-aware monotonicity tolerance

**Comparison with Related Methods:**

|            |                  |                  |           |
|------------|------------------|------------------|-----------|
| **Method** | **Optimization** | **Monotonicity** | **Speed** |
| JEDI       | Joint/Global     | Adaptive         | Medium    |
| IVB        | DP (Exact)       | Enforced         | Slow      |
| GMB        | Greedy/Local     | Enforced         | Fast      |
| ChiMerge   | Statistical      | Optional         | Fast      |

## References

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*
(2nd ed.). Wiley-Interscience.
[doi:10.1002/047174882X](https://doi.org/10.1002/047174882X)

Kullback, S. (1959). *Information Theory and Statistics*. Wiley.

Navas-Palencia, G. (2020). Optimal binning: mathematical programming
formulation and solution approach. *Expert Systems with Applications*,
158, 113508.
[doi:10.1016/j.eswa.2020.113508](https://doi.org/10.1016/j.eswa.2020.113508)

Good, I. J. (1965). *The Estimation of Probabilities: An Essay on Modern
Bayesian Methods*. MIT Press.

Zeng, G. (2014). A necessary condition for a good binning algorithm in
credit scoring. *Applied Mathematical Sciences*, 8(65), 3229-3242.

## See also

[`ob_categorical_ivb`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_ivb.md)
for Information Value DP optimization,
[`ob_categorical_gmb`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_gmb.md)
for greedy merge binning,
[`ob_categorical_dp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_dp.md)
for general dynamic programming,
[`ob_categorical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_cm.md)
for ChiMerge-based binning

## Examples

``` r
# \donttest{
# Example 1: Basic JEDI optimization
set.seed(42)
n_obs <- 1500

# Simulate employment types with risk gradient
employment <- c(
  "Permanent", "Contract", "Temporary", "SelfEmployed",
  "Unemployed", "Student", "Retired"
)
risk_rates <- c(0.03, 0.08, 0.15, 0.12, 0.35, 0.25, 0.10)

cat_feature <- sample(employment, n_obs,
  replace = TRUE,
  prob = c(0.35, 0.20, 0.15, 0.12, 0.08, 0.06, 0.04)
)
bin_target <- sapply(cat_feature, function(x) {
  rbinom(1, 1, risk_rates[which(employment == x)])
})

# Apply JEDI algorithm
result_jedi <- ob_categorical_jedi(
  cat_feature,
  bin_target,
  min_bins = 3,
  max_bins = 5
)

# Display results
print(data.frame(
  Bin = result_jedi$bin,
  WoE = round(result_jedi$woe, 3),
  IV = round(result_jedi$iv, 4),
  Count = result_jedi$count,
  EventRate = round(result_jedi$count_pos / result_jedi$count, 3)
))
#>                                 Bin    WoE     IV Count EventRate
#> 1                         Permanent -1.841 0.6226   544     0.020
#> 2 Retired%;%SelfEmployed%;%Contract -0.026 0.0002   540     0.113
#> 3                         Temporary  0.376 0.0218   200     0.160
#> 4                           Student  1.050 0.1183   110     0.273
#> 5                        Unemployed  1.489 0.2595   106     0.368

cat("\nTotal IV (jointly optimized):", round(result_jedi$total_iv, 4), "\n")
#> 
#> Total IV (jointly optimized): 1.0224 
cat("Converged:", result_jedi$converged, "\n")
#> Converged: TRUE 
cat("Iterations:", result_jedi$iterations, "\n")
#> Iterations: 0 

# Example 2: Method comparison (JEDI vs alternatives)
set.seed(123)
n_obs_comp <- 2000

departments <- c(
  "Sales", "IT", "HR", "Finance", "Operations",
  "Marketing", "Legal", "R&D"
)
cat_feature_comp <- sample(departments, n_obs_comp, replace = TRUE)
bin_target_comp <- rbinom(n_obs_comp, 1, 0.12)

# JEDI (joint optimization)
result_jedi_comp <- ob_categorical_jedi(
  cat_feature_comp, bin_target_comp,
  min_bins = 3, max_bins = 4
)

# IVB (exact DP)
result_ivb_comp <- ob_categorical_ivb(
  cat_feature_comp, bin_target_comp,
  min_bins = 3, max_bins = 4
)

# GMB (greedy)
result_gmb_comp <- ob_categorical_gmb(
  cat_feature_comp, bin_target_comp,
  min_bins = 3, max_bins = 4
)

cat("\nMethod comparison (Total IV):\n")
#> 
#> Method comparison (Total IV):
cat(
  "  JEDI:", round(result_jedi_comp$total_iv, 4),
  "- converged:", result_jedi_comp$converged, "\n"
)
#>   JEDI: 0.0427 - converged: TRUE 
cat(
  "  IVB:", round(result_ivb_comp$total_iv, 4),
  "- converged:", result_ivb_comp$converged, "\n"
)
#>   IVB: 0.0427 - converged: TRUE 
cat(
  "  GMB:", round(result_gmb_comp$total_iv, 4),
  "- converged:", result_gmb_comp$converged, "\n"
)
#>   GMB: 0.0436 - converged: TRUE 

# Example 3: Bayesian smoothing with sparse data
set.seed(789)
n_obs_sparse <- 400

# Small sample with rare events
categories <- c("A", "B", "C", "D", "E", "F", "G")
cat_probs <- c(0.25, 0.20, 0.18, 0.15, 0.12, 0.07, 0.03)

cat_feature_sparse <- sample(categories, n_obs_sparse,
  replace = TRUE,
  prob = cat_probs
)
bin_target_sparse <- rbinom(n_obs_sparse, 1, 0.05) # 5% event rate

result_jedi_sparse <- ob_categorical_jedi(
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
cat("  Total events:", sum(bin_target_sparse), "\n")
#>   Total events: 19 
cat("  Event rate:", round(mean(bin_target_sparse), 4), "\n")
#>   Event rate: 0.0475 
cat("  Bins created:", length(result_jedi_sparse$bin), "\n\n")
#>   Bins created: 4 
#> 

# Show how smoothing prevents extreme WoE values
for (i in seq_along(result_jedi_sparse$bin)) {
  cat(sprintf(
    "  Bin %d: events=%d/%d, WoE=%.3f (smoothed)\n",
    i,
    result_jedi_sparse$count_pos[i],
    result_jedi_sparse$count[i],
    result_jedi_sparse$woe[i]
  ))
}
#>   Bin 1: events=1/78, WoE=-1.353 (smoothed)
#>   Bin 2: events=3/111, WoE=-0.606 (smoothed)
#>   Bin 3: events=8/151, WoE=0.090 (smoothed)
#>   Bin 4: events=7/60, WoE=0.944 (smoothed)

# Example 4: Violation detection and repair
set.seed(456)
n_obs_viol <- 1200

# Create feature with non-monotonic risk pattern
risk_categories <- c(
  "VeryLow", "Low", "MediumHigh", "Medium", # Intentional non-monotonic
  "High", "VeryHigh"
)
actual_risks <- c(0.02, 0.05, 0.20, 0.12, 0.25, 0.40) # MediumHigh > Medium

cat_feature_viol <- sample(risk_categories, n_obs_viol, replace = TRUE)
bin_target_viol <- sapply(cat_feature_viol, function(x) {
  rbinom(1, 1, actual_risks[which(risk_categories == x)])
})

result_jedi_viol <- ob_categorical_jedi(
  cat_feature_viol,
  bin_target_viol,
  min_bins = 3,
  max_bins = 5,
  max_iterations = 50
)

cat("\nViolation detection and repair:\n")
#> 
#> Violation detection and repair:
cat("  Original categories:", length(unique(cat_feature_viol)), "\n")
#>   Original categories: 6 
cat("  Final bins:", length(result_jedi_viol$bin), "\n")
#>   Final bins: 5 
cat("  Iterations to convergence:", result_jedi_viol$iterations, "\n")
#>   Iterations to convergence: 0 
cat("  Monotonicity achieved:", result_jedi_viol$converged, "\n\n")
#>   Monotonicity achieved: TRUE 
#> 

# Check final WoE monotonicity
woe_diffs <- diff(result_jedi_viol$woe)
cat(
  "  WoE differences between bins:",
  paste(round(woe_diffs, 3), collapse = ", "), "\n"
)
#>   WoE differences between bins: 0.721, 0.344, 0.978, 0.964 
cat("  All positive (monotonic):", all(woe_diffs >= -1e-6), "\n")
#>   All positive (monotonic): TRUE 

# Example 5: High cardinality performance
set.seed(321)
n_obs_hc <- 3000

# Simulate product categories (high cardinality)
products <- paste0("Product_", sprintf("%03d", 1:50))

cat_feature_hc <- sample(products, n_obs_hc, replace = TRUE)
bin_target_hc <- rbinom(n_obs_hc, 1, 0.08)

# Measure JEDI performance
time_jedi_hc <- system.time({
  result_jedi_hc <- ob_categorical_jedi(
    cat_feature_hc,
    bin_target_hc,
    min_bins = 4,
    max_bins = 7,
    max_n_prebins = 20,
    bin_cutoff = 0.02
  )
})

cat("\nHigh cardinality performance:\n")
#> 
#> High cardinality performance:
cat("  Original categories:", length(unique(cat_feature_hc)), "\n")
#>   Original categories: 50 
cat("  Final bins:", length(result_jedi_hc$bin), "\n")
#>   Final bins: 7 
cat("  Execution time:", round(time_jedi_hc[3], 3), "seconds\n")
#>   Execution time: 0 seconds
cat("  Total IV:", round(result_jedi_hc$total_iv, 4), "\n")
#>   Total IV: 0.1435 
cat("  Converged:", result_jedi_hc$converged, "\n")
#>   Converged: TRUE 

# Show merged categories
for (i in seq_along(result_jedi_hc$bin)) {
  n_merged <- length(strsplit(result_jedi_hc$bin[i], "%;%")[[1]])
  if (n_merged > 1) {
    cat(sprintf("  Bin %d: %d categories merged\n", i, n_merged))
  }
}
#>   Bin 1: 4 categories merged
#>   Bin 2: 6 categories merged
#>   Bin 3: 26 categories merged
#>   Bin 4: 5 categories merged
#>   Bin 5: 3 categories merged
#>   Bin 6: 3 categories merged
#>   Bin 7: 3 categories merged

# Example 6: Convergence behavior
set.seed(555)
n_obs_conv <- 1000

education_levels <- c(
  "Elementary", "HighSchool", "Vocational",
  "Bachelor", "Master", "PhD"
)

cat_feature_conv <- sample(education_levels, n_obs_conv,
  replace = TRUE,
  prob = c(0.10, 0.30, 0.20, 0.25, 0.12, 0.03)
)
bin_target_conv <- rbinom(n_obs_conv, 1, 0.15)

# Test different convergence thresholds
thresholds <- c(1e-3, 1e-6, 1e-9)

for (thresh in thresholds) {
  result_conv <- ob_categorical_jedi(
    cat_feature_conv,
    bin_target_conv,
    min_bins = 2,
    max_bins = 4,
    convergence_threshold = thresh,
    max_iterations = 100
  )

  cat(sprintf("\nThreshold %.0e:\n", thresh))
  cat("  Final bins:", length(result_conv$bin), "\n")
  cat("  Total IV:", round(result_conv$total_iv, 4), "\n")
  cat("  Converged:", result_conv$converged, "\n")
  cat("  Iterations:", result_conv$iterations, "\n")
}
#> 
#> Threshold 1e-03:
#>   Final bins: 4 
#>   Total IV: 0.0206 
#>   Converged: TRUE 
#>   Iterations: 0 
#> 
#> Threshold 1e-06:
#>   Final bins: 4 
#>   Total IV: 0.0206 
#>   Converged: TRUE 
#>   Iterations: 0 
#> 
#> Threshold 1e-09:
#>   Final bins: 4 
#>   Total IV: 0.0206 
#>   Converged: TRUE 
#>   Iterations: 0 

# Example 7: Missing value handling
set.seed(999)
cat_feature_na <- cat_feature
na_indices <- sample(n_obs, 75) # 5% missing
cat_feature_na[na_indices] <- NA

result_jedi_na <- ob_categorical_jedi(
  cat_feature_na,
  bin_target,
  min_bins = 3,
  max_bins = 5
)

# Locate NA bin
na_bin_idx <- grep("NA", result_jedi_na$bin)
if (length(na_bin_idx) > 0) {
  cat("\nMissing value treatment:\n")
  cat("  NA bin:", result_jedi_na$bin[na_bin_idx], "\n")
  cat("  NA count:", result_jedi_na$count[na_bin_idx], "\n")
  cat(
    "  NA event rate:",
    round(result_jedi_na$count_pos[na_bin_idx] /
      result_jedi_na$count[na_bin_idx], 3), "\n"
  )
  cat("  NA WoE:", round(result_jedi_na$woe[na_bin_idx], 3), "\n")
  cat("  NA IV contribution:", round(result_jedi_na$iv[na_bin_idx], 4), "\n")
}
#> 
#> Missing value treatment:
#>   NA bin: Retired%;%SelfEmployed%;%NA%;%Contract 
#>   NA count: 587 
#>   NA event rate: 0.112 
#>   NA WoE: -0.031 
#>   NA IV contribution: 4e-04 
# }
```
