# Unified Optimal Binning and Weight of Evidence Transformation

Master interface for optimal discretization and Weight of Evidence (WoE)
computation across numerical and categorical predictors. This function
serves as the primary entry point for the **OptimalBinningWoE** package,
providing automatic feature type detection, intelligent algorithm
selection, and unified output structures for seamless integration into
credit scoring and predictive modeling workflows.

## Usage

``` r
obwoe(
  data,
  target,
  feature = NULL,
  min_bins = 2,
  max_bins = 7,
  algorithm = "auto",
  control = control.obwoe()
)
```

## Arguments

- data:

  A `data.frame` containing the predictor variables (features) and the
  response variable (target). All features to be binned must be present
  in this data frame. The data frame should not contain list-columns.

- target:

  Character string specifying the column name of the response variable.
  Must be a binary outcome encoded as integers `0` (non-event) and `1`
  (event), or a multinomial outcome encoded as integers
  `0, 1, 2, ..., K`. Missing values in the target are not permitted.

- feature:

  Optional character vector specifying which columns to process. If
  `NULL` (default), all columns except `target` are processed. Features
  containing only missing values are automatically skipped with a
  warning.

- min_bins:

  Integer specifying the minimum number of bins. Must satisfy \\2 \le\\
  `min_bins` \\\le\\ `max_bins`. Algorithms may produce fewer bins if
  the data has insufficient unique values. Default is 2.

- max_bins:

  Integer specifying the maximum number of bins. Controls the
  granularity of discretization. Higher values capture more detail but
  risk overfitting. Typical values range from 5 to 10 for credit scoring
  applications. Default is 7.

- algorithm:

  Character string specifying the binning algorithm. Use `"auto"`
  (default) for automatic selection based on target type: `"jedi"` for
  binary targets, `"jedi_mwoe"` for multinomial. See Details for the
  complete algorithm taxonomy.

- control:

  A list of algorithm-specific control parameters created by
  [`control.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/control.obwoe.md).
  Provides fine-grained control over convergence thresholds, bin
  cutoffs, and other optimization parameters.

## Value

An S3 object of class `"obwoe"` containing:

- `results`:

  Named list where each element contains the binning result for a single
  feature, including:

  `bin`

  :   Character vector of bin labels/intervals

  `woe`

  :   Numeric vector of Weight of Evidence per bin

  `iv`

  :   Numeric vector of Information Value contribution per bin

  `count`

  :   Integer vector of observation counts per bin

  `count_pos`

  :   Integer vector of positive (event) counts per bin

  `count_neg`

  :   Integer vector of negative (non-event) counts per bin

  `cutpoints`

  :   Numeric vector of bin boundaries (numerical only)

  `converged`

  :   Logical indicating algorithm convergence

  `iterations`

  :   Integer count of optimization iterations

- `summary`:

  Data frame with one row per feature containing: `feature` (name),
  `type` (numerical/categorical), `algorithm` (used), `n_bins` (count),
  `total_iv` (sum), `error` (logical flag)

- `target`:

  Name of the target column

- `target_type`:

  Detected type: `"binary"` or `"multinomial"`

- `n_features`:

  Number of features processed

- `call`:

  The matched function call for reproducibility

## Details

### Theoretical Foundation

Weight of Evidence (WoE) transformation is a staple of credit scoring
methodology, originating from information theory and the concept of
evidential support (Good, 1950; Kullback, 1959). For a bin \\i\\, the
WoE is defined as:

\$\$WoE_i = \ln\left(\frac{p_i}{n_i}\right) =
\ln\left(\frac{N\_{i,1}/N_1}{N\_{i,0}/N_0}\right)\$\$

where:

- \\N\_{i,1}\\ = number of events (target=1) in bin \\i\\

- \\N\_{i,0}\\ = number of non-events (target=0) in bin \\i\\

- \\N_1\\, \\N_0\\ = total events and non-events, respectively

- \\p_i = N\_{i,1}/N_1\\ = proportion of events in bin \\i\\

- \\n_i = N\_{i,0}/N_0\\ = proportion of non-events in bin \\i\\

The Information Value (IV) quantifies the total predictive power of a
binning:

\$\$IV = \sum\_{i=1}^{k} (p_i - n_i) \times WoE_i = \sum\_{i=1}^{k}
(p_i - n_i) \times \ln\left(\frac{p_i}{n_i}\right)\$\$

where \\k\\ is the number of bins. IV is equivalent to the
Kullback-Leibler divergence between the event and non-event
distributions.

### Algorithm Taxonomy

The package provides 28 algorithms organized by supported feature types:

**Universal Algorithms** (both numerical and categorical):

|             |                                  |                                |
|-------------|----------------------------------|--------------------------------|
| **ID**      | **Full Name**                    | **Method**                     |
| `jedi`      | Joint Entropy-Driven Information | Heuristic + IV optimization    |
| `jedi_mwoe` | JEDI Multinomial WoE             | Extension for K\>2 classes     |
| `cm`        | ChiMerge                         | Bottom-up chi-squared merging  |
| `dp`        | Dynamic Programming              | Exact optimal IV partitioning  |
| `dmiv`      | Decision Tree MIV                | Recursive partitioning         |
| `fetb`      | Fisher's Exact Test              | Statistical significance-based |
| `mob`       | Monotonic Optimal Binning        | IV-optimal with monotonicity   |
| `sketch`    | Sketching                        | Probabilistic data structures  |
| `udt`       | Unsupervised Decision Tree       | Entropy-based without target   |

**Numerical-Only Algorithms**:

|             |                                    |
|-------------|------------------------------------|
| **ID**      | **Description**                    |
| `bb`        | Branch and Bound (exact search)    |
| `ewb`       | Equal Width Binning (unsupervised) |
| `fast_mdlp` | Fast MDLP with pruning             |
| `ir`        | Isotonic Regression                |
| `kmb`       | K-Means Binning                    |
| `ldb`       | Local Density Binning              |
| `lpdb`      | Local Polynomial Density           |
| `mblp`      | Monotonic Binning LP               |
| `mdlp`      | Minimum Description Length         |
| `mrblp`     | Monotonic Regression LP            |
| `oslp`      | Optimal Supervised LP              |
| `ubsd`      | Unsupervised Std-Dev Based         |

**Categorical-Only Algorithms**:

|        |                              |
|--------|------------------------------|
| **ID** | **Description**              |
| `gmb`  | Greedy Monotonic Binning     |
| `ivb`  | Information Value DP (exact) |
| `mba`  | Modified Binning Algorithm   |
| `milp` | Mixed Integer LP             |
| `sab`  | Simulated Annealing          |
| `sblp` | Similarity-Based LP          |
| `swb`  | Sliding Window Binning       |

### Automatic Type Detection

Feature types are detected as follows:

- **Numerical**: `numeric` or `integer` vectors not of class `factor`

- **Categorical**: `character`, `factor`, or `logical` vectors

When `algorithm = "auto"`, the function selects:

- `"jedi"` for binary targets (recommended for most use cases)

- `"jedi_mwoe"` for multinomial targets (K \> 2 classes)

### IV Interpretation Guidelines

Siddiqi (2006) provides the following IV thresholds for variable
selection:

|              |                                 |
|--------------|---------------------------------|
| **IV Range** | **Predictive Power**            |
| \< 0.02      | Unpredictive                    |
| 0.02 - 0.10  | Weak                            |
| 0.10 - 0.30  | Medium                          |
| 0.30 - 0.50  | Strong                          |
| \> 0.50      | Suspicious (likely overfitting) |

### Computational Considerations

Time complexity varies by algorithm:

- **JEDI, ChiMerge, MOB**: \\O(n \log n + k^2 m)\\ where \\n\\ =
  observations, \\k\\ = bins, \\m\\ = iterations

- **Dynamic Programming**: \\O(n \cdot k^2)\\ for exact solution

- **Equal Width**: \\O(n)\\ (fastest, but unsupervised)

- **MILP, SBLP**: Potentially exponential (NP-hard problems)

For large datasets (\\n \> 10^6\\), consider:

1.  Using `algorithm = "sketch"` for approximate streaming

2.  Reducing `max_n_prebins` via
    [`control.obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/control.obwoe.md)

3.  Sampling the data before binning

## References

Good, I. J. (1950). Probability and the Weighing of Evidence. *Griffin,
London*.

Kullback, S. (1959). Information Theory and Statistics. *Wiley, New
York*.

Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
Intelligent Credit Scoring. *John Wiley & Sons*.
[doi:10.1002/9781119201731](https://doi.org/10.1002/9781119201731)

Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and
Its Applications. *SIAM Monographs on Mathematical Modeling and
Computation*.
[doi:10.1137/1.9780898718317](https://doi.org/10.1137/1.9780898718317)

Navas-Palencia, G. (2020). Optimal Binning: Mathematical Programming
Formulation and Solution Approach. *Expert Systems with Applications*,
158, 113508.
[doi:10.1016/j.eswa.2020.113508](https://doi.org/10.1016/j.eswa.2020.113508)

Zeng, G. (2014). A Necessary Condition for a Good Binning Algorithm in
Credit Scoring. *Applied Mathematical Sciences*, 8(65), 3229-3242.

## See also

[`control.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/control.obwoe.md)
for algorithm-specific parameters,
[`obwoe_algorithms`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_algorithms.md)
to list all available algorithms with capabilities,
[`print.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/print.obwoe.md)
for display methods,
[`ob_apply_woe_num`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_num.md)
and
[`ob_apply_woe_cat`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_cat.md)
to apply WoE transformations to new data.

For individual algorithms with full parameter control:
[`ob_numerical_jedi`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_jedi.md),
[`ob_categorical_jedi`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_jedi.md),
[`ob_numerical_mdlp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mdlp.md),
[`ob_categorical_ivb`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_ivb.md).

## Examples

``` r
# \donttest{
# =============================================================================
# Example 1: Basic Usage with Mixed Feature Types
# =============================================================================
set.seed(42)
n <- 2000

# Simulate credit scoring data
df <- data.frame(
  # Numerical features
  age = pmax(18, pmin(80, rnorm(n, 45, 15))),
  income = exp(rnorm(n, 10, 0.8)),
  debt_ratio = rbeta(n, 2, 5),
  credit_history_months = rpois(n, 60),

  # Categorical features
  education = sample(c("High School", "Bachelor", "Master", "PhD"),
    n,
    replace = TRUE, prob = c(0.35, 0.40, 0.20, 0.05)
  ),
  employment = sample(c("Employed", "Self-Employed", "Unemployed", "Retired"),
    n,
    replace = TRUE, prob = c(0.60, 0.20, 0.10, 0.10)
  ),

  # Binary target (default probability varies by features)
  target = rbinom(n, 1, 0.15)
)

# Process all features with automatic algorithm selection
result <- obwoe(df, target = "target")
print(result)
#> Optimal Binning Weight of Evidence
#> ===================================
#> 
#> Target: target ( binary )
#> Features processed: 6 
#> 
#> Results:  6  successful
#> 
#> Top features by IV:
#>   employment: IV = 0.0269 (4 bins, jedi)
#>   credit_history_months: IV = 0.0193 (4 bins, jedi)
#>   income: IV = 0.0092 (4 bins, jedi)
#>   debt_ratio: IV = 0.0091 (4 bins, jedi)
#>   education: IV = 0.0048 (4 bins, jedi)
#>   ... and 1 more

# View detailed summary
print(result$summary)
#>                 feature        type algorithm n_bins    total_iv converged
#> 1                   age   numerical      jedi      3 0.002240807      TRUE
#> 2                income   numerical      jedi      4 0.009202297      TRUE
#> 3            debt_ratio   numerical      jedi      4 0.009121470      TRUE
#> 4 credit_history_months   numerical      jedi      4 0.019333587      TRUE
#> 5             education categorical      jedi      4 0.004793127      TRUE
#> 6            employment categorical      jedi      4 0.026934398      TRUE
#>   iterations error
#> 1          2 FALSE
#> 2          2 FALSE
#> 3          2 FALSE
#> 4          2 FALSE
#> 5          0 FALSE
#> 6          0 FALSE

# Access results for a specific feature
age_bins <- result$results$age
print(data.frame(
  bin = age_bins$bin,
  woe = round(age_bins$woe, 3),
  iv = round(age_bins$iv, 4),
  count = age_bins$count
))
#>                     bin    woe     iv count
#> 1      (-Inf;40.127923] -0.022 0.0002   741
#> 2 (40.127923;59.856538] -0.020 0.0002   953
#> 3      (59.856538;+Inf]  0.109 0.0019   306

# =============================================================================
# Example 2: Using a Specific Algorithm
# =============================================================================

# Use MDLP for numerical features (entropy-based)
result_mdlp <- obwoe(df,
  target = "target",
  feature = c("age", "income"),
  algorithm = "mdlp",
  min_bins = 3,
  max_bins = 6
)

cat("\nMDLP Results:\n")
#> 
#> MDLP Results:
print(result_mdlp$summary)
#>   feature      type algorithm n_bins     total_iv converged iterations error
#> 1     age numerical      mdlp      3 0.0003386014      TRUE         16 FALSE
#> 2  income numerical      mdlp      3 0.0022781944      TRUE         16 FALSE

# =============================================================================
# Example 3: Custom Control Parameters
# =============================================================================

# Fine-tune algorithm behavior
ctrl <- control.obwoe(
  bin_cutoff = 0.02, # Minimum 2% per bin
  max_n_prebins = 30, # Allow more initial bins
  convergence_threshold = 1e-8
)

result_custom <- obwoe(df,
  target = "target",
  feature = "debt_ratio",
  algorithm = "jedi",
  control = ctrl
)

cat("\nCustom JEDI Result:\n")
#> 
#> Custom JEDI Result:
print(result_custom$results$debt_ratio$bin)
#> [1] "(-Inf;0.085180]"     "(0.085180;0.119174]" "(0.119174;0.345076]"
#> [4] "(0.345076;+Inf]"    

# =============================================================================
# Example 4: Comparing Multiple Algorithms
# =============================================================================

algorithms <- c("jedi", "mdlp", "ewb", "mob")
iv_comparison <- sapply(algorithms, function(algo) {
  tryCatch(
    {
      res <- obwoe(df, target = "target", feature = "income", algorithm = algo)
      res$summary$total_iv
    },
    error = function(e) NA_real_
  )
})

cat("\nAlgorithm Comparison (IV for 'income'):\n")
#> 
#> Algorithm Comparison (IV for 'income'):
print(sort(iv_comparison, decreasing = TRUE))
#>         jedi          ewb         mdlp          mob 
#> 0.0092022973 0.0028174937 0.0023796596 0.0003402702 

# =============================================================================
# Example 5: Feature Selection Based on IV
# =============================================================================

# Process all features and select those with IV > 0.02
result_all <- obwoe(df, target = "target")

strong_features <- result_all$summary[
  result_all$summary$total_iv >= 0.02 & !result_all$summary$error,
  c("feature", "total_iv", "n_bins")
]
strong_features <- strong_features[order(-strong_features$total_iv), ]

cat("\nFeatures with IV >= 0.02 (predictive):\n")
#> 
#> Features with IV >= 0.02 (predictive):
print(strong_features)
#>      feature  total_iv n_bins
#> 6 employment 0.0269344      4

# =============================================================================
# Example 6: Handling Algorithm Compatibility
# =============================================================================

# MDLP only works for numerical - will fail for categorical
result_mixed <- obwoe(df,
  target = "target",
  algorithm = "mdlp"
)

# Check for errors
cat("\nCompatibility check:\n")
#> 
#> Compatibility check:
print(result_mixed$summary[, c("feature", "type", "error")])
#>                 feature        type error
#> 1                   age   numerical FALSE
#> 2                income   numerical FALSE
#> 3            debt_ratio   numerical FALSE
#> 4 credit_history_months   numerical FALSE
#> 5             education categorical  TRUE
#> 6            employment categorical  TRUE
# }
```
