# Data Preprocessor for Optimal Binning

Prepares features for optimal binning by handling missing values and
optionally detecting/treating outliers. Supports both numerical and
categorical variables with configurable preprocessing strategies.

## Usage

``` r
ob_preprocess(
  feature,
  target,
  num_miss_value = -999,
  char_miss_value = "N/A",
  outlier_method = "iqr",
  outlier_process = FALSE,
  preprocess = "both",
  iqr_k = 1.5,
  zscore_threshold = 3,
  grubbs_alpha = 0.05
)
```

## Arguments

- feature:

  Vector (numeric, character, or factor) to be preprocessed. Type is
  automatically detected.

- target:

  Numeric or integer vector of binary target values (0/1). Must have the
  same length as `feature`. Used for validation but not directly in
  preprocessing.

- num_miss_value:

  Numeric value to replace missing (`NA`) values in numerical features
  (default: `-999.0`). Choose a value outside the expected range of the
  feature.

- char_miss_value:

  Character string to replace missing (`NA`) values in categorical
  features (default: `"N/A"`).

- outlier_method:

  Character string specifying the outlier detection method for numerical
  features (default: `"iqr"`). Options:

  - `"iqr"`: Interquartile Range method. Outliers are values \\\< Q_1 -
    k \times IQR\\ or \\\> Q_3 + k \times IQR\\.

  - `"zscore"`: Z-score method. Outliers are values with \\\|z\| \>
    \text{threshold}\\ where \\z = (x - \mu) / \sigma\\.

  - `"grubbs"`: Grubbs' test for outliers (iterative). Removes the most
    extreme value if it exceeds the critical G-statistic at significance
    level `grubbs_alpha`.

- outlier_process:

  Logical flag to enable outlier detection and treatment (default:
  `FALSE`). Only applies to numerical features.

- preprocess:

  Character vector specifying output components (default: `"both"`):

  - `"feature"`: Return preprocessed feature data only.

  - `"report"`: Return preprocessing report only (summary statistics,
    counts).

  - `"both"`: Return both preprocessed data and report.

- iqr_k:

  Multiplier for the IQR method (default: 1.5). Larger values are more
  conservative (fewer outliers). Common values: 1.5 (standard), 3.0
  (extreme).

- zscore_threshold:

  Z-score threshold for outlier detection (default: 3.0). Values with
  \\\|z\| \> \text{threshold}\\ are considered outliers.

- grubbs_alpha:

  Significance level for Grubbs' test (default: 0.05). Lower values are
  more conservative (fewer outliers detected).

## Value

A list with up to two elements (depending on `preprocess`):

- preprocess:

  Data frame with columns:

  - `feature`: Original feature values.

  - `feature_preprocessed`: Preprocessed feature values (NAs replaced,
    outliers capped or removed).

- report:

  Data frame with one row containing:

  - `variable_type`: `"numeric"` or `"categorical"`.

  - `missing_count`: Number of `NA` values replaced.

  - `outlier_count`: Number of outliers detected (numeric only, `NA` for
    categorical).

  - `original_stats`: String representation of summary statistics before
    preprocessing (min, Q1, median, mean, Q3, max for numeric).

  - `preprocessed_stats`: Summary statistics after preprocessing.

## Details

**Preprocessing Pipeline**:

1.  **Type Detection**: Automatically classifies `feature` as numeric or
    categorical based on R type.

2.  **Missing Value Handling**: Replaces `NA` with `num_miss_value`
    (numeric) or `char_miss_value` (categorical).

3.  **Outlier Detection** (if `outlier_process = TRUE` for numeric):

    - **IQR Method**: Caps outliers at boundaries \\\[Q_1 - k \times
      IQR, Q_3 + k \times IQR\]\\.

    - **Z-score Method**: Caps outliers at \\\[\mu - t \times \sigma,
      \mu + t \times \sigma\]\\.

    - **Grubbs' Test**: Iteratively removes the most extreme value if
      \\G = \frac{\max\|x_i - \bar{x}\|}{s} \> G\_{\text{critical}}\\.

4.  **Summary Calculation**: Computes statistics before and after
    preprocessing for validation.

**Outlier Treatment Strategies**:

- IQR and Z-score: **Winsorization** (capping at boundaries).

- Grubbs: **Removal** (replaced with `num_miss_value`).

**Use Cases**:

- **Before binning**: Stabilize binning algorithms by removing extreme
  values that could create singleton bins.

- **Data quality audit**: Identify features with excessive missingness
  or outliers.

- **Model deployment**: Ensure test data undergoes identical
  preprocessing as training data.

## References

- Grubbs, F. E. (1950). "Sample Criteria for Testing Outlying
  Observations". *Annals of Mathematical Statistics*, 21(1), 27-58.

- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
  \[IQR method\]

## Examples

``` r
# \donttest{
# Numerical feature with outliers
set.seed(123)
feature_num <- c(rnorm(95, 50, 10), NA, NA, 200, -100, 250)
target <- sample(0:1, 100, replace = TRUE)

# Preprocess with IQR outlier detection
result_iqr <- ob_preprocess(
  feature = feature_num,
  target = target,
  outlier_process = TRUE,
  outlier_method = "iqr",
  iqr_k = 1.5
)

print(result_iqr$report)
#>   variable_type missing_count outlier_count
#> 1       numeric             2             5
#>                                                                                            original_stats
#> 1 { min: -100.000000, Q1: 45.061458, median: 50.905956, mean: 52.773778, Q3: 57.210082, max: 250.000000 }
#>                                                                                     preprocessed_stats
#> 1 { min: 25.774368, Q1: 44.575383, median: 50.617563, mean: 50.502606, Q3: 56.981770, max: 75.553623 }
# Shows: missing_count = 2, outlier_count = 3

# Categorical feature
feature_cat <- c(rep("A", 30), rep("B", 40), rep("C", 28), NA, NA)
target_cat <- sample(0:1, 100, replace = TRUE)

result_cat <- ob_preprocess(
  feature = feature_cat,
  target = target_cat,
  char_miss_value = "Missing"
)

# Compare original vs preprocessed
head(result_cat$preprocess)
#>   feature feature_preprocessed
#> 1       A                    A
#> 2       A                    A
#> 3       A                    A
#> 4       A                    A
#> 5       A                    A
#> 6       A                    A
# Shows NA replaced with "Missing"

# Return only report (no data)
result_report <- ob_preprocess(
  feature = feature_num,
  target = target,
  preprocess = "report",
  outlier_process = TRUE
)

# Grubbs' test (most conservative)
result_grubbs <- ob_preprocess(
  feature = feature_num,
  target = target,
  outlier_process = TRUE,
  outlier_method = "grubbs",
  grubbs_alpha = 0.01 # Very strict
)
# }
```
