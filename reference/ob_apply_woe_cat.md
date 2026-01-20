# Apply Optimal Weight of Evidence (WoE) to a Categorical Feature

Transforms a categorical feature into its corresponding Weight of
Evidence (WoE) values using pre-computed binning results from an optimal
binning algorithm (e.g.,
[`ob_categorical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_cm.md)).

## Usage

``` r
ob_apply_woe_cat(
  obresults,
  feature,
  bin_separator = "%;%",
  missing_values = c("NA", "Missing", "")
)
```

## Arguments

- obresults:

  List output from an optimal binning function for categorical
  variables. Must contain elements `bin` (character vector of bin
  labels) and `woe` (numeric vector of WoE values). Bins may represent
  individual categories or merged groups separated by `bin_separator`.

- feature:

  Character or factor vector of categorical values to be transformed.
  Automatically coerced to character if provided as factor.

- bin_separator:

  Character string used to separate multiple categories within a single
  bin label (default: `"%;%"`). For example, a bin `"A%;%B%;%C"`
  contains categories A, B, and C.

- missing_values:

  Character vector specifying which values should be treated as missing
  (default: `c("NA", "Missing", "")`). These values are matched against
  a special bin labeled `"NA"` or `"Missing"` in `obresults`.

## Value

Numeric vector of WoE values with the same length as `feature`.
Categories not found in `obresults` will produce `NA` values with a
warning.

## Details

This function is typically used in a two-step workflow:

1.  Train binning on training data:
    `bins <- ob_categorical_cm(feature_train, target_train)`

2.  Apply WoE to new data:
    `woe_test <- ob_apply_woe_cat(bins, feature_test)`

The function performs exact string matching between categories in
`feature` and the bin labels in `obresults$bin`. For merged bins
(containing `bin_separator`), the string is split and each component is
matched individually.

## Examples

``` r
# \donttest{
# Mock data
train_data <- data.frame(
  category = c("A", "B", "A", "C", "B", "A"),
  default = c(0, 1, 0, 1, 0, 0)
)
test_data <- data.frame(
  category = c("A", "C", "B")
)

# Train binning on training set
train_bins <- ob_categorical_cm(
  feature = train_data$category,
  target = train_data$default
)

# Apply to test set
test_woe <- ob_apply_woe_cat(
  obresults = train_bins,
  feature = test_data$category
)

# Handle custom missing indicators
test_woe <- ob_apply_woe_cat(
  obresults = train_bins,
  feature = test_data$category,
  missing_values = c("NA", "Unknown", "N/A", "")
)
# }
```
