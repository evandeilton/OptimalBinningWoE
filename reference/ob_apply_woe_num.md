# Apply Optimal Weight of Evidence (WoE) to a Numerical Feature

Transforms a numerical feature into its corresponding Weight of Evidence
(WoE) values using pre-computed binning results from an optimal binning
algorithm (e.g.,
[`ob_numerical_mdlp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mdlp.md),
[`ob_numerical_mob`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mob.md)).

## Usage

``` r
ob_apply_woe_num(
  obresults,
  feature,
  include_upper_bound = TRUE,
  missing_values = c(-999)
)
```

## Arguments

- obresults:

  List output from an optimal binning function for numerical variables.
  Must contain elements `cutpoints` (numeric vector of bin boundaries)
  and `woe` (numeric vector of WoE values). The number of WoE values
  should equal `length(cutpoints) + 1`.

- feature:

  Numeric vector of values to be transformed. Automatically coerced to
  numeric if provided in another type.

- include_upper_bound:

  Logical flag controlling interval boundary behavior (default: `TRUE`):

  - `TRUE`: Intervals are `(lower, upper]` (right-closed).

  - `FALSE`: Intervals are `[lower, upper)` (left-closed).

  This must match the convention used during binning.

- missing_values:

  Numeric vector of values to be treated as missing (default:
  `c(-999)`). These values are assigned the WoE of the special missing
  bin if it exists in `obresults`, or `NA` otherwise.

## Value

Numeric vector of WoE values with the same length as `feature`. Values
outside the range of `cutpoints` are assigned to the first or last bin.
`NA` values in `feature` are propagated to the output unless explicitly
listed in `missing_values`.

## Details

This function is typically used in a two-step workflow:

1.  Train binning on training data:
    `bins <- ob_numerical_mdlp(feature_train, target_train)`

2.  Apply WoE to new data:
    `woe_test <- ob_apply_woe_num(bins, feature_test)`

**Bin Assignment Logic**: For `k` cutpoints \\c_1 \< c_2 \< \cdots \<
c_k\\, values are assigned as:

- Bin 1: \\x \le c_1\\ (if `include_upper_bound = TRUE`)

- Bin i: \\c\_{i-1} \< x \le c_i\\ for \\i = 2, \ldots, k\\

- Bin k+1: \\x \> c_k\\

**Handling of Edge Cases**:

- Values in `missing_values` are matched against a bin labeled `"NA"` or
  `"Missing"` in `obresults$bin` (if available).

- `Inf` and `-Inf` are assigned to the last and first bins,
  respectively.

- Values exactly equal to cutpoints follow the `include_upper_bound`
  convention.

## See also

[`ob_numerical_mdlp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mdlp.md)
for MDLP binning,
[`ob_numerical_mob`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mob.md)
for monotonic binning,
[`ob_apply_woe_cat`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_cat.md)
for applying WoE to categorical features.

## Examples

``` r
# \donttest{
# Mock data
train_data <- data.frame(
  income = c(50000, 75000, 30000, 45000, 80000, 60000),
  default = c(0, 0, 1, 1, 0, 0)
)
test_data <- data.frame(
  income = c(55000, 35000, 90000)
)

# Train binning on training set
train_bins <- ob_numerical_mdlp(
  feature = train_data$income,
  target = train_data$default
)

# Apply to test set
test_woe <- ob_apply_woe_num(
  obresults = train_bins,
  feature = test_data$income
)

# Handle custom missing indicators (e.g., -999, -1)
test_woe <- ob_apply_woe_num(
  obresults = train_bins,
  feature = test_data$income,
  missing_values = c(-999, -1, -9999)
)

# Use left-closed intervals (match scikit-learn convention)
test_woe <- ob_apply_woe_num(
  obresults = train_bins,
  feature = test_data$income,
  include_upper_bound = FALSE
)
# }
```
