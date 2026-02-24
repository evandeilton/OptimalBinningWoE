# Apply Weight of Evidence Transformations to New Data

Applies the binning and Weight of Evidence (WoE) transformations learned
by
[`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
to new data. This is the scoring function for deploying WoE-based models
in production. For each feature, the function assigns observations to
bins and maps them to their corresponding WoE values.

## Usage

``` r
obwoe_apply(
  data,
  obj,
  suffix_bin = "_bin",
  suffix_woe = "_woe",
  keep_original = TRUE,
  na_woe = 0
)
```

## Arguments

- data:

  A `data.frame` containing the features to transform. Must include all
  features present in the `obj` results. The target column is optional;
  if present, it will be included in the output.

- obj:

  An object of class `"obwoe"` returned by
  [`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md).

- suffix_bin:

  Character string suffix for bin columns. Default is `"_bin"`.

- suffix_woe:

  Character string suffix for WoE columns. Default is `"_woe"`.

- keep_original:

  Logical. If `TRUE` (default), include the original feature columns in
  the output. If `FALSE`, only bin and WoE columns are returned.

- na_woe:

  Numeric value to assign when an observation cannot be mapped to a bin
  (e.g., new categories not seen during training). Default is 0.

## Value

A `data.frame` containing:

- `target`:

  The target column (if present in `data`)

- `<feature>`:

  Original feature values (if `keep_original = TRUE`)

- `<feature>_bin`:

  Assigned bin label for each observation

- `<feature>_woe`:

  Weight of Evidence value for the assigned bin

## Details

### Bin Assignment Logic

**Numerical Features**: Observations are assigned to bins based on
cutpoints stored in the `obwoe` object. The
[`cut()`](https://rdrr.io/r/base/cut.html) function is used with
intervals \\(a_i, a\_{i+1}\]\\ where \\a_0 = -\infty\\ and \\a_k =
+\infty\\.

**Categorical Features**: Categories are matched directly to bin labels.
Categories not seen during training are assigned `NA` for bin and
`na_woe` for WoE.

### Production Deployment

For production scoring, it is recommended to:

1.  Train the binning model using
    [`obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
    on the training set

2.  Save the fitted object with
    [`saveRDS()`](https://rdrr.io/r/base/readRDS.html)

3.  Load and apply using `obwoe_apply()` on new data

The WoE-transformed features can be used directly as inputs to logistic
regression or other linear models, enabling interpretable credit
scorecards.

## References

Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
Intelligent Credit Scoring. *John Wiley & Sons*.
[doi:10.1002/9781119201731](https://doi.org/10.1002/9781119201731)

## See also

[`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
for fitting the binning model,
[`summary.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/summary.obwoe.md)
for model diagnostics.

## Examples

``` r
# \donttest{
# =============================================================================
# Example 1: Basic Usage - Train and Apply
# =============================================================================
set.seed(42)
n <- 1000

# Training data
train_df <- data.frame(
  age = rnorm(n, 40, 15),
  income = exp(rnorm(n, 10, 0.8)),
  education = sample(c("HS", "BA", "MA", "PhD"), n, replace = TRUE),
  target = rbinom(n, 1, 0.15)
)

# Fit binning model
model <- obwoe(train_df, target = "target")

# New data for scoring (could be validation/test set)
new_df <- data.frame(
  age = c(25, 45, 65),
  income = c(20000, 50000, 80000),
  education = c("HS", "MA", "PhD")
)

# Apply transformations
scored <- obwoe_apply(new_df, model)
print(scored)
#>   age               age_bin     age_woe income                 income_bin
#> 1  25      (-Inf;39.802484] -0.07802808  20000 (8069.329816;61783.844034]
#> 2  45 (41.347493;52.361109]  0.03006841  50000 (8069.329816;61783.844034]
#> 3  65      (52.361109;+Inf]  0.13769907  80000        (61783.844034;+Inf]
#>    income_woe education education_bin education_woe
#> 1 -0.03653409        HS            HS   -0.43188518
#> 2 -0.03653409        MA            MA   -0.09570827
#> 3 -0.04630457       PhD           PhD    0.19851891

# Use WoE features for downstream modeling
woe_cols <- grep("_woe$", names(scored), value = TRUE)
print(woe_cols)
#> [1] "age_woe"       "income_woe"    "education_woe"

# =============================================================================
# Example 2: Without Original Features
# =============================================================================

scored_compact <- obwoe_apply(new_df, model, keep_original = FALSE)
print(scored_compact)
#>                 age_bin     age_woe                 income_bin  income_woe
#> 1      (-Inf;39.802484] -0.07802808 (8069.329816;61783.844034] -0.03653409
#> 2 (41.347493;52.361109]  0.03006841 (8069.329816;61783.844034] -0.03653409
#> 3      (52.361109;+Inf]  0.13769907        (61783.844034;+Inf] -0.04630457
#>   education_bin education_woe
#> 1            HS   -0.43188518
#> 2            MA   -0.09570827
#> 3           PhD    0.19851891
# }
```
