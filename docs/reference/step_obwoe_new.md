# Internal Constructor for step_obwoe

Creates a new step_obwoe object. This is an internal function and should
not be called directly by users.

## Usage

``` r
step_obwoe_new(
  terms,
  role,
  trained,
  outcome,
  algorithm,
  min_bins,
  max_bins,
  bin_cutoff,
  output,
  suffix_woe,
  suffix_bin,
  na_woe,
  control,
  binning_results,
  skip,
  id
)
```

## Arguments

- terms:

  A list of quosures specifying the variables to transform.

- role:

  For variables created by this step, what role should they have?

  Default is `"predictor"`.

- trained:

  A logical indicating whether the step has been trained (fitted). This
  should not be set manually.

- outcome:

  A character string specifying the name of the binary or multinomial
  response variable. This argument is **required** as all binning
  algorithms are supervised. The outcome must exist in the training data
  provided to
  [`prep()`](https://recipes.tidymodels.org/reference/prep.html). The
  outcome should be encoded as a factor (standard for tidymodels
  classification) or as integers 0/1 for binary, 0/1/2/... for
  multinomial.

- algorithm:

  Character string specifying the binning algorithm to use. Use `"auto"`
  (default) for automatic selection based on target type: `"jedi"` for
  binary targets, `"jedi_mwoe"` for multinomial.

  Available algorithms are organized by supported feature types:

  **Universal (numerical and categorical):** `"auto"`, `"jedi"`,
  `"jedi_mwoe"`, `"cm"`, `"dp"`, `"dmiv"`, `"fetb"`, `"mob"`,
  `"sketch"`, `"udt"`

  **Numerical only:** `"bb"`, `"ewb"`, `"fast_mdlp"`, `"ir"`, `"kmb"`,
  `"ldb"`, `"lpdb"`, `"mblp"`, `"mdlp"`, `"mrblp"`, `"oslp"`, `"ubsd"`

  **Categorical only:** `"gmb"`, `"ivb"`, `"mba"`, `"milp"`, `"sab"`,
  `"sblp"`, `"swb"`

  This parameter is tunable with `tune()`.

- min_bins:

  Integer specifying the minimum number of bins to create. Must be at
  least 2. Default is 2. This parameter is tunable with `tune()`.

- max_bins:

  Integer specifying the maximum number of bins to create. Must be
  greater than or equal to `min_bins`. Default is 10. This parameter is
  tunable with `tune()`.

- bin_cutoff:

  Numeric value between 0 and 1 (exclusive) specifying the minimum
  proportion of total observations that each bin must contain. Bins with
  fewer observations are merged with adjacent bins. This serves as a
  regularization mechanism to prevent overfitting and ensure statistical
  stability of WoE estimates. Default is 0.05 (5%). This parameter is
  tunable with `tune()`.

- output:

  Character string specifying the transformation output format:

  `"woe"`

  :   Replaces the original variable with WoE values (default). This is
      the standard choice for logistic regression scorecards.

  `"bin"`

  :   Replaces the original variable with bin labels (character). Useful
      for tree-based models or exploratory analysis.

  `"both"`

  :   Keeps the original column and adds two new columns with suffixes
      `_woe` and `_bin`. Useful for model comparison or audit trails.

- suffix_woe:

  Character string suffix appended to create WoE column names when
  `output = "both"`. Default is `"_woe"`.

- suffix_bin:

  Character string suffix appended to create bin column names when
  `output = "both"`. Default is `"_bin"`.

- na_woe:

  Numeric value to assign to observations that cannot be mapped to a bin
  during [`bake()`](https://recipes.tidymodels.org/reference/bake.html).
  This includes missing values (`NA`) and unseen categories not present
  in the training data. Default is 0, which represents neutral evidence
  (neither good nor bad).

- control:

  A named list of additional control parameters passed to
  [`control.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/control.obwoe.md).
  These provide fine-grained control over algorithm behavior such as
  convergence thresholds and maximum pre-bins. Parameters specified
  directly in
  [`step_obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md)
  (e.g., `bin_cutoff`) take precedence over values in this list.

- binning_results:

  Internal storage for fitted binning models after
  [`prep()`](https://recipes.tidymodels.org/reference/prep.html). Do not
  set manually.

- skip:

  Logical. Should this step be skipped when
  [`bake()`](https://recipes.tidymodels.org/reference/bake.html) is
  called on new data? Default is `FALSE`. Setting to `TRUE` is rarely
  needed for WoE transformations but may be useful in specialized
  workflows.

- id:

  A unique character string to identify this step. If not provided, a
  random identifier is generated.

## Value

A step_obwoe object.
