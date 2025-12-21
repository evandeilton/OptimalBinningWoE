# Optimal Binning and WoE Transformation Step

`step_obwoe()` creates a *specification* of a recipe step that
discretizes predictor variables using one of 28 state-of-the-art optimal
binning algorithms and transforms them into Weight of Evidence (WoE)
values. This step fully integrates the **OptimalBinningWoE** package
into the `tidymodels` framework, supporting supervised discretization
for both binary and multinomial classification targets with extensive
hyperparameter tuning capabilities.

## Usage

``` r
step_obwoe(
  recipe,
  ...,
  role = "predictor",
  trained = FALSE,
  outcome = NULL,
  algorithm = "auto",
  min_bins = 2L,
  max_bins = 10L,
  bin_cutoff = 0.05,
  output = c("woe", "bin", "both"),
  suffix_woe = "_woe",
  suffix_bin = "_bin",
  na_woe = 0,
  control = list(),
  binning_results = NULL,
  skip = FALSE,
  id = recipes::rand_id("obwoe")
)
```

## Arguments

- recipe:

  A recipe object. The step will be added to the sequence of operations
  for this recipe.

- ...:

  One or more selector functions to choose variables for this step. See
  [`selections`](https://recipes.tidymodels.org/reference/selections.html)
  for available selectors. Common choices include
  [`all_predictors()`](https://recipes.tidymodels.org/reference/has_role.html),
  [`all_numeric_predictors()`](https://recipes.tidymodels.org/reference/has_role.html),
  or
  [`all_nominal_predictors()`](https://recipes.tidymodels.org/reference/has_role.html).
  Ensure the selected variables are compatible with the chosen
  `algorithm` (e.g., do not apply `"mdlp"` to categorical data).

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
  directly in `step_obwoe()` (e.g., `bin_cutoff`) take precedence over
  values in this list.

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

An updated `recipe` object with the new step appended.

## Details

### Weight of Evidence Transformation

Weight of Evidence (WoE) is a supervised encoding technique that
transforms categorical and continuous variables into a scale that
measures the predictive strength of each value or bin relative to the
target variable. For a bin \\i\\, the WoE is defined as:

\$\$WoE_i = \ln\left(\frac{\text{Distribution of
Events}\_i}{\text{Distribution of Non-Events}\_i}\right)\$\$

Positive WoE values indicate the bin has a higher proportion of events
(e.g., defaults) than the overall population, while negative values
indicate lower risk.

### Algorithm Selection Strategy

The `algorithm` parameter provides access to 28 binning algorithms:

- Use `algorithm = "auto"` (default) for automatic selection: `"jedi"`
  for binary targets, `"jedi_mwoe"` for multinomial.

- Use `algorithm = "mob"` (Monotonic Optimal Binning) when monotonic WoE
  trends are required for regulatory compliance (Basel/IFRS 9).

- Use `algorithm = "mdlp"` for entropy-based discretization of numerical
  variables (requires
  [`all_numeric_predictors()`](https://recipes.tidymodels.org/reference/has_role.html)).

- Use `algorithm = "dp"` (Dynamic Programming) for exact optimal
  solutions when computational cost is acceptable.

If an incompatible algorithm is applied to a variable (e.g., `"mdlp"` on
a factor), the step will issue a warning during
[`prep()`](https://recipes.tidymodels.org/reference/prep.html) and skip
that variable, leaving it untransformed.

### Handling New Data

During [`bake()`](https://recipes.tidymodels.org/reference/bake.html),
observations are mapped to bins learned during
[`prep()`](https://recipes.tidymodels.org/reference/prep.html):

- **Numerical variables**: Values are assigned to bins based on the
  learned cutpoints using interval notation.

- **Categorical variables**: Categories are matched to their
  corresponding bins. Categories not seen during training receive the
  `na_woe` value.

- **Missing values**: Always receive the `na_woe` value.

### Tuning with tune

This step is fully compatible with the `tune` package. The following
parameters support `tune()`:

- `algorithm`: See
  [`obwoe_algorithm`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_algorithm.md).

- `min_bins`: See
  [`obwoe_min_bins`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_min_bins.md).

- `max_bins`: See
  [`obwoe_max_bins`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_max_bins.md).

- `bin_cutoff`: See
  [`obwoe_bin_cutoff`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_bin_cutoff.md).

### Case Weights

This step does not currently support case weights. All observations are
treated with equal weight during binning optimization.

## References

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

## See also

[`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
for the underlying binning engine,
[`control.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/control.obwoe.md)
for advanced control parameters,
[`obwoe_algorithm`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_algorithm.md),
[`obwoe_min_bins`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_min_bins.md),
[`obwoe_max_bins`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_max_bins.md),
[`obwoe_bin_cutoff`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_bin_cutoff.md)
for tuning parameter definitions,
[`recipe`](https://recipes.tidymodels.org/reference/recipe.html),
[`prep`](https://recipes.tidymodels.org/reference/prep.html),
[`bake`](https://recipes.tidymodels.org/reference/bake.html) for the
tidymodels recipe framework.

## Examples

``` r
# \donttest{
library(recipes)
#> Loading required package: dplyr
#> 
#> Attaching package: ‘dplyr’
#> The following objects are masked from ‘package:stats’:
#> 
#>     filter, lag
#> The following objects are masked from ‘package:base’:
#> 
#>     intersect, setdiff, setequal, union
#> 
#> Attaching package: ‘recipes’
#> The following object is masked from ‘package:stats’:
#> 
#>     step

# Simulated credit data
set.seed(123)
credit_data <- data.frame(
  age = rnorm(500, 45, 12),
  income = exp(rnorm(500, 10, 0.6)),
  employment = sample(c("Employed", "Self-Employed", "Unemployed"),
    500,
    replace = TRUE, prob = c(0.7, 0.2, 0.1)
  ),
  education = factor(c("HighSchool", "Bachelor", "Master", "PhD")[
    sample(1:4, 500, replace = TRUE, prob = c(0.3, 0.4, 0.2, 0.1))
  ]),
  default = factor(rbinom(500, 1, 0.15),
    levels = c(0, 1),
    labels = c("No", "Yes")
  )
)

# Example 1: Basic usage with automatic algorithm selection
rec_basic <- recipe(default ~ ., data = credit_data) %>%
  step_obwoe(all_predictors(), outcome = "default")

rec_prepped <- prep(rec_basic)
baked_data <- bake(rec_prepped, new_data = NULL)
head(baked_data)
#> # A tibble: 6 × 5
#>        age income employment education default
#>      <dbl>  <dbl>      <dbl>     <dbl> <fct>  
#> 1   0.0599 -0.115    0.00761   -0.0945 No     
#> 2   0.0599 -0.115    0.00761   -0.0979 Yes    
#> 3   0.0599  0.230    0.00761   -0.0945 No     
#> 4   0.0599  0.230    0.00761   -0.0504 No     
#> 5   0.0599 -0.310    0.00761   -0.0979 No     
#> 6 -20.2     0        0.00761   -0.0979 No     

# View binning details
tidy(rec_prepped, number = 1)
#> # A tibble: 15 × 5
#>    terms      bin                               woe        iv id         
#>    <chr>      <chr>                           <dbl>     <dbl> <chr>      
#>  1 age        (-Inf;65.213228]              0.0599  0.00348   obwoe_sOPpR
#>  2 age        (65.213228;+Inf]            -20.2     1.17      obwoe_sOPpR
#>  3 income     (-Inf;11272.824005]          -0.310   0.0129    obwoe_sOPpR
#>  4 income     (11272.824005;20785.031269]  -0.115   0.00384   obwoe_sOPpR
#>  5 income     (20785.031269;24008.757553]   0       0         obwoe_sOPpR
#>  6 income     (24008.757553;32373.582575]   0       0         obwoe_sOPpR
#>  7 income     (32373.582575;59462.758585]   0.230   0.0114    obwoe_sOPpR
#>  8 income     (59462.758585;+Inf]           0.429   0.0107    obwoe_sOPpR
#>  9 employment Self-Employed                -0.0873  0.00136   obwoe_sOPpR
#> 10 employment Employed                      0.00761 0.0000409 obwoe_sOPpR
#> 11 employment Unemployed                    0.0382  0.000164  obwoe_sOPpR
#> 12 education  HighSchool                   -0.0979  0.00273   obwoe_sOPpR
#> 13 education  Master                       -0.0945  0.00187   obwoe_sOPpR
#> 14 education  Bachelor                     -0.0504  0.000965  obwoe_sOPpR
#> 15 education  PhD                           0.490   0.0298    obwoe_sOPpR

# Example 2: Numerical-only algorithm on numeric predictors
rec_mdlp <- recipe(default ~ age + income, data = credit_data) %>%
  step_obwoe(all_numeric_predictors(),
    outcome = "default",
    algorithm = "mdlp",
    min_bins = 3,
    max_bins = 6
  )

# Example 3: Output both bins and WoE
rec_both <- recipe(default ~ age, data = credit_data) %>%
  step_obwoe(age,
    outcome = "default",
    output = "both"
  )

baked_both <- bake(prep(rec_both), new_data = NULL)
names(baked_both)
#> [1] "age"     "age_woe" "age_bin" "default"
# Contains: default, age, age_woe, age_bin

# Example 4: Custom control parameters
rec_custom <- recipe(default ~ ., data = credit_data) %>%
  step_obwoe(all_predictors(),
    outcome = "default",
    algorithm = "mob",
    bin_cutoff = 0.03,
    control = list(
      max_n_prebins = 30,
      convergence_threshold = 1e-8
    )
  )

# Example 5: Tuning specification (for use with tune package)
# rec_tune <- recipe(default ~ ., data = credit_data) %>%
#   step_obwoe(all_predictors(),
#              outcome = "default",
#              algorithm = tune(),
#              min_bins = tune(),
#              max_bins = tune())
# }
```
