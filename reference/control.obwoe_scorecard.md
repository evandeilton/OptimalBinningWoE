# Control Parameters for the Scorecard Pipeline

Settings for the stages of
[`obwoe_scorecard`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scorecard.md)
that have no home in
[`control.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/control.obwoe.md)
or
[`obwoe_select`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_select.md).

## Usage

``` r
control.obwoe_scorecard(
  corr_cutoff = 0.7,
  corr_method = "pearson",
  pdo = 20,
  score_ref = 600,
  odds_ref = 50,
  direction = "higher_is_safer",
  n_groups = 10L,
  na_woe = 0,
  drop_negative = TRUE,
  max_abs_coef = 15,
  min_events_per_variable = 20,
  engine_fallback = FALSE,
  overwrite = TRUE,
  digits = 6L
)
```

## Arguments

- corr_cutoff:

  Numeric in \\(0, 1\]\\. Absolute correlation in the WoE space at or
  above which two variables are treated as redundant. Default `0.70`.
  Set to `1` to disable pruning.

- corr_method:

  Character string passed to
  [`obcorr`](https://evandeilton.github.io/OptimalBinningWoE/reference/obcorr.md).
  Default `"pearson"`.

- pdo, score_ref, odds_ref, direction:

  Passed to
  [`obwoe_scale`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scale.md).

- n_groups:

  Integer. Number of score bands in the gains and stability tables.
  Default `10`.

- na_woe:

  Numeric. WoE given to a value that falls in no fitted bin — an unseen
  category, or a missing value where the binner built no missing bin.
  Default `0`, which is the population-average contribution.

- drop_negative:

  Logical. Refit without any variable whose coefficient on WoE comes out
  negative? Default `TRUE`. A negative coefficient means the variable is
  fighting the rest of the model, and a scorecard with one cannot be
  signed off.

- max_abs_coef:

  Numeric. Coefficients above this in absolute value are flagged as
  separation. Default `15`.

- min_events_per_variable:

  Numeric. Events per variable in the model below which the fit is
  flagged as thin. Default `20`, the usual rule of thumb.

- engine_fallback:

  Logical. If the requested engine's package is not installed, fall back
  to `"glm"` with a warning instead of failing? Default `FALSE`.

- overwrite:

  Logical. Overwrite `file` if it exists? Default `TRUE`.

- digits:

  Integer. Rounding used in the workbook. Default `6`.

## Value

An object of class `"obwoe_scorecard_control"`.

## See also

[`obwoe_scorecard`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scorecard.md)

## Examples

``` r
control.obwoe_scorecard(pdo = 25, score_ref = 700, corr_cutoff = 0.8)
#> $corr_cutoff
#> [1] 0.8
#> 
#> $corr_method
#> [1] "pearson"
#> 
#> $pdo
#> [1] 25
#> 
#> $score_ref
#> [1] 700
#> 
#> $odds_ref
#> [1] 50
#> 
#> $direction
#> [1] "higher_is_safer"
#> 
#> $n_groups
#> [1] 10
#> 
#> $na_woe
#> [1] 0
#> 
#> $drop_negative
#> [1] TRUE
#> 
#> $max_abs_coef
#> [1] 15
#> 
#> $min_events_per_variable
#> [1] 20
#> 
#> $engine_fallback
#> [1] FALSE
#> 
#> $overwrite
#> [1] TRUE
#> 
#> $digits
#> [1] 6
#> 
#> attr(,"class")
#> [1] "obwoe_scorecard_control"
```
