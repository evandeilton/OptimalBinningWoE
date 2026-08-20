# Automated Scorecard Pipeline

Runs a credit or fraud scorecard end to end in one call: optimal
binning, variable screening by Information Value and bin ordering,
redundancy pruning in the Weight of Evidence space, model fitting, and
score scaling to points. Optionally writes the whole thing to a
multi-sheet `.xlsx` workbook that a model committee, a validation team
and a deployment engineer can each read.

## Usage

``` r
obwoe_scorecard(
  data,
  target,
  feature = NULL,
  exclude = NULL,
  split = 0.7,
  validation = NULL,
  file = NULL,
  binning = list(),
  screening = list(),
  engine = "glm",
  engine_args = list(),
  control = control.obwoe_scorecard(),
  seed = NULL,
  verbose = FALSE
)
```

## Arguments

- data:

  A `data.frame` holding the development sample.

- target:

  Character string naming the target column. Binary; a factor's
  **second** level is the event, and which level that was is recorded in
  the result.

- feature:

  Character vector of candidate predictors. `NULL` (default) uses every
  column except the target, the split column and anything named in
  `exclude`.

- exclude:

  Character vector of columns to keep out of the model — identifiers,
  dates, and above all fields populated after the outcome was known.
  Default `NULL`.

- split:

  How the development sample is divided. One of:

  a number in \\(0, 1)\\

  :   proportion of rows used for training, sampled at random with the
      event rate preserved (default `0.7`);

  a column name

  :   an out-of-time split: the earliest value of that column trains,
      the rest validates;

  an integer or logical vector

  :   the training rows, given explicitly;

  `NULL`

  :   no split — everything trains, and every reported metric is
      in-sample. Flagged as such throughout.

- validation:

  Optional named list of extra `data.frame`s to score and report on.
  They never influence any fitted quantity.

- file:

  Optional path to a `.xlsx` file. `NULL` (default) computes everything
  and writes nothing; pass the result to
  [`obwoe_report`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_report.md)
  later to produce the workbook.

- binning:

  Named list of arguments for
  [`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
  — typically `min_bins`, `max_bins`, `algorithm`, `control`.

- screening:

  Named list of arguments for
  [`obwoe_select`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_select.md)
  — typically `iv_min`, `iv_max`, `require_monotonic`, `min_bin_pct`,
  `top_n`.

- engine:

  Character string naming the model engine — `"glm"` (default),
  `"obwoe"` (the package's own C++ logistic regression) or `"glmnet"` —
  or a list supplying `fit`, `link` and `coef` of your own. See Details.

- engine_args:

  Named list forwarded to the engine's fitting call.

- control:

  An `"obwoe_scorecard_control"` object from
  [`control.obwoe_scorecard`](https://evandeilton.github.io/OptimalBinningWoE/reference/control.obwoe_scorecard.md).

- seed:

  Optional integer seed, for a reproducible split.

- verbose:

  Logical. Report progress? Default `FALSE`.

## Value

An object of class `"obwoe_scorecard"`: a list with

- `call`, `version`, `built_on`, `seed`:

  provenance

- `target`, `event_level`, `candidates`, `selected`, `final`:

  the funnel, as character vectors

- `binning`:

  the `"obwoe"` object, fitted on the training rows only

- `screening`, `screening_bins`:

  [`obwoe_select`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_select.md)
  at both levels of detail, plus a `stage` column naming the step that
  rejected each variable: `"in_model"`, `"sign_rejected"` (negative WoE
  coefficient), `"corr_pruned"`, `"constant_woe"` (one WoE value after
  the transform) or `"screened_out"` (failed the IV or ordering rules)

- `correlation`:

  the
  [`obwoe_prune`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_prune.md)
  result

- `control`:

  the resolved `"obwoe_scorecard_control"` this object was actually
  built with (as of 1.13.1) –
  [`predict()`](https://rdrr.io/r/stats/predict.html) and
  [`obwoe_report`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_report.md)
  read `na_woe` from here so every output agrees on the fallback for an
  unseen category or an unmodeled missing value

- `engine`:

  name, requested versus used, and whether additive

- `model`:

  the raw fitted object — engine-specific and outside the supported
  interface

- `coefficients`, `diagnostics`:

  the additive fit and its sign, separation and convergence checks

- `scaling`:

  the
  [`obwoe_scale`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scale.md)
  constants

- `points`:

  the fixed points table, one row per variable and bin

- `samples`:

  per sample: scores, gains table, headline metrics

- `stability`:

  PSI for the score and for every variable in the model

- `warnings`:

  everything the pipeline wants a human to read

## Details

### The split comes first

Binning is **supervised**: the cut points and the WoE are both chosen
using the target. Fitting them on rows that are later used to measure
performance is leakage, and it is the failure this function is built to
make impossible — the split happens before anything is fitted, the
binning sees the training rows only, and every other sample is
transformed through
[`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md).
Screening statistics are computed on the training rows for the same
reason.

### The binning is fitted once

One `"obwoe"` object drives screening, correlation, the model, the
points, the stability report and the generated SQL. Refitting the
binning for the model — with, say, a different `bin_cutoff` — produces a
workbook whose evidence describes a transform the deployed model never
saw, and every number in it looks internally consistent.

### Points

With \\k\\ variables in the model, an intercept \\\alpha\\ and slopes
\\\beta_j\\,

\$\$\mathrm{points}\_{ij} = \frac{\mathrm{Offset}}{k} -
\mathrm{Factor}\left(\beta_j\\\mathrm{WoE}\_{ij} +
\frac{\alpha}{k}\right)\$\$

so the points of the bins an applicant falls into sum to their score.
The table is **fixed**: each bin rounds to one integer, the same for
everyone, because a bin whose value depended on who was being scored
would not be a scorecard. Rounding \\k\\ terms moves the total by at
most \\k/2\\ points against the unrounded model score; that drift is
measured and reported rather than hidden, and the card — the sum of the
integers — is the deployed definition.

### Engines

An engine is three functions: `fit(x, y, args)`, `link(object, x)`
returning the log-odds of the event, and `coef(object)` returning the
additive coefficients **or `NULL`**. Returning `NULL` declares the model
non-additive: gains, stability and score bands are still produced, but
no points table is, because a model that is not a sum of per-variable
terms cannot be decomposed into one. Supplying `engine` as a list of
those three functions is how a gradient-boosted or tidymodels fit is
plugged in without the package depending on either.

### What it refuses to do

The pipeline stops, rather than producing a plausible workbook, when the
shortlist is empty, when the fit does not converge, or when the
requested engine's package is missing. It drops constant predictors
before fitting and, by default, refits without any variable whose WoE
coefficient comes out negative. Everything it does silently absorb is
listed in `$warnings` and written to the workbook.

## References

Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
Intelligent Credit Scoring. *John Wiley & Sons*.
[doi:10.1002/9781119201731](https://doi.org/10.1002/9781119201731)

Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and
Its Applications. *SIAM*.
[doi:10.1137/1.9780898718317](https://doi.org/10.1137/1.9780898718317)

Anderson, R. (2007). The Credit Scoring Toolkit. *Oxford University
Press*.

## See also

[`obwoe_report`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_report.md)
to write the workbook,
[`predict.obwoe_scorecard`](https://evandeilton.github.io/OptimalBinningWoE/reference/predict.obwoe_scorecard.md)
to score new data,
[`obwoe_select`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_select.md)
for the screening rules,
[`obwoe_sql`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_sql.md)
for the deployment SQL,
[`obwoe_scale`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scale.md)
and
[`obwoe_psi`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_psi.md).

## Examples

``` r
# \donttest{
german <- read.csv(
  gzfile(system.file("extdata", "germancredit.csv.gz",
    package = "OptimalBinningWoE"
  )),
  stringsAsFactors = FALSE
)
german$default <- 1L - german$credit_risk
german$credit_risk <- NULL

sc <- obwoe_scorecard(german, target = "default", seed = 42)
#> Warning: 210 events for 15 variables (14.0 per variable): below the 20-events-per-variable rule of thumb, so the coefficients are unstable.
sc
#> Scorecard
#> ============================================================ 
#> Target      : default (event = 1)
#> Split       : stratified random, 70% training
#> Funnel      : 20 candidates -> 15 screened -> 15 in model
#> Engine      : glm (additive, points available)
#> Scaling     : PDO 20, 600 points at 50:1
#> 
#> Performance:
#>   sample   n events     ks   gini    auc
#>    train 699    210 0.5238 0.6244 0.8122
#>  holdout 301     90 0.3592 0.4455 0.7227
#> 
#> Points      : 64 rows; max card-vs-model drift 2.82 (bound 7.5)
#> 
#> 1 warning(s); see $warnings.

head(sc$points[, c("variable", "bin", "woe", "points")])
#>   variable                   bin         woe points
#> 1 duration       (-Inf;7.000000] -1.81150195     77
#> 2 duration  (7.000000;11.000000] -0.56165869     47
#> 3 duration (11.000000;16.000000] -0.25335733     40
#> 4 duration (16.000000;27.000000]  0.08396997     32
#> 5 duration (27.000000;36.000000]  0.61699631     20
#> 6 duration      (36.000000;+Inf]  0.91194633     13
sc$samples$train$metrics
#> $n
#> [1] 699
#> 
#> $events
#> [1] 210
#> 
#> $event_rate
#> [1] 0.3004292
#> 
#> $auc
#> [1] 0.8121823
#> 
#> $gini
#> [1] 0.6243646
#> 
#> $ks
#> [1] 0.5238095
#> 

# write the workbook
# obwoe_report(sc, file = "scorecard.xlsx")
# }
```
