# Write a Scorecard Workbook

Turns a fitted
[`obwoe_scorecard`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scorecard.md)
into a multi-sheet `.xlsx` file: the points table a branch officer
reads, the evidence a validation team asks for, and the SQL a deployment
engineer runs.

## Usage

``` r
obwoe_report(
  x,
  file,
  control = NULL,
  table = "your_table",
  dialect = "ansi",
  keep_columns = NULL
)
```

## Arguments

- x:

  An `"obwoe_scorecard"` object.

- file:

  Path to the `.xlsx` file to write.

- control:

  An `"obwoe_scorecard_control"` object; only `digits` and `overwrite`
  are used. Defaults to the settings stored on `x` where available.

- table:

  Character string naming the source table used in the generated SQL.
  Default `"your_table"`.

- dialect:

  SQL dialect for the deployment sheets, passed to
  [`obwoe_sql`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_sql.md).
  Default `"ansi"`.

- keep_columns:

  Columns the generated SQL carries through unchanged.

## Value

The path written, invisibly.

## Details

The workbook has one sheet per stage, in the order a reviewer reads
them:

|  |  |
|----|----|
| `01_Model_Summary` | provenance, the funnel, headline metrics |
| `02_Scorecard` | **the deliverable**: one row per variable and bin, with the integer points |
| `03_Coefficients` | the fit, with the sign check and standard errors |
| `04_Bin_Statistics` | the gains table of every binned variable used in training |
| `05_Screening` | every candidate and why it lived or died |
| `06_Correlations` | redundancy in the WoE space and what was pruned |
| `07_Score_Gains` | rank ordering per sample, with observed versus predicted |
| `08_Stability_PSI` | the score and every variable, per sample |
| `09_Cutoff_Strategy` | approval rate, bad rate and swap set by cutoff |
| `10_SQL_WoE` | [`obwoe_sql`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_sql.md) output: the WoE transform |
| `11_SQL_Points` | the same bins returning integer points, summed into the score |
| `12_Reproducibility` | the call, versions, and every warning raised |

Splitting the deployment SQL in two is deliberate. The WoE sheet
reproduces the model exactly and is what a data scientist re-scores
with; the points sheet reproduces the *card*, which is what the business
signed and what the decision engine should run.

## See also

[`obwoe_scorecard`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scorecard.md),
[`obwoe_sql`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_sql.md)

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

sc <- obwoe_scorecard(german, target = "default", seed = 1)
#> Warning: 210 events for 13 variables (16.2 per variable): below the 20-events-per-variable rule of thumb, so the coefficients are unstable.
obwoe_report(sc, file = file.path(tempdir(), "scorecard.xlsx"))
# }
```
