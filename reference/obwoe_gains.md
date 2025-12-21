# Gains Table Statistics for Credit Risk Scorecard Evaluation

Computes a comprehensive gains table (also known as a lift table or
decile analysis) for evaluating the discriminatory power of credit
scoring models and optimal binning transformations. The gains table is a
fundamental tool in credit risk management for model validation, cutoff
selection, and regulatory reporting (Basel II/III, IFRS 9).

This function accepts three types of input:

1.  An `"obwoe"` object from
    [`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
    (uses stored binning)

2.  A `data.frame` from
    [`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
    (uses bin/WoE columns)

3.  Any `data.frame` with a grouping variable (e.g., score deciles)

## Usage

``` r
obwoe_gains(
  obj,
  target = NULL,
  feature = NULL,
  use_column = c("auto", "bin", "woe", "direct"),
  sort_by = c("id", "woe", "event_rate", "bin"),
  n_groups = NULL
)
```

## Arguments

- obj:

  Input object: an `"obwoe"` object, a `data.frame` from
  [`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md),
  or any `data.frame` containing a grouping variable and target values.

- target:

  Integer vector of binary target values (0/1) or the name of the target
  column in `obj`. Required for `data.frame` inputs. For `"obwoe"`
  objects, the target is extracted automatically.

- feature:

  Character string specifying the feature/variable to analyze. For
  `"obwoe"` objects: defaults to the feature with highest IV. For
  `data.frame` objects: can be any column name representing groups
  (e.g., `"age_bin"`, `"age_woe"`, `"score_decile"`).

- use_column:

  Character string specifying which column type to use when `obj` is a
  `data.frame` from
  [`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md):

  `"bin"`

  :   Use the `<feature>_bin` column (default)

  `"woe"`

  :   Use the `<feature>_woe` column (groups by WoE values)

  `"auto"`

  :   Automatically detect: use `_bin` if available

  `"direct"`

  :   Use the `feature` column name directly (for any variable)

- sort_by:

  Character string specifying sort order for bins:

  `"woe"`

  :   Descending WoE (highest risk first) - default

  `"event_rate"`

  :   Descending event rate

  `"bin"`

  :   Alphabetical/natural order

- n_groups:

  Integer. For continuous variables (e.g., scores), the number of groups
  (deciles) to create. Default is `NULL` (use existing groups). Set to
  10 for standard decile analysis.

## Value

An S3 object of class `"obwoe_gains"` containing:

- `table`:

  Data frame with 18 statistics per bin (see Details)

- `metrics`:

  Named list of global performance metrics:

  `ks`

  :   Kolmogorov-Smirnov statistic (%)

  `gini`

  :   Gini coefficient (%)

  `auc`

  :   Area Under ROC Curve

  `total_iv`

  :   Total Information Value

  `ks_bin`

  :   Bin where maximum KS occurs

- `feature`:

  Feature/variable name analyzed

- `n_bins`:

  Number of bins/groups

- `n_obs`:

  Total observations

- `event_rate`:

  Overall event rate

## Details

### Gains Table Construction

The gains table is constructed by:

1.  Sorting observations by risk score or WoE (highest risk first)

2.  Grouping into bins (pre-defined or created via quantiles)

3.  Computing bin-level and cumulative statistics

The table enables assessment of model rank-ordering ability: a
well-calibrated model should show monotonically increasing event rates
as risk score increases.

### Bin-Level Statistics (18 metrics)

|                |                                |                                  |
|----------------|--------------------------------|----------------------------------|
| **Column**     | **Formula**                    | **Description**                  |
| `bin`          | \-                             | Bin label or interval            |
| `count`        | \\n_i\\                        | Total observations in bin        |
| `count_pct`    | \\n_i / N\\                    | Proportion of total population   |
| `pos_count`    | \\n\_{i,1}\\                   | Event count (Bad, target=1)      |
| `neg_count`    | \\n\_{i,0}\\                   | Non-event count (Good, target=0) |
| `pos_rate`     | \\n\_{i,1} / n_i\\             | Event rate (Bad rate) in bin     |
| `neg_rate`     | \\n\_{i,0} / n_i\\             | Non-event rate (Good rate)       |
| `pos_pct`      | \\n\_{i,1} / N_1\\             | Distribution of events           |
| `neg_pct`      | \\n\_{i,0} / N_0\\             | Distribution of non-events       |
| `odds`         | \\n\_{i,1} / n\_{i,0}\\        | Odds of event                    |
| `log_odds`     | \\\ln(\text{odds})\\           | Log-odds (logit)                 |
| `woe`          | \\\ln(p_i / q_i)\\             | Weight of Evidence               |
| `iv`           | \\(p_i - q_i) \cdot WoE_i\\    | Information Value contribution   |
| `cum_pos_pct`  | \\\sum\_{j \le i} p_j\\        | Cumulative events captured       |
| `cum_neg_pct`  | \\\sum\_{j \le i} q_j\\        | Cumulative non-events            |
| `ks`           | \\\|F_1(i) - F_0(i)\|\\        | KS statistic at bin              |
| `lift`         | \\\text{pos\\rate} / \bar{p}\\ | Lift over random                 |
| `capture_rate` | \\cum\\pos\\pct\\              | Cumulative capture rate          |

### Global Performance Metrics

**Kolmogorov-Smirnov (KS) Statistic**: Maximum absolute difference
between cumulative distributions of events and non-events. Measures the
model's ability to separate populations.

\$\$KS = \max_i \|F_1(i) - F_0(i)\|\$\$

|              |                                     |
|--------------|-------------------------------------|
| **KS Range** | **Interpretation**                  |
| \< 20%       | Poor discrimination                 |
| 20-40%       | Acceptable                          |
| 40-60%       | Good                                |
| 60-75%       | Very good                           |
| \> 75%       | Excellent (verify for data leakage) |

**Gini Coefficient**: Measure of inequality between event and non-event
distributions. Equivalent to 2\*AUC - 1, representing the area between
the Lorenz curve and the line of equality.

\$\$Gini = 2 \times AUC - 1\$\$

**Area Under ROC Curve (AUC)**: Probability that a randomly chosen event
is ranked higher than a randomly chosen non-event. Computed via the
trapezoidal rule.

**Total Information Value (IV)**: Sum of IV contributions across all
bins. See
[`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
for interpretation guidelines.

### Use Cases

**Model Validation**: Verify rank-ordering (monotonic event rates) and
acceptable KS/Gini.

**Cutoff Selection**: Identify the bin where the model provides optimal
separation for business rules (e.g., auto-approve above score X).

**Population Stability**: Compare gains tables over time to detect model
drift.

**Regulatory Reporting**: Generate metrics required by Basel II/III and
IFRS 9 frameworks.

## References

Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
Intelligent Credit Scoring. *John Wiley & Sons*.
[doi:10.1002/9781119201731](https://doi.org/10.1002/9781119201731)

Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and
Its Applications. *SIAM Monographs on Mathematical Modeling and
Computation*.
[doi:10.1137/1.9780898718317](https://doi.org/10.1137/1.9780898718317)

Anderson, R. (2007). The Credit Scoring Toolkit: Theory and Practice for
Retail Credit Risk Management. *Oxford University Press*.

Hand, D. J., & Henley, W. E. (1997). Statistical Classification Methods
in Consumer Credit Scoring: A Review. *Journal of the Royal Statistical
Society: Series A*, 160(3), 523-541.
[doi:10.1111/j.1467-985X.1997.00078.x](https://doi.org/10.1111/j.1467-985X.1997.00078.x)

## See also

[`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
for optimal binning,
[`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
for scoring new data,
[`plot.obwoe_gains`](https://evandeilton.github.io/OptimalBinningWoE/reference/plot.obwoe_gains.md)
for visualization (cumulative gains, KS, lift).

## Examples

``` r
# \donttest{
# =============================================================================
# Example 1: From obwoe Object (Standard Usage)
# =============================================================================
set.seed(42)
n <- 1000
df <- data.frame(
  age = rnorm(n, 40, 15),
  income = exp(rnorm(n, 10, 0.8)),
  score = rnorm(n, 600, 100),
  target = rbinom(n, 1, 0.15)
)

model <- obwoe(df, target = "target")
gains <- obwoe_gains(model, feature = "age")
print(gains)
#> Gains Table: age 
#> ================================================== 
#> 
#> Observations: 1000  |  Bins: 4
#> Total IV: 0.0180
#> 
#> Performance Metrics:
#>   KS Statistic: 5.47%
#>   Gini Coefficient: 6.76%
#>   AUC: 0.4662
#> 
#>                    bin count pos_rate     woe     iv cum_pos_pct   ks lift
#>       (-Inf;37.699632]   450   14.00% -0.1270 0.0069       40.4% 5.5%  0.9
#>  (37.699632;59.237138]   450   16.22%  0.0465 0.0010       87.2% 3.3%  0.9
#>  (59.237138;62.999207]    50   20.00%  0.3020 0.0050       93.6% 1.7%  0.9
#>       (62.999207;+Inf]    50   20.00%  0.3020 0.0050      100.0% 0.0%  0.9

# Access metrics
cat("KS:", gains$metrics$ks, "%\n")
#> KS: 5.468465 %
cat("Gini:", gains$metrics$gini, "%\n")
#> Gini: 6.759631 %

# =============================================================================
# Example 2: From obwoe_apply Output - Using Bin Column
# =============================================================================
scored <- obwoe_apply(df, model)

# Default: uses age_bin column
gains_bin <- obwoe_gains(scored,
  target = df$target, feature = "age",
  use_column = "bin"
)

# =============================================================================
# Example 3: From obwoe_apply Output - Using WoE Column
# =============================================================================
# Group by WoE values (continuous analysis)
gains_woe <- obwoe_gains(scored,
  target = df$target, feature = "age",
  use_column = "woe", n_groups = 5
)
#> Warning: NAs introduced by coercion

# =============================================================================
# Example 4: Any Variable - Score Decile Analysis
# =============================================================================
# Create score deciles manually
df$score_decile <- cut(df$score,
  breaks = quantile(df$score, probs = seq(0, 1, 0.1)),
  include.lowest = TRUE, labels = 1:10
)

# Analyze score deciles directly
gains_score <- obwoe_gains(df,
  target = "target", feature = "score_decile",
  use_column = "direct"
)
print(gains_score)
#> Gains Table: score_decile 
#> ================================================== 
#> 
#> Observations: 1000  |  Bins: 10
#> Total IV: 0.0517
#> 
#> Performance Metrics:
#>   KS Statistic: 5.16%
#>   Gini Coefficient: 0.91%
#>   AUC: 0.5046
#> 
#>  bin count pos_rate     woe     iv cum_pos_pct   ks lift
#>    1   100   15.00% -0.0463 0.0002        9.6% 0.5% 0.96
#>    2   100   15.00% -0.0463 0.0002       19.2% 0.9% 0.96
#>    3   100   14.00% -0.1270 0.0015       28.2% 2.1% 0.96
#>    4   100   19.00%  0.2383 0.0062       40.4% 0.5% 0.96
#>    5   100   13.00% -0.2127 0.0042       48.7% 1.5% 0.96
#>    6   100   21.00%  0.3634 0.0149       62.2% 2.6% 0.96
#>    7   100   19.00%  0.2383 0.0062       74.4% 5.2% 0.96
#>    8   100   11.00% -0.4024 0.0141       81.4% 1.7% 0.96
#>    9   100   13.00% -0.2127 0.0042       89.7% 0.3% 0.96
#>   10   100   16.00%  0.0301 0.0001      100.0% 0.0% 0.96

# =============================================================================
# Example 5: Automatic Decile Creation
# =============================================================================
# Use n_groups to automatically create quantile groups
gains_auto <- obwoe_gains(df,
  target = "target", feature = "score",
  use_column = "direct", n_groups = 10
)
# }
```
