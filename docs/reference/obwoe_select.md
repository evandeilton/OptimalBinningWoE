# Automated Variable Selection for Weight of Evidence Scorecards

Screens every binned feature of an `"obwoe"` model against the two
criteria that govern variable admission in credit risk practice:
**predictive strength**, measured by the Information Value and graded
with the Siddiqi (2006) bands, and **guaranteed rank ordering**,
measured by monotonicity of the bin event rate. The function never drops
rows: a base with 500 candidate variables yields 500 rows, each carrying
a selection flag and the exact reason behind the decision, so the
analyst can override the automatic verdict at will.

## Usage

``` r
obwoe_select(
  obj,
  detail = c("summary", "full"),
  iv_min = 0.02,
  iv_max = 0.5,
  require_monotonic = c("numeric", "all", "none"),
  monotonicity = c("weak", "strict"),
  min_bins = 2L,
  max_bins = Inf,
  min_bin_pct = 0,
  allow_degenerate = FALSE,
  top_n = NULL,
  sort_by = c("iv", "ks", "gini", "auc", "feature", "none"),
  decreasing = TRUE,
  bin_separator = "%;%"
)
```

## Arguments

- obj:

  An object of class `"obwoe"` returned by
  [`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md).
  The target must be binary; multinomial models are reported as
  unsupported rather than silently mis-scored.

- detail:

  Character string controlling the granularity of the result:

  `"summary"`

  :   One row per feature with its headline metrics and the selection
      verdict (default).

  `"full"`

  :   One row per feature *and* optimised bin, carrying the complete
      gains table of every variable alongside the feature-level verdict.

- iv_min:

  Numeric. Minimum admissible total Information Value (inclusive).
  Default `0.02`, the *Unpredictive*/*Weak* boundary.

- iv_max:

  Numeric. Exclusive upper bound on the total Information Value. Default
  `0.50`, the *Suspicious* threshold: variables at or above it are
  rejected because such strength usually signals target leakage rather
  than genuine signal. Use `Inf` to disable.

- require_monotonic:

  Character string selecting where the ordering constraint applies:

  `"numeric"`

  :   Only numerical features must be monotonic (default). Nominal
      categories carry no intrinsic order, so their bins can always be
      sorted by WoE and are trivially rank-orderable.

  `"all"`

  :   Categorical features must also be monotonic in the bin order
      returned by the algorithm.

  `"none"`

  :   The ordering constraint is reported but not enforced.

- monotonicity:

  Character string. `"weak"` (default) accepts non-decreasing or
  non-increasing profiles; `"strict"` rejects ties between adjacent
  bins.

- min_bins:

  Integer. Minimum number of bins a feature must have to be selected.
  Default `2`.

- max_bins:

  Numeric. Maximum admissible number of bins. Default `Inf`.

- min_bin_pct:

  Numeric in \\\[0, 1)\\. Minimum population share of the smallest bin.
  Default `0` (no constraint); `0.05` is the usual scorecard convention.

- allow_degenerate:

  Logical. If `FALSE` (default), features with a bin containing no
  events or no non-events are rejected, since their WoE is not finite
  and the bin cannot be scored reliably out of sample.

- top_n:

  Integer or `NULL` (default). When supplied, only the `top_n` best
  features among those passing every rule stay selected; the remainder
  are flagged `"NOT_IN_TOP_N"`. Ranking follows `sort_by`, falling back
  to alphabetical order when `sort_by` names no metric.

- sort_by:

  Character string giving the ordering of the output and the ranking
  used by `top_n`: `"iv"` (default), `"ks"`, `"gini"`, `"auc"`,
  `"feature"` or `"none"`.

- decreasing:

  Logical. Sort the output in decreasing order? Default `TRUE`; ignored
  when `sort_by` is `"feature"` or `"none"`. It affects presentation
  only: `rank` and the `top_n` cut always keep the highest values of the
  ranking metric.

- bin_separator:

  Character string separating merged categories inside a categorical bin
  label. Default `"%;%"`, matching
  [`control.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/control.obwoe.md).

## Value

A `data.table` when the data.table package is installed and a
`data.frame` otherwise (`data.table` inherits from `data.frame`, so
either object supports the usual accessors).

With `detail = "summary"` the table holds one row per feature:

|  |  |
|----|----|
| **Column** | **Meaning** |
| `feature` | Variable name |
| `type` | `"numerical"` or `"categorical"` |
| `algorithm` | Binning algorithm used |
| `n_bins` | Number of optimised bins |
| `n_obs`, `n_pos`, `n_neg` | Population and class counts |
| `event_rate` | Overall event rate |
| `total_iv` | Total Information Value |
| `iv_class` | Siddiqi strength band |
| `ks` | Kolmogorov-Smirnov statistic in \\\[0, 1\]\\ |
| `gini`, `auc` | Discrimination of the binned score |
| `max_lift` | Largest bin lift over the base rate |
| `min_bin_pct`, `min_bin_count` | Smallest bin size |
| `n_degenerate_bins` | Bins with zero events or zero non-events |
| `woe_min`, `woe_max`, `woe_range` | Spread of the WoE profile |
| `monotonic`, `monotonic_strict` | Event-rate ordering flags |
| `monotonic_direction` | `"increasing"`, `"decreasing"`, `"constant"` or `"non-monotonic"` |
| `n_violations` | Adjacent pairs breaking the dominant trend |
| `spearman` | Rank correlation between bin order and event rate |
| `woe_monotonic` | Ordering of the WoE actually applied |
| `converged`, `iterations` | Optimiser diagnostics |
| `error`, `error_msg` | Binning failure flag and message |
| `quality` | Excellence tier (see Details) |
| `selected` | **Selection flag** |
| `reason` | **Machine-readable reason codes** |
| `reason_desc` | **Human-readable justification** |
| `rank` | Position under `sort_by` among selectable features |

With `detail = "full"` the table holds one row per feature *and*
optimised bin. Each row carries the complete gains table produced by
[`obwoe_gains_score`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_gains_score.md)
(`count`, `pos`, `neg`, `woe`, `iv`, `total_iv`, `pos_rate`, `lift`,
`cum_pos_perc`, `precision`, `recall`, `f1_score`, `kl_divergence`,
`js_divergence`, and so on) plus:

|  |  |
|----|----|
| `bin_id`, `bin` | Bin position and label |
| `bin_lower`, `bin_upper` | Interval bounds of a numerical bin, on the half-open convention \\(lower, upper\]\\ |
| `n_categories`, `categories` | Categories merged into a categorical bin |
| `woe_model` | WoE as reported by the binning algorithm, i.e. the value actually applied by [`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md) |
| `ks_bin` | Cumulative KS *at* this bin (the gains-table `ks`), renamed so that `ks` keeps its feature-level meaning |

The feature-level verdict (`iv_class`, `ks`, `gini`, `auc`, the
monotonicity flags, `quality`, `selected`, `reason`, `reason_desc`,
`rank`) is repeated on every bin row. Variables whose binning failed
have no bins and appear as a single placeholder row, so the row count
never hides a candidate variable.

## Details

### Predictive strength

The Information Value of a binned variable aggregates the divergence
between the event and non-event distributions:

\$\$IV = \sum\_{i=1}^{k} \left( \frac{n\_{1,i}}{N_1} -
\frac{n\_{0,i}}{N_0} \right) \ln\\\left( \frac{n\_{1,i}/N_1}
{n\_{0,i}/N_0} \right)\$\$

where \\n\_{1,i}\\ and \\n\_{0,i}\\ are the event and non-event counts
of bin \\i\\. This is the symmetrised Kullback-Leibler divergence
(Jeffreys divergence) between the two conditional distributions. Counts
are taken from the fitted binning, so the value is invariant to any
smoothing an individual algorithm may apply to its reported WoE.

Bands follow Siddiqi (2006): \\IV \< 0.02\\ unpredictive, \\\[0.02,
0.10)\\ weak, \\\[0.10, 0.30)\\ medium, \\\[0.30, 0.50)\\ strong, \\\ge
0.50\\ suspicious. The upper band is excluded by default: an IV above
0.5 on a single variable is, in practice, far more often a symptom of
leakage than of a genuinely dominant predictor.

### Guaranteed ordering

A variable is rank-ordering when its event rate moves in one direction
across the bin sequence. Writing the bin odds as \\\theta_i = n\_{1,i} /
n\_{0,i}\\, both quantities used in scorecards are strictly increasing
transformations of \\\theta_i\\:

\$\$\pi_i = \frac{n\_{1,i}}{n\_{1,i} + n\_{0,i}} = \frac{\theta_i}{1 +
\theta_i}, \qquad WoE_i = \ln \theta_i + \ln \frac{N_0}{N_1}\$\$

so monotonicity of the event rate \\\pi_i\\ and of the empirical WoE are
the same statement. The check is therefore performed once, on \\\pi_i\\,
which stays finite even when a bin holds a single class; the ordering of
the WoE actually applied by the algorithm is reported separately in
`woe_monotonic`.

For numerical variables the bin sequence is intrinsic (ordered
intervals), making monotonicity a genuine constraint. For nominal
categorical variables the sequence is a free relabelling, so a
non-monotonic profile can always be repaired by sorting; the default
`require_monotonic = "numeric"` reflects this asymmetry.

### Discrimination metrics

`ks`, `auc` and `gini` describe the *deployed* score, that is the WoE
the algorithm attaches to each bin — exactly what
[`obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
writes out and what a downstream logistic regression consumes. Bins are
ranked by that WoE and merged when tied, then

\$\$KS = \max_i \left\| F_1(i) - F_0(i) \right\|, \qquad AUC =
\frac{1}{N_1 N_0} \sum_i n\_{1,i} \left( \sum\_{j : s_j \< s_i}
n\_{0,j} + \tfrac{1}{2} n\_{0,i} \right), \qquad Gini = 2\\AUC - 1\$\$

where \\s_i\\ is the bin WoE and \\F_1, F_0\\ are the cumulative event
and non-event distributions in that order. The AUC expression is the
tie-corrected Mann-Whitney U statistic and reproduces, to machine
precision, both the trapezoidal area under the binned ROC curve and the
rank-based AUC computed on the WoE-transformed observations. Ranking by
WoE rather than by bin id makes `ks` the true KS of the transformed
variable even when the binning is not monotonic.

### Excellence tiers

`quality` summarises how cleanly the algorithm categorised a variable:

|  |  |
|----|----|
| **Tier** | **Definition** |
| `Excellent` | Selected, strictly monotonic, \\IV \ge 0.10\\, at least 3 bins, no degenerate bin, smallest bin \\\ge 5\\\\ |
| `Good` | Selected, monotonic, \\IV \ge 0.10\\ |
| `Fair` | Selected under every active rule but failing one of the structural refinements above |
| `Rejected` | Not selected |

### Reason codes

`reason` concatenates every violated rule with `";"`, so a variable
rejected on two counts reports both:

|                  |                                           |
|------------------|-------------------------------------------|
| `OK`             | Passed every active rule                  |
| `BINNING_ERROR`  | The algorithm failed on this variable     |
| `IV_BELOW_MIN`   | \\IV \<\\ `iv_min`                        |
| `IV_SUSPICIOUS`  | \\IV \ge\\ `iv_max` (leakage risk)        |
| `NOT_MONOTONIC`  | Event rate not ordered across bins        |
| `TOO_FEW_BINS`   | Fewer bins than `min_bins`                |
| `TOO_MANY_BINS`  | More bins than `max_bins`                 |
| `SMALL_BIN`      | Smallest bin below `min_bin_pct`          |
| `DEGENERATE_BIN` | A bin holds no events or no non-events    |
| `NOT_IN_TOP_N`   | Passed the rules but ranked below `top_n` |

## References

Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
Intelligent Credit Scoring. *John Wiley & Sons*.
[doi:10.1002/9781119201731](https://doi.org/10.1002/9781119201731)

Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and
Its Applications. *SIAM Monographs on Mathematical Modeling and
Computation*.
[doi:10.1137/1.9780898718317](https://doi.org/10.1137/1.9780898718317)

Hand, D. J., & Till, R. J. (2001). A Simple Generalisation of the Area
Under the ROC Curve for Multiple Class Classification Problems. *Machine
Learning*, 45(2), 171-186.
[doi:10.1023/A:1010920819831](https://doi.org/10.1023/A%3A1010920819831)

Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency.
*The Annals of Mathematical Statistics*, 22(1), 79-86.
[doi:10.1214/aoms/1177729694](https://doi.org/10.1214/aoms/1177729694)

## See also

[`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
for fitting the binning,
[`obwoe_sql`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_sql.md)
for exporting the selected variables as SQL,
[`obwoe_gains_score`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_gains_score.md)
for the underlying bin-level engine,
[`summary.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/summary.obwoe.md)
for the compact model overview.

## Examples

``` r
# \donttest{
set.seed(42)
n <- 2000
df <- data.frame(
  age = rnorm(n, 40, 12),
  income = exp(rnorm(n, 10, 0.7)),
  noise = rnorm(n),
  region = sample(c("N", "S", "E", "W"), n, replace = TRUE)
)
df$target <- rbinom(n, 1, plogis(-1.2 + 0.05 * (df$age - 40)))

model <- obwoe(df, target = "target", max_bins = 6)

# One row per variable, every candidate kept
sel <- obwoe_select(model)
sel[, c("feature", "total_iv", "iv_class", "ks", "monotonic", "selected", "reason")]
#>    feature    total_iv     iv_class         ks monotonic selected       reason
#>     <char>       <num>       <fctr>      <num>    <lgcl>   <lgcl>       <char>
#> 1:     age 0.229744253       Medium 0.13648979      TRUE     TRUE           OK
#> 2:   noise 0.005735885 Unpredictive 0.01852158      TRUE    FALSE IV_BELOW_MIN
#> 3:  income 0.002517288 Unpredictive 0.02450547      TRUE    FALSE IV_BELOW_MIN
#> 4:  region 0.001436911 Unpredictive 0.01814545      TRUE    FALSE IV_BELOW_MIN

# Full gains detail for every optimised bin
det <- obwoe_select(model, detail = "full")
head(det[, c("feature", "bin", "count", "pos_rate", "woe", "iv", "selected")])
#>    feature                   bin count  pos_rate         woe           iv
#>     <char>                <char> <num>     <num>       <num>        <num>
#> 1:     age      (-Inf;20.248542]   100 0.0600000 -1.52620628 0.0726263739
#> 2:     age (20.248542;24.835653]   100 0.1300000 -0.67562973 0.0186743347
#> 3:     age (24.835653;29.663522]   200 0.1350000 -0.63212570 0.0331425501
#> 4:     age (29.663522;52.019257]  1300 0.2246154 -0.01364062 0.0001204926
#> 5:     age (52.019257;58.908730]   200 0.3750000  0.71450341 0.0602643766
#> 6:     age      (58.908730;+Inf]   100 0.4100000  0.86136365 0.0449161254
#>    selected
#>      <lgcl>
#> 1:     TRUE
#> 2:     TRUE
#> 3:     TRUE
#> 4:     TRUE
#> 5:     TRUE
#> 6:     TRUE

# Stricter policy: monotonic everywhere, 5% minimum bin, top 10 by KS
strict <- obwoe_select(model,
  require_monotonic = "all", monotonicity = "strict",
  min_bin_pct = 0.05, top_n = 10, sort_by = "ks"
)
# }
```
