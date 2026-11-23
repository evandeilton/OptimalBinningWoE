# Optimal Binning for Categorical Variables using Order-Restricted Bayesian Evidence

Groups the categories of a feature by exact search on a Bayesian score
of a constant event rate per group. Categories are merged when the score
supports it beyond its sampling error. Reports posterior-mean WoE with
credible intervals.

## Usage

``` r
ob_categorical_orbe(
  feature,
  target,
  min_bins = 2,
  max_bins = 10,
  bin_cutoff = 0.05,
  max_n_prebins = 200,
  bin_separator = "%;%",
  prior_strength = 1,
  credible_level = 0.95,
  bin_penalty = 0,
  criterion = "predictive",
  se_rule = 1
)
```

## Arguments

- feature:

  A character vector or factor of categories. Missing values form the
  category `"NA"`.

- target:

  A binary vector (0/1 or logical) with the same length as `feature`.
  Missing values are an error.

- min_bins:

  Minimum number of bins (default: 2). It is honored whenever a grouping
  with that many bins satisfies `bin_cutoff` and has strictly increasing
  WoE; otherwise the largest feasible number of bins is returned. With
  `min_bins = 1` and `se_rule = 0` a single bin is returned when the
  score does not support any split.

- max_bins:

  Maximum number of bins (default: 10). The number of bins is selected
  by the score between `min_bins` and `max_bins`.

- bin_cutoff:

  Minimum fraction of the rows in a bin, in \\\[0, 1)\\ (default: 0.05).
  It is a constraint of the optimization. With `bin_cutoff = 0` rare
  categories are merged only by the score.

- max_n_prebins:

  Maximum number of atoms, the candidate units the bins are built from
  (default: 200). A feature with at most this many categories is
  searched over every boundary between categories in the order of their
  event rates. Above it, categories adjacent in that order are grouped
  into atoms of about equal frequency. Values above 1000 are treated as
  1000, and the limit is lowered further when `max_bins` is large, to
  bound memory.

- bin_separator:

  String placed between the categories of a merged bin in its label
  (default: `"%;%"`). A warning is issued when it occurs inside a
  category name.

- prior_strength:

  Weight of the prior, in pseudo-events of the minority class (default:
  1). Larger values shrink rare categories more strongly toward the
  overall event rate.

- credible_level:

  Probability content of the WoE credible intervals, in \\(0, 1)\\
  (default: 0.95).

- bin_penalty:

  Cost of each bin beyond the first, in units of the log score, at most
  1e6 (default: 0). Positive values give fewer bins and guard against
  splitting a feature unrelated to the target.

- criterion:

  Score of a bin: `"predictive"` (default), the leave-one-out log
  predictive probability of its rows, or `"evidence"`, its log marginal
  likelihood. `"predictive"` favors out-of-sample accuracy and returns
  more bins; `"evidence"` is more parsimonious. See Details.

- se_rule:

  Tolerance of the selection of the number of bins, in standard errors,
  in \\\[0, 10^6\]\\ (default: 1). Among the groupings whose score is
  within `se_rule` standard errors of the best score, the one with the
  most bins is returned. With `se_rule = 0` the grouping with the best
  score is returned. Used only with `criterion = "predictive"`. See
  Details.

## Value

A list with one element per bin in the vectors below, in increasing
order of WoE:

- id:

  Bin identifiers (1-based).

- bin:

  Bin labels: the categories joined by `bin_separator`.

- woe:

  Posterior-mean Weight of Evidence.

- iv:

  Information Value contribution, \\(p_i - q_i) \\ \mathrm{WoE}\_i\\,
  with \\p_i\\ and \\q_i\\ the observed shares of events and non-events.

- count, count_pos, count_neg:

  Rows, events and non-events.

- event_rate:

  Observed event rate.

- woe_se:

  Posterior standard deviation of the WoE.

- woe_lower, woe_upper:

  Limits of the equal-tailed credible interval of the WoE.

and the following:

- total_iv:

  Sum of `iv`.

- total_iv_corrected:

  `total_iv` minus the first-order sampling bias of a plug-in IV,
  floored at zero.

- log_score:

  Score of the grouping returned.

- log_score_by_bins:

  Score of the best grouping with \\1, 2, \ldots\\ bins; `NA` where no
  grouping is feasible.

- criterion:

  The criterion used.

- order_probability:

  Posterior probability that the event rates of the bins are in the
  reported order.

- converged:

  Always `TRUE`.

- iterations:

  Number of bin counts evaluated.

## Details

**Model.** The event rates of the groups are independent
\\\mathrm{Beta}(a, b)\\ with mean equal to the overall event rate
\\\pi_0\\ and \\\min(a, b)\\ equal to `prior_strength`. For a group with
\\n_i\\ rows and \\e_i\\ events, write \\A_i = a + e_i\\ and \\B_i = b +
n_i - e_i\\.

**Criterion.** A grouping is scored by \\\sum_i f(n_i, e_i)\\ minus
`bin_penalty` for each group beyond the first. With
`criterion = "predictive"`, \\f\\ is the leave-one-out log predictive
probability of the rows of the group, \$\$f(n_i, e_i) = e_i
\log\frac{A_i - 1}{A_i + B_i - 1} + (n_i - e_i) \log\frac{B_i - 1}{A_i +
B_i - 1} ,\$\$ and with `criterion = "evidence"` it is the log marginal
likelihood \$\$f(n_i, e_i) = \log \frac{B(A_i, B_i)}{B(a, b)} .\$\$ Both
scores fall when a group is split into groups with equal observed event
rates: by about one unit for the predictive score, and by an amount that
grows as \\\frac{1}{2}\log n_i\\ for the evidence, which therefore
returns fewer groups in large samples. The grouping is chosen among many
candidates, so with `bin_penalty = 0` a feature unrelated to the target
can still be split. A positive `bin_penalty` controls this: it
multiplies the prior weight of a grouping by \\e^{-\mathrm{penalty}}\\
for each additional group.

**Selection.** Let \\S_K\\ be the score of the best grouping with \\K\\
bins and \\K^\*\\ the number of bins with the largest score between
`min_bins` and `max_bins`. Two groupings that differ in score by less
than the sampling error of that difference predict equally well, and the
one with fewer bins discards differences between categories that the
other one keeps. With `criterion = "predictive"` the score is, apart
from the bin penalty, a sum over the rows, so the difference between the
scores of two groupings is a sum of row-wise differences \\d_1, \ldots,
d_N\\ with standard error \\\sqrt{N}\\ \mathrm{sd}(d)\\. The grouping
returned is the one with the largest \\K \ge K^\*\\ such that \$\$S_K
\ge S\_{K^\*} - \mathtt{se\\rule} \times \mathrm{se}(S_K - S\_{K^\*})
.\$\$ With the default `se_rule = 1`, categories are merged when keeping
them apart lowers the score by more than one standard error, and are
kept apart otherwise. `log_score` is the score of the grouping returned,
which can be below `max(log_score_by_bins)`. The standard error is large
in small samples, where the rule keeps more groups. A positive `se_rule`
also returns more groups for a feature unrelated to the target;
`se_rule = 0` or a positive `bin_penalty` gives the most parsimonious
grouping.

**Search.** Categories are ordered by their posterior-mean event rate
\\(a + e)/(a + b + n)\\, which shrinks rare categories toward \\\pi_0\\
before they are ordered. Ties are broken by count and then by name, so
the result does not depend on the row order. The engine then finds, by
dynamic programming, the grouping into bins contiguous in that order
that maximizes the score, subject to `bin_cutoff`, the limits on the
number of bins and strictly increasing posterior WoE. The search is
exact among groupings contiguous in that order; groupings that are not
contiguous are not examined.

**Rare categories.** Merging two groups changes the score by \\f(n_g +
n_h, e_g + e_h) - f(n_g, e_g) - f(n_h, e_h)\\. A category with few rows
rarely raises the score as a group of its own, so with `bin_cutoff = 0`
and `se_rule = 0` it is merged by the score, not by a frequency
threshold. With a positive `se_rule` it is kept apart when the loss in
score is within the tolerance.

**WoE.** As in
[`ob_numerical_orbe`](https://evandeilton.github.io/OptimalBinningWoE/dev/reference/ob_numerical_orbe.md),
\$\$\mathrm{WoE}\_i = \psi(A_i) - \psi(B_i) - \log\frac{\pi_0}{1 -
\pi_0},\$\$ with posterior variance \\\psi'(A_i) + \psi'(B_i)\\. The
credible intervals are conditional on the selected bins.

## References

Boullé, M. (2005). A Bayes optimal approach for partitioning the values
of categorical attributes. *Journal of Machine Learning Research*, 6,
1431-1452.

Geisser, S., & Eddy, W. F. (1979). A predictive approach to model
selection. *Journal of the American Statistical Association*, 74(365),
153-160.

## See also

[`ob_numerical_orbe`](https://evandeilton.github.io/OptimalBinningWoE/dev/reference/ob_numerical_orbe.md)
for numerical features and
[`ob_categorical_dp`](https://evandeilton.github.io/OptimalBinningWoE/dev/reference/ob_categorical_dp.md)
for dynamic programming on the Information Value.

## Examples

``` r
set.seed(123)
levels <- c(paste0("region_", 1:8), "rare_a", "rare_b")
prob <- c(rep(0.1, 8), 0.01, 0.01)
feature <- sample(levels, 3000, replace = TRUE, prob = prob)
rate <- c(0.05, 0.06, 0.10, 0.11, 0.20, 0.21, 0.35, 0.36, 0.15, 0.15)
target <- rbinom(3000, 1, rate[match(feature, levels)])

# bin_cutoff = 0 and se_rule = 0 leave the rare categories to the score
fit <- ob_categorical_orbe(feature, target, bin_cutoff = 0, se_rule = 0)

data.frame(
  bin = fit$bin, count = fit$count,
  woe = round(fit$woe, 3),
  lower = round(fit$woe_lower, 3),
  upper = round(fit$woe_upper, 3)
)
#>                            bin count    woe  lower  upper
#> 1          region_1%;%region_2   738 -1.095 -1.389 -0.818
#> 2          region_3%;%region_4   713 -0.809 -1.073 -0.558
#> 3 rare_a%;%region_6%;%region_5   771  0.040 -0.143  0.219
#> 4 rare_b%;%region_8%;%region_7   778  0.946  0.800  1.092

# The default se_rule = 1 keeps more categories apart
length(ob_categorical_orbe(feature, target, bin_cutoff = 0)$bin)
#> [1] 6
```
