# Optimal Binning for Numerical Variables using Order-Restricted Bayesian Evidence

Chooses the cutpoints and the number of bins that maximize a Bayesian
score of a piecewise-constant event rate, with the shape of the Weight
of Evidence (monotone, peak or valley) enforced inside an exact dynamic
program. Reports posterior-mean WoE with credible intervals.

## Usage

``` r
ob_numerical_orbe(
  feature,
  target,
  min_bins = 2,
  max_bins = 10,
  bin_cutoff = 0.05,
  max_n_prebins = 200,
  monotonic_trend = "auto",
  prior_strength = 1,
  credible_level = 0.95,
  bin_penalty = 0,
  criterion = "predictive",
  se_rule = 0
)
```

## Arguments

- feature:

  A numeric vector of feature values. `NA` and `NaN` rows are excluded
  from the fit. `-Inf` and `+Inf` are kept and fall in the first and
  last bin.

- target:

  A binary vector (0/1 or logical) with the same length as `feature`.
  Missing values are an error.

- min_bins:

  Minimum number of bins (default: 2). It is honored whenever a
  partition with that many bins satisfies the shape and `bin_cutoff`;
  otherwise the largest feasible number of bins is returned. With
  `min_bins = 1` a single bin is returned when the score does not
  support any cut.

- max_bins:

  Maximum number of bins (default: 10). The number of bins is selected
  by the score between `min_bins` and `max_bins`, so `max_bins` is an
  upper limit, not a target.

- bin_cutoff:

  Minimum fraction of the non-missing rows in a bin, in \\\[0, 1)\\
  (default: 0.05). It is a constraint of the optimization.

- max_n_prebins:

  Maximum number of atoms, the candidate units the bins are built from
  (default: 200). A feature with at most this many distinct values is
  searched over every boundary between distinct values. Above it,
  adjacent distinct values are grouped into atoms of about equal
  frequency, which restricts the candidate cutpoints. Values above 1000
  are treated as 1000, and the limit is lowered further when `max_bins`
  is large, to bound memory.

- monotonic_trend:

  Shape of the WoE across bins: `"auto"` (default; ascending or
  descending, whichever has the larger score), `"ascending"`,
  `"descending"`, `"peak"` (increasing then decreasing), `"valley"`
  (decreasing then increasing) or `"none"`.

- prior_strength:

  Weight of the prior, in pseudo-events of the minority class (default:
  1). Larger values shrink the WoE of small bins more strongly toward
  zero.

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
  more bins; `"evidence"` is more parsimonious and, with a positive
  `bin_penalty`, recovers the number of distinct event-rate levels more
  often. See Details.

- se_rule:

  Tolerance of the selection of the number of bins, in standard errors,
  in \\\[0, 10^6\]\\ (default: 0). With `se_rule = 0` the partition with
  the best score is returned. With a positive value, the partition with
  the most bins among those whose score is within `se_rule` standard
  errors of the best score is returned. Used only with
  `criterion = "predictive"`. See Details.

## Value

A list with one element per bin in the vectors below:

- id:

  Bin identifiers (1-based).

- bin:

  Interval labels, e.g. `"(-Inf;1.500000]"`.

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

- cutpoints:

  Upper limits of all bins but the last.

- total_iv:

  Sum of `iv`.

- total_iv_corrected:

  `total_iv` minus the first-order sampling bias of a plug-in IV,
  floored at zero.

- log_score:

  Score of the selected partition.

- log_score_by_bins:

  Score of the best partition with \\1, 2, \ldots\\ bins under the
  selected shape; `NA` where no partition is feasible.

- criterion:

  The criterion used.

- monotonic_trend:

  Shape of the selected model.

- order_probability:

  Posterior probability that the bin event rates follow the selected
  shape.

- converged:

  Always `TRUE`: the search is exact.

- iterations:

  Number of bin counts evaluated.

## Details

**Model.** Let bin \\i\\ hold \\n_i\\ rows and \\e_i\\ events, and let
\\\pi_0\\ be the overall event rate. The event rates of the bins are
independent \\\mathrm{Beta}(a, b)\\ with mean \\\pi_0\\ and \\\min(a,
b)\\ equal to `prior_strength`, which is the uniform prior when \\\pi_0
= 0.5\\. Write \\A_i = a + e_i\\ and \\B_i = b + n_i - e_i\\.

**Criterion.** A partition is scored by \\\sum_i f(n_i, e_i)\\ minus
`bin_penalty` for each bin beyond the first. With
`criterion = "predictive"`, \\f\\ is the leave-one-out log predictive
probability of the rows of the bin, each row predicted from the
posterior given the other rows of its bin: \$\$f(n_i, e_i) = e_i
\log\frac{A_i - 1}{A_i + B_i - 1} + (n_i - e_i) \log\frac{B_i - 1}{A_i +
B_i - 1} .\$\$ With `criterion = "evidence"`, \\f\\ is the log marginal
likelihood \$\$f(n_i, e_i) = \log \frac{B(A_i, B_i)}{B(a, b)} .\$\$ Both
scores fall when a bin is split into bins with equal observed event
rates: by about one unit for the predictive score, and by an amount that
grows as \\\frac{1}{2}\log n_i\\ for the evidence, which therefore
returns fewer bins in large samples. The cutpoints are chosen among many
candidates, so with `bin_penalty = 0` a feature unrelated to the target
can still be split, more often under `"predictive"`. A positive
`bin_penalty` controls this: it multiplies the prior weight of a
partition by \\e^{-\mathrm{penalty}}\\ for each additional bin.

**Optimization.** The engine maximizes the score over all partitions
into contiguous bins that respect `bin_cutoff`, the limits on the number
of bins and the shape. The state of the dynamic program is the last bin
of a partial partition, and a bin may follow another only when its
posterior WoE respects the shape, so the reported WoE satisfies the
shape strictly. The search is exact over the atoms and takes \\O(K u^2
\log u)\\ time for \\u\\ atoms.

**Order restriction.** Restricting \\K\\ exchangeable rates to an order
multiplies the marginal likelihood by the ratio of the posterior to the
prior probability of that order. The prior probability is \\1/K!\\ for a
given monotone direction, \\2/K!\\ for `"auto"` and \\2^{K-1}/K!\\ for a
peak or a valley. The logarithm of this ratio is added to the score of
the best partition of each size before the number of bins and, for
`"auto"`, the direction are selected. `order_probability` is the
posterior probability in that ratio.

**Selection.** Let \\S_K\\ be the score of the best partition with \\K\\
bins and \\K^\*\\ the number of bins with the largest score between
`min_bins` and `max_bins`. With `criterion = "predictive"` and a
positive `se_rule`, the partition returned is the one with the largest
\\K \ge K^\*\\ such that \\S_K \ge S\_{K^\*} - \mathtt{se\\rule} \times
\mathrm{se}\\, where \\\mathrm{se} = \sqrt{N}\\ \mathrm{sd}(d)\\ and
\\d_1, \ldots, d_N\\ are the row-wise differences between the
leave-one-out log predictive probabilities of the two partitions. The
bin penalty and the order-restriction term enter \\S_K\\ but not the
standard error. For `"auto"` the best partition of each direction is a
candidate for each \\K\\, the one with the larger score first, and the
direction returned is that of the selected partition.

**WoE.** The reported WoE is the posterior mean \$\$\mathrm{WoE}\_i =
\psi(A_i) - \psi(B_i) - \log\frac{\pi_0}{1 - \pi_0},\$\$ where \\\psi\\
is the digamma function. Its posterior variance is \\\psi'(A_i) +
\psi'(B_i)\\ and the interval limits are the logits of the Beta
quantiles. The WoE is finite for bins without events or without
non-events. The credible intervals are conditional on the selected bins.

## References

Boullé, M. (2006). MODL: A Bayes optimal discretization method for
continuous attributes. *Machine Learning*, 65(1), 131-165.

Geisser, S., & Eddy, W. F. (1979). A predictive approach to model
selection. *Journal of the American Statistical Association*, 74(365),
153-160.

Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D.
(1972). *Statistical inference under order restrictions*. John Wiley &
Sons.

## See also

[`ob_categorical_orbe`](https://evandeilton.github.io/OptimalBinningWoE/dev/reference/ob_categorical_orbe.md)
for categorical features,
[`ob_numerical_ir`](https://evandeilton.github.io/OptimalBinningWoE/dev/reference/ob_numerical_ir.md)
for isotonic regression binning and
[`ob_numerical_mdlp`](https://evandeilton.github.io/OptimalBinningWoE/dev/reference/ob_numerical_mdlp.md)
for entropy-based binning.

## Examples

``` r
set.seed(123)
feature <- rnorm(2000)
target <- rbinom(2000, 1, plogis(-2 + 0.8 * feature))

fit <- ob_numerical_orbe(feature, target)

# Bins, posterior-mean WoE and its credible interval
data.frame(
  bin = fit$bin, count = fit$count,
  woe = round(fit$woe, 3),
  lower = round(fit$woe_lower, 3),
  upper = round(fit$woe_upper, 3)
)
#>                     bin count    woe  lower  upper
#> 1      (-Inf;-1.185480]   230 -1.703 -2.550 -0.993
#> 2 (-1.185480;-0.457761]   390 -1.100 -1.578 -0.667
#> 3 (-0.457761;-0.191775]   200 -0.649 -1.194 -0.159
#> 4 (-0.191775;-0.060822]   110 -0.169 -0.767  0.368
#> 5  (-0.060822;0.354352]   320 -0.037 -0.369  0.275
#> 6   (0.354352;0.818828]   310  0.314  0.017  0.597
#> 7   (0.818828;1.181181]   200  0.451  0.099  0.786
#> 8   (1.181181;1.532424]   110  0.781  0.353  1.188
#> 9       (1.532424;+Inf]   130  1.330  0.979  1.674

# Score of the best partition for each number of bins
round(fit$log_score_by_bins, 1)
#>  [1] -790.7 -742.3 -726.4 -722.6 -720.0 -717.7 -716.1 -715.3 -714.7 -714.7
fit$monotonic_trend
#> [1] "ascending"

# A more parsimonious fit
fit_ev <- ob_numerical_orbe(feature, target, criterion = "evidence")
length(fit_ev$bin)
#> [1] 8

# A non-monotone relationship
target2 <- rbinom(2000, 1, plogis(-1 - feature^2))
fit2 <- ob_numerical_orbe(feature, target2, monotonic_trend = "peak")
round(fit2$woe, 3)
#>  [1] -3.582 -0.908  0.064  0.456  0.889  0.365  0.127 -0.643 -1.159 -3.582
```
