# Internal: Area Under the ROC Curve for a Binned Score

Exact tie-corrected AUC of a score that is constant within each bin.
Bins sharing the same score value are merged before accumulation, so the
result equals both the trapezoidal area under the binned ROC curve and
the normalised Mann-Whitney U statistic.

\$\$AUC = \frac{1}{N_1 N_0} \sum\_{i} n\_{1,i} \left( \sum\_{j : s_j \<
s_i} n\_{0,j} + \tfrac{1}{2} n\_{0,i} \right)\$\$

## Usage

``` r
.ob_auc_ks_binned(pos, neg, score)
```

## Arguments

- pos:

  Integer/numeric vector of event counts per bin.

- neg:

  Integer/numeric vector of non-event counts per bin.

- score:

  Numeric vector giving the score attached to each bin. Higher values
  must indicate higher event propensity.

## Value

A list with `auc`, `gini` and `ks`, or `NA` components when the target
is degenerate.
