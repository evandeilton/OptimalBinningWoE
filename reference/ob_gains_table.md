# Compute Comprehensive Gains Table from Binning Results

This function serves as a high-performance engine (implemented in C++)
to calculate a comprehensive set of credit scoring and classification
metrics based on pre-aggregated binning results. It takes a list of bin
counts and computes metrics such as Information Value (IV), Weight of
Evidence (WoE), Kolmogorov-Smirnov (KS), Gini, Lift, and various
entropy-based divergence measures.

## Usage

``` r
ob_gains_table(binning_result)
```

## Arguments

- binning_result:

  A named `list` or `data.frame` containing the following atomic vectors
  (all must have the same length):

  `id`

  :   Numeric vector of bin identifiers. Determines the sort order for
      cumulative metrics (e.g., KS, Recall).

  `bin`

  :   Character vector of bin labels/intervals.

  `count`

  :   Numeric vector of total observations per bin (\\O_i\\).

  `count_pos`

  :   Numeric vector of positive (event) counts per bin (\\E_i\\).

  `count_neg`

  :   Numeric vector of negative (non-event) counts per bin (\\NE_i\\).

## Value

A `data.frame` with the following columns (metrics calculated per bin):

- **Identifiers**:

  `id`, `bin`

- **Counts & Rates**:

  `count`, `pos`, `neg`, `pos_rate` (\\\pi_i\\), `neg_rate`
  (\\1-\pi_i\\), `count_perc` (\\O_i / O\_{total}\\)

- **Distributions (Shares)**:

  `pos_perc` (\\D_1(i)\\: Share of Bad), `neg_perc` (\\D_0(i)\\: Share
  of Good)

- **Cumulative Statistics**:

  `cum_pos`, `cum_neg`, `cum_pos_perc` (\\CDF_1\\), `cum_neg_perc`
  (\\CDF_0\\), `cum_count_perc`

- **Credit Scoring Metrics**:

  `woe`, `iv`, `total_iv`, `ks`, `lift`, `odds_pos`, `odds_ratio`

- **Advanced Metrics**:

  `gini_contribution`, `log_likelihood`, `kl_divergence`,
  `js_divergence`

- **Classification Metrics**:

  `precision`, `recall`, `f1_score`

## Details

### Mathematical Definitions

Let \\E_i\\ and \\NE_i\\ be the number of events and non-events in bin
\\i\\, and \\E\_{total}\\, \\NE\_{total}\\ be the population totals.

**Weight of Evidence (WoE) & Information Value (IV):** \$\$WoE_i =
\ln\left(\frac{E_i / E\_{total}}{NE_i / NE\_{total}}\right)\$\$ \$\$IV_i
= \left(\frac{E_i}{E\_{total}} - \frac{NE_i}{NE\_{total}}\right) \times
WoE_i\$\$

**Kolmogorov-Smirnov (KS):** \$\$KS_i = \left\| \sum\_{j=1}^i
\frac{E_j}{E\_{total}} - \sum\_{j=1}^i \frac{NE_j}{NE\_{total}}
\right\|\$\$

**Lift:** \$\$Lift_i = \frac{E_i / (E_i + NE_i)}{E\_{total} /
(E\_{total} + NE\_{total})}\$\$

**Kullback-Leibler Divergence (Bernoulli):** Measures the divergence
between the bin's event rate \\p_i\\ and the global event rate
\\p\_{global}\\: \$\$KL_i = p_i
\ln\left(\frac{p_i}{p\_{global}}\right) + (1-p_i)
\ln\left(\frac{1-p_i}{1-p\_{global}}\right)\$\$

## Examples

``` r
# Manually constructed binning result
bin_res <- list(
  id = 1:3,
  bin = c("Low", "Medium", "High"),
  count = c(100, 200, 50),
  count_pos = c(5, 30, 20),
  count_neg = c(95, 170, 30)
)

gt <- ob_gains_table(bin_res)
print(gt[, c("bin", "woe", "iv", "ks")])
#>      bin         woe          iv        ks
#> 1    Low -1.26479681 0.292325919 0.2311248
#> 2 Medium -0.05495888 0.001693648 0.2619414
#> 3   High  1.27417706 0.333759785 0.0000000
```
