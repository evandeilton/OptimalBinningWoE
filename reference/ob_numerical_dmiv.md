# Optimal Binning using Metric Divergence Measures (Zeng, 2013)

Performs supervised discretization of continuous numerical variables
using the theoretical framework proposed by Zeng (2013). This method
creates bins that maximize a specified divergence measure (e.g.,
Kullback-Leibler, Hellinger) between the distributions of positive and
negative cases, effectively maximizing the Information Value (IV) or
other discriminatory statistics.

## Usage

``` r
ob_numerical_dmiv(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  is_monotonic = TRUE,
  convergence_threshold = 1e-06,
  max_iterations = 1000,
  bin_method = c("woe1", "woe"),
  divergence_method = c("l2", "he", "kl", "tr", "klj", "sc", "js", "l1", "ln")
)
```

## Arguments

- feature:

  A numeric vector representing the continuous predictor variable.
  Missing values (`NA`/`NaN`) are excluded from the fit; `-Inf`/`Inf`
  fall in the first/last bin.

- target:

  An integer vector of binary outcomes (0/1) corresponding to each
  observation in `feature`. Must have the same length as `feature`.

- min_bins:

  Integer. The minimum number of bins to produce. Must be \\\ge\\ 2.
  Defaults to 3.

- max_bins:

  Integer. The maximum number of bins to produce. Must be \\\ge\\
  `min_bins`. Defaults to 5.

- bin_cutoff:

  Numeric. The minimum fraction of total observations required for a bin
  to be considered valid. Bins with frequency \< `bin_cutoff` will be
  merged. Value must be in (0, 1). Defaults to 0.05.

- max_n_prebins:

  Integer. The number of initial quantiles to generate during the
  pre-binning phase. Defaults to 20.

- is_monotonic:

  Logical. If `TRUE`, the algorithm enforces a strict monotonic
  relationship (increasing or decreasing) between the bin indices and
  their WoE values. Defaults to `TRUE`.

- convergence_threshold:

  Numeric. Accepted for backward compatibility and not used: merging
  always continues until the number of bins reaches `max_bins` (or
  `max_iterations` is exhausted). Defaults to 1e-6.

- max_iterations:

  Integer. Safety limit for the maximum number of merging iterations.
  Defaults to 1000.

- bin_method:

  Character string specifying the formula for Weight of Evidence
  calculation:

  - `"woe"`: Standard definition \\\ln((p_i/P) / (n_i/N))\\.

  - `"woe1"`: Zeng's definition \\\ln(p_i / n_i)\\ (direct log odds).

  Defaults to `"woe1"`.

- divergence_method:

  Character string specifying the divergence measure to maximize.
  Available options:

  - `"he"`: Hellinger Distance.

  - `"kl"`: Kullback-Leibler Divergence.

  - `"tr"`: Triangular Discrimination.

  - `"klj"`: Jeffrey's Divergence (Symmetric KL).

  - `"sc"`: Symmetric Chi-Square Divergence.

  - `"js"`: Jensen-Shannon Divergence.

  - `"l1"`: Manhattan Distance (L1 Norm).

  - `"l2"`: Euclidean Distance (L2 Norm).

  - `"ln"`: Chebyshev Distance (L-infinity Norm).

  Defaults to `"l2"`.

## Value

A list containing the binning results:

- `id`: Integer vector of bin identifiers.

- `bin`: Character vector of bin labels in interval notation.

- `woe`: Numeric vector of Weight of Evidence for each bin.

- `iv`: Numeric vector of standard Information Value per bin.

- `divergence`: Numeric vector of the chosen divergence contribution per
  bin.

- `count`: Integer vector of total observations per bin.

- `count_pos`: Integer vector of positive cases.

- `count_neg`: Integer vector of negative cases.

- `cutpoints`: Numeric vector of upper boundaries (excluding Inf).

- `converged`: Logical indicating if the algorithm converged.

- `iterations`: Integer count of merging iterations performed.

- `total_divergence`: The sum of the divergence measure across all bins.

- `total_iv`: The total standard Information Value.

- `bin_method`: The WoE calculation method used.

- `divergence_method`: The divergence measure used.

## Details

This algorithm implements the "Metric Divergence Measures" framework.
Unlike standard ChiMerge which uses statistical significance, this
method merges adjacent bins greedily to minimize the loss of a specific
divergence metric when merging bins.

**The Process:**

1.  **Pre-binning:** Generates granular bins based on quantiles.

2.  **Rare Merging:** Merges bins smaller than `bin_cutoff`.

3.  **Pure bins:** A bin with no events (or no non-events) stays
    isolated only if its purity is significant at the 5% level under the
    overall event rate \\r_0\\: \\(1 - r_0)^n \le 0.05\\ for a bin of
    \\n\\ observations without events (\\r_0^n \le 0.05\\ without
    non-events). Otherwise it is merged with the neighbour whose merge
    keeps the larger total divergence, while more than `min_bins` bins
    remain.

4.  **Monotonicity:** If `is_monotonic = TRUE`, forces the WoE trend to
    be monotonic by merging "violating" bins in the direction that
    maximizes the total divergence.

5.  **Optimization:** Iteratively merges a pair of adjacent bins until
    `max_bins` is reached. For the divergences `"he"`, `"kl"`, `"tr"`,
    `"klj"`, `"sc"` and `"js"` it is the pair whose merge keeps the
    largest total of the selected divergence. The distances `"l1"`,
    `"l2"` and `"ln"` are norms of the per-bin differences \\p_i -
    q_i\\, and merging two bins on the same side of \\p = q\\ never
    lowers them (L1 and L-infinity are unchanged, L2 grows), so the
    distance cannot rank those merges: among such pairs the engine
    merges the one whose event rates are least distinguishable (smallest
    Pearson chi-square of the \\2 \times 2\\ table), and the selected
    distance decides only when every adjacent pair lies on opposite
    sides of \\p = q\\.

## References

Zeng, G. (2013). Metric Divergence Measures and Information Value in
Credit Scoring. *Journal of the Operational Research Society*, 64(5),
712-731.

## Examples

``` r
# Example using the "he" (Hellinger) distance
set.seed(123)
feature <- rnorm(1000)
target <- rbinom(1000, 1, plogis(feature))

result <- ob_numerical_dmiv(feature, target,
  min_bins = 3,
  max_bins = 5,
  divergence_method = "he",
  bin_method = "woe"
)

print(result$bin)
#> [1] "(-Inf;-1.049677]"      "(-1.049677;-0.486452]" "(-0.486452;0.373732]" 
#> [4] "(0.373732;1.254752]"   "(1.254752;+Inf]"      
print(result$divergence)
#> [1] 0.0306124252 0.0126253512 0.0005722359 0.0156672829 0.0398556402
print(paste("Total Hellinger Distance:", round(result$total_divergence, 4)))
#> [1] "Total Hellinger Distance: 0.0993"
```
