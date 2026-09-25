# Optimal Binning for Numerical Variables using Sketch-based Algorithm

Implements optimal binning using the \*\*KLL Sketch\*\* (Karnin, Lang,
Liberty, 2016), a probabilistic data structure for quantile
approximation in data streams. This is the **only method in the
package** that uses a fundamentally different algorithmic approach
(streaming algorithms) compared to batch processing methods (MOB, MDLP,
etc.).

The sketch-based approach enables:

- **Sublinear space complexity**: O(k log N) vs O(N) for batch methods

- **Single-pass processing**: Suitable for streaming data

- **Provable approximation guarantees**: Quantile error \\\epsilon
  \approx O(1/k)\\

The method combines KLL Sketch for candidate generation with either
Dynamic Programming (for small N \<= 50) or greedy IV-based selection
(for larger datasets), followed by monotonicity enforcement via the Pool
Adjacent Violators Algorithm (PAVA).

## Usage

``` r
ob_numerical_sketch(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  monotonic = TRUE,
  convergence_threshold = 1e-06,
  max_iterations = 1000,
  sketch_k = 200
)
```

## Arguments

- feature:

  Numeric vector of feature values. Missing values (NA/NaN) are dropped
  silently: those rows are counted in no bin and the counts add up to
  the number of non-missing rows. `-Inf` and `+Inf` are kept as extreme
  values in the first and last bin and never become cutpoints. A feature
  whose values are all missing is an error. The sketch summarises the
  finite values; with fewer than two distinct finite values a single bin
  is returned.

- target:

  Integer vector of binary target values (must contain only 0 and 1).
  Must have the same length as `feature`. Missing values are not
  permitted.

- min_bins:

  Minimum number of bins (default: 3). Must be at least 2.

- max_bins:

  Maximum number of bins (default: 5). Must be \>= `min_bins`.

- bin_cutoff:

  Minimum fraction of total observations per bin (default: 0.05). Must
  be in (0, 1). Bins with fewer observations will be merged with
  neighbors.

- max_n_prebins:

  Accepted for compatibility with the other binning functions (default:
  20; must be in \[2, 1000\]). The candidate grid is a fixed set of
  about 40 sketch quantiles.

- monotonic:

  Logical flag to enforce WoE monotonicity (default: TRUE). Uses PAVA
  (Pool Adjacent Violators Algorithm) for enforcement. Direction
  (increasing/ decreasing) is automatically detected from the data.

- convergence_threshold:

  Convergence threshold (default: 1e-6). Accepted for compatibility: the
  selected cutpoints never exceed `max_bins - 1`, so no iterative
  bin-count reduction is needed.

- max_iterations:

  Maximum iterations for bin optimization (default: 1000). Prevents
  infinite loops in the optimization process.

- sketch_k:

  Integer parameter controlling sketch accuracy (default: 200). Larger
  values improve quantile precision but increase memory usage.
  **Approximation error**: observed rank error about \\1/k\\ of the
  sample size (200 → about 0.5%). **Valid range**: \[10, 1000\]. Typical
  values: 50 (fast), 200 (balanced), 500 (precise).

## Value

A list of class `c("OptimalBinningSketch", "OptimalBinning", "list")`
containing:

- id:

  Numeric vector of bin identifiers (1-based indexing).

- bin:

  Character vector of right-closed bin labels `"(lower;upper]"`; the
  first bin is labelled from `-Inf` and the last one up to `+Inf`.

- bin_lower:

  Numeric vector of lower bin boundaries: the observed minimum
  (included) for the first bin, the previous cutpoint (excluded)
  otherwise.

- bin_upper:

  Numeric vector of upper bin boundaries (included); the observed
  maximum for the last bin.

- woe:

  Numeric vector of Weight of Evidence values. Monotonic if
  `monotonic = TRUE`.

- iv:

  Numeric vector of Information Value contributions per bin.

- count:

  Integer vector of total observations per bin.

- count_pos:

  Integer vector of positive class (target = 1) counts per bin.

- count_neg:

  Integer vector of negative class (target = 0) counts per bin.

- total_iv:

  Total Information Value (sum of bin IVs).

- cutpoints:

  Numeric vector of bin split points (length = number of bins - 1).
  These are the internal boundaries between bins.

- converged:

  Logical flag indicating whether optimization converged.

- iterations:

  Integer number of optimization iterations performed.

## Details

**Algorithm Overview**

The sketch-based binning algorithm executes in four phases:

**Phase 1: KLL Sketch Construction**

The KLL Sketch maintains a compressed, multi-level representation of the
data distribution:

\$\$\text{Sketch} = \\\text{Compactor}\_0, \text{Compactor}\_1, \ldots,
\text{Compactor}\_L\\\$\$

where each \\\text{Compactor}\_\ell\\ stores items with weight
\\2^\ell\\. When a compactor exceeds capacity \\k\\ (controlled by
`sketch_k`), it is compacted.

**Theoretical Guarantees** (Karnin et al., 2016):

For a quantile \\q\\ with estimated value \\\hat{q}\\:

\$\$\|\text{rank}(\hat{q}) - q \cdot N\| \le \epsilon \cdot N\$\$

where \\\epsilon \approx O(1/k)\\ and space complexity is \\O(k
\log(N/k))\\.

This implementation compacts deterministically (the kept element of each
pair alternates between levels) instead of using KLL's random coin, so
results are reproducible without a seed. Its worst-case rank error is
\\O(\log(N/k)/k)\\; the observed error is about \\1/k\\. The quantiles
only propose candidate cutpoints: every bin statistic is then computed
exactly from the data.

**Phase 2: Candidate Extraction**

Approximately 40 quantiles are extracted from the sketch using a
non-uniform grid with higher resolution in distribution tails.

**Phase 3: Optimal Cutpoint Selection**

For small datasets (N \<= 50), Dynamic Programming maximizes total IV.
For larger datasets, a greedy IV-based selection is used.

**Phase 4: Bin Refinement**

Bins are refined through frequency constraint enforcement, monotonicity
enforcement (if requested). Bins that hold no observation are always
merged.

**Computational Complexity**

- **Time**: \\O(N \log k + N \times C + k^2 \times I)\\

- **Space**: \\O(k \log N)\\ for large N

**When to Use Sketch-based Binning**

- **Use**: Large datasets (N \> 10^6) with memory constraints or
  streaming data

- **Avoid**: Small datasets (N \< 1000) where approximation error may
  dominate

## References

- Karnin, Z., Lang, K., & Liberty, E. (2016). "Optimal Quantile
  Approximation in Streams". *Proceedings of the 57th Annual IEEE
  Symposium on Foundations of Computer Science (FOCS)*, 71-78.
  [doi:10.1109/FOCS.2016.20](https://doi.org/10.1109/FOCS.2016.20)

- Greenwald, M., & Khanna, S. (2001). "Space-efficient online
  computation of quantile summaries". *ACM SIGMOD Record*, 30(2), 58-66.
  [doi:10.1145/376284.375670](https://doi.org/10.1145/376284.375670)

- Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D.
  (1972). *Statistical Inference Under Order Restrictions*. Wiley.

- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and
  Implementing Intelligent Credit Scoring*. Wiley.
  [doi:10.1002/9781119201731](https://doi.org/10.1002/9781119201731)

## See also

[`ob_numerical_mdlp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mdlp.md),
[`ob_numerical_mblp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mblp.md)

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
# Example 1: Basic usage with simulated data
set.seed(123)
feature <- rnorm(500, mean = 100, sd = 20)
target <- rbinom(500, 1, prob = plogis((feature - 100) / 20))

result <- ob_numerical_sketch(
  feature = feature,
  target = target,
  min_bins = 3,
  max_bins = 5
)

# Display results
print(data.frame(
  Bin = result$id,
  Count = result$count,
  WoE = round(result$woe, 4),
  IV = round(result$iv, 4)
))
#>   Bin Count     WoE     IV
#> 1   1   199 -1.1027 0.4361
#> 2   2    24  0.0391 0.0001
#> 3   3   277  0.7370 0.2900

# Example 2: Comparing different sketch_k values
set.seed(456)
x <- rnorm(1000, 50, 15)
y <- rbinom(1000, 1, prob = 0.3)

result_k50 <- ob_numerical_sketch(x, y, sketch_k = 50)
result_k200 <- ob_numerical_sketch(x, y, sketch_k = 200)

cat("K=50 IV:", sum(result_k50$iv), "\n")
#> K=50 IV: 0.0126231 
cat("K=200 IV:", sum(result_k200$iv), "\n")
#> K=200 IV: 0.008483146 
# }
```
