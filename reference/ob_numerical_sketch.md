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
  special_codes = "",
  monotonic = TRUE,
  convergence_threshold = 1e-06,
  max_iterations = 1000,
  sketch_k = 200
)
```

## Arguments

- feature:

  Numeric vector of feature values. Missing values (NA) are **not
  permitted** and will trigger an error. For streaming applications,
  pre-filter NAs.

- target:

  Integer vector of binary target values (must contain only 0 and 1).
  Must have the same length as `feature`.

- min_bins:

  Minimum number of bins (default: 3). Must be at least 2.

- max_bins:

  Maximum number of bins (default: 5). Must be \>= `min_bins`.

- bin_cutoff:

  Minimum fraction of total observations per bin (default: 0.05). Must
  be in (0, 1).

- special_codes:

  Character string for special value handling (default: ""). Currently
  unused; reserved for future extensions.

- monotonic:

  Logical flag to enforce WoE monotonicity (default: TRUE). Uses PAVA
  for enforcement.

- convergence_threshold:

  Convergence threshold for IV change (default: 1e-6).

- max_iterations:

  Maximum iterations for bin optimization (default: 1000).

- sketch_k:

  Integer parameter controlling sketch accuracy (default: 200). Larger
  values improve quantile precision but increase memory. Typical range:
  50-500. **Approximation error**: \\\epsilon \approx 1/k\\ (200 → 0.5%
  error).

## Value

A list containing:

- id:

  Integer bin identifiers (1-based).

- bin_lower:

  Numeric lower bounds of bins.

- bin_upper:

  Numeric upper bounds of bins.

- woe:

  Numeric WoE values (monotonic if `monotonic = TRUE`).

- iv:

  Numeric IV contributions per bin.

- count:

  Integer total observations per bin.

- count_pos:

  Integer positive class counts.

- count_neg:

  Integer negative class counts.

- cutpoints:

  Numeric vector of bin boundaries (internal splits only).

- converged:

  Logical convergence flag.

- iterations:

  Integer iteration count.

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
`sketch_k`), it is compacted:

1.  Sort items in \\\text{Compactor}\_\ell\\

2.  Merge adjacent pairs based on level parity:

    - Even levels (\\\ell \bmod 2 = 0\\): Merge pairs at even indices

    - Odd levels (\\\ell \bmod 2 = 1\\): Merge pairs at odd indices

3.  Promote merged items to \\\text{Compactor}\_{\ell+1}\\

**Theoretical Guarantees** (Karnin et al., 2016):

For a quantile \\q\\ with estimated value \\\hat{q}\\:

\$\$\|\text{rank}(\hat{q}) - q \cdot N\| \le \epsilon \cdot N \quad
\text{w.p. } \ge 1 - \delta\$\$

where \\\epsilon \approx O(1/k)\\ and space complexity is \\O(k
\log(N/k))\\.

**Phase 2: Candidate Extraction**

Approximately 40 quantiles are extracted from the sketch using a
non-uniform grid:

- **Tail regions** (0.01-0.1, 0.9-0.99): 10 quantiles per tail (step
  0.01)

- **Central region** (0.1-0.9): 17 quantiles (step 0.05)

This adaptive grid ensures higher resolution in distribution tails where
extreme values may significantly impact WoE.

**Phase 3: Optimal Cutpoint Selection**

Two strategies are employed based on dataset size:

**3a. Dynamic Programming (N \<= 50)**:

For small datasets, an exact DP solution maximizes total IV:

\$\$\text{dp}\[i\]\[j\] = \max\_{l \< i} \left\\ \text{dp}\[l\]\[j-1\] +
\text{IV}(\text{bin from } l+1 \text{ to } i) \right\\\$\$

where \\\text{dp}\[i\]\[j\]\\ is the maximum IV using \\i\\ observations
in \\j\\ bins.

**Note**: The DP implementation has known bugs for \\N \> 50\\
(conservative limit to prevent crashes). For larger datasets, fallback
uses uniform quantiles.

**3b. Greedy IV-based Selection (N \> 50)**:

For each candidate cutpoint \\c\\, compute the split IV:

\$\$\text{IV}\_{\text{split}}(c) = \text{IV}\_{\text{left}} +
\text{IV}\_{\text{right}}\$\$

where left/right refer to observations \\\le c\\ and \\\> c\\,
respectively.

Candidates are ranked by \\\text{IV}\_{\text{split}}\\ (descending), and
the top `max_bins - 1` are selected.

**Phase 4: Bin Refinement**

**4a. Frequency Constraint Enforcement**:

Bins with count \\\< \text{bin\\cutoff} \times N\\ are merged with the
adjacent bin having the most similar event rate:

\$\$\text{merge\\with} = \arg\min\_{j \in \\i-1, i+1\\}
\|\text{event\\rate}\_i - \text{event\\rate}\_j\|\$\$

**4b. Monotonicity Enforcement (PAVA)**:

If `monotonic = TRUE`, the Pool Adjacent Violators Algorithm ensures:

\$\$\text{WoE}\_1 \le \text{WoE}\_2 \le \cdots \le \text{WoE}\_k \quad
\text{(increasing)}\$\$

or the reverse for decreasing patterns. Direction is auto-detected via:

\$\$\text{increasing} = \begin{cases} \text{TRUE} & \text{if }
\text{WoE}\_{\text{last}} \ge \text{WoE}\_{\text{first}} \\ \text{FALSE}
& \text{otherwise} \end{cases}\$\$

Violations are resolved by merging adjacent bins iteratively.

**4c. Bin Count Optimization**:

If the number of bins exceeds `max_bins`, bins are merged to minimize IV
loss:

\$\$\Delta \text{IV}\_{i,i+1} = \text{IV}\_i + \text{IV}\_{i+1} -
\text{IV}\_{\text{merged}}\$\$

The pair with smallest \\\Delta \text{IV}\\ is merged iteratively until
\\k \le \text{max\\bins}\\.

**Computational Complexity**

- **Time**:

  - Sketch construction: \\O(N \log k)\\ for \\N\\ updates

  - Candidate evaluation: \\O(N \times C)\\ where \\C \approx 40\\

  - DP (if applicable): \\O(N^2 \times k)\\

  - PAVA: \\O(k^2)\\ worst case

  - **Total**: \\O(N \log k + N \times C + k^2 \times I)\\ where \\I\\
    is iterations

- **Space**:

  - Sketch: \\O(k \log N)\\ vs \\O(N)\\ for batch methods

  - DP table (if N \<= 50): \\O(N \times k)\\

  - **Total**: \\O(k \log N)\\ for large N

**Comparison with Batch Methods**

|            |            |            |                              |                 |
|------------|------------|------------|------------------------------|-----------------|
| **Method** | **Space**  | **Passes** | **Guarantees**               | **Scalability** |
| Sketch     | O(k log N) | 1          | Probabilistic (\\\epsilon\\) | Streaming-ready |
| MDLP       | O(N)       | 1          | Deterministic (MDL)          | Batch only      |
| MOB/MBLP   | O(N)       | Multiple   | Heuristic                    | Batch only      |

**When to Use Sketch-based Binning**

- **Use Sketch**: For very large datasets (N \> 10^6) where memory is
  constrained, or for streaming data where single-pass processing is
  required.

- **Use MDLP**: For moderate datasets (N \< 10^5) where exact quantiles
  and deterministic results are preferred.

- **Avoid Sketch**: For small datasets (N \< 1000) where approximation
  error may dominate, or when reproducibility with exact quantiles is
  critical.

**Tuning sketch_k**

The `sketch_k` parameter controls the accuracy-memory tradeoff:

- **k = 50**: Fast, low memory, \\\epsilon \approx 2\\\\ (suitable for
  exploration)

- **k = 200** (default): Balanced, \\\epsilon \approx 0.5\\\\
  (production)

- **k = 500**: High precision, \\\epsilon \approx 0.2\\\\ (critical
  applications)

## References

- Karnin, Z., Lang, K., & Liberty, E. (2016). "Optimal Quantile
  Approximation in Streams". *Proceedings of the 57th Annual IEEE
  Symposium on Foundations of Computer Science (FOCS)*, pp. 71-78.

- Greenwald, M., & Khanna, S. (2001). "Space-efficient online
  computation of quantile summaries". *ACM SIGMOD Record*, 30(2), 58-66.

- Cormode, G., & Duffield, N. (2014). "Sampling for Big Data: A
  Tutorial". *Proceedings of the 20th ACM SIGKDD*, pp. 1975-1975.

- Munro, J. I., & Paterson, M. S. (1980). "Selection and sorting with
  limited storage". *Theoretical Computer Science*, 12(3), 315-323.

- Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D.
  (1972). *Statistical Inference Under Order Restrictions*. Wiley.

- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.

## See also

[`ob_numerical_mdlp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mdlp.md)
for deterministic binning with exact quantiles,
[`ob_numerical_mblp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mblp.md)
for batch processing with monotonicity.

## Author

Lopes, J. E. (KLL Sketch implementation based on Karnin et al., 2016)

## Examples

``` r
if (FALSE) { # \dontrun{
# Simulate large-scale credit scoring data
set.seed(2024)
n <- 100000 # Large dataset where sketch shines

feature <- c(
  rnorm(40000, mean = 580, sd = 60),
  rnorm(40000, mean = 680, sd = 50),
  rnorm(20000, mean = 750, sd = 40)
)

target <- c(
  rbinom(40000, 1, 0.30),
  rbinom(40000, 1, 0.12),
  rbinom(20000, 1, 0.04)
)

# Apply sketch-based binning
result <- ob_numerical_sketch(
  feature = feature,
  target = target,
  min_bins = 3,
  max_bins = 5,
  sketch_k = 200, # Standard accuracy
  monotonic = TRUE
)

# Inspect results
print(result$woe)
print(result$cutpoints)
cat(sprintf(
  "Converged: %s (iterations: %d)\n",
  result$converged, result$iterations
))

# Compare sketch_k values
result_k50 <- ob_numerical_sketch(feature, target, sketch_k = 50)
result_k500 <- ob_numerical_sketch(feature, target, sketch_k = 500)

# Check cutpoint stability (higher k → more stable)
data.frame(
  k = c(50, 200, 500),
  N_Bins = c(
    length(result_k50$woe),
    length(result$woe),
    length(result_k500$woe)
  ),
  First_Cutpoint = c(
    result_k50$cutpoints[1],
    result$cutpoints[1],
    result_k500$cutpoints[1]
  )
)

# Memory comparison (conceptual - not executed)
# Sketch: ~200 items x 4 levels x 16 bytes = approx 12 KB
# Batch:  100,000 items x 16 bytes = approx 1.6 MB
# Ratio:  ~0.75% of batch memory
} # }
```
