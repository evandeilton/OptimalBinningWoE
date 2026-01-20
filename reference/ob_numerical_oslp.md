# Optimal Binning for Numerical Variables using Optimal Supervised Learning Partitioning

Implements a greedy binning algorithm with quantile-based pre-binning
and monotonicity enforcement. **Important Note**: Despite "Optimal
Supervised Learning Partitioning" and "LP" in the name, the algorithm
uses **greedy heuristics** without formal Linear Programming or convex
optimization. The method is functionally equivalent to
[`ob_numerical_mrblp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mrblp.md)
with minor differences in pre-binning strategy and bin reduction
criteria.

Users seeking true optimization-based binning should consider
Mixed-Integer Programming (MIP) implementations (e.g., via `ompr` or
`lpSolve` packages), though these scale poorly beyond N \> 10,000
observations.

## Usage

``` r
ob_numerical_oslp(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  convergence_threshold = 1e-06,
  max_iterations = 1000,
  laplace_smoothing = 0.5
)
```

## Arguments

- feature:

  Numeric vector of feature values. Missing values (NA) and infinite
  values are **not permitted** and will trigger an error.

- target:

  Integer or numeric vector of binary target values (must contain only 0
  and 1). Must have the same length as `feature`. Unlike other binning
  methods, OSLP internally uses `double` for target, allowing implicit
  conversion from integer.

- min_bins:

  Minimum number of bins (default: 3). Must be at least 2.

- max_bins:

  Maximum number of bins (default: 5). Must be greater than or equal to
  `min_bins`.

- bin_cutoff:

  Minimum fraction of total observations per bin (default: 0.05). Must
  be in (0, 1).

- max_n_prebins:

  Maximum number of pre-bins (default: 20). Must be at least equal to
  `min_bins`.

- convergence_threshold:

  Convergence threshold for IV change (default: 1e-6).

- max_iterations:

  Maximum iterations (default: 1000).

- laplace_smoothing:

  Laplace smoothing parameter (default: 0.5). Must be non-negative.

## Value

A list containing:

- id:

  Integer bin identifiers (1-based).

- bin:

  Character bin intervals `"[lower;upper)"`.

- woe:

  Numeric WoE values (guaranteed monotonic).

- iv:

  Numeric IV contributions per bin.

- count:

  Integer total observations per bin.

- count_pos:

  Integer positive class counts.

- count_neg:

  Integer negative class counts.

- event_rate:

  Numeric event rates.

- cutpoints:

  Numeric bin boundaries (excluding ±Inf).

- total_iv:

  Total Information Value.

- converged:

  Logical convergence flag.

- iterations:

  Integer iteration count.

## Details

**Algorithm Overview**

OSLP executes in five phases:

**Phase 1: Quantile-Based Pre-binning**

Unlike equal-frequency methods that ensure balanced bin sizes, OSLP
places cutpoints at quantiles of **unique feature values**:

\$\$\text{edge}\_i = \text{unique\\vals}\left\[\left\lfloor p_i \times
(n\_{\text{unique}} - 1) \right\rfloor\right\]\$\$

where \\p_i = i / \text{max\\n\\prebins}\\.

**Critique**: If unique values are clustered (e.g., many observations at
specific values), bins may have vastly different sizes, violating the
equal-frequency principle that ensures statistical stability.

**Phase 2: Rare Bin Merging**

Bins with \\n_i / N \< \text{bin\\cutoff}\\ are merged. The merge
direction minimizes IV loss:

\$\$\Delta \text{IV} = \text{IV}\_i + \text{IV}\_{i+d} -
\text{IV}\_{\text{merged}}\$\$

where \\d \in \\-1, +1\\\\ (left or right neighbor).

**Phase 3: Initial WoE/IV Calculation**

Standard WoE with Laplace smoothing:

\$\$\text{WoE}\_i = \ln\left(\frac{n_i^{+} + \alpha}{n^{+} + k\alpha}
\bigg/ \frac{n_i^{-} + \alpha}{n^{-} + k\alpha}\right)\$\$

**Phase 4: Monotonicity Enforcement**

Direction determined via majority vote (identical to MRBLP):

\$\$\text{increasing} = \begin{cases} \text{TRUE} & \text{if } \sum_i
\mathbb{1}\_{\\\text{WoE}\_i \> \text{WoE}\_{i-1}\\} \ge \sum_i
\mathbb{1}\_{\\\text{WoE}\_i \< \text{WoE}\_{i-1}\\} \\ \text{FALSE} &
\text{otherwise} \end{cases}\$\$

Violations are merged iteratively.

**Phase 5: Bin Count Reduction**

If \\k \> \text{max\\bins}\\, merge bins with the **smallest combined
IV**:

\$\$\text{merge\\idx} = \arg\min\_{i=0}^{k-2} \left( \text{IV}\_i +
\text{IV}\_{i+1} \right)\$\$

**Rationale**: Assumes bins with low total IV contribute least to
predictive power. However, this ignores the interaction between bins; a
low-IV bin may be essential for monotonicity or preventing gaps.

**Theoretical Foundations**

Despite the name "Optimal Supervised Learning Partitioning", the
algorithm lacks:

- **Global optimality guarantees**: Greedy merging is myopic

- **Formal loss function**: No explicit objective being minimized

- **LP formulation**: No constraint matrix, simplex solver, or dual
  variables

A true optimal partitioning approach would formulate the problem as:

\$\$\min\_{\mathbf{z}, \mathbf{b}} \left\\ -\sum\_{i=1}^{k}
\text{IV}\_i(\mathbf{b}) + \lambda k \right\\\$\$

subject to: \$\$\sum\_{j=1}^{k} z\_{ij} = 1 \quad \forall i \in \\1,
\ldots, N\\\$\$ \$\$\text{WoE}\_j \le \text{WoE}\_{j+1} \quad \forall
j\$\$ \$\$z\_{ij} \in \\0, 1\\, \quad b_j \in \mathbb{R}\$\$

where \\z\_{ij}\\ indicates observation \\i\\ assigned to bin \\j\\, and
\\\lambda\\ is a complexity penalty. This requires MILP solvers (CPLEX,
Gurobi) and is intractable for \\N \> 10^4\\.

**Comparison with Related Methods**

|            |                        |                |                         |                 |
|------------|------------------------|----------------|-------------------------|-----------------|
| **Method** | **Pre-binning**        | **Direction**  | **Merge (max_bins)**    | **Target Type** |
| OSLP       | Quantile (unique vals) | Majority vote  | Min (IV(i) + IV(i+1))   | double          |
| MRBLP      | Equal-frequency        | Majority vote  | Min \|IV(i) - IV(i+1)\| | int             |
| MOB        | Equal-frequency        | First two bins | Min IV loss             | int             |
| MBLP       | Quantile (data)        | Correlation    | Min IV loss             | int             |

**When to Use OSLP**

- **Use OSLP**: Never. Use MBLP or MOB instead for better pre-binning
  and merge strategies.

- **Use MBLP**: For robust direction detection via correlation.

- **Use MDLP**: For information-theoretic stopping criteria.

- **Use True LP**: For small datasets (N \< 1000) where global
  optimality is critical and computational cost is acceptable.

## References

- Mironchyk, P., & Tchistiakov, V. (2017). "Monotone optimal binning
  algorithm for credit risk modeling". *Frontiers in Applied Mathematics
  and Statistics*, 3, 2.

- Zeng, G. (2014). "A Necessary Condition for a Good Binning Algorithm
  in Credit Scoring". *Applied Mathematical Sciences*, 8(65), 3229-3242.

- Fayyad, U. M., & Irani, K. B. (1993). "Multi-Interval Discretization
  of Continuous-Valued Attributes". *IJCAI*, pp. 1022-1027.

- Good, I. J. (1952). "Rational Decisions". *Journal of the Royal
  Statistical Society B*, 14(1), 107-114.

- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.

## See also

[`ob_numerical_mrblp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mrblp.md)
for nearly identical algorithm with better pre-binning,
[`ob_numerical_mblp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mblp.md)
for correlation-based direction detection,
[`ob_numerical_mdlp`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mdlp.md)
for information-theoretic optimality.

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
set.seed(123)
n <- 5000
feature <- c(
  rnorm(2000, 600, 50),
  rnorm(2000, 680, 40),
  rnorm(1000, 740, 30)
)
target <- c(
  rbinom(2000, 1, 0.25),
  rbinom(2000, 1, 0.10),
  rbinom(1000, 1, 0.03)
)

result <- ob_numerical_oslp(
  feature = feature,
  target = target,
  min_bins = 3,
  max_bins = 5
)

print(result$woe)
#> [1]  0.8025010  0.5114407 -0.2910587 -1.2743136 -1.3945886
print(result$total_iv)
#> [1] 0.3691969

# Compare with MRBLP (should be nearly identical)
result_mrblp <- ob_numerical_mrblp(
  feature = feature,
  target = target,
  min_bins = 3,
  max_bins = 5
)

data.frame(
  Method = c("OSLP", "MRBLP"),
  Total_IV = c(result$total_iv, result_mrblp$total_iv),
  N_Bins = c(length(result$woe), length(result_mrblp$woe))
)
#>   Method  Total_IV N_Bins
#> 1   OSLP 0.3691969      5
#> 2  MRBLP 0.3847959      5
# }
```
