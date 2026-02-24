# Optimal Binning for Categorical Variables using Divergence Measures

Performs supervised discretization of categorical variables using a
divergence-based hierarchical merging algorithm. This implementation
supports multiple information-theoretic and metric divergence measures
as described by Zeng (2013), enabling flexible optimization of binning
structures for credit scoring and binary classification tasks.

## Usage

``` r
ob_categorical_dmiv(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  bin_separator = "%;%",
  convergence_threshold = 1e-06,
  max_iterations = 1000,
  bin_method = "woe1",
  divergence_method = "l2"
)
```

## Arguments

- feature:

  A character vector or factor representing the categorical predictor
  variable to be binned. Missing values are automatically converted to
  the category `"NA"`.

- target:

  An integer vector of binary outcomes (0/1) corresponding to each
  observation in `feature`. Missing values are not permitted.

- min_bins:

  Integer. Minimum number of bins to produce. Must be \>= 2. If the
  final number of bins after merging falls below this threshold, the
  algorithm will attempt to split bins. Defaults to 3.

- max_bins:

  Integer. Maximum number of bins to produce. Must be \>= `min_bins`.
  The algorithm performs hierarchical merging until this constraint is
  satisfied. Defaults to 5.

- bin_cutoff:

  Numeric. Frequency threshold for rare category handling. Categories
  with relative frequency below this value are candidates for
  pre-binning. Must be in (0, 1). Defaults to 0.05.

- max_n_prebins:

  Integer. Maximum number of initial bins before the main merging phase.
  When unique categories exceed this limit, rare categories are
  pre-merged into an "other" bin. Must be \>= 2. Defaults to 20.

- bin_separator:

  Character string used to concatenate category names when multiple
  categories are merged into a single bin. Defaults to "%;%".

- convergence_threshold:

  Numeric. Convergence tolerance for the iterative merging process.
  Merging stops when the change in minimum divergence between iterations
  falls below this threshold. Must be \> 0. Defaults to 1e-6.

- max_iterations:

  Integer. Maximum number of merge operations allowed. Prevents infinite
  loops in edge cases. Must be \> 0. Defaults to 1000.

- bin_method:

  Character string specifying the Weight of Evidence calculation method.
  Must be one of:

  `"woe"`

  :   Traditional WoE: \\\ln\left(\frac{p_i/P}{n_i/N}\right)\\

  `"woe1"`

  :   Smoothed WoE (Zeng): \\\ln\left(\frac{g_i + 0.5}{b_i +
      0.5}\right)\\

  The smoothed variant provides numerical stability for sparse bins.
  Defaults to `"woe1"`.

- divergence_method:

  Character string specifying the divergence measure used for
  determining bin similarity. Must be one of:

  `"he"`

  :   Hellinger Distance: \\\sum(\sqrt{p_i} - \sqrt{n_i})^2\\

  `"kl"`

  :   Symmetrized Kullback-Leibler Divergence

  `"klj"`

  :   Jeffreys J-Divergence: \\(p-n)\ln(p/n)\\

  `"tr"`

  :   Triangular Discrimination: \\(p-n)^2/(p+n)\\

  `"sc"`

  :   Symmetric Chi-Square: \\(p-n)^2(p+n)/(pn)\\

  `"js"`

  :   Jensen-Shannon Divergence

  `"l1"`

  :   L1 Metric (Manhattan Distance): \\\|p-n\|\\

  `"l2"`

  :   L2 Metric (Euclidean Distance): \\\sqrt{\sum(p-n)^2}\\

  `"ln"`

  :   L-infinity Metric (Chebyshev Distance): \\\max\|p-n\|\\

  Defaults to `"l2"`.

## Value

A list containing the binning results with the following components:

- `id`:

  Integer vector of bin identifiers (1-indexed)

- `bin`:

  Character vector of bin labels (merged category names)

- `woe`:

  Numeric vector of Weight of Evidence values per bin

- `divergence`:

  Numeric vector of divergence contribution per bin

- `count`:

  Integer vector of total observations per bin

- `count_pos`:

  Integer vector of positive cases (target=1) per bin

- `count_neg`:

  Integer vector of negative cases (target=0) per bin

- `converged`:

  Logical indicating algorithm convergence

- `iterations`:

  Integer count of merge operations performed

- `total_divergence`:

  Numeric total divergence of the binning solution

- `bin_method`:

  Character string of WoE method used

- `divergence_method`:

  Character string of divergence measure used

## Details

The algorithm implements a hierarchical agglomerative approach where
bins are iteratively merged based on minimum pairwise divergence until
the `max_bins` constraint is satisfied or convergence is achieved.

**Algorithm Workflow:**

1.  Input validation and frequency computation

2.  Pre-binning of rare categories (if unique categories \>
    `max_n_prebins`)

3.  Initialization of pairwise divergence matrix

4.  Iterative merging of most similar bin pairs

5.  Splitting of heterogeneous bins (if bins \< `min_bins`)

6.  Final metric computation and WoE-based sorting

**Divergence Measure Selection:** The choice of divergence measure
affects the binning structure:

- Information-theoretic measures (`"kl"`, `"js"`, `"klj"`): Emphasize
  distributional differences; sensitive to rare events

- Metric measures (`"l1"`, `"l2"`, `"ln"`): Provide geometric
  interpretation; robust to outliers

- Chi-square family (`"sc"`, `"tr"`): Balance between information
  content and robustness

- Hellinger distance (`"he"`): Bounded measure; suitable for probability
  distributions

**Pre-binning Strategy:** When the number of unique categories exceeds
`max_n_prebins`, categories with fewer than 5 observations are
aggregated into a special "PREBIN_OTHER" bin to control computational
complexity.

## References

Zeng, G. (2013). Metric Divergence Measures and Information Value in
Credit Scoring. *Journal of Mathematics*, 2013, Article ID 848271.
[doi:10.1155/2013/848271](https://doi.org/10.1155/2013/848271)

Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency.
*The Annals of Mathematical Statistics*, 22(1), 79-86.

Lin, J. (1991). Divergence Measures Based on the Shannon Entropy. *IEEE
Transactions on Information Theory*, 37(1), 145-151.

## See also

[`ob_categorical_cm`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_cm.md)
for ChiMerge-based categorical binning

## Examples

``` r
# \donttest{
# Example 1: Basic usage with synthetic credit data
set.seed(42)
n <- 1000

# Simulate occupation categories with varying default rates
occupations <- c(
  "Engineer", "Doctor", "Teacher", "Sales",
  "Manager", "Clerk", "Other"
)
default_probs <- c(0.05, 0.03, 0.08, 0.15, 0.07, 0.12, 0.20)

feature <- sample(occupations, n,
  replace = TRUE,
  prob = c(0.15, 0.10, 0.20, 0.18, 0.12, 0.15, 0.10)
)
target <- sapply(feature, function(x) {
  rbinom(1, 1, default_probs[which(occupations == x)])
})

# Apply optimal binning with L2 divergence
result <- ob_categorical_dmiv(feature, target,
  min_bins = 2,
  max_bins = 4,
  divergence_method = "l2"
)

# Examine binning results
print(data.frame(
  bin = result$bin,
  woe = round(result$woe, 3),
  count = result$count,
  event_rate = round(result$count_pos / result$count, 3)
))
#>                  bin    woe count event_rate
#> 1             Doctor -3.780   111      0.018
#> 2 Engineer%;%Teacher -2.562   355      0.070
#> 3    Manager%;%Clerk -2.086   275      0.109
#> 4      Sales%;%Other -1.524   259      0.178

# Example 2: Comparing divergence methods
result_js <- ob_categorical_dmiv(feature, target,
  divergence_method = "js",
  max_bins = 4
)
result_kl <- ob_categorical_dmiv(feature, target,
  divergence_method = "kl",
  max_bins = 4
)

cat("Jensen-Shannon bins:", length(result_js$bin), "\n")
#> Jensen-Shannon bins: 4 
cat("Kullback-Leibler bins:", length(result_kl$bin), "\n")
#> Kullback-Leibler bins: 4 

# Example 3: High cardinality feature with pre-binning
set.seed(123)
postal_codes <- paste0("ZIP_", sprintf("%03d", 1:50))
feature_high_card <- sample(postal_codes, 2000, replace = TRUE)
target_high_card <- rbinom(2000, 1, 0.1)

result_prebin <- ob_categorical_dmiv(
  feature_high_card,
  target_high_card,
  max_n_prebins = 15,
  max_bins = 5
)
#> Info: Number of unique categories (50) exceeds max_n_prebins (15). Pre-binning rare categories.
#> Info: Pre-binning reduced categories from 50 to 50 initial bins.
#> Info: Converged after 1 iterations (divergence change < threshold).

cat("Final bins after pre-binning:", length(result_prebin$bin), "\n")
#> Final bins after pre-binning: 49 
cat("Algorithm converged:", result_prebin$converged, "\n")
#> Algorithm converged: TRUE 
# }
```
