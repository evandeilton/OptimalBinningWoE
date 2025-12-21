# Optimal Binning for Categorical Variables using Enhanced ChiMerge Algorithm

Performs supervised discretization of categorical variables using an
enhanced implementation of the ChiMerge algorithm (Kerber, 1992) with
optional Chi2 extension (Liu & Setiono, 1995). This method optimally
groups categorical levels based on their relationship with a binary
target variable to maximize predictive power while maintaining
statistical significance.

## Usage

``` r
ob_categorical_cm(
  feature,
  target,
  min_bins = 3,
  max_bins = 5,
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  bin_separator = "%;%",
  convergence_threshold = 1e-06,
  max_iterations = 1000,
  chi_merge_threshold = 0.05,
  use_chi2_algorithm = FALSE
)
```

## Arguments

- feature:

  A character vector or factor representing the categorical predictor
  variable to be binned.

- target:

  An integer vector of binary outcomes (0/1) corresponding to each
  observation in `feature`.

- min_bins:

  Integer. Minimum number of bins to produce. Must be \>= 2. Defaults to
  3.

- max_bins:

  Integer. Maximum number of bins to produce. Must be \>= `min_bins`.
  Defaults to 5.

- bin_cutoff:

  Numeric. Threshold for treating categories as rare. Categories with
  frequency \< `bin_cutoff` will be merged with their most similar
  neighbors. Value must be in (0, 1). Defaults to 0.05.

- max_n_prebins:

  Integer. Maximum number of initial pre-bins before merging. Controls
  computational complexity. Must be \>= 2. Defaults to 20.

- bin_separator:

  String. Separator used when combining multiple categories into a
  single bin label. Defaults to "%;%".

- convergence_threshold:

  Numeric. Convergence tolerance for iterative merging process. Smaller
  values require stricter convergence. Must be \> 0. Defaults to 1e-6.

- max_iterations:

  Integer. Maximum iterations for the merging algorithm. Prevents
  infinite loops. Must be \> 0. Defaults to 1000.

- chi_merge_threshold:

  Numeric. Statistical significance level (p-value) for chi-square tests
  during merging. Higher values create fewer bins. Value must be in (0,
  1). Defaults to 0.05.

- use_chi2_algorithm:

  Logical. If TRUE, uses the Chi2 variant which performs multi-pass
  merging with decreasing significance thresholds. Defaults to FALSE.

## Value

A list containing binning results with the following components:

- `id`: Integer vector of bin identifiers (1:n_bins)

- `bin`: Character vector of bin labels (merged category names)

- `woe`: Numeric vector of Weight of Evidence for each bin

- `iv`: Numeric vector of Information Value contribution per bin

- `count`: Integer vector of total observations per bin

- `count_pos`: Integer vector of positive cases per bin

- `count_neg`: Integer vector of negative cases per bin

- `converged`: Logical indicating if algorithm converged

- `iterations`: Integer count of algorithm iterations performed

- `algorithm`: Character string identifying algorithm used

- `warnings`: Character vector of any warnings encountered

- `metadata`: List with additional diagnostic information:

  - `total_iv`: Total Information Value of the binned variable

  - `n_bins`: Final number of bins produced

  - `unique_categories`: Number of unique input categories

  - `total_obs`: Total number of observations processed

  - `execution_time_ms`: Processing time in milliseconds

  - `monotonic`: Direction of WoE monotonicity
    ("increasing"/"decreasing")

## Details

The algorithm implements two main approaches:

1\. Standard ChiMerge: Iteratively merges adjacent bins with lowest
chi-square statistics until all remaining pairs are statistically
distinguishable at the specified significance level.

2\. Chi2 Algorithm (when `use_chi2_algorithm = TRUE`): Performs multiple
passes with decreasing significance thresholds (0.5 → 0.001), creating
more robust binning structures particularly for noisy data.

Key features include:

- Rare category handling through pre-merging

- Monotonicity enforcement of Weight of Evidence

- Numerical stability with underflow protection

- Efficient chi-square caching for performance

- Comprehensive input validation and error handling

Information Value interpretation:

- \< 0.02: Predictive power not useful

- 0.02-0.1: Weak predictive power

- 0.1-0.3: Medium predictive power

- 0.3-0.5: Strong predictive power

- \> 0.5: Suspiciously high (potential overfitting)

## References

Kerber, R. (1992). ChiMerge: Discretization of numeric attributes. In
*Proceedings of the Tenth National Conference on Artificial
Intelligence* (pp. 123-128).

Liu, B., & Setiono, R. (1995). Chi2: Feature selection and
discretization of numeric attributes. In *Proceedings of the Seventh
IEEE International Conference on Tools with Artificial Intelligence*
(pp. 372-377).

## Author

Developed as part of the OptimalBinningWoE package

## Examples

``` r
# Example 1: Basic usage with synthetic data
set.seed(123)
n <- 1000
categories <- c("A", "B", "C", "D", "E", "F", "G", "H")
feature <- sample(categories, n, replace = TRUE, prob = c(
  0.2, 0.15, 0.15,
  0.1, 0.1, 0.1,
  0.1, 0.1
))
# Create target with some association to categories
probs <- c(0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85) # increasing probability
target <- sapply(seq_along(feature), function(i) {
  cat_idx <- which(categories == feature[i])
  rbinom(1, 1, probs[cat_idx])
})

result <- ob_categorical_cm(feature, target)
print(result[c("bin", "woe", "iv", "count")])
#> $bin
#> [1] "A"     "B"     "C%;%D" "E%;%G" "F%;%H"
#> 
#> $woe
#> [1] -1.18847425 -0.80239061 -0.01771012  0.87051615  1.37170669
#> 
#> $iv
#> [1] 2.801835e-01 1.052373e-01 7.275865e-05 1.299588e-01 3.041406e-01
#> 
#> $count
#> [1] 198 159 242 195 206
#> 

# View metadata
print(paste("Total IV:", round(result$metadata$total_iv, 3)))
#> [1] "Total IV: 0.82"
print(paste("Algorithm converged:", result$converged))
#> [1] "Algorithm converged: TRUE"

# Example 2: Using Chi2 algorithm for more conservative binning
result_chi2 <- ob_categorical_cm(feature, target,
  use_chi2_algorithm = TRUE,
  max_bins = 6
)

# Compare number of bins
cat("Standard ChiMerge bins:", result$metadata$n_bins, "\n")
#> Standard ChiMerge bins: 5 
cat("Chi2 algorithm bins:", result_chi2$metadata$n_bins, "\n")
#> Chi2 algorithm bins: 6 
```
