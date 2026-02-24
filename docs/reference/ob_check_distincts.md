# Check Distinct Length

Internal utility function that counts the number of distinct (unique)
values in a feature vector. Used for preprocessing validation before
applying optimal binning algorithms to determine if the feature has
sufficient variability.

## Usage

``` r
ob_check_distincts(x, target)
```

## Arguments

- x:

  Vector (numeric, character, or factor) whose unique values are to be
  counted. Accepts any R vector type that can be compared for equality.

- target:

  Integer vector of binary target values (0/1). Must have the same
  length as `x`. While not used in the distinct count calculation, it is
  required for interface consistency and may be used for future
  extensions (e.g., counting distinct values per class).

## Value

Integer scalar representing the number of unique values in `x`,
excluding `NA` values. Returns 0 if `x` is empty or contains only `NA`
values.

## Details

This function is typically used internally by optimal binning algorithms
to:

- Validate that the feature has at least 2 distinct values (required for
  binning).

- Determine if special handling is needed for low-cardinality features
  (e.g., \\\le 2\\ unique values).

- Decide between binning strategies (continuous vs categorical).

**Handling of Missing Values**: `NA`, `NaN`, and `Inf` values are
excluded from the count. To include missing values as a distinct
category, preprocess `x` by converting missings to a placeholder (e.g.,
`"-999"` for numeric, `"Missing"` for character).

## Examples

``` r
# \donttest{
# Continuous feature with many unique values
x_continuous <- rnorm(1000)
target <- sample(0:1, 1000, replace = TRUE)
ob_check_distincts(x_continuous, target)
#> [1] 1000    1
# Returns: ~1000 (approximately all unique due to floating point)

# Low-cardinality feature
x_binary <- sample(c("Yes", "No"), 1000, replace = TRUE)
ob_check_distincts(x_binary, target)
#> [1] 2 2
# Returns: 2

# Feature with missing values
x_with_na <- c(1, 2, NA, 2, 3, NA, 1)
target_short <- c(1, 0, 1, 0, 1, 0, 1)
ob_check_distincts(x_with_na, target_short)
#> [1] 3 1
# Returns: 3 (counts: 1, 2, 3; NAs excluded)

# Empty or all-NA feature
x_empty <- rep(NA, 100)
ob_check_distincts(x_empty, sample(0:1, 100, replace = TRUE))
#> [1] 0 0
# Returns: 0
# }
```
