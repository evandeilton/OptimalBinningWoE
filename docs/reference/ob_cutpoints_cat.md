# Binning Categorical Variables using Custom Cutpoints

This function applies user-defined binning to a categorical variable by
grouping specified categories into bins and calculating Weight of
Evidence (WoE) and Information Value (IV) for each bin.

## Usage

``` r
ob_cutpoints_cat(feature, target, cutpoints)
```

## Arguments

- feature:

  A character vector or factor representing the categorical predictor
  variable.

- target:

  An integer vector containing binary outcome values (0 or 1). Must be
  the same length as `feature`.

- cutpoints:

  A character vector where each element defines a bin by concatenating
  the original category names with "+" as separator.

## Value

A list containing:

- `woefeature`:

  Numeric vector of WoE values corresponding to each observation in the
  input `feature`

- `woebin`:

  Data frame with one row per bin containing:

  - `bin`: The bin definition (original categories joined by "+")

  - `count`: Total number of observations in the bin

  - `count_pos`: Number of positive outcomes (target=1) in the bin

  - `count_neg`: Number of negative outcomes (target=0) in the bin

  - `woe`: Weight of Evidence for the bin

  - `iv`: Information Value contribution of the bin

## Details

The function takes a character vector defining how categories should be
grouped. Each element in the `cutpoints` vector defines one bin by
listing the original categories that should be merged, separated by "+"
signs.

For example, if you want to create two bins from categories "A", "B",
"C", "D":

- Bin 1: "A+B"

- Bin 2: "C+D"

## Note

- Target variable must contain only 0 and 1 values.

- Every unique category in `feature` must be included in exactly one bin
  definition in `cutpoints`.

- Categories not mentioned in `cutpoints` will be assigned to bin 0
  (which may lead to unexpected results).

## Examples

``` r
# Sample data
feature <- c("A", "B", "C", "D", "A", "B", "C", "D")
target <- c(1, 0, 1, 0, 1, 1, 0, 0)

# Define custom bins: (A,B) and (C,D)
cutpoints <- c("A+B", "C+D")

# Apply binning
result <- ob_cutpoints_cat(feature, target, cutpoints)

# View bin statistics
print(result$woebin)
#>   bin count count_pos count_neg       woe        iv
#> 1 A+B     4         3         1  1.098612 0.5493061
#> 2 C+D     4         1         3 -1.098612 0.5493061

# View WoE-transformed feature
print(result$woefeature)
#> [1]  1.098612  1.098612 -1.098612 -1.098612  1.098612  1.098612 -1.098612
#> [8] -1.098612
```
