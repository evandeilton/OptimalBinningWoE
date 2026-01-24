# Compute Multiple Robust Correlations Between Numeric Variables

This function computes various correlation coefficients between all
pairs of numeric variables in a data frame. It implements several
classical and robust correlation measures, including Pearson, Spearman,
Kendall, Hoeffding's D, Distance Correlation, Biweight Midcorrelation,
and Percentage Bend correlation.

## Usage

``` r
obcorr(df, method = "all", threads = 0L)
```

## Arguments

- df:

  A data frame containing numeric variables. Non-numeric columns will be
  automatically excluded. At least two numeric variables are required.

- method:

  A character string specifying which correlation method(s) to compute.
  Possible values are:

  - `"all"`: Compute all available correlation methods (default).

  - `"pearson"`: Compute only Pearson correlation.

  - `"spearman"`: Compute only Spearman correlation.

  - `"kendall"`: Compute only Kendall correlation.

  - `"hoeffding"`: Compute only Hoeffding's D.

  - `"distance"`: Compute only distance correlation.

  - `"biweight"`: Compute only biweight midcorrelation.

  - `"pbend"`: Compute only percentage bend correlation.

  - `"robust"`: Compute robust correlations (biweight and pbend).

  - `"alternative"`: Compute alternative correlations (hoeffding and
    distance).

- threads:

  An integer specifying the number of threads to use for parallel
  computation. If 0 (default), uses all available cores. Ignored if
  OpenMP is not available.

## Value

A data frame with the following columns:

- `x`, `y`:

  Names of the variable pairs being correlated.

- `pearson`:

  Pearson correlation coefficient.

- `spearman`:

  Spearman rank correlation coefficient.

- `kendall`:

  Kendall's tau-b correlation coefficient.

- `hoeffding`:

  Hoeffding's D statistic (scaled).

- `distance`:

  Distance correlation coefficient.

- `biweight`:

  Biweight midcorrelation coefficient.

- `pbend`:

  Percentage bend correlation coefficient.

The exact columns returned depend on the `method` parameter.

## Details

The function supports multiple correlation methods simultaneously and
utilizes OpenMP for parallel computation when available.

Available correlation methods:

- **Pearson**: Standard linear correlation coefficient.

- **Spearman**: Rank-based correlation coefficient.

- **Kendall**: Kendall's tau-b correlation coefficient.

- **Hoeffding**: Hoeffding's D statistic (scaled by 30).

- **Distance**: Distance correlation (Székely et al., 2007).

- **Biweight**: Biweight midcorrelation (robust alternative).

- **Pbend**: Percentage bend correlation (robust alternative).

## Note

- Missing values (NA) are handled appropriately for each correlation
  method.

- For robust methods (biweight, pbend), fallback to Pearson correlation
  occurs when there are insufficient data points or numerical
  instability.

- Hoeffding's D requires at least 5 complete pairs.

- Distance correlation is computed without forming NxN distance matrices
  for memory efficiency.

- When OpenMP is available, computations are automatically parallelized
  across variable pairs.

## References

Székely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007). Measuring and
testing dependence by correlation of distances. The Annals of
Statistics, 35(6), 2769-2794.

Wilcox, R.R. (1994). The percentage bend correlation coefficient.
Psychometrika, 59(4), 601-616.

## Examples

``` r
# Create sample data
set.seed(123)
n <- 100
df <- data.frame(
  x1 = rnorm(n),
  x2 = rnorm(n),
  x3 = rt(n, df = 3), # Heavy-tailed distribution
  x4 = sample(c(0, 1), n, replace = TRUE), # Binary variable
  category = sample(letters[1:3], n, replace = TRUE) # Non-numeric column
)

# Add some relationships
df$x2 <- df$x1 + rnorm(n, 0, 0.5)
df$x3 <- df$x1^2 + rnorm(n, 0, 0.5)

# Compute all correlations
result_all <- obcorr(df)
head(result_all)
#>    x  y     pearson    spearman      kendall    hoeffding  distance    biweight
#> 1 x1 x4  0.11946052  0.11432107  0.093808315 -0.005827769 0.2600756  0.11946052
#> 2 x1 x2  0.86981390  0.87913591  0.700202020  0.397341844 1.4207684  0.87255023
#> 3 x1 x3  0.09556132  0.06517852  0.048080808  0.047496185 0.7460724  0.11266947
#> 4 x3 x4 -0.09009510 -0.06720693 -0.055147919 -0.006921340 0.1976969 -0.09009510
#> 5 x2 x3  0.10385820  0.07159916  0.050505051  0.016285581 0.5003787  0.15306172
#> 6 x2 x4 -0.02020361 -0.01039282 -0.008528029 -0.008435515 0.1420518 -0.02020361
#>         pbend
#> 1  0.09814141
#> 2  0.88270163
#> 3  0.06725434
#> 4 -0.04556955
#> 5  0.08868743
#> 6 -0.01342052

# Compute only robust correlations
result_robust <- obcorr(df, method = "robust")

# Compute only Pearson correlation with 2 threads
result_pearson <- obcorr(df, method = "pearson", threads = 2)
```
