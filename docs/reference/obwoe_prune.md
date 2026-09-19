# Prune Redundant Variables by Correlation

Greedy, iterative removal of variables that carry the same information
as a better-ranked one. Information Value ranks variables one at a time;
two variables can both be strong and say the same thing, which a model
on Weight of Evidence shows as an unstable or sign-flipped coefficient.

## Usage

``` r
obwoe_prune(x, ranking, cutoff = 0.7, method = "pearson")
```

## Arguments

- x:

  A `data.frame` of numeric columns to be compared — normally the
  WoE-transformed predictors, which is the space the model sees — or a
  pre-computed pairwise table from
  [`obcorr`](https://evandeilton.github.io/OptimalBinningWoE/reference/obcorr.md)
  with columns `x`, `y` and a correlation column.

- ranking:

  Character vector of variable names, best first. Variables absent from
  `ranking` are treated as ranked last.

- cutoff:

  Numeric in \\(0, 1\]\\. Absolute correlation at or above which two
  variables are considered redundant. Default `0.70`.

- method:

  Character string passed to
  [`obcorr`](https://evandeilton.github.io/OptimalBinningWoE/reference/obcorr.md)
  when `x` is a `data.frame` of columns. Default `"pearson"`.

## Value

A list with:

- `keep`:

  Character vector of surviving variables.

- `dropped`:

  `data.frame` of `variable`, `correlated_with` and `correlation` — one
  row per removal, in the order the removals happened.

- `pairs`:

  The pairwise table, with an `abs_corr` column.

- `cutoff`:

  The cutoff used.

## Details

The pass is iterative on purpose. Evaluating every pair independently
removes both \\B\\ and \\C\\ from a chain \\A \sim B \sim C\\ even when
\\B\\ and \\C\\ are unrelated to each other: once \\B\\ is gone, the
pair \\(B, C)\\ no longer exists. Each iteration therefore drops one
variable — the worst-ranked member of the strongest surviving pair — and
recomputes what is left.

## See also

[`obcorr`](https://evandeilton.github.io/OptimalBinningWoE/reference/obcorr.md),
[`obwoe_select`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_select.md)

## Examples

``` r
set.seed(1)
n <- 500
a <- rnorm(n)
df <- data.frame(a = a, b = a + rnorm(n, 0, 0.2), c = rnorm(n))

# b duplicates a; a is ranked better, so b goes
obwoe_prune(df, ranking = c("a", "b", "c"), cutoff = 0.7)
#> $keep
#> [1] "a" "c"
#> 
#> $dropped
#>   variable correlated_with correlation
#> 1        b               a   0.9785343
#> 
#> $pairs
#>   x y     pearson   abs_corr
#> 1 a b  0.97853425 0.97853425
#> 2 a c -0.02845298 0.02845298
#> 3 b c -0.04396917 0.04396917
#> 
#> $cutoff
#> [1] 0.7
#> 
```
