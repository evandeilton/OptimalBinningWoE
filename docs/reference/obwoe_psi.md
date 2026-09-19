# Population Stability Index

Compares how a variable, or a score, is distributed between two samples
— typically the development window and a later vintage.

## Usage

``` r
obwoe_psi(base, compare, levels = NULL, breaks = NULL, n_groups = 10L)
```

## Arguments

- base:

  Vector observed in the reference sample.

- compare:

  Vector observed in the sample being monitored.

- levels:

  Character vector of the categories or bin labels to compare over.
  Required when the inputs are not numeric; defaults to the union of the
  values seen.

- breaks:

  Numeric vector of cut points used to band numeric inputs. `NULL`
  (default) bands them at the deciles of `base`.

- n_groups:

  Integer. Number of bands when `breaks` is `NULL`. Default `10`.

## Value

A list with `psi` (the total), `table` (the contribution of each band)
and `flag`, one of `"stable"`, `"watch"` or `"act"` at the conventional
0.10 and 0.25 thresholds.

## Details

\$\$\mathrm{PSI} = \sum_i (p_i - q_i)\\\ln\frac{p_i}{q_i}\$\$

which is the same symmetrised Kullback-Leibler divergence that defines
Information Value, applied to two vintages of one variable instead of to
two classes of one sample. The conventional reading is \\\<0.10\\
stable, \\0.10\\–\\0.25\\ worth watching, \\\>0.25\\ act.

A band that is populated in one sample and empty in the other makes the
divergence infinite. That is reported as `Inf` rather than smoothed
away, because a segment that has vanished is exactly what monitoring
exists to catch; the per-band table shows which one.

## See also

[`obwoe_scorecard`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scorecard.md)

## Examples

``` r
set.seed(1)
obwoe_psi(rnorm(1000), rnorm(1000))
#> $psi
#> [1] 0.01130065
#> 
#> $flag
#> [1] "stable"
#> 
#> $table
#>                band pct_base pct_compare     psi_band
#> 1      [-Inf,-1.34]      0.1       0.103 8.867641e-05
#> 2    (-1.34,-0.882]      0.1       0.099 1.005034e-05
#> 3   (-0.882,-0.511]      0.1       0.111 1.147960e-03
#> 4   (-0.511,-0.296]      0.1       0.081 4.003700e-03
#> 5  (-0.296,-0.0353]      0.1       0.105 2.439508e-04
#> 6   (-0.0353,0.245]      0.1       0.112 1.359944e-03
#> 7     (0.245,0.536]      0.1       0.085 2.437784e-03
#> 8     (0.536,0.854]      0.1       0.096 1.632880e-04
#> 9      (0.854,1.32]      0.1       0.113 1.588829e-03
#> 10      (1.32, Inf]      0.1       0.095 2.564665e-04
#> 
obwoe_psi(rnorm(1000), rnorm(1000, mean = 0.6))
#> $psi
#> [1] 0.3198493
#> 
#> $flag
#> [1] "act"
#> 
#> $table
#>                band pct_base pct_compare     psi_band
#> 1      [-Inf,-1.29]      0.1       0.032 0.0774815313
#> 2    (-1.29,-0.836]      0.1       0.044 0.0459749109
#> 3   (-0.836,-0.483]      0.1       0.063 0.0170953120
#> 4   (-0.483,-0.245]      0.1       0.061 0.0192775566
#> 5  (-0.245,-0.0055]      0.1       0.075 0.0071920518
#> 6   (-0.0055,0.233]      0.1       0.089 0.0012818720
#> 7     (0.233,0.538]      0.1       0.108 0.0006156883
#> 8     (0.538,0.919]      0.1       0.144 0.0160442970
#> 9      (0.919,1.38]      0.1       0.149 0.0195400299
#> 10      (1.38, Inf]      0.1       0.235 0.1153460693
#> 
```
