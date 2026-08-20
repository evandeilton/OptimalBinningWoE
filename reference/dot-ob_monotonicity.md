# Internal: Monotonicity Diagnostics of a Bin Profile

Assesses whether a per-bin quantity is ordered across the natural bin
sequence. Because the Weight of Evidence is a strictly increasing
function of the bin odds, and the event rate is likewise strictly
increasing in the odds, monotonicity of the event rate and of the
empirical WoE are equivalent statements.

## Usage

``` r
.ob_monotonicity(v)
```

## Arguments

- v:

  Numeric vector indexed by bin id (already in bin order).

## Value

A list with `monotonic`, `strict`, `direction`, `n_violations` and
`spearman`.
