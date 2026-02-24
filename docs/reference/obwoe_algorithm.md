# Binning Algorithm Parameter

A qualitative tuning parameter for selecting the optimal binning
algorithm in
[`step_obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md).

## Usage

``` r
obwoe_algorithm(values = NULL)
```

## Arguments

- values:

  A character vector of algorithm names to include in the parameter
  space. If `NULL` (default), includes all 29 algorithms (28 specific
  algorithms plus `"auto"`).

## Value

A `dials` qualitative parameter object.

## Details

The algorithms are organized into three groups:

**Universal** (support both numerical and categorical features):
`"auto"`, `"jedi"`, `"jedi_mwoe"`, `"cm"`, `"dp"`, `"dmiv"`, `"fetb"`,
`"mob"`, `"sketch"`, `"udt"`

**Numerical only**: `"bb"`, `"ewb"`, `"fast_mdlp"`, `"ir"`, `"kmb"`,
`"ldb"`, `"lpdb"`, `"mblp"`, `"mdlp"`, `"mrblp"`, `"oslp"`, `"ubsd"`

**Categorical only**: `"gmb"`, `"ivb"`, `"mba"`, `"milp"`, `"sab"`,
`"sblp"`, `"swb"`

When tuning with mixed feature types, consider restricting `values` to
universal algorithms only.

## See also

[`step_obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md),
[`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)

## Examples

``` r
# Default: all algorithms
obwoe_algorithm()
#> Binning Algorithm (qualitative)
#> 29 possible values include:
#> 'auto', 'jedi', 'jedi_mwoe', 'cm', 'dp', 'dmiv', 'fetb', 'mob', 'sketch',
#> 'udt', 'gmb', 'ivb', 'mba', 'milp', 'sab', 'sblp', 'swb', 'bb', …, 'oslp', and
#> 'ubsd'

# Restrict to universal algorithms for mixed data
obwoe_algorithm(values = c("jedi", "mob", "dp", "cm"))
#> Binning Algorithm (qualitative)
#> 4 possible values include:
#> 'jedi', 'mob', 'dp', and 'cm'

# Numerical-only algorithms
obwoe_algorithm(values = c("mdlp", "fast_mdlp", "ewb", "ir"))
#> Binning Algorithm (qualitative)
#> 4 possible values include:
#> 'mdlp', 'fast_mdlp', 'ewb', and 'ir'
```
