# List Available Algorithms

Returns a data frame with all available binning algorithms.

## Usage

``` r
obwoe_algorithms()
```

## Value

A data frame with algorithm information.

## Examples

``` r
obwoe_algorithms()
#>    algorithm numerical categorical multinomial
#> 1         cm      TRUE        TRUE       FALSE
#> 2       dmiv      TRUE        TRUE       FALSE
#> 3         dp      TRUE        TRUE       FALSE
#> 4       fetb      TRUE        TRUE       FALSE
#> 5       jedi      TRUE        TRUE       FALSE
#> 6  jedi_mwoe      TRUE        TRUE        TRUE
#> 7        mob      TRUE        TRUE       FALSE
#> 8       orbe      TRUE        TRUE       FALSE
#> 9     sketch      TRUE        TRUE       FALSE
#> 10       udt      TRUE        TRUE       FALSE
#> 11       gmb     FALSE        TRUE       FALSE
#> 12       ivb     FALSE        TRUE       FALSE
#> 13       mba     FALSE        TRUE       FALSE
#> 14      milp     FALSE        TRUE       FALSE
#> 15       sab     FALSE        TRUE       FALSE
#> 16      sblp     FALSE        TRUE       FALSE
#> 17       swb     FALSE        TRUE       FALSE
#> 18        bb      TRUE       FALSE       FALSE
#> 19       ewb      TRUE       FALSE       FALSE
#> 20 fast_mdlp      TRUE       FALSE       FALSE
#> 21        ir      TRUE       FALSE       FALSE
#> 22       kmb      TRUE       FALSE       FALSE
#> 23       ldb      TRUE       FALSE       FALSE
#> 24      lpdb      TRUE       FALSE       FALSE
#> 25      mblp      TRUE       FALSE       FALSE
#> 26      mdlp      TRUE       FALSE       FALSE
#> 27     mrblp      TRUE       FALSE       FALSE
#> 28      oslp      TRUE       FALSE       FALSE
#> 29      ubsd      TRUE       FALSE       FALSE
```
