# Internal: Fixed-Notation Decimal With a Given Number of Decimals

Formats finite doubles with `nd` decimal places and drops the trailing
zeros. [`sprintf()`](https://rdrr.io/r/base/sprintf.html) is used rather
than [`format()`](https://rdrr.io/r/base/format.html) because it
delegates to the C library's correctly rounded binary-to-decimal
conversion, which behaves identically on every platform R runs on.

## Usage

``` r
.ob_sql_decimal(v, nd)
```

## Arguments

- v:

  Numeric vector of finite values.

- nd:

  Number of decimal places; recycled against `v`.

## Value

A character vector in plain decimal notation.
