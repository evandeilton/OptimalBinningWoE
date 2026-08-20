# Internal: Round-Trip-Safe SQL Numeric Literal

Renders a double as the shortest fixed-notation decimal string that
parses back to the identical binary value. Cut points and WoE values
must survive the round trip exactly, otherwise an observation sitting on
a boundary could fall into the wrong bin.

## Usage

``` r
.ob_sql_num(x, digits = NULL)
```

## Arguments

- x:

  Numeric vector.

- digits:

  Optional integer. When supplied, values are rounded to that many
  significant digits instead of being written exactly.

## Value

Character vector of SQL numeric literals.
