# Internal: SQL Numeric Literal That Survives the Round Trip

Renders a double as a plain decimal string that any conforming reader
parses back to the identical binary value. Cut points and WoE values
must survive the round trip exactly, otherwise an observation sitting on
a boundary could fall into the wrong bin.

Seventeen significant digits identify an IEEE 754 double uniquely, so
that width is always safe and is what a value falls back to. Shorter
strings are preferred when one of them is exact, which keeps a cut point
of `2.3` reading as `2.3`.

Whether a shorter string is exact is decided without
[`as.numeric()`](https://rdrr.io/r/base/numeric.html). R accumulates
decimal digits in `long double`, which on aarch64 macOS is no wider than
a `double`, so there
[`as.numeric()`](https://rdrr.io/r/base/numeric.html) can land one bit
from the nearest double – accepting a literal that does not round trip
and rejecting one that does. A candidate with `nd` decimals is instead
checked as `m / 10^nd`, where `m` is its digits read as an integer:
while `m` is below \\2^{53}\\ and `nd` at most 22 both operands are
exact, so IEEE 754 gives the correctly rounded quotient – the nearest
double to the candidate – on every platform. Candidates outside those
bounds are not judged, they simply lose to the fallback.

[`as.numeric()`](https://rdrr.io/r/base/numeric.html) still has to agree
before a candidate is accepted. It is not trusted to decide, but a
literal R itself reads back as a different double would be a poor thing
to write into an audit artifact, and on x86_64 its 80-bit division
rounded down to 64 bits disagrees with the correctly rounded one about
twice in every hundred thousand values.

## Usage

``` r
.ob_sql_num(x, digits = NULL)
```

## Arguments

- x:

  Numeric vector.

- digits:

  Optional integer. When supplied, values are written with that many
  decimal places instead of at full precision.

## Value

Character vector of SQL numeric literals.
