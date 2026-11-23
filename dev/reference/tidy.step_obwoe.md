# Tidy Method for step_obwoe

Returns a tibble with information about the binning transformation. For
trained steps, returns one row per bin per feature, including bin
labels, WoE values, and IV contributions. For untrained steps, returns a
placeholder tibble.

## Usage

``` r
# S3 method for class 'step_obwoe'
tidy(x, ...)
```

## Arguments

- x:

  A step_obwoe object.

- ...:

  Additional arguments (currently unused).

## Value

A tibble with columns:

- terms:

  Character. Feature name.

- bin:

  Character. Bin label or interval.

- woe:

  Numeric. Weight of Evidence value for the bin.

- iv:

  Numeric. Information Value contribution of the bin.

- id:

  Character. Step identifier.
