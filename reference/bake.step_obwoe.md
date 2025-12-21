# Apply the Optimal Binning Transformation

Applies the learned binning and WoE transformation to new data. This
method is called by
[`bake`](https://recipes.tidymodels.org/reference/bake.html) and should
not be invoked directly.

## Usage

``` r
# S3 method for class 'step_obwoe'
bake(object, new_data, ...)
```

## Arguments

- object:

  A trained step_obwoe object.

- new_data:

  A tibble or data frame to transform.

- ...:

  Additional arguments (currently unused).

## Value

A tibble with transformed columns according to the `output` parameter.
