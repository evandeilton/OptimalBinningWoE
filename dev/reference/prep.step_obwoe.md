# Prepare the Optimal Binning Step

Fits the optimal binning models on training data. This method is called
by [`prep`](https://recipes.tidymodels.org/reference/prep.html) and
should not be invoked directly.

## Usage

``` r
# S3 method for class 'step_obwoe'
prep(x, training, info = NULL, ...)
```

## Arguments

- x:

  A step_obwoe object.

- training:

  A tibble or data frame containing the training data.

- info:

  A tibble with column metadata from the recipe.

- ...:

  Additional arguments (currently unused).

## Value

A trained step_obwoe object with `binning_results` populated.
