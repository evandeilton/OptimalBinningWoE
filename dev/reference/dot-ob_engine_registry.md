# Internal: The Engine Contract

Every engine is three functions and nothing else:

- `fit(x, y, args)`:

  `x` is a numeric matrix of Weight of Evidence columns with no
  intercept column, `y` an integer 0/1 vector. Returns an opaque fitted
  object.

- `link(object, x)`:

  Returns the log-odds of the **event**, one per row of `x`. Mandatory.

- `coef(object)`:

  Returns a named numeric vector with `"(Intercept)"` first, **or
  `NULL`** to declare that the model is not additive in the WoE
  predictors.

The `NULL` return is the honest-degradation switch. A scorecard's points
table exists only because the model is a sum of per-variable terms; a
tree ensemble is not, so it cannot be decomposed into points and the
pipeline must say so rather than fabricate a table.

## Usage

``` r
.ob_engine_registry()
```

## Value

A named list of engine definitions.
