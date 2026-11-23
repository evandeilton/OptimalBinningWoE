# Internal: Resolve an Engine

Internal: Resolve an Engine

## Usage

``` r
.ob_engine_get(engine = "glm", fallback = FALSE)
```

## Arguments

- engine:

  Character string naming a registered engine, or a list supplying
  `fit`, `link` and `coef` directly.

- fallback:

  Logical. Fall back to `"glm"` when the engine's package is not
  installed?

## Value

An engine definition, with `name`, `requested` and `used` recorded.

## Details

A missing engine package is an error, not a silent substitution. A
scorecard is a governance artefact: a workbook documenting a model the
analyst did not ask for is worse than a call that fails.
`fallback = TRUE` opts into the substitution and records both the
requested and the used engine so the workbook can say which one produced
it.
