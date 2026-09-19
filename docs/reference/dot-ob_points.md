# Internal: Siddiqi Points Allocation

Distributes a fitted additive model over its bins as a fixed points
table.

With \\k\\ variables in the model, an intercept \\\alpha\\ and slopes
\\\beta_j\\ on the Weight of Evidence,

\$\$\mathrm{points}\_{ij} = \frac{\mathrm{Offset}}{k} -
\mathrm{Factor}\left(\beta_j\\\mathrm{WoE}\_{ij} +
\frac{\alpha}{k}\right)\$\$

so that summing one row per variable reproduces the score exactly:
\\\sum_j \mathrm{points}\_{ij} = \mathrm{Offset} -
\mathrm{Factor}(\alpha + \sum_j \beta_j \mathrm{WoE}\_{ij}) =
\mathrm{Offset} - \mathrm{Factor}\\\eta\\.

## Usage

``` r
.ob_points(
  binning,
  coefficients,
  features,
  scaling,
  na_woe = 0,
  bin_separator = "%;%"
)
```

## Arguments

- binning:

  An `"obwoe"` object.

- coefficients:

  Named numeric with `"(Intercept)"` first.

- features:

  Character vector of the variables in the model, in the order their
  coefficients appear.

- scaling:

  An `"obwoe_scaling"` object.

- na_woe:

  Weight of Evidence assigned to a value that falls in no fitted bin.
  The per-variable points that follow from it are attached to the result
  as the `"points_na"` attribute, so that the R card score and the
  generated SQL fall back on one and the same number.

- bin_separator:

  Separator inside merged categorical bin labels.

## Value

A `data.frame` with one row per variable and bin, carrying a named
numeric `"points_na"` attribute.
