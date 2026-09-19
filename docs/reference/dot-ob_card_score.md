# Internal: Score From the Fixed Points Table

Internal: Score From the Fixed Points Table

## Usage

``` r
.ob_card_score(bins, points, features, na_points = attr(points, "points_na"))
```

## Details

A value in no fitted bin scores that variable's `"points_na"` fallback
rather than `NA`, which is what the deployed SQL does and what a
production scorer has to do: a single unseen category cannot void the
whole application's score.
