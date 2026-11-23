# Internal: Resolve the Target into a 0/1 Event Vector

Records which level was treated as the event, because getting this wrong
inverts the score while leaving every other statistic — IV, KS, Gini,
the coefficient signs — completely unchanged.

## Usage

``` r
.ob_resolve_target(y)
```

## Arguments

- y:

  The raw target column.

## Value

A list with `y` (integer 0/1) and `event_level`.
