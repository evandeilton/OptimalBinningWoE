# Points-to-Double-the-Odds Score Scaling

Computes the two constants that turn a model's log-odds into scorecard
points: the *factor* that fixes how many points double the odds, and the
*offset* that anchors a reference score to a reference odds.

## Usage

``` r
obwoe_scale(
  pdo = 20,
  score_ref = 600,
  odds_ref = 50,
  direction = c("higher_is_safer", "higher_is_riskier")
)
```

## Arguments

- pdo:

  Numeric. Points to Double the Odds. Default `20`.

- score_ref:

  Numeric. The reference score. Default `600`.

- odds_ref:

  Numeric. The good-to-bad odds at `score_ref`, expressed as a ratio
  (`50` means 50 non-events per event). Default `50`.

- direction:

  Character string fixing which end of the scale is safe:
  `"higher_is_safer"` (default, the scorecard convention) or
  `"higher_is_riskier"`.

## Value

An object of class `"obwoe_scaling"`: a list with `pdo`, `score_ref`,
`odds_ref`, `direction`, `factor` and `offset`.

## Details

Scorecards are scaled on the **good-to-bad** odds:

\$\$\mathrm{Score} = \mathrm{Offset} + \mathrm{Factor}\cdot
\ln(\mathrm{odds}\_{good:bad}), \qquad \mathrm{Factor} =
\frac{\mathrm{PDO}}{\ln 2}, \qquad \mathrm{Offset} = \mathrm{Score}\_0 -
\mathrm{Factor}\cdot\ln(\mathrm{Odds}\_0)\$\$

Every model in this package predicts the log-odds of the **event** (the
bad), which is the other direction, so the deployed form is

\$\$\mathrm{Score} = \mathrm{Offset} - \mathrm{Factor}\cdot \eta\$\$

The minus sign is the whole point: a bin with a high Weight of Evidence
is a risky bin, and a risky bin must score *fewer* points.

Two identities pin the scaling completely, and
[`obwoe_score`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_score.md)
is tested against both:

- **Anchor.** A case whose odds are `odds_ref` scores exactly
  `score_ref`.

- **Scale.** Doubling the good-to-bad odds adds exactly `pdo` points, at
  every point of the scale.

## References

Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
Intelligent Credit Scoring. *John Wiley & Sons*.
[doi:10.1002/9781119201731](https://doi.org/10.1002/9781119201731)

## See also

[`obwoe_score`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_score.md)
to apply the scaling,
[`obwoe_scorecard`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scorecard.md)
for the full pipeline.

## Examples

``` r
scaling <- obwoe_scale(pdo = 20, score_ref = 600, odds_ref = 50)
scaling
#> Scorecard scaling
#>   20 points double the odds; 600 points at 50:1 good:bad
#>   factor = 28.853901, offset = 487.122876
#>   score = offset - factor * link   (higher_is_safer)

# a case at the reference odds scores the reference score
obwoe_score(-log(50), scaling)
#> [1] 600

# halving the odds of being good costs exactly one PDO
obwoe_score(-log(25), scaling)
#> [1] 580
```
