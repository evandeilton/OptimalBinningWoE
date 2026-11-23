# Internal: Stability Against the Training Sample

PSI for the score and for every variable in the model, comparing each
sample back to the training rows. Variables are compared on their bin
shares — the bins are what the model consumes, the comparison works for
numerical and categorical variables alike, and a shift that does not
cross a cut point is a shift the model never sees.

## Usage

``` r
.ob_stability(scored, binning, features, breaks)
```
