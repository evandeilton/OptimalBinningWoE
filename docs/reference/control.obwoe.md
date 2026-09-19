# Control Parameters for Optimal Binning Algorithms

Constructs a validated list of control parameters for the
[`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
master interface. These parameters govern the behavior of all supported
binning algorithms, including convergence criteria, minimum bin sizes,
and optimization limits.

## Usage

``` r
control.obwoe(
  bin_cutoff = 0.05,
  max_n_prebins = 20,
  convergence_threshold = 1e-06,
  max_iterations = 1000,
  bin_separator = "%;%",
  verbose = FALSE,
  ...
)
```

## Arguments

- bin_cutoff:

  Numeric value in \\(0, 1)\\ specifying the minimum proportion of total
  observations that a bin must contain. Bins with fewer observations are
  merged with adjacent bins. Serves as a regularization mechanism to
  prevent overfitting and ensure statistical stability of WoE estimates.
  Recommended range: 0.02 to 0.10. Default is 0.05 (5%).

- max_n_prebins:

  Integer specifying the maximum number of initial bins created before
  optimization. For high-cardinality categorical features, categories
  with similar event rates are pre-merged until this limit is reached.
  Higher values preserve more granularity but increase computational
  cost. Typical range: 10 to 50. Default is 20.

- convergence_threshold:

  Numeric value specifying the tolerance for algorithm convergence.
  Iteration stops when the absolute change in Information Value between
  successive iterations falls below this threshold: \\\|IV\_{t} -
  IV\_{t-1}\| \< \epsilon\\. Smaller values yield more precise solutions
  at higher computational cost. Typical range: \\10^{-4}\\ to
  \\10^{-8}\\. Default is \\10^{-6}\\.

- max_iterations:

  Integer specifying the maximum number of optimization iterations.
  Prevents infinite loops in degenerate cases. If the algorithm does not
  converge within this limit, it returns the best solution found.
  Typical range: 100 to 10000. Default is 1000.

- bin_separator:

  Character string used to concatenate category names when multiple
  categories are merged into a single bin. Should be a string unlikely
  to appear in actual category names. Default is `"%;%"`.

- verbose:

  Logical indicating whether to print progress messages during feature
  processing. Useful for debugging or monitoring long-running jobs.
  Default is `FALSE`.

- ...:

  Additional named parameters reserved for algorithm-specific
  extensions. Currently unused but included for forward compatibility.

## Value

An S3 object of class `"obwoe_control"` containing all specified
parameters. This object is validated and can be passed directly to
[`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md).

## Details

### Parameter Impact on Results

**bin_cutoff**: Lower values allow smaller bins, which may capture
subtle patterns but risk unstable WoE estimates. The variance of WoE
estimates increases as \\1/n_i\\ where \\n_i\\ is the bin size. For bins
with fewer than ~30 observations, consider using Laplace or Bayesian
smoothing (applied automatically by most algorithms).

**max_n_prebins**: Critical for categorical features with many levels.
If a feature has 100 categories, setting `max_n_prebins = 20` will
pre-merge similar categories into 20 groups before optimization.

For *numerical* features this parameter is a modelling decision, not an
implementation detail, and the default of 20 is not neutral. Pre-binning
runs before any algorithm sees the data, so a heavy-tailed variable
whose signal sits in the upper tail can have that tail smeared across a
single quantile cell and lost before optimization begins. The effect is
silent: the fit reports `converged = TRUE` and no warning.

Benchmarks on two open datasets (76,020 x 369 and 590,540 x 454,
five-fold held-out IV, `min_bins` and `max_bins` held fixed) show the
size of it. Raising `max_n_prebins` from 20 to 200 moved the median
held-out IV of heavy-tailed numerical predictors by **+129%** and
**+39%** respectively, while leaving discrete and categorical predictors
untouched. The bin count barely moved, so this is the placement of cut
points changing, not extra splits.

It moves results in **both** directions, which is why the default is
left alone. On one of the two benchmarks, raising it improved the
portfolio but cost 15% of held-out IV on the twenty-five strongest
predictors – exactly the ones a scorecard would keep – while inflating
IV on weak predictors no model would select. No cheap rule separated the
two cases: gating on the tail ratio worked cleanly on one dataset and
captured almost nothing on the other, and for roughly half of all
predictors the default of 20 was already the best value tested.

The practical advice is therefore per-variable rather than global: leave
the default, and tune it against *held-out* data for continuous
predictors with a long tail. Tuning on in-sample IV will always favour
more resolution and is not evidence of anything.

**convergence_threshold**: Trade-off between precision and speed. For
exploratory analysis, \\10^{-4}\\ is sufficient. For production models
requiring reproducibility, use \\10^{-8}\\ or smaller.

## See also

[`obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
for the main binning interface.

## Examples

``` r
# Default control parameters
ctrl_default <- control.obwoe()
print(ctrl_default)
#> $bin_cutoff
#> [1] 0.05
#> 
#> $max_n_prebins
#> [1] 20
#> 
#> $convergence_threshold
#> [1] 1e-06
#> 
#> $max_iterations
#> [1] 1000
#> 
#> $bin_separator
#> [1] "%;%"
#> 
#> $verbose
#> [1] FALSE
#> 
#> attr(,"class")
#> [1] "obwoe_control"

# Conservative settings for production
ctrl_production <- control.obwoe(
  bin_cutoff = 0.03,
  max_n_prebins = 30,
  convergence_threshold = 1e-8,
  max_iterations = 5000
)

# Aggressive settings for exploration
ctrl_explore <- control.obwoe(
  bin_cutoff = 0.01,
  max_n_prebins = 50,
  convergence_threshold = 1e-4,
  max_iterations = 500
)
```
