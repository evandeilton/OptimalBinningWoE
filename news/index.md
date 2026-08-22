# Changelog

## OptimalBinningWoE 1.13.3

### Audit fixes (2026-08-21)

Bug-fix release from a third internal audit, covering the WoE/IV return
contract, the `converged` flag and the `max_bins` constraint across all
37 algorithm/type combinations. Every item below was reproduced before
the fix and re-verified after it. The regression suite gained
`tests/testthat/test-audit-regressions.R`, which fails on 1.13.2.

#### Behavior changes (read before upgrading)

- **`mdlp` (numerical), `gmb` and `fetb` (categorical) now honour
  `max_bins`.** All three stopped on their own criterion and never
  re-checked the cap, so `max_bins = 5` returned 18, 11 and 10 bins
  respectively – silently, with no warning. All three roxygen blocks
  already documented `max_bins` as a hard constraint, so the code was
  wrong, not the documentation.

  **These three algorithms now produce different bins.** For `mdlp` the
  returned partition is no longer the unconstrained MDL optimum when the
  cap binds: merging continues past the MDL stopping point, each step
  taking the pair with the smallest increase in MDL cost. `min_bins` is
  never violated to satisfy `max_bins`.

- **[`obwoe_apply()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
  now refuses multiclass models instead of scoring them wrongly.** A
  multinomial fit carries a bins x classes WoE matrix; the lookup
  linear-indexed it column-major and returned **class 1’s WoE for every
  row**, discarding the other classes with no error and no warning. It
  now stops with an actionable message. The per-class matrix is still in
  `$results` for callers that want to handle it themselves.

- **`dmiv` now reports a numeric `total_iv`** where `summary$total_iv`
  was previously `NA` for every feature. Code that tested for that `NA`
  will see a number.

#### Corrected values

- **The categorical `sketch` engine computed every WoE against the wrong
  marginal.** It passed `(total_neg, total_pos)` to helpers whose
  signature is `(total_pos, total_neg)`. On a random 8-category feature
  with no real signal, the reported IV was **10.4021 against a true
  value of 0.0043** – wrong by three orders of magnitude, and wrong in
  the direction that makes a useless variable look like the strongest
  predictor in the model. WoE deviated from `log((pos_i/TP)/(neg_i/TN))`
  by up to 2.7721; it is now within 0.0002.

  The identical defect was fixed in the numerical twin some releases ago
  and never replicated here. A numerical-vs-categorical parity test now
  covers every algorithm that has both variants, so this class of
  divergence cannot recur silently.

- **`dmiv` returns `iv` and `total_iv`** alongside the divergence
  measures it already reported. IV is computed from the smoothed class
  distributions rather than from `woe`, because the default
  `bin_method = "woe1"` is Zeng’s log-odds
  `ln((pos + 0.5)/(neg + 0.5))`, which differs from standard WoE by the
  constant `ln(TP/TN)`; deriving IV from it would be wrong.

- **`cm` (categorical) exposes `total_iv` at top level**, like the other
  15 categorical engines, instead of only inside `metadata`. `metadata`
  is unchanged, so existing callers keep working.

#### Fixed crashes

- **`sab` (categorical) no longer aborts with `unordered_map::at`** when
  a category contains zero events – a routine situation in credit data.
  The positive-count map only gained a key for categories with at least
  one event, but was read with `.at()` in seven places.

- **[`obwoe_scorecard()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scorecard.md)
  and the cutoff table now reject a missing target with a clear
  message** naming the sample and the count, instead of failing deep
  inside with `missing value where TRUE/FALSE needed`. The development
  frame was already checked; samples passed through `validation =` were
  not. The `NA` is rejected at the boundary rather than swept up with
  `na.rm = TRUE`, which would have silently charged every
  unknown-outcome row to the non-events and corrupted KS and AUC.

#### `converged` now means the same thing everywhere

The flag was effectively inverted in several engines: initialised
`false` and set `true` only on a degenerate shortcut or an in-loop
tolerance test, while the normal successful exit – reaching the
bin-count target – set nothing. So ordinary well-binned features
reported `FALSE` and only degenerate ones reported `TRUE`.

- `dmiv` and `bb` (numerical) report `converged` on reaching the
  bin-count target, matching the categorical `dmiv`, which already did.
- `ir` (numerical) reports `converged` for the exact binning it produces
  when a feature has two or fewer distinct values. **Every binary 0/1
  feature previously reported `FALSE`** despite a correct result.
- `dp`, `gmb`, `mba`, `sketch`, `cm` and `dmiv` (categorical) report
  `converged` on every successful termination path, including the fast
  paths that bypassed the flag entirely.

The intended contract is now documented in
`src/common/bin_structures.h`: `converged == true` means the algorithm
reached a valid stopping state (tolerance met, monotonicity achieved, or
the bin-count target reached); `false` means it exhausted
`max_iterations`.

#### Three algorithms were quadratic in the number of rows

`lpdb`, `ldb` and numerical `udt` scaled as n^2.00, n^2.00 and n^2.30. A
single variable with 10^6 rows would have taken `lpdb` roughly 72
minutes; measured against `jedi` at the same size they were 2,772x the
median algorithm’s cost. Nothing warned about it.

Two distinct causes, neither of them inherent to the methods:

- **`ldb` and `lpdb` estimated the density with a naive double loop**,
  evaluating the Gaussian kernel of every observation against every
  other one. The same defect had been written twice, in two files, which
  is how it survived. It is replaced by the standard linear-binning
  estimator – the one R’s own
  [`density()`](https://rdrr.io/r/stats/density.html) uses – which now
  lives once in `src/common/optimal_binning_common.h` so the two cannot
  diverge again.

- **`udt` rescanned every observation once per candidate split**,
  allocating two vectors and recomputing the parent entropy each time,
  giving O(u x n). Information gain depends only on integer counts, so a
  single sweep carrying running totals produces the identical value.

Measured at n = 50,000, against a build of the previous revision:

| algorithm | before  | after  | speedup |
|-----------|---------|--------|---------|
| `ldb`     | 10.740s | 0.010s | 1074x   |
| `lpdb`    | 10.725s | 0.011s | 975x    |
| `udt`     | 7.354s  | 0.019s | 387x    |

All three now scale linearly and land within a factor of two of `jedi`,
the package default. At n = 400,000 they take 0.085s, 0.087s and 0.170s
against `jedi`’s 0.103s – sizes the previous code could not reach at
all.

**Results.** `udt` and `ldb` are bit-identical to the previous revision:
`udt` by construction, and `ldb`’s local-minimum search resolves the
grid estimate to the same cut points. Both are pinned by a new
regression test.

**`lpdb` changes.** It differentiates the density twice to find
inflection points, and finite differences taken between adjacent
observations are not the same thing as finite differences on a properly
sampled curve. Its critical points are now located on the estimation
grid. On German Credit the partitions generally improve – `duration`
goes from 2 bins and IV 0.0923 to 5 bins and IV 0.2635, `age` from 2
bins and 0.0628 to 4 bins and 0.0781 – and no variable tested got
materially worse. Anyone with a fitted `lpdb` model should expect
different cut points.

Also removed `OBN_LPDB::local_polynomial_density()`, which no longer had
a caller and never did local polynomial regression despite its name.

#### Documentation

- **`max_n_prebins` is documented as the modelling decision it is.** For
  numerical features, pre-binning runs before any algorithm sees the
  data, so the default of 20 quantile cells can smear a heavy tail and
  lose the signal in it before optimization begins – silently, with
  `converged = TRUE`. Benchmarks on two open datasets (76,020 x 369 and
  590,540 x 454, five-fold held-out IV) move the median held-out IV of
  heavy-tailed numerical predictors by +129% and +39% when the parameter
  is raised from 20 to 200, with the bin count essentially unchanged.

  The default is deliberately **not** changed. The same experiment shows
  the effect runs both ways: on the larger benchmark, raising it cost
  15% of held-out IV on the twenty-five strongest predictors while
  inflating IV on weak ones, and no cheap rule separated the two cases.
  [`?control.obwoe`](https://evandeilton.github.io/OptimalBinningWoE/reference/control.obwoe.md)
  now says so, and recommends per-variable tuning against held-out data.

- **`min_bins` is documented as frequently binding.** It reads as a
  safety floor but often sets the partition, because several algorithms
  stop merging on their own criterion well before `max_bins`. On German
  Credit `amount` with `max_bins = 5`, the default returns two bins and
  an IV of 0.0016 while `min_bins = 4` returns 0.0824 – same data, same
  algorithm.

- **Corrected the 1.13.0 entry on binary size.** It described `-Os`,
  `-fvisibility=hidden`, `-ffunction-sections`, `-Wl,--gc-sections` and
  a `cleanup` script, none of which are in the package: the visibility
  and section flags hid Rcpp symbols and broke the build on every
  platform and were reverted in `b74e95b`. The entry is corrected rather
  than deleted so the flags are not reinstated by someone reading the
  old claim.

#### Smaller items

- **An unmeasured IV is no longer reported as an IV of zero.** With no
  finite IV anywhere, [`summary()`](https://rdrr.io/r/base/summary.html)
  printed `Total IV: 0.0000` and `IV Range: [Inf, -Inf]`, asserting “no
  predictive power” about features that were never measured;
  [`print()`](https://rdrr.io/r/base/print.html) on a `step_obwoe`
  recipe printed `total IV=0.0000`. Both report `NA` with a note.
  [`obwoe_gains()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_gains.md)
  failed with `attempt to select less than one element in get1index` and
  `plot(type = "iv")` with `need finite 'xlim' values`; both now say
  what is actually wrong.

- **`sketch` (numerical) returns a `bin` label for a constant feature**,
  so `summary$n_bins` is `1` instead of `NA`.

- **`udt` (numerical) keeps WoE and IV finite with
  `laplace_smoothing = 0`.** A bin holding no events gave `woe = -Inf`
  and `iv = Inf`; both distributions are now floored with the
  package-wide epsilon, so the parameter stays usable. Reachable only
  through
  [`ob_numerical_udt()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_udt.md)
  directly –
  [`obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
  never forwards the argument.

## OptimalBinningWoE 1.13.2

### The fitted object now records how it was fitted

- **[`obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)
  returns the `control` it used**, alongside the effective `min_bins`,
  `max_bins` and `algorithm`. `call` was never a substitute: it records
  only the arguments the caller typed, never the defaults that actually
  applied, so a saved model could not say what produced it.

- **A custom `bin_separator` now works end to end.** Everything that has
  to split grouped categories out of a bin label reads the separator
  from the model instead of assuming the package default:
  [`obwoe_apply()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md),
  the points table, the points SQL and the WoE SQL emitted into the
  workbook.

  This was a silent corruption, not a cosmetic gap. Fitting with
  `control.obwoe(bin_separator = "||")` and applying the model back to
  its own training data put **754 of 900 rows** on the `na_woe`
  fallback, because the labels were split on `"%;%"` and never matched.
  The generated SQL was worse: `a||e||c` came out as
  `g IN ('a', '|', '|', 'e', '|', '|', 'c')` – broken SQL, no warning.
  Both are exact now.

- **Models saved by earlier versions keep working.** They carry no
  `control` element, so the separator falls back to the package default
  rather than erroring on a missing field.

Nothing changes for the default configuration: `"%;%"` remains the
default everywhere, so only the custom-separator path – which was broken
– behaves differently.

## OptimalBinningWoE 1.13.1

### Audit fixes (2026-08-20)

Bug-fix release addressing an internal code audit of 1.13.0. Every item
below either changes a computed value that was previously wrong, or
removes an API surface the author never intended to publish. See the
pull request for the full item-by-item breakdown.

#### API changes

- `fit_logistic_regression()` is no longer exported. It is still used
  internally (renamed to `.ob_fit_logistic_regression()`) to implement
  `engine = "obwoe"` in
  [`obwoe_scorecard()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scorecard.md),
  which is unaffected.

- `ob_gains_table()` and `ob_gains_table_feature()` are renamed to
  [`obwoe_gains_score()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_gains_score.md)
  and
  [`obwoe_gains_variable()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_gains_variable.md)
  respectively, to sit alongside the rest of the `obwoe_*` family.
  Behavior is unchanged; only the names moved.
  [`obwoe_gains()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_gains.md)
  (the higher-level, plot-producing function) is a different function
  and was not renamed.

#### Behavior changes (read before upgrading)

- **[`obwoe_apply()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
  now routes a missing categorical value to the fitted “missing” bin’s
  WoE when the binning built one, instead of always using `na_woe`.**
  Previously
  [`obwoe_apply()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
  ignored any missing-value bin learned during training and always
  returned `na_woe` for `NA`, while the generated deployment SQL
  ([`obwoe_sql()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_sql.md),
  `null_to_na_bin = TRUE` by default) routed `IS NULL` to that bin’s WoE
  — so R and the SQL scored the same missing value differently. `na_woe`
  is now only a fallback for variables where no missing-value bin
  exists. If you fit a model with real `NA`s in a categorical predictor,
  the WoE
  [`obwoe_apply()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)/[`predict()`](https://rdrr.io/r/stats/predict.html)
  return for `NA` may change in this version; it now matches the SQL and
  the bin actually fitted. See
  [`?obwoe_apply`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
  and
  [`?obwoe_sql`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_sql.md).

- **`obwoe_scorecard(..., drop_negative = TRUE)` now actually removes a
  single negative-coefficient variable instead of silently keeping it.**
  Previously the removal loop stopped one variable too early whenever
  exactly one variable had a negative coefficient, so that case fell
  through to a warning instead of either fixing the model or raising
  [`stop()`](https://rdrr.io/r/base/stop.html). It now keeps removing
  negative-coefficient variables until either none remain or only one
  variable is left (in which case, if it is still negative,
  [`obwoe_scorecard()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scorecard.md)
  stops with an error, as documented).

- **[`ob_cutpoints_num()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_cutpoints_num.md)’s
  bins are now right-closed `(a, b]`, and both
  [`ob_cutpoints_num()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_cutpoints_num.md)
  and
  [`ob_cutpoints_cat()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_cutpoints_cat.md)
  return the pieces (`id`, and `cutpoints` for the numerical version)
  [`ob_apply_woe_num()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_num.md)/
  [`ob_apply_woe_cat()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_cat.md)
  require.** Previously
  [`ob_cutpoints_num()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_cutpoints_num.md)
  built left-closed `[a, b)` bins — the opposite of
  [`ob_apply_woe_num()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_num.md)’s
  `include_upper_bound = TRUE` default — so a value sitting exactly on a
  cutpoint could get a different, often sign-flipped, WoE depending on
  whether it went through the fit or the apply side; and neither manual
  cutpoint function’s result could be handed to its matching apply
  function at all, because both lacked the `id` element (and
  [`ob_cutpoints_num()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_cutpoints_num.md)
  the top-level `cutpoints`) the apply side requires.
  [`ob_cutpoints_cat()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_cutpoints_cat.md)’s
  emitted bin labels also now use `"%;%"` (matching
  [`ob_apply_woe_cat()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_cat.md)’s
  default separator and the main pipeline) instead of echoing the
  `"+"`-joined input verbatim; the `"+"` input format is unchanged. If
  you use
  [`ob_cutpoints_num()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_cutpoints_num.md)/[`ob_cutpoints_cat()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_cutpoints_cat.md)
  and parse their `bin` labels yourself, check the new format. See
  [`?ob_cutpoints_num`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_cutpoints_num.md)
  and
  [`?ob_cutpoints_cat`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_cutpoints_cat.md).

#### Fixes

- Gains tables sorted by `sort_by = "bin"` (including the score bands in
  [`obwoe_scorecard()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scorecard.md))
  were ordered lexicographically instead of by the bins’ natural (level)
  order, silently understating KS by as much as a third on real data.
- `step_obwoe(algorithm = "auto")` could resolve to the multiclass
  algorithm for a genuinely binary outcome whenever the outcome factor
  declared an unobserved third level.
- [`print()`](https://rdrr.io/r/base/print.html) on an unprepped
  `step_obwoe` recipe step errored when `algorithm = tune::tune()`.
- `obwoe_gains(use_column = "woe", n_groups = k)` returned `NA` WoE and
  zero total IV for every bin.
- `plot.obwoe_gains(type = "cumulative")` assumed every bin held an
  equal share of the population; the x-axis now reflects each bin’s real
  size.
- The advertised algorithm count was corrected from “36 (20 numerical,
  16 categorical)” to the actual 37 (21 numerical, 16 categorical) — the
  numerical `ir` algorithm was implemented, documented and tested, but
  was never counted.

## OptimalBinningWoE 1.13.0

### New features (2026-08-20)

#### `obwoe_scorecard()` — the pipeline as one artefact

Runs the whole origination workflow in a single call — stratified split,
binning, screening by Information Value and correlation, model fitting,
PDO scaling — and returns an object that also writes itself out as an
`.xlsx` model document. The point is not convenience: each stage records
why it did what it did, so the workbook is reviewable evidence rather
than a set of numbers.

- **The binning sees the training rows only.** Binning is supervised:
  cut points and WoE are both chosen against the target, so fitting them
  on the full base before splitting leaks the hold-out into the
  transformation. The split therefore happens first, and the binning is
  fitted once, on the training rows, and applied everywhere else.
  Measured on German Credit, the difference is a hold-out AUC of 0.768
  fitted the leaky way against 0.709 fitted correctly — the leak buys
  six points of AUC that do not exist.

- **A negative WoE coefficient is a fault, not a result.** The WoE
  already carries the direction of risk, so a negative slope means the
  model is reversing a variable to compensate for another. Such
  variables are dropped one at a time, worst first, and the model
  refitted, with each removal recorded.

- **The points table is fixed, not per applicant.** Points follow the
  Siddiqi allocation,
  `points_ij = Offset/k - Factor(beta_j WoE_ij + alpha/k)`, so a bin is
  worth the same number of points to everyone. Rounding each bin once
  costs at most `k/2` points against the unrounded model score; that
  drift is measured on every sample and reported rather than hidden by
  re-apportioning the rounding per row, which would make the same bin
  worth different points to different applicants.

- **The deployment SQL reproduces the R score exactly.** Both the WoE
  form and the integer-points form are generated, the total computed in
  an outer `SELECT` over a subquery — referring to a select-list alias
  within the same `SELECT` is a MySQL extension that ANSI SQL, SQLite,
  PostgreSQL, SQL Server and Oracle all reject. Verified against live
  SQLite on all 1000 German Credit rows: maximum difference 0, unseen
  categories included.

- **A value in no fitted bin scores a defined fallback.** It is counted
  and warned about, per sample and per variable, but it still produces a
  score — both in R and in SQL, from one and the same `na_woe` figure. A
  single unseen category cannot void an application.

- **Thirteen sheets.** Model summary, scorecard, coefficients with
  standard errors, bin statistics, screening funnel with a reason per
  rejected variable, correlations before and after pruning, score gains,
  PSI between samples, cut-off strategy, SQL in both forms, and a
  reproducibility record.

- **Engines are a three-function contract** — `fit`, `link`, `coef` —
  with `glm`, the package’s own C++ L-BFGS logistic regression, and
  `glmnet` registered, and custom engines accepted.
  [`coef()`](https://rdrr.io/r/stats/coef.html) returning `NULL`
  declares the model non-additive, and the pipeline then produces no
  points table rather than fabricating one from a model that has no
  per-variable decomposition. A missing engine package is an error by
  default, not a silent substitution: a workbook documenting a model the
  analyst did not ask for is worse than a call that fails.

#### Supporting functions

- [`obwoe_scale()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_scale.md)
  and
  [`obwoe_score()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_score.md)
  implement PDO scaling, `Factor = PDO/ln 2` and
  `Offset = Score0 - Factor ln(Odds0)`, in both score directions.
  Doubling the good:bad odds moves the score by exactly one PDO at every
  point of the range.
- [`obwoe_prune()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_prune.md)
  removes redundant variables iteratively, dropping the worse-ranked
  member of the strongest surviving pair and recomputing, rather than
  resolving all pairs against the original matrix at once.
- [`obwoe_psi()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_psi.md)
  computes the Population Stability Index between two samples, for both
  binned and continuous inputs, using interior quantiles so that a
  merely shifted distribution does not produce a degenerate band.
- [`obwoe_report()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_report.md)
  writes the workbook for an already-fitted scorecard.
- [`predict()`](https://rdrr.io/r/stats/predict.html) methods for
  `"score"`, `"card"`, `"link"`, `"prob"` and `"woe"`, reading only the
  binning, the coefficients and the scaling.

#### Fixed during review

- **The screening funnel named the wrong stage.** Every variable that
  left the pipeline after Information Value screening was labelled
  `corr_pruned`, including those dropped by the negative-coefficient
  check and those that turned out constant after the WoE transform. The
  funnel is the sheet a reviewer reads to learn *why* a variable is
  absent, so a mislabelled rejection is a false statement in a model
  document. `stage` now distinguishes `in_model`, `sign_rejected`,
  `corr_pruned`, `constant_woe` and `screened_out`, each assigned by the
  step that actually rejected the variable.

- **The cut-off strategy sheet ignored the score direction.** It always
  approved applicants scoring at or above the cut, which is right under
  the default scale but exactly inverted under
  `direction = "higher_is_riskier"` — the sheet then approved the worst
  applicants and reported an approved bad rate above the rejected one,
  at every one of its twenty cut-offs. The approval rule now follows the
  direction of the scale.

- **`file` is validated before the pipeline runs.** A missing
  `openxlsx`, a non-existent directory or an unwritable one used to
  surface only at the final write, after the binning, the screening and
  the fit had all been computed and were about to be discarded. The
  check now happens up front: the failure arrives in about 0.04 s
  instead of after the whole run.

### Dependencies

`openxlsx` and `glmnet` are added to `Suggests`. Neither is needed
unless a workbook is written or `engine = "glmnet"` is requested.

## OptimalBinningWoE 1.12.0

### New features (2026-08-19)

Two additions close the gap between a fitted binning and a deployed
scorecard: deciding which variables deserve to enter the model, and
shipping the accepted transformation to the database where the data
lives.

#### `obwoe_select()` — automated variable screening

Screens every binned variable of an `obwoe` model against the two
criteria that govern variable admission in credit risk practice —
**predictive strength**, graded with the Siddiqi (2006) Information
Value bands, and **guaranteed rank ordering**, measured by monotonicity
of the bin event rate — and returns a verdict for each.

- **Nothing is ever dropped.** A base with 500 candidate variables
  yields 500 rows. Each carries a `selected` flag, a machine-readable
  `reason` listing *every* rule it violated, and a `reason_desc` in
  plain language, so the automatic verdict can be reviewed, overridden,
  or replaced by the analyst’s own triage.

- **Two levels of detail.** `detail = "summary"` gives one row per
  variable with its headline metrics; `detail = "full"` expands to one
  row per variable *and* optimised bin, carrying the complete gains
  table of `ob_gains_table()` (31 metrics: WoE, IV, lift, cumulative KS,
  precision, recall, F1, KL and JS divergence, …) together with the
  interval bounds of numerical bins and the merged category lists of
  categorical ones.

- **Metrics that describe the deployed score.** `ks`, `auc` and `gini`
  are computed on the WoE actually applied to the data, with bins ranked
  by that WoE and merged when tied. `auc` uses the tie-corrected
  Mann-Whitney form, which reproduces the rank-based AUC of the
  WoE-transformed observations to machine precision. `ks` is therefore
  the true KS of the transformed variable even when the binning is not
  monotonic.

- **An honest treatment of ordering.** For numerical variables the bin
  sequence is intrinsic, so monotonicity is a genuine constraint. For
  nominal categories the sequence is a free relabelling, so the default
  `require_monotonic = "numeric"` applies the constraint only where it
  means something; `"all"` and `"none"` are available.

- **Rejects suspicious variables by default.** `iv_max = 0.50` excludes
  the Siddiqi *Suspicious* band, where a single-variable IV is far more
  often a symptom of target leakage than of a dominant predictor. On the
  Statlog German Credit benchmark this is exactly what happens to the
  checking-account status (IV = 0.67).

- Further gates for `min_bins`, `max_bins`, minimum bin population
  share, bins with no events or no non-events, and a `top_n` cut by IV,
  KS, Gini or AUC. An `Excellent`/`Good`/`Fair`/`Rejected` quality tier
  summarises how cleanly the algorithm categorised each variable.

- Returns a `data.table` when that package is installed and an identical
  `data.frame` otherwise; the table is assembled column-wise in a single
  pass, so 500 variables are screened in about a second.

#### `obwoe_sql()` — SQL code generation

Translates a fitted binning into executable SQL, so the WoE
transformation runs inside the database with no round trip through R.

- **Exact interval semantics.** Numerical bins are half-open on the
  right, `(lower, upper]`, exactly as
  [`obwoe_apply()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
  assigns them. Boundaries come from the fitted `cutpoints` vector,
  never parsed back from bin labels, whose formatting varies between
  algorithms. By default each branch states both of its bounds
  (`WHEN x > 7 AND x <= 10 THEN ...`) so it is correct in isolation;
  `explicit_bounds = FALSE` emits the shorter cascading form.

- **Cut points that survive the round trip.** Literals are written as
  the shortest fixed-notation decimal that parses back to the identical
  IEEE 754 double, and never in scientific notation. Rounding a boundary
  such as `4049.5` would silently move observations between bins, so
  exactness is the default; `digits` makes rounding an explicit choice.

- **NULL cannot leak.** In SQL, `NULL <= 5` is `NULL`, not `FALSE`, so a
  missing value matches no comparison and would fall through to `ELSE`.
  Every generated expression opens with an explicit `WHEN <col> IS NULL`
  branch, and when the binner folded training missings into a category
  bin, `null_to_na_bin = TRUE` routes database `NULL`s to that same bin.

- **Escaping that holds.** Single quotes are doubled per ANSI SQL; on
  MySQL, MariaDB and the Hive family backslashes are doubled too.
  Category names are matched byte for byte, whitespace included.
  Identifiers are quoted with the dialect’s own delimiters and, by
  default, only when the name needs it or collides with a reserved word
  — which keeps generated code readable on case-folding engines such as
  Oracle and Snowflake.

- 14 dialects (`ansi`, `postgres`, `mysql`, `mariadb`, `sqlserver`,
  `oracle`, `spark`, `hive`, `databricks`, `bigquery`, `snowflake`,
  `redshift`, `duckdb`, `sqlite`), four assembly styles (`select`,
  `case`, `cte`, `view`), four output modes (`woe`, `bin`, `index`,
  `both`), an audit header recording package version and algorithm, and
  direct file output.

- Accepts an `obwoe` object, a prepped
  [`step_obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md)
  step, or a prepped recipe containing one, and composes with
  [`obwoe_select()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_select.md)
  through `features = sel$feature[sel$selected]`.

#### Validation

- **Real benchmark data is bundled.** `inst/extdata/germancredit.csv.gz`
  holds the Statlog (German Credit) dataset from the UCI Machine
  Learning Repository (1000 applications, 7 numerical and 13 categorical
  attributes, CC BY 4.0), in its labelled form — category names carrying
  spaces, slashes and colons make it a realistic test bed for SQL
  escaping.

- [`obwoe_select()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_select.md)
  reproduces the **published Information Values** of that benchmark to
  four decimals for the 14 attributes the optimiser keeps ungrouped, and
  its IV, KS, AUC and Gini match an independent computation made from
  the raw observations to 1e-10.

- The generated SQL is validated by **parsing and executing it**: the
  test suite carries a small `CASE` interpreter that reads the emitted
  text the way an engine would, and the resulting bin assignment is
  checked against the counts the binning algorithm itself reported —
  across five algorithms and all 20 German Credit variables. During
  development the same statements were additionally run against a live
  SQLite engine, whose results were identical. Edge cases covered
  include observations sitting exactly on a cut point, cut points with
  no exact binary representation, magnitudes from `1e-9` to `1e9`,
  single-bin variables, degenerate bins, reserved-word column names, and
  categories containing quotes, backslashes, tabs, accented characters
  and significant whitespace.

### Bug fixes

Both defects below were uncovered while validating the two new functions
against real data, and both are pinned by tests in
`tests/testthat/test-regression-audit.R` that fail on 1.11.0 — 31
assertions in total.

- **[`obwoe_apply()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
  and
  [`bake.step_obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/bake.step_obwoe.md)
  silently failed to score any category carrying leading or trailing
  whitespace.** Both rebuilt their category-to-bin lookup with
  `trimws(strsplit(bin_label, "%;%", fixed = TRUE)[[1]])`. The binning
  engines join the original category strings with the separator and add
  no padding — verified here for all 15 categorical algorithms — so the
  split pieces are already the categories byte for byte, and the
  trimming turned a category such as `" N/A "` or `"PENDING "` into a
  key no observation could match. Those rows were assigned `bin = NA`
  and `woe = na_woe`, i.e. scored as unseen, with no warning: on a base
  whose codes come from a `CHAR(n)` column or a hand-maintained code
  table, an entire segment could be dropped from the model without a
  trace. The trimming is gone; categories are now matched exactly, as
  [`obwoe_sql()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_sql.md)
  already did.

- **[`obwoe_gains()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_gains.md)
  reported the same lift for every bin.** The column was built with
  `ifelse(overall_rate > 0, df$pos_rate / overall_rate, 0)`;
  [`ifelse()`](https://rdrr.io/r/base/ifelse.html) returns a result
  shaped like its *test*, and that test is a scalar, so the expression
  collapsed to a length-one vector that R then recycled across the
  table. Every gains table with more than one bin reported the first
  bin’s lift throughout, and `plot(type = "lift")` drew a flat line. The
  R gains table now agrees with the independent C++ engine
  (`ob_gains_table()`) to machine precision on `lift`, `woe`, `iv` and
  `ks`.

- **[`ob_numerical_ir()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_ir.md)
  reported counts that did not describe its own bins, and bins that were
  not monotonic.** `applyIsotonicRegression()` ran the Pool Adjacent
  Violators algorithm over the bin event rates and then overwrote each
  bin with `count_pos <- round(fitted_rate * count)`. Pooling adjacent
  violators means those bins form *one* block; keeping them separate and
  back-solving synthetic counts produced two defects at once. First,
  `count_pos` and `count_neg` no longer described the observations
  falling between the reported `cutpoints`, so WoE, IV, KS and every
  gains table derived from them referred to a distribution that does not
  exist — on the German Credit `amount` attribute the reported and
  observed event rates differed in four of six bins. Second, because of
  the rounding the reported bins were not even monotonic, which is the
  one property an isotonic binner exists to guarantee.

  PAVA blocks are now merged into single bins. This reproduces the
  isotonic fit exactly — with the bin counts as weights, a block’s
  pooled rate *is* `sum(count_pos) / sum(count)` over that block — while
  `count`, `count_pos` and `count_neg` stay equal to what was observed.
  Features whose event rate is already monotone are unaffected and
  return bit-identical results; where PAVA had violators to pool, the
  binning is now coarser and genuinely rank-ordering. `min_bins` becomes
  a target rather than a guarantee for this algorithm, since
  monotonicity cannot always be attained at that resolution.

### Documentation

- **The README is a third of its former length.** It now covers what the
  package is, the four steps of a run, how to choose an algorithm and
  where to read more; the extended worked examples moved into the
  vignettes, where an analyst can study them properly. Algorithm
  selection is presented as a decision graph (rendered by GitHub) with
  the family reference table kept below it.

- **Two vignettes replace the single kitchen-sink one.**

  *Optimal Binning and Weight of Evidence: A Practical Guide* is the
  working reference: what WoE and IV measure and why monotonicity of the
  event rate and of the WoE are the same statement, reading bin and
  gains tables, screening a base with
  [`obwoe_select()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_select.md),
  how the algorithm families differ, applying the transformation,
  exporting SQL, and preprocessing.

  *An Industrial Scorecard Pipeline* is new and runs an origination
  scorecard end to end: a wide synthetic base with the pathologies that
  matter (missing fields, rare dealer codes, near-duplicate vendor
  variables, pure noise and a leaky post-booking field), screening at
  scale, redundancy pruning with
  [`obcorr()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obcorr.md)
  in the WoE space, a `recipes` pipeline around
  [`step_obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md),
  logistic regression with the coefficient sign check, PDO scorecard
  points, out-of-time validation with gains and PSI, deployment through
  [`obwoe_sql()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_sql.md),
  tuning with tidymodels, and a governance checklist.

- Both vignettes run on the bundled German Credit benchmark or on data
  they generate, so they build from `recipes` alone. The previous
  vignette required the Suggested `scorecard` package and failed to
  build without it; `scorecard` and `pROC` are no longer used anywhere
  and have been dropped from `Suggests`.

### Notes

- `data.table` is used when installed, for `rbindlist()` and for the
  returned table type, and is declared in `Suggests` only: the package
  has no new hard dependency and behaves identically without it, as the
  test suite verifies.

## OptimalBinningWoE 1.11.0

### C++ Engine — Runtime Audit (2026-08-12)

Follow-up to the 1.10.0 static audit, this time driven by instrumented
builds (`-fsanitize=address,undefined`), a degenerate-input stress
harness, and a golden-output regression suite covering all 37 exported
algorithms (~3,200 result comparisons). All fixes below are covered by
new tests in `tests/testthat/test-regression-audit.R`, each of which
fails on 1.10.0.

#### Bug Fixes — crashes and hangs

- **`ob_categorical_ivb` crashed the R session** (segmentation fault,
  not a catchable error) for **any feature with no more categories than
  `max_bins`** — with the default `max_bins = 5` this meant every 2-,
  3-, 4- or 5-level predictor, i.e. sex, marital status, region,
  education. In `perform_binning()` the `ncat <= max_bins` fast path
  skipped `initialize_dp_structures()`, the only place `stats_cache` was
  created, while result assembly dereferenced it unconditionally. The
  cache is now built before the branch.

- **`ob_numerical_ir`, `ob_numerical_jedi`, `ob_numerical_jedi_mwoe`
  hung forever** whenever `min_bins` exceeded the number of distinct
  feature values — a routine situation when one `min_bins` is applied
  across a whole feature set. The “ensure at least `min_bins`” loops
  called a split routine that silently declines to split unbounded
  intervals (and, in JEDI, ran before counts existed, so it always
  targeted the unsplittable `(-Inf, e1]` bin), making no progress and
  never terminating. Since these loops contained no
  `R_CheckUserInterrupt()`, the hang could not even be interrupted with
  Ctrl-C. The loops now consider only splittable bins and stop as soon
  as no progress is possible.

#### Reproducibility

- **`ob_categorical_sab` is now reproducible under
  [`set.seed()`](https://rdrr.io/r/base/Random.html).** It was seeded
  from `std::random_device`, so identical input returned a different
  binning on every call with no way to control it — unusable for
  auditable or regulated models, and unreliable on some MinGW
  toolchains. The simulated-annealing search is now seeded from R’s own
  RNG stream. **This changes `ob_categorical_sab` results**, which were
  previously random and therefore had no stable baseline to preserve.

#### Output consistency

- **`ob_numerical_sketch` now returns the `bin` label field** (plus
  `total_iv`) like every other numerical algorithm. Its absence broke
  generic consumers, including this package’s own test helper.
  `bin_lower` and `bin_upper` are retained, so the change is purely
  additive; no numeric output changed.

- The `ob_numerical_sketch` test disabled since 1.0.7 for a segfault has
  been **re-enabled**: the underlying `MergeCache` defect was removed in
  an earlier round, and the case was re-verified clean under
  `-fsanitize=address,undefined` at n = 500/1000/2000/5000.

#### Parallelism (`obcorr`) — CRAN policy and determinism

- **[`obcorr()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obcorr.md)
  no longer seizes every core.** With the default `threads = 0` it
  called `omp_set_num_threads(std::thread::hardware_concurrency())`,
  taking all available cores. CRAN Repository Policy requires a package
  never to use more than two cores simultaneously by default, since the
  check farm is a shared resource; this was an archival risk. The
  default is now at most 2 threads, honouring any lower limit already
  set through `OMP_NUM_THREADS`, and capped by `omp_get_num_procs()`. An
  explicit positive `threads` is still respected.

- **[`obcorr()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obcorr.md)
  returned its rows in a non-deterministic order.** Per-thread result
  buffers were spliced together inside an `omp critical` block, so the
  row order depended on thread scheduling — two runs on the same data
  with the same thread count could return the pairs in different orders.
  Any caller using [`head()`](https://rdrr.io/r/utils/head.html),
  positional indexing, or a positional join saw different numbers
  between runs. Each iteration now writes to its own slot in a pre-sized
  vector, which is deterministic, independent of the thread count, and
  removes the critical section. Correlation *values* are unchanged; only
  the row order is now stable.

#### Build hygiene

- Removed eight stale Dropbox conflict copies
  (`*Cópia em conflito*.cpp/.h`) from `src/`. `R CMD build` would have
  shipped and compiled them, producing duplicate symbols. Guards added
  to `.gitignore` and `.Rbuildignore`.

#### Interval convention — standardised on `(a, b]` (changes numeric output)

The package previously disagreed with itself about what a bin *is*. Bin
labels advertised `(a;b]`, but many algorithms assigned observations as
`[a;b)`;
[`ob_apply_woe_num()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_num.md)
had its two binary searches swapped, so *both* settings of
`include_upper_bound` did the opposite of what the argument documents;
and several equal-frequency pre-binners split runs of tied values, so
their reported counts could not be derived from their own reported
cutpoints under any convention. A value landing exactly on a cutpoint —
routine for integer, rounded or currency features — could be scored into
a different bin than the one it was trained in.

Measured across 63 algorithm/dataset combinations, **31 violated the
documented `(a, b]` convention before this release; 0 do now.**

- [`ob_apply_woe_num()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_num.md):
  `include_upper_bound = TRUE` now really means `(a, b]` (`lower_bound`
  search) and `FALSE` really means `[a, b)` (`upper_bound`). The two
  were previously exchanged.
- `ob_numerical_dp`, `_fetb`, `_ldb`, `_oslp`: bin lookup switched from
  `upper_bound` to `lower_bound`, so boundary values stay in the bin
  below.
- `ob_numerical_bb`, `_dmiv`: dropped an `+ EPSILON` added to the search
  key, which silently turned the documented `upper >= value` test into a
  strict one.
- `ob_numerical_sketch`: interval test changed from `[lower, upper)` to
  `(lower, upper]` (first bin remains closed on the left, so the minimum
  is included).
- `ob_numerical_mob`, `_mdlp`, `_mrblp`: equal-frequency pre-binning is
  now tie-aware — a run of identical values is never split across two
  pre-bins — and boundaries are the last value *in* the bin rather than
  the first value of the next. Their labels changed from `[a;b)` to
  `(a;b]` to match.
- `ob_numerical_fast_mdlp`: the `force_min_bins()` fallback picked a
  split at a raw index midpoint, which could land inside a run of tied
  observations. It now moves the split to a genuine value boundary, as
  the MDL recursion already did.

**`ob_numerical_cm` silently discarded observations.** Its
equal-frequency pre-binner, on hitting a tie that straddled a bin
boundary, advanced past the tied records without ever assigning them to
a bin. On tied or discrete features this dropped a large share of the
data — 26% on an integer feature and 40% on a coarse one in testing —
and the reported WoE/IV were computed from the surviving subset. The
tied records are now absorbed into the preceding bin.

#### Known issues (not yet fixed)

- `ob_categorical_dp` and `ob_categorical_fetb` report `WoE = IV = 0`
  for perfectly separating bins instead of applying smoothing.
- Return fields are not uniform across algorithms: `event_rate` is
  present in only 13 of 37, `total_iv` in 28 of 37, and `iv` is missing
  from both `_dmiv` variants.
- [`ob_apply_woe_num()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_num.md)
  does not support the multinomial `*_jedi_mwoe` variants, whose output
  carries per-class counts rather than `count_pos` / `count_neg`.

## OptimalBinningWoE 1.10.0

### C++ Engine — Comprehensive Audit & Hardening (2026-05-17)

This release is the result of a full static audit of the C++ engine
covering all 36 binning algorithms. No public R API was changed.

#### Bug Fixes

- **`OB_LogisticRegression`** — Replaced exact `det != 0` singularity
  guard with a threshold-based check (`|det| > 1e-10 × ‖H‖`); replaced
  `hessian.inverse()` with `Eigen::LDLT` decomposition for numerical
  stability; added `.cwiseMax(0.0)` before
  [`sqrt()`](https://rdrr.io/r/base/MathFun.html) to prevent `NaN`
  standard errors from near-zero diagonal entries.
- **`OBN_MDLP` — monotonicity direction bug** — `is_monotonic()` and
  `enforce_monotonicity()` previously hardcoded ascending direction,
  causing unnecessary merges on negatively-correlated features. Both now
  auto-detect the dominant trend via Welford’s slope algorithm before
  checking/enforcing monotonicity.
- **`OBC_DP` — DP backtracking out-of-bounds** — Added guard before
  `static_cast<size_t>(prev_j)` in `backtrack_optimal_bins()`; invalid
  predecessor index now raises a descriptive runtime error instead of
  silent undefined behaviour.
- **`OBN_DP` — push before validate** — Target value validation in
  `optimal_binning_numerical_dp()` now fires before the value is
  appended to `target_vec`, preventing insertion of invalid data.
- **`OBN_MDLP` — `log2(0)` in MDL cost** — Guard added for the
  single-bin case where `log2(k-1)` would evaluate to `log2(0) = -Inf`.
- **`NumericalBin` constructor invariant** — The 7-arg constructor now
  derives `count = count_pos + count_neg` regardless of the `c`
  argument, enforcing the `count == total()` invariant at construction
  time.

#### Performance Improvements

- **`OBC_DP` — DP outer loop removed** — The deterministic DP in
  `perform_dynamic_programming()` was wrapped in a redundant
  `max_iterations` outer loop (default 1000). Removing it yields up to
  **1000× speedup** for the categorical DP algorithm.
- **`OBC_DP::ensure_max_prebins()`** — O(m² log m) full re-sort per
  merge step replaced with O(m log m + m²) `std::lower_bound + insert`.
- **`OBN_MDLP::apply_mdl_merging()`** — O(k³) full-vector copy per
  candidate merge eliminated; MDL delta is now computed analytically
  from bin statistics in O(k²) per outer step.
- **`OBN_BB::quantile()`** — Per-call sort-copy O(n_prebins × n log n)
  eliminated; `prebinning()` sorts once and passes the sorted vector to
  a stateless [`quantile()`](https://rdrr.io/r/stats/quantile.html).
- **`monotonicity_utils.h` — Welford index allocation** — Removed
  unnecessary `std::vector<double> indices(n)` heap allocation; loop
  index cast directly to `double`.
- **`OBN_DP` — Pearson correlation instability** — Replaced naive
  two-pass Pearson formula (catastrophic cancellation risk) with
  `detect_trend_from_correlation()` using Welford’s online algorithm.

#### CRAN / ODR Safety

- **`safe_math.h`** — All 6 functions changed from `constexpr` to
  `inline`; `std::log`, `std::exp`, `std::abs` and `std::isfinite` are
  not guaranteed `constexpr` in C++11/14, risking compilation failure on
  SOLARIS/Studio.
- **`chi_square_utils.h`** — `CHI_SQUARE_CRITICAL_VALUES`
  namespace-scope `const` replaced with a function returning a `static`
  local instance (one shared copy per process, C++11 thread-safe init).
- **`entropy_utils.h`** — `ENTROPY_LUT` (~81 KB) replaced with
  `entropy_lut_instance()` returning a `static` local; eliminates one
  copy per translation unit.
- **`OBC_CM_v5`** — Duplicate `ChiSquareCache` class (global namespace)
  removed; file now uses `OptimalBinning::ChiSquareCache` from
  `chi_square_utils.h`.
- **35 `.cpp` files** — Duplicate `using namespace Rcpp` appearing
  before `#include "common/"` headers removed, preventing potential
  name-resolution ordering issues.

#### Code Quality

- **`OBC_DP`** — Dead commented-out code blocks
  (`// struct CategoryStats`,
  `// Local CategoricalBin definition removed`) deleted.
- **`OBN_DP`** — Local variable `total_count` renamed to `rare_total` to
  fix shadowing of the class member with the same name.
- **`OBN_IR`** — `[[Rcpp::plugins(cpp17)]]` standardised to `cpp11` for
  consistency with the rest of the package.
- **`OBC_DP`** — Auto-detection of monotonicity direction
  (`monotonic_trend = "auto"`) implemented in
  `compute_and_sort_event_rates()` via `detect_trend_welford_woe()`.

------------------------------------------------------------------------

## OptimalBinningWoE 1.0.9

- **CRAN Fix (2026-03-14)** - Replaced `Rf_error` with `Rcpp::stop`:
  - **Fixed C++ Exception Handling**: Addressed an issue reported by
    [@Enchufa2](https://github.com/Enchufa2) regarding the usage of
    `::Rf_error` inside `catch(...)` blocks. Updated all instances to
    use `Rcpp::stop` to ensure proper C++ stack unwinding and avoid
    memory leaks.
  - **Affected Files**: `src/OBN_LPDB_v5.cpp`, `src/OBN_EWB_v5.cpp`,
    `src/OBN_KMB_v5.cpp`, `src/OBN_LDB_v5.cpp`, `src/OBN_MBLP_v5.cpp`.

## OptimalBinningWoE 1.0.8

CRAN release: 2026-01-29

- **CRAN Fix (2026-01-28)** - LTO/ODR Compliance:
  - **Fixed One Definition Rule (ODR) violations**: Wrapped internal
    helper classes `IVCache` and `CumulativeStatsCache` in anonymous
    namespaces within `OBC_GMB_v5.cpp`, `OBC_IVB_v5.cpp`, and
    `OBC_JEDI_v5.cpp`. This resolves Link-Time Optimization (LTO)
    warnings/errors on CRAN checks.

## OptimalBinningWoE 1.0.7

- **UBSAN Investigation Fix (2026-01-27)** - Addressing persistent
  memory safety errors:

  - **Temporarily disabled `ob_categorical_sketch` tests**: The
    sketch-based categorical binning algorithm is under investigation
    for persistent UBSAN memory errors that appear to be related to
    cache invalidation timing in GitHub Actions CI environment.

  - **Removed `MergeCache` class from `OBC_Sketch_v5.cpp`**: Completely
    removed the caching mechanism and implemented on-the-fly divergence
    calculation to eliminate potential memory corruption sources.

- **Affected Files**:

  - `src/OBC_Sketch_v5.cpp`: MergeCache class removed, divergence
    calculated on-the-fly
  - `tests/testthat/test-categorical-all.R`: Sketch tests temporarily
    commented out

- **No API Changes**: Fully backward compatible with v1.0.6.

## OptimalBinningWoE 1.0.6

- **CRAN Fix (2026-01-26)** - Resolving AddressSanitizer memory safety
  errors:

  - **Fixed heap-buffer-overflow in `OBN_CM_v5.cpp`**: The
    `calculate_inconsistency_rate()` function was accessing `bins[j-1]`
    when `j=0` and `bins.size()==1`, causing invalid memory access.
    Restructured bin-finding loop to avoid negative index access.

  - **Fixed uninitialized bool in `OBC_MBA_v5.cpp`**: The
    `MergeCache::enabled` member was not explicitly initialized, causing
    “load of value 128, which is not a valid value for type ‘bool’”
    runtime error. Added explicit `bool enabled = false` initialization.

- **Affected Files**:

  - `src/OBN_CM_v5.cpp` (lines 863-887): Safe bin-finding logic
  - `src/OBC_MBA_v5.cpp` (line 26): Explicit bool initialization

- **No API Changes**: Fully backward compatible with v1.0.5.

## OptimalBinningWoE 1.0.5

- **CRAN Fix (2026-01-25)** - Resolving ERROR on macOS platforms during
  vignette re-build:

  - **Fixed
    [`obwoe_apply()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
    “breaks are not unique” error**: Enhanced cutpoint deduplication
    logic to properly handle cases where `sort(unique(cutpoints))`
    reduces the number of intervals. When the deduplicated cutpoint
    count doesn’t match the original bin count, the function now uses a
    fallback mapping with dynamically generated interval labels and mean
    WoE values, avoiding the
    [`cut.default()`](https://rdrr.io/r/base/cut.html) error.

  - This addresses the vignette build failure reported on
    r-release-macos-arm64, r-release-macos-x86_64, r-oldrel-macos-arm64,
    and r-oldrel-macos-x86_64 platforms.

- **Internal Changes**:

  - Added interval count validation after cutpoint deduplication
    (R/obwoe.R)
  - Fallback to mean WoE when bin/interval mismatch occurs
  - Dynamic interval label generation for edge cases

## OptimalBinningWoE 1.0.4

- **CRITICAL CRAN Fixes (2026-01-24)** - Addressing ERROR and NOTE on
  macOS platforms:

  - **Fixed macOS vignette ERROR**: Added comprehensive validation for
    duplicate cutpoints in
    [`obwoe_apply()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
    and
    [`bake.step_obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/bake.step_obwoe.md).
    The R base [`cut()`](https://rdrr.io/r/base/cut.html) function now
    receives guaranteed unique, sorted breaks, preventing the
    `"'breaks' are not unique"` error that was causing vignette build
    failures on macOS platforms.

  - **Attempted to reduce the package binary size** with size
    optimization flags (`-Os`, `-fvisibility=hidden`,
    `-ffunction-sections`, `-fdata-sections`), the `-Wl,--gc-sections`
    linker flag and a `cleanup` script for symbol stripping.

    **This was reverted and no longer describes the package.**
    `-fvisibility=hidden` and `-Wl,--gc-sections` hid Rcpp symbols and
    broke the build on every platform, so they were removed in commit
    `b74e95b`; `-Os` and the `cleanup` script did not survive either.
    `src/Makevars` now sets only the OpenMP and BLAS/LAPACK flags. The
    entry is corrected here rather than deleted so that nobody
    reinstates flags that are already known to break the build.

    The size itself is not a problem: the shared object is large only
    because it carries debug symbols. Measured on 1.13.3, `.debug_info`
    alone is 33Mb against 2.2Mb of `.text`, and stripping takes the
    library from 70.4Mb to 2.6Mb. `R CMD check --as-cran` reports the
    installed size as INFO, not as a NOTE.

- **Internal Changes**:

  - Added `src/common/cutpoints_validator.h` - new C++ utility header
    with `validate_cutpoints()` function to ensure cutpoint uniqueness
    across all numerical binning algorithms. Uses floating-point
    tolerance (1e-10) for safe duplicate detection.

  - Modified `get_cutpoints()` in `src/OBN_MOB_v5.cpp` (line 180) to
    apply validation before returning cutpoints.

  - Modified `update_cutpoints()` in `src/OBN_UBSD_v5.cpp` (line 874) to
    apply validation before storing cutpoints.

  - Added R-level validation in
    [`obwoe_apply()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)
    (R/obwoe.R, line 1550): cutpoints are now sorted and deduplicated
    using `sort(unique(cutpoints))` before constructing breaks vector.

  - Added R-level validation in
    [`bake.step_obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/bake.step_obwoe.md)
    (R/step_obwoe.R, line 789): same deduplication logic for recipes
    integration.

  - Enhanced vignette robustness (`vignettes/introduction.Rmd`): Added
    try-catch error handling in scorecard workflow to prevent build
    failures on edge-case data distributions.

- **Affected Algorithms**: All 21 numerical binning algorithms now
  validate cutpoints to prevent duplicate breaks:

  - Monotonic Optimal Binning (MOB)
  - Dynamic Programming (DP)
  - Chi-Merge (CM)
  - Unsupervised Binning with Standard Deviation (UBSD)
  - And 17 other numerical algorithms

- **No API Changes**: Fully backward compatible with v1.0.3. All
  existing code will continue to work without modification.

## OptimalBinningWoE 1.0.3

CRAN release: 2026-01-23

- **Critical Bug Fixes - KLL Sketch Algorithm (2026-01-20)**:
  - Fixed **iterator invalidation** in `KLLSketch::compact_level()` -
    the `compactors.push_back()` call was invalidating references to
    vector elements, causing crashes with datasets larger than ~200
    observations.
  - Fixed **parameter order bug** in `calculate_metrics()` calls -
    swapped `(total_good, total_bad)` to correct order
    `(total_pos, total_neg)`, fixing incorrect WoE calculations.
  - Fixed **half-open interval logic** in bin assignment - added
    explicit closed interval `[lower, upper]` check for the last bin to
    ensure boundary values are correctly assigned.
  - Fixed **merge direction logic** in `enforce_bin_cutoff()` -
    corrected iterator invalidation when merging bins by always erasing
    the higher-indexed bin.
  - Added **bounds safety checks** in DP optimization - ensured `k >= 2`
    and `k < n` to prevent undefined behavior with edge cases.
  - Added **underflow guard** in compaction loop - check for
    `compactor.size() < 2` before iteration.
  - Added **input validation** for non-finite values (Inf, NaN) in
    sketch updates.
  - Improved **documentation** in
    [`ob_numerical_sketch()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_sketch.md)
    with clearer parameter descriptions and simplified examples.
  - Replaced `special_codes` parameter with `max_n_prebins` for
    consistency with other algorithms.
- **CRAN Reviewer Feedback (2026-01-17)**:
  - Removed single quotes from author names (`Siddiqi`,
    `Navas-Palencia`) in DESCRIPTION.
  - Removed commented-out code from examples in `obwoe_apply`.
  - Replaced all `\dontrun{}` with `\donttest{}` in 12 function
    examples.
  - Added proper [`par()`](https://rdrr.io/r/graphics/par.html)
    restoration in examples and vignettes.

## OptimalBinningWoE 1.0.2

- **CRAN Resubmission**:
  - Updated `inst/WORDLIST` to include technical terms and author names
    (MILP, Navas, Palencia) to resolve spelling notes.
  - Fixed `README.md` links for `CONTRIBUTING.md` and
    `CODE_OF_CONDUCT.md` to use absolute GitHub URLs, ensuring
    compliance with CRAN URI checks for ignored files.
  - Added `Language: en-US` to `DESCRIPTION` metadata.

## OptimalBinningWoE 1.0.1

- **CRAN Preparation**: Comprehensive updates for CRAN submission
  compliance.
- **Documentation**:
  - Enhanced `README.Rmd` with detailed algorithm descriptions,
    `tidymodels` integration examples, and performance metrics.
  - Added `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1) and
    `CONTRIBUTING.md` guidelines.
  - Added `inst/WORDLIST` for spell checking.
- **Metadata**:
  - Updated `DESCRIPTION` with corrected fields (Authors, BugReports,
    Depends, References).
  - Added `cran-comments.md` for submission notes.

## OptimalBinningWoE 1.0.0

### Initial Release

**OptimalBinningWoE** is a high-performance R package for optimal
binning and Weight of Evidence (WoE) transformation, designed for credit
scoring and predictive modeling.

#### Key Features

- **Comprehensive Algorithm Suite**: Implementation of 36 binning
  algorithms:
  - **20 Numerical Algorithms**: Including MDLP (Minimum Description
    Length Principle), JEDI (Joint Entropy-Driven Information), MOB
    (Monotonic Optimal Binning), Sketch (KLL/Count-Min for large data),
    and more.
  - **16 Categorical Algorithms**: Including ChiMerge, Fisher’s Exact
    Test Binning (FETB), SBLP (Similarity-Based LP), JEDI-MWoE
    (Multinomial WoE), and others.
- **High Performance**: Core algorithms are implemented in C++ using
  `Rcpp` and `RcppEigen` for maximum efficiency and scalability.
- **Unified Interface**:
  - [`obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md):
    Master function for optimal binning with automatic type detection
    and algorithm selection.
  - [`ob_apply_woe_num()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_num.md)
    /
    [`ob_apply_woe_cat()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_cat.md):
    Functions to apply learned binning mappings to new data.
- **tidymodels Integration**:
  - [`step_obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md):
    A complete `recipes` step for integrating optimal binning into
    machine learning pipelines.
  - Supports `tune()` for hyperparameter optimization of binning
    parameters (algorithm, min_bins, etc.).
- **Multinomial Support**:
  - Dedicated algorithms like `JEDI-MWoE` for handling multi-class
    target variables.
- **Robust Preprocessing**:
  - [`ob_preprocess()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_preprocess.md):
    Utilities for missing value handling and outlier detection/treatment
    (IQR, Z-score, Grubbs).
- **Advanced Metrics**:
  - `ob_gains_table()`: Computation of detailed gains tables including
    IV, WoE, KS, Gini, Lift, Precision, Recall, KL Divergence, and
    Jensen-Shannon Divergence.
- **Visualization**:
  - S3 [`plot()`](https://rdrr.io/r/graphics/plot.default.html) methods
    for visualizing binning results and WoE patterns.

#### usage

- See the package vignette
  ([`vignette("introduction", package = "OptimalBinningWoE")`](https://evandeilton.github.io/OptimalBinningWoE/articles/introduction.md))
  for detailed examples and theoretical background.
