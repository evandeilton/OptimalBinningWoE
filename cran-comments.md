# CRAN Submission Comments — OptimalBinningWoE 1.13.2

## Summary

This release adds user-facing functionality on top of a mostly-unchanged C++
engine, followed by two bug-fix passes from an internal code audit. Existing
code continues to work: nothing was removed or renamed except two
internal-only API adjustments requested by the author (see `NEWS.md`), and a
number of previously-wrong computed values now compute correctly — each one
flagged in `NEWS.md` where the corrected value differs from what a caller may
already have worked around.

Four waves are included:

* **1.12.0** — `obwoe_select()`, automated variable screening by Information
  Value strength and bin ordering, and `obwoe_sql()`, generation of the
  equivalent SQL `CASE` expressions for in-database scoring (14 dialects).
* **1.13.0** — `obwoe_scorecard()`, an end-to-end scorecard pipeline
  (split, binning, screening, fitting, PDO scaling) whose result can be
  written as a multi-sheet `.xlsx` model document by `obwoe_report()`, plus
  the supporting `obwoe_scale()`, `obwoe_score()`, `obwoe_prune()` and
  `obwoe_psi()`.
* **1.13.1** — Bug-fix release from an internal audit: correctness fixes to
  gains-table ordering and KS, `step_obwoe()`'s `"auto"` algorithm
  resolution, `obwoe_gains()`'s WoE regrouping, missing-value handling in
  `obwoe_apply()`/`obwoe_sql()`, and several smaller items. Two internal
  functions were un-exported or renamed (neither advertised as stable public
  API — see `NEWS.md`).
* **1.13.2** — Second audit pass. `obwoe()` now returns the `control` it was
  fitted with, so a saved model records its own configuration; everything
  that has to split grouped categories out of a bin label reads the
  separator from there instead of assuming the package default. This fixed a
  silent defect rather than adding a feature: a model fitted with a custom
  `bin_separator` previously mis-scored most of its own training rows and
  emitted structurally broken SQL, with no error and no warning.

The full list is in `NEWS.md`, with behavior changes called out at the top of
each section.

### Note on the version jump

The last version accepted on CRAN is 1.0.8; this submission is 1.13.2. The
intermediate versions were developed and used outside CRAN while the
package's scope grew substantially (the scorecard pipeline, SQL code
generation, `tidymodels` integration via `step_obwoe()`), and were never
submitted. There is no CRAN-visible history between 1.0.8 and this
submission; `NEWS.md` documents every version in between for completeness.

This produces the expected `Version jumps in minor` NOTE below.

---

## R CMD check results

### Local check

x86_64-pc-linux-gnu, R 4.6.x, GCC, `R CMD check --as-cran` on the built
tarball with vignettes:

```
0 errors | 0 warnings | 2 notes
```

Tests, examples (including `--run-donttest`) and both vignettes build and run
cleanly. The test suite is 1933 assertions, all passing, with no warnings.

**NOTE 1 — `Version jumps in minor (submitted: 1.13.2, existing: 1.0.8)`.**
Expected; explained above.

**NOTE 2 — `checking HTML version of manual`.** The check machine has no
`tidy` binary and no `V8` package, so HTML validation and math rendering were
skipped. This is a property of the machine, not of the package.

A third note, `Found the following hidden files and directories: .git`,
appeared in an earlier run of this check and has been eliminated. It was an
artifact of building the tarball from a `git worktree`, where `.git` is a
small *file* pointing at the real git directory rather than a directory
itself. `R CMD build` skips a `.git` directory automatically but not a file
of that name, so the pointer was packaged. `.Rbuildignore` now carries
`^\.git$`, which excludes it either way, and a rebuild from the same
worktree confirms it is gone.

### INFO — installed size

```
installed size is 71.9Mb
sub-directories of 1Mb or more:
  libs  70.3Mb
```

The package compiles 37 binning algorithms as separate translation units, so
the shared object is large. Essentially all of the reported size is debug
symbols from the local `-g` compiler default. Measured on this build:

| | size |
|---|---|
| `OptimalBinningWoE.so` as built | 70.3 MB |
| same object after `strip --strip-debug` | 2.8 MB |

So ~96% of the figure is symbols that a stripping install removes. We have
not added `-Os` or an explicit strip step to `src/Makevars`, since CRAN
policy asks packages not to override the platform's compiler flags.

---

## Test environments

* **Local**: x86_64-pc-linux-gnu, R 4.6.x, GCC — `0 errors | 0 warnings | 2 notes`
* **GitHub Actions**, all passing on the submitted commit:
  * ubuntu-latest — R devel, release, oldrel-1, oldrel-2, oldrel-3
  * windows-latest — R release
* **win-builder (R-release)**: to be verified before submission
* **win-builder (R-devel)**: to be verified before submission
* **macOS builder**: **not yet verified.** macOS is currently disabled in the
  CI matrix, so no macOS build has been exercised for this release. The
  package is C++17 and links against `RcppEigen` and `RcppNumerical`, so this
  is the platform most likely to surface a compilation difference. It will be
  checked on the macOS builder before submission.

---

## Dependencies

No new `Imports` or `Depends`.

`Suggests` changed since 1.13.0:

* `dplyr` **removed** — it was declared but never used.
* `tune` **added** — used by one regression test that constructs
  `step_obwoe(algorithm = tune::tune())`, guarded with
  `skip_if_not_installed("tune")`.

`openxlsx` and `glmnet` remain optional, as added in 1.13.0: `openxlsx` is
needed only to write the `.xlsx` workbook, `glmnet` only for
`engine = "glmnet"`. Each is guarded with `requireNamespace()` and produces
an informative error naming the package when absent.

---

## Downstream dependencies

This package currently has no reverse dependencies on CRAN.
