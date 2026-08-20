# CRAN Submission Comments — OptimalBinningWoE 1.13.0

## Summary

This is a minor release (1.11.0 → 1.13.0) adding user-facing functionality on
top of an unchanged C++ engine. Existing code continues to work: nothing was
removed, renamed or given different behaviour.

Two feature waves are included:

* **1.12.0** — `obwoe_select()`, automated variable screening by Information
  Value strength and bin ordering, and `obwoe_sql()`, generation of the
  equivalent SQL `CASE` expressions for in-database scoring (14 dialects).
* **1.13.0** — `obwoe_scorecard()`, an end-to-end scorecard pipeline
  (split, binning, screening, fitting, PDO scaling) whose result can be
  written as a multi-sheet `.xlsx` model document by `obwoe_report()`, plus
  the supporting `obwoe_scale()`, `obwoe_score()`, `obwoe_prune()` and
  `obwoe_psi()`.

The full list is in `NEWS.md`.

---

## R CMD check results

### Local check

x86_64-pc-linux-gnu, R 4.5.x, GCC, `R CMD check --no-manual` on the built
tarball with vignettes:

```
0 errors | 1 warning | 2 notes
```

Tests, examples and both vignettes build and run cleanly.

**WARNING — `checking R files for syntax errors`.** No syntax error is
reported. The only output is:

```
Warning in Sys.setlocale("LC_CTYPE", "en_US.UTF-8") :
  OS reports request to set locale to "en_US.UTF-8" cannot be honored
```

The check container has no `en_US.UTF-8` locale installed. This is a property
of the machine, not of the package, and does not reproduce where the locale is
available.

**NOTE — `installed size is 66.6Mb`, `libs 65.1Mb`.** The package compiles 36
binning algorithms as separate translation units, so the shared object is
large; but essentially all of the reported size is debug symbols from the
local `-g` compiler default. Measured on this build:

| | size |
|---|---|
| `OptimalBinningWoE.so` as built | 68.1 MB |
| same object after `strip --strip-debug` | 2.7 MB |

So ~96% of the note is symbols that a stripping install removes. We have not
added `-Os` or an explicit strip step to `src/Makevars`, since CRAN policy
asks packages not to override the platform's compiler flags.

**NOTE — `Packages suggested but not available for checking: 'tidymodels',
'workflows', 'parsnip'`.** The check container has no network access to CRAN,
so these optional packages could not be installed. They are used only in the
`step_obwoe()` tests and vignette sections, all of which are guarded by
`skip_if_not_installed()` / `eval=` conditions.

---

## Test environments

* **Local**: x86_64-pc-linux-gnu, GCC — `0 errors | 1 warning (locale) | 2 notes`
* **win-builder (R-release)**: to be verified before submission
* **win-builder (R-devel)**: to be verified before submission
* **macOS builder**: to be verified before submission

---

## Dependencies

`openxlsx` and `glmnet` were added to `Suggests` in 1.13.0. Both are optional:
`openxlsx` is needed only to write the `.xlsx` workbook, `glmnet` only for
`engine = "glmnet"`. Each is guarded with `requireNamespace()` and produces an
informative error naming the package when absent.

No new `Imports` or `Depends`.

---

## Downstream dependencies

This package currently has no reverse dependencies on CRAN.
