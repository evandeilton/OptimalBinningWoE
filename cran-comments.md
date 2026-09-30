# CRAN Submission Comments — OptimalBinningWoE 1.14.0

## Summary

This release no longer uses `RcppEigen` or `RcppNumerical`. The `RcppEigen`
maintainers asked for it on CRAN before `RcppEigen` moves to Eigen 5.0.1
(<https://github.com/RcppCore/RcppEigen/issues/151>,
<https://github.com/evandeilton/OptimalBinningWoE/issues/19>).

1.13.5 fails to install against the Eigen 5.0.1 release candidate of
`RcppEigen` (0.4.9.9-2): it uses `MappedSparseMatrix`, which Eigen 5 removed.
The only Eigen code was the scorecard engine's logistic regression, now a
plain C++ Newton-IRLS. 1.14.0 includes no Eigen header and no longer depends
on `RcppEigen`.

The release also fixes crashes, hangs and out-of-bounds reads in several
binners, applies one NA/Inf rule to all numerical binners, and corrects the
ChiMerge, Fisher and MDLP criteria. No argument was added or removed. Details
are in `NEWS.md`.

---

## R CMD check results

### Local check

x86_64-pc-linux-gnu, R 4.6.1, GCC 15.2, `R CMD check --as-cran
--run-donttest` on the built tarball with vignettes:

```
0 errors | 0 warnings | 1 note
```

**NOTE — `checking HTML version of manual`.** The check machine has no `tidy`
binary and no `V8` package, so HTML validation and math rendering were skipped.

**INFO — installed size 53.7 MB (`libs` 51.9 MB).** Almost all of it is debug
symbols from the local `-g` default; the shared object is 2.9 MB after
`strip --strip-debug`. `src/Makevars` does not override the platform's
compiler flags.

---

## Test environments

* **Local**: x86_64-pc-linux-gnu, R 4.6.1, GCC 15.2.
* **GitHub Actions**: ubuntu-latest (R devel, release, oldrel-1, oldrel-2,
  oldrel-3), windows-latest (R release) and macos-latest (R release, Apple
  silicon).
* **win-builder (R-release and R-devel)**: submitted 2026-09-30; results to be
  confirmed before submission.

---

## Dependencies

`RcppEigen` and `RcppNumerical` removed from `LinkingTo`. No new dependencies.

---

## Downstream dependencies

This package currently has no reverse dependencies on CRAN.
