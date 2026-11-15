# CRAN Submission Comments — OptimalBinningWoE 1.14.1

## Summary

Bug-fix release. Fourteen categorical binners ordered tied categories by
hash-table or first-appearance order, so the fitted bins could depend on the
row order of the data and differ between platforms (Linux and macOS). Ties are
now broken by count, then by category name, so the result no longer depends
on the row order. `ob_categorical_dp()` with `monotonic_trend = "descending"`
and at most `max_bins` categories now returns the bins in decreasing event
rate. The release also corrects help pages, vignettes and `DESCRIPTION` where
they disagreed with the code. No argument was added or removed. Details are in
`NEWS.md`.

---

## R CMD check results

### Local check

x86_64-pc-linux-gnu, R 4.6.1, GCC 15.2, `R CMD check --as-cran` on the built
tarball with vignettes:

```
0 errors | 0 warnings | 1 note
```

**NOTE — `checking HTML version of manual`.** The check machine has no `tidy`
binary, so HTML validation was skipped.

---

## Test environments

* **Local**: x86_64-pc-linux-gnu, R 4.6.1, GCC 15.2.
* **GitHub Actions**: ubuntu-latest (R devel, release, oldrel-1, oldrel-2,
  oldrel-3), windows-latest (R release) and macos-latest (R release, Apple
  silicon).

---

## Dependencies

No change.

---

## Downstream dependencies

One reverse dependency on CRAN: `scorecraft` (Imports).
