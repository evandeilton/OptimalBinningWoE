# CRAN Submission Comments — OptimalBinningWoE 1.15.0

## Summary

Correctness release for the C++ binning engines, following a forensic audit.
Bins without events (or non-events) no longer get a neutral WoE of 0;
several engines now honour `min_bins` / `bin_cutoff` as documented; merge
criteria and the exact dynamic programs optimize the IV they report, with a
minimum-evidence rule (empirical-Bayes smoothing, BIC or a binomial test) so
small samples and rare events are not overfitted; nine numerical engines
return a verified `monotonic` flag. Fitted bins change for the engines listed
in `NEWS.md`; every other engine gives the same result as 1.14.1. No argument
was added or removed.

---

## R CMD check results

GitHub Actions, `R CMD check` on the matrix below: 0 errors | 0 warnings.

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
