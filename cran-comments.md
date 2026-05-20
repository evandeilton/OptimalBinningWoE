# CRAN Submission Comments — OptimalBinningWoE 1.10.0

## Summary

This is a new minor release (1.0.9 → 1.10.0) consisting entirely of internal
C++ engine improvements. There are **no changes to the public R API** —
all existing user code is fully backward compatible.

---

## R CMD check results

### Local check (x86_64-pc-linux-gnu, R 4.5.2, GCC 13.3.0)

```
0 errors | 0 warnings | 1 note
```

**NOTE:** `checking for future file timestamps ... unable to verify current time`

This note is caused by the check machine being unable to reach a time server
(NTP) during the check run. It is an environmental network issue, not a
package defect. The `Date` field in DESCRIPTION is correct (2026-05-20).

---

## Changes in this version

### Bug fixes

* **`OB_LogisticRegression`**: Replaced exact `det != 0` singularity guard
  with threshold-based check (`|det| > 1e-10 * ||H||`); replaced
  `hessian.inverse()` with `Eigen::LDLT` decomposition; added `cwiseMax(0.0)`
  before `sqrt()` to prevent `NaN` standard errors from near-zero diagonal
  entries in the information matrix.

* **`OBN_MDLP` — monotonicity direction**: `is_monotonic()` and
  `enforce_monotonicity()` previously hardcoded ascending direction, causing
  unnecessary merges on negatively-correlated features. Both now auto-detect
  the dominant trend via Welford's online slope algorithm.

* **`OBC_DP` — DP backtracking**: Added guard before
  `static_cast<size_t>(prev_j)` to prevent undefined behaviour from a
  negative predecessor index.

* **`OBN_DP` — push before validate**: Target value validation now fires
  before appending to `target_vec`, preventing storage of invalid data.

* **`OBN_MDLP` — `log2(0)` in MDL cost**: Guard added for the single-bin
  case where `log2(k-1)` would evaluate to `log2(0) = -Inf`.

* **`NumericalBin` constructor**: `count` always derived from
  `count_pos + count_neg` to enforce the internal invariant at construction
  time.

### Performance improvements

* **`OBC_DP` — redundant DP outer loop removed**: The deterministic DP in
  `perform_dynamic_programming()` was wrapped in a redundant
  `max_iterations`-bounded outer loop (default 1000). Removing it yields up
  to 1000× speedup for large categorical datasets.

* **`OBC_DP::ensure_max_prebins()`**: O(m² log m) full re-sort per merge
  step replaced with O(m log m + m²) `std::lower_bound` + `insert`.

* **`OBN_MDLP::apply_mdl_merging()`**: O(k³) full-vector copy per candidate
  merge eliminated; MDL delta computed analytically in O(k²) per outer step.

* **`OBN_BB::quantile()`**: Per-call sort-and-copy O(n_prebins × n log n)
  eliminated; `prebinning()` sorts once and passes the sorted vector.

* **`OBN_DP` correlation**: Replaced naive two-pass Pearson formula
  (catastrophic cancellation risk on large-magnitude features) with
  Welford's numerically stable online algorithm.

### CRAN / ODR safety

* **`safe_math.h`**: Six helper functions changed from `constexpr` to
  `inline`; `std::log`, `std::exp`, `std::abs`, and `std::isfinite` are not
  guaranteed `constexpr` in C++11/14 and trigger compilation errors on
  SOLARIS/Oracle Studio.

* **`chi_square_utils.h`**: Namespace-scope `const` map replaced with a
  function returning a `static` local instance, eliminating one copy per
  translation unit (C++11 thread-safe initialisation).

* **`entropy_utils.h`**: 81 KB `ENTROPY_LUT` static object replaced with a
  shared `static` local instance via `entropy_lut_instance()`.

* **`OBC_CM_v5`**: Duplicate `ChiSquareCache` class (global namespace)
  removed; file now uses `OptimalBinning::ChiSquareCache` from the shared
  header, eliminating an ODR violation.

* **35 `.cpp` files**: Duplicate `using namespace Rcpp` appearing before
  `#include "common/"` headers removed.

---

## Test environments

* **Local**: x86_64-pc-linux-gnu, Zorin OS 18.1 (Ubuntu 24.04 base),
  R 4.5.2, GCC 13.3.0 — `0 errors | 0 warnings | 1 note (NTP)`
* **win-builder (R-release)**: to be verified before acceptance
* **win-builder (R-devel)**: to be verified before acceptance

---

## Downstream dependencies

This package currently has no reverse dependencies on CRAN.
