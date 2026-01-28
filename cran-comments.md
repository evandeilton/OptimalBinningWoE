## Resubmission (Version 1.0.6)

This is a resubmission addressing AddressSanitizer memory safety errors identified in CRAN check results for version 1.0.5.

### Fixed ERROR (AddressSanitizer - linux-x86_64-fedora-gcc)

* **Issue 1**: Heap-buffer-overflow in `OBN_CM_v5.cpp:870`
  - **Cause**: `calculate_inconsistency_rate()` accessed `bins[j-1]` when `j=0` and `bins.size()==1`, causing invalid memory access.
  - **Fix**: Restructured bin-finding loop to check `j==0` first and avoid negative index access (lines 863-887).

* **Issue 2**: Invalid bool value in `OBC_MBA_v5.cpp:53`
  - **Cause**: `MergeCache::enabled` member was not explicitly initialized, causing "load of value 128, which is not a valid value for type 'bool'" runtime error.
  - **Fix**: Added explicit initialization: `bool enabled = false` (line 26).

* **Verification**: The previously failing example now completes successfully:
  ```r
  ob_numerical_cm(feature, target, min_bins = 3, max_bins = 6, use_chi2_algorithm = TRUE)
  ```

## R CMD check results

0 errors ✓ | 0 warnings ✓ | 0 notes ✓

## Test environments

* local Windows 11, R 4.5.2
* win-builder (devel, release, oldrel)
* GitHub Actions:
  - macOS-latest (release)
  - windows-latest (release)
  - ubuntu-latest (devel, release, oldrel)

## Downstream dependencies

This is a CRAN package with no reverse dependencies.

## Additional Notes

* **No API changes**: Version 1.0.6 is fully backward compatible with v1.0.5.
* **All previous CRAN feedback remains addressed**.

---

## Previous Version History

### Version 1.0.5 (2026-01-25)

Fixed macOS vignette ERROR - improved cutpoint deduplication handling in `obwoe_apply()`.

### Version 1.0.4 (2026-01-24)

Initial cutpoint deduplication fix and package size optimization.

### Version 1.0.3 (2026-01-20)

Critical bug fixes in KLL Sketch algorithm.

### Version 1.0.2 (2026-01-17)

WORDLIST updates and README URL fixes for CRAN compliance.

### Version 1.0.1 (2026-01-11)

Initial CRAN submission preparation.
