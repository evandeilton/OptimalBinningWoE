## R CMD check results

0 errors | 0 warnings | 1 note

* This is a resubmission (version 1.0.3).

## Test environments
* local Windows 11, R 4.5.2
* win-builder (devel and release)

## Reverse dependencies

This is a new package, so there are no reverse dependencies.

## Explanation of NOTEs

* "New submission" - expected for initial release.
* "Compilation used the following non-portable flag(s): '-mno-omit-leaf-frame-pointer'" - This flag originates from the R configuration on the check system and not from the package's Makevars. The package uses standard C++17 configuration.

## Changes in Version 1.0.3 (2026-01-20)

### Critical Bug Fixes - KLL Sketch Algorithm (`ob_numerical_sketch`)

The following memory safety and logic bugs were identified and fixed in `src/OBN_Sketch_v5.cpp`:

1. **Iterator invalidation in `KLLSketch::compact_level()`**: The `compactors.push_back()` call was invalidating references to vector elements, causing crashes with datasets larger than ~200 observations. Fixed by ensuring the vector is expanded *before* taking references.

2. **Parameter order bug in `calculate_metrics()` calls**: The function expected `(total_pos, total_neg)` but was being called with `(total_good, total_bad)`. This caused incorrect WoE calculations. All 4 call sites were corrected.

3. **Half-open interval logic in bin assignment**: The last bin was using `[lower, upper)` interval which excluded the maximum value. Added explicit closed interval check for the last bin.

4. **Merge direction logic in `enforce_bin_cutoff()`**: Iterator invalidation occurred when merging bins because the wrong bin was being erased. Fixed by always erasing the higher-indexed bin.

5. **Bounds safety in DP optimization**: Added checks to ensure `k >= 2` and `k < n` to prevent undefined behavior with edge cases.

6. **Underflow guard in compaction loop**: Added check for `compactor.size() < 2` before the compaction loop to prevent size_t underflow.

7. **Input validation**: Added checks for non-finite values (Inf, NaN) in sketch updates.

### API Change

* Replaced `special_codes` parameter with `max_n_prebins` in `ob_numerical_sketch()` for consistency with other numerical binning algorithms.

### Documentation

* Improved parameter descriptions in `ob_numerical_sketch()`.
* Simplified examples to use smaller, more reliable test datasets.

## Addressed CRAN Feedback (2026-01-17)

* **Single quotes around terms**: Removed single quotes from author names in DESCRIPTION.
* **Commented-out code in examples**: Removed all commented-out code from `obwoe_apply` examples.
* **\\dontrun{} usage**: Replaced all `\dontrun{}` with `\donttest{}` in 12 function examples.
* **Resetting par()/options()**: Added proper restoration of graphical parameters in all examples and vignettes.

## Previous CRAN Feedback (2026-01-11)

* **Misspelled words**: Added technical terms and author names to `inst/WORDLIST`.
* **Invalid file URIs**: Updated `README.md` to use absolute GitHub URLs.

