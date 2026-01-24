# Changelog

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

  - **Reduced package binary size from 42.7MB to ~15-18MB** (60%
    reduction): Implemented size optimization flags (`-Os`,
    `-fvisibility=hidden`, `-ffunction-sections`, `-fdata-sections`) in
    `src/Makevars` and `src/Makevars.win`. Added linker flag
    `-Wl,--gc-sections` to remove unused code sections. Created
    `cleanup` script for automatic symbol stripping on Linux/macOS
    builds.

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
  - [`ob_gains_table()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_gains_table.md):
    Computation of detailed gains tables including IV, WoE, KS, Gini,
    Lift, Precision, Recall, KL Divergence, and Jensen-Shannon
    Divergence.
- **Visualization**:
  - S3 [`plot()`](https://rdrr.io/r/graphics/plot.default.html) methods
    for visualizing binning results and WoE patterns.

#### usage

- See the package vignette
  ([`vignette("introduction", package = "OptimalBinningWoE")`](https://evandeilton.github.io/OptimalBinningWoE/articles/introduction.md))
  for detailed examples and theoretical background.
