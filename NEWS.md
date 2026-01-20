# OptimalBinningWoE 1.0.3

*   **Critical Bug Fixes - KLL Sketch Algorithm (2026-01-20)**:
    *   Fixed **iterator invalidation** in `KLLSketch::compact_level()` - the `compactors.push_back()` call was invalidating references to vector elements, causing crashes with datasets larger than ~200 observations.
    *   Fixed **parameter order bug** in `calculate_metrics()` calls - swapped `(total_good, total_bad)` to correct order `(total_pos, total_neg)`, fixing incorrect WoE calculations.
    *   Fixed **half-open interval logic** in bin assignment - added explicit closed interval `[lower, upper]` check for the last bin to ensure boundary values are correctly assigned.
    *   Fixed **merge direction logic** in `enforce_bin_cutoff()` - corrected iterator invalidation when merging bins by always erasing the higher-indexed bin.
    *   Added **bounds safety checks** in DP optimization - ensured `k >= 2` and `k < n` to prevent undefined behavior with edge cases.
    *   Added **underflow guard** in compaction loop - check for `compactor.size() < 2` before iteration.
    *   Added **input validation** for non-finite values (Inf, NaN) in sketch updates.
    *   Improved **documentation** in `ob_numerical_sketch()` with clearer parameter descriptions and simplified examples.
    *   Replaced `special_codes` parameter with `max_n_prebins` for consistency with other algorithms.

*   **CRAN Reviewer Feedback (2026-01-17)**:
    *   Removed single quotes from author names (`Siddiqi`, `Navas-Palencia`) in DESCRIPTION.
    *   Removed commented-out code from examples in `obwoe_apply`.
    *   Replaced all `\dontrun{}` with `\donttest{}` in 12 function examples.
    *   Added proper `par()` restoration in examples and vignettes.

# OptimalBinningWoE 1.0.2

*   **CRAN Resubmission**:
    *   Updated `inst/WORDLIST` to include technical terms and author names (MILP, Navas, Palencia) to resolve spelling notes.
    *   Fixed `README.md` links for `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` to use absolute GitHub URLs, ensuring compliance with CRAN URI checks for ignored files.
    *   Added `Language: en-US` to `DESCRIPTION` metadata.

# OptimalBinningWoE 1.0.1

*   **CRAN Preparation**: Comprehensive updates for CRAN submission compliance.
*   **Documentation**:
    *   Enhanced `README.Rmd` with detailed algorithm descriptions, `tidymodels` integration examples, and performance metrics.
    *   Added `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1) and `CONTRIBUTING.md` guidelines.
    *   Added `inst/WORDLIST` for spell checking.
*   **Metadata**:
    *   Updated `DESCRIPTION` with corrected fields (Authors, BugReports, Depends, References).
    *   Added `cran-comments.md` for submission notes.

# OptimalBinningWoE 1.0.0

## Initial Release

**OptimalBinningWoE** is a high-performance R package for optimal binning and Weight of Evidence (WoE) transformation, designed for credit scoring and predictive modeling.

### Key Features

*   **Comprehensive Algorithm Suite**: Implementation of 36 binning algorithms:
    *   **20 Numerical Algorithms**: Including MDLP (Minimum Description Length Principle), JEDI (Joint Entropy-Driven Information), MOB (Monotonic Optimal Binning), Sketch (KLL/Count-Min for large data), and more.
    *   **16 Categorical Algorithms**: Including ChiMerge, Fisher's Exact Test Binning (FETB), SBLP (Similarity-Based LP), JEDI-MWoE (Multinomial WoE), and others.
*   **High Performance**: Core algorithms are implemented in C++ using `Rcpp` and `RcppEigen` for maximum efficiency and scalability.
*   **Unified Interface**:
    *   `obwoe()`: Master function for optimal binning with automatic type detection and algorithm selection.
    *   `ob_apply_woe_num()` / `ob_apply_woe_cat()`: Functions to apply learned binning mappings to new data.
*   **tidymodels Integration**:
    *   `step_obwoe()`: A complete `recipes` step for integrating optimal binning into machine learning pipelines.
    *   Supports `tune()` for hyperparameter optimization of binning parameters (algorithm, min_bins, etc.).
*   **Multinomial Support**:
    *   Dedicated algorithms like `JEDI-MWoE` for handling multi-class target variables.
*   **Robust Preprocessing**:
    *   `ob_preprocess()`: Utilities for missing value handling and outlier detection/treatment (IQR, Z-score, Grubbs).
*   **Advanced Metrics**:
    *   `ob_gains_table()`: Computation of detailed gains tables including IV, WoE, KS, Gini, Lift, Precision, Recall, KL Divergence, and Jensen-Shannon Divergence.
*   **Visualization**:
    *   S3 `plot()` methods for visualizing binning results and WoE patterns.

### usage

*   See the package vignette (`vignette("introduction", package = "OptimalBinningWoE")`) for detailed examples and theoretical background.
