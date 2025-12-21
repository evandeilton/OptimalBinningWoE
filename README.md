
<!-- README.md is generated from README.Rmd. Please edit that file -->

# OptimalBinningWoE <a href="https://evandeilton.github.io/OptimalBinningWoE/"><img src="inst/figures/obwoe.png" align="right" height="139" alt="OptimalBinningWoE website" /></a>

<!-- badges: start -->

[![CRAN
status](https://www.r-pkg.org/badges/version/OptimalBinningWoE)](https://CRAN.R-project.org/package=OptimalBinningWoE)
[![R-CMD-check](https://github.com/evandeilton/OptimalBinningWoE/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/evandeilton/OptimalBinningWoE/actions/workflows/R-CMD-check.yaml)
[![Downloads](https://cranlogs.r-pkg.org/badges/grand-total/OptimalBinningWoE)](https://cran.r-project.org/package=OptimalBinningWoE)
[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- badges: end -->

## Overview

`OptimalBinningWoE` is a high-performance R package for optimal binning
and Weight of Evidence (WoE) calculation. It provides **36 advanced
binning algorithms** implemented in C++ via Rcpp for both numerical and
categorical variables, designed for credit scoring, risk assessment, and
predictive modeling applications.

### Key Features

- **36 Binning Algorithms**: 16 for categorical + 20 for numerical
  variables
- **Binary & Multinomial Support**: Traditional WoE for binary targets
  and M-WoE for multinomial classification
- **High Performance**: Core algorithms implemented in C++ with
  RcppEigen
- **Automatic Method Selection**: Test multiple algorithms and select
  the best based on Information Value
- **Monotonicity Enforcement**: Ensure interpretable, monotonic WoE
  patterns
- **Robust Preprocessing**: Handle missing values, outliers, and rare
  categories
- **Comprehensive Metrics**: Gains tables with KS, Gini, lift,
  divergence measures

## Key Concepts

### Weight of Evidence (WoE)

$$\text{WoE}_i = \ln\left(\frac{P(X_i | Y = 1)}{P(X_i | Y = 0)}\right)$$

### Information Value (IV)

$$\text{IV}_{\text{total}} = \sum_{i=1}^{n} \left(P(X_i | Y = 1) - P(X_i | Y = 0)\right) \times \text{WoE}_i$$

| IV Range   | Interpretation                    |
|------------|-----------------------------------|
| \< 0.02    | Not Predictive                    |
| 0.02 - 0.1 | Weak Predictive Power             |
| 0.1 - 0.3  | Medium Predictive Power           |
| 0.3 - 0.5  | Strong Predictive Power           |
| ≥ 0.5      | Suspicious (possible overfitting) |

## Algorithms for Categorical Variables (16)

| Acronym | Function | Full Name | Description |
|----|----|----|----|
| **CM** | `ob_categorical_cm()` | ChiMerge | Merges categories based on chi-square statistics |
| **DMIV** | `ob_categorical_dmiv()` | Decision Tree MIV | Decision Tree with Minimum Information Value |
| **DP** | `ob_categorical_dp()` | Dynamic Programming | Optimal binning with local constraints |
| **FETB** | `ob_categorical_fetb()` | Fisher’s Exact Test | Merges categories with similar target distributions |
| **GMB** | `ob_categorical_gmb()` | Greedy Monotonic | Creates monotonic bins using greedy approach |
| **IVB** | `ob_categorical_ivb()` | Information Value | Maximizes Information Value |
| **JEDI** | `ob_categorical_jedi()` | Joint Entropy-Driven | Information maximization with Bayesian smoothing |
| **JEDI-MWoE** | `ob_categorical_jedi_mwoe()` | JEDI Multinomial | Multinomial WoE for multi-class targets |
| **MBA** | `ob_categorical_mba()` | Modified Binning | Modified approach for categorical binning |
| **MILP** | `ob_categorical_milp()` | Mixed Integer LP | Mixed Integer Linear Programming optimization |
| **MOB** | `ob_categorical_mob()` | Monotonic Optimal | Ensures monotonicity in WoE across categories |
| **SAB** | `ob_categorical_sab()` | Simulated Annealing | Stochastic optimization for binning |
| **SBLP** | `ob_categorical_sblp()` | Similarity-Based LP | Similarity-based logistic partitioning |
| **Sketch** | `ob_categorical_sketch()` | Count-Min Sketch | Sketch-based for large-scale/streaming data |
| **SWB** | `ob_categorical_swb()` | Sliding Window | Sliding window method for categoricals |
| **UDT** | `ob_categorical_udt()` | Unsupervised DT | Unsupervised decision tree binning |

## Algorithms for Numerical Variables (20)

| Acronym | Function | Full Name | Description |
|----|----|----|----|
| **BB** | `ob_numerical_bb()` | Branch and Bound | Exact optimization algorithm |
| **CM** | `ob_numerical_cm()` | ChiMerge | Chi-square based merging for numericals |
| **DMIV** | `ob_numerical_dmiv()` | Decision Tree MIV | Decision Tree with Minimum Information Value |
| **DP** | `ob_numerical_dp()` | Dynamic Programming | Optimal binning with local constraints |
| **EWB** | `ob_numerical_ewb()` | Equal Width | Creates bins of equal width |
| **Fast-MDLP** | `ob_numerical_fast_mdlp()` | Fast MDLP | Optimized MDLP implementation |
| **FETB** | `ob_numerical_fetb()` | Fisher’s Exact Test | Fisher’s exact test for numericals |
| **IR** | `ob_numerical_ir()` | Isotonic Regression | Isotonic regression for binning |
| **JEDI** | `ob_numerical_jedi()` | Joint Entropy-Driven | Information maximization with Bayesian smoothing |
| **JEDI-MWoE** | `ob_numerical_jedi_mwoe()` | JEDI Multinomial | Multinomial WoE for multi-class targets |
| **KMB** | `ob_numerical_kmb()` | K-Means Binning | K-means clustering for binning |
| **LDB** | `ob_numerical_ldb()` | Local Density | Local density estimation binning |
| **LPDB** | `ob_numerical_lpdb()` | Local Polynomial | Local polynomial density binning |
| **MBLP** | `ob_numerical_mblp()` | Monotonic LP | Monotonic binning via linear programming |
| **MDLP** | `ob_numerical_mdlp()` | Min Description Length | Minimum Description Length Principle |
| **MOB** | `ob_numerical_mob()` | Monotonic Optimal | Monotonic optimal binning |
| **MRBLP** | `ob_numerical_mrblp()` | Monotonic Regression LP | Monotonic regression with LP |
| **OSLP** | `ob_numerical_oslp()` | Optimal Supervised LP | Optimal supervised learning path |
| **Sketch** | `ob_numerical_sketch()` | KLL Sketch | Quantile approximation for large data |
| **UBSD** | `ob_numerical_ubsd()` | Unsupervised StdDev | Standard deviation-based intervals |
| **UDT** | `ob_numerical_udt()` | Unsupervised DT | Unsupervised decision tree binning |

## Utility Functions

| Function | Description |
|----|----|
| `ob_preprocess()` | Data preprocessing with missing value and outlier handling (IQR, Z-score, Grubbs) |
| `ob_gains_table()` | Comprehensive gains table with KS, Gini, lift, KL/JS divergence |
| `ob_gains_table_feature()` | Per-feature gains table generation |
| `ob_apply_woe_num()` | Apply WoE transformation to numerical features |
| `ob_apply_woe_cat()` | Apply WoE transformation to categorical features |
| `ob_binning_cutpoints_num()` | Extract cutpoints from numerical binning results |
| `ob_binning_cutpoints_cat()` | Extract cutpoints from categorical binning results |
| `ob_check_distincts()` | Check distinct values in features |

## Installation

``` r
# install.packages("devtools")
devtools::install_github("evandeilton/OptimalBinningWoE")
```

## Quick Start

### Numerical Binning

``` r
library(OptimalBinningWoE)

result <- ob_numerical_jedi(
  feature = numeric_vector,
  target = binary_target,
  min_bins = 3,
  max_bins = 5
)
```

### Categorical Binning

``` r
result <- ob_categorical_sblp(
  feature = categorical_vector,
  target = binary_target,
  min_bins = 3,
  max_bins = 5
)
```

### Preprocessing

``` r
preprocessed <- ob_preprocess(
  feature = feature_vector,
  target = target_vector,
  outlier_method = "iqr",
  outlier_process = TRUE
)
```

### Gains Table

``` r
gains <- ob_gains_table(binning_result)
# Includes: WoE, IV, KS, Gini, lift, precision, recall, 
# KL divergence, Jensen-Shannon divergence
```

### Multinomial Classification

``` r
result <- ob_categorical_jedi_mwoe(
  feature = categorical_vector,
  target = multiclass_target,  # Values: 0, 1, 2, ...
  min_bins = 3,
  max_bins = 5
)
# Returns M-WoE and IV matrices (n_bins × n_classes)
```

## Algorithm Selection Guide

| Use Case                | Recommended Algorithms |
|-------------------------|------------------------|
| **Credit Scoring**      | JEDI, MBLP, MOB, SBLP  |
| **Large Datasets**      | Sketch, Fast-MDLP, EWB |
| **Streaming Data**      | Sketch (Count-Min/KLL) |
| **Multinomial Targets** | JEDI-MWoE              |
| **Strong Monotonicity** | MOB, MBLP, MRBLP, IR   |
| **Interpretability**    | DP, MDLP, UDT          |
| **Global Optimization** | MILP, SAB, BB          |

## Technical Details

### C++ Implementation

Core algorithms use Rcpp, RcppEigen, and RcppNumerical for high
performance.

### Bayesian Smoothing

Many algorithms employ Bayesian smoothing for robust WoE estimation with
small samples.

### Sketch-Based Algorithms

- **Count-Min Sketch**: Frequency estimation for categorical data
- **KLL Sketch**: Quantile approximation for numerical data

Both provide sublinear memory usage with single-pass processing.

## References

- Siddiqi, N. (2006). **Credit Risk Scorecards**. John Wiley & Sons.
- Thomas, L. C., et al. (2002). **Credit Scoring and Its Applications**.
  SIAM.
- Beltrami, M., et al. (2021). **Monotonic Optimal Binning Algorithm**.
  Risks, 9(3), 58.
- Navas-Palencia, G. (2020). **Optimal binning: mathematical programming
  formulations**. arXiv:2001.08025.
- Cormode, G., & Muthukrishnan, S. (2005). **Count-min sketch**. Journal
  of Algorithms, 55(1).

## Contributing

Contributions welcome! Open an issue or PR on
[GitHub](https://github.com/evandeilton/OptimalBinningWoE).

## License

MIT License - see [LICENSE](https://opensource.org/licenses/MIT).
