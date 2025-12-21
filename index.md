# OptimalBinningWoE [![OptimalBinningWoE website](inst/figures/obwoe.png)](https://evandeilton.github.io/OptimalBinningWoE/)

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

$$\text{WoE}_{i} = \ln\left( \frac{P\left( X_{i}|Y = 1 \right)}{P\left( X_{i}|Y = 0 \right)} \right)$$

### Information Value (IV)

$$\text{IV}_{\text{total}} = \sum\limits_{i = 1}^{n}\left( P\left( X_{i}|Y = 1 \right) - P\left( X_{i}|Y = 0 \right) \right) \times \text{WoE}_{i}$$

| IV Range   | Interpretation                    |
|------------|-----------------------------------|
| \< 0.02    | Not Predictive                    |
| 0.02 - 0.1 | Weak Predictive Power             |
| 0.1 - 0.3  | Medium Predictive Power           |
| 0.3 - 0.5  | Strong Predictive Power           |
| ≥ 0.5      | Suspicious (possible overfitting) |

## Algorithms for Categorical Variables (16)

| Acronym       | Function                                                                                                              | Full Name            | Description                                         |
|---------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|-----------------------------------------------------|
| **CM**        | [`ob_categorical_cm()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_cm.md)               | ChiMerge             | Merges categories based on chi-square statistics    |
| **DMIV**      | [`ob_categorical_dmiv()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_dmiv.md)           | Decision Tree MIV    | Decision Tree with Minimum Information Value        |
| **DP**        | [`ob_categorical_dp()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_dp.md)               | Dynamic Programming  | Optimal binning with local constraints              |
| **FETB**      | [`ob_categorical_fetb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_fetb.md)           | Fisher’s Exact Test  | Merges categories with similar target distributions |
| **GMB**       | [`ob_categorical_gmb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_gmb.md)             | Greedy Monotonic     | Creates monotonic bins using greedy approach        |
| **IVB**       | [`ob_categorical_ivb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_ivb.md)             | Information Value    | Maximizes Information Value                         |
| **JEDI**      | [`ob_categorical_jedi()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_jedi.md)           | Joint Entropy-Driven | Information maximization with Bayesian smoothing    |
| **JEDI-MWoE** | [`ob_categorical_jedi_mwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_jedi_mwoe.md) | JEDI Multinomial     | Multinomial WoE for multi-class targets             |
| **MBA**       | [`ob_categorical_mba()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_mba.md)             | Modified Binning     | Modified approach for categorical binning           |
| **MILP**      | [`ob_categorical_milp()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_milp.md)           | Mixed Integer LP     | Mixed Integer Linear Programming optimization       |
| **MOB**       | [`ob_categorical_mob()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_mob.md)             | Monotonic Optimal    | Ensures monotonicity in WoE across categories       |
| **SAB**       | [`ob_categorical_sab()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_sab.md)             | Simulated Annealing  | Stochastic optimization for binning                 |
| **SBLP**      | [`ob_categorical_sblp()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_sblp.md)           | Similarity-Based LP  | Similarity-based logistic partitioning              |
| **Sketch**    | [`ob_categorical_sketch()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_sketch.md)       | Count-Min Sketch     | Sketch-based for large-scale/streaming data         |
| **SWB**       | [`ob_categorical_swb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_swb.md)             | Sliding Window       | Sliding window method for categoricals              |
| **UDT**       | [`ob_categorical_udt()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_udt.md)             | Unsupervised DT      | Unsupervised decision tree binning                  |

## Algorithms for Numerical Variables (20)

| Acronym       | Function                                                                                                          | Full Name               | Description                                      |
|---------------|-------------------------------------------------------------------------------------------------------------------|-------------------------|--------------------------------------------------|
| **BB**        | [`ob_numerical_bb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_bb.md)               | Branch and Bound        | Exact optimization algorithm                     |
| **CM**        | [`ob_numerical_cm()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_cm.md)               | ChiMerge                | Chi-square based merging for numericals          |
| **DMIV**      | [`ob_numerical_dmiv()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_dmiv.md)           | Decision Tree MIV       | Decision Tree with Minimum Information Value     |
| **DP**        | [`ob_numerical_dp()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_dp.md)               | Dynamic Programming     | Optimal binning with local constraints           |
| **EWB**       | [`ob_numerical_ewb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_ewb.md)             | Equal Width             | Creates bins of equal width                      |
| **Fast-MDLP** | [`ob_numerical_fast_mdlp()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_fast_mdlp.md) | Fast MDLP               | Optimized MDLP implementation                    |
| **FETB**      | [`ob_numerical_fetb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_fetb.md)           | Fisher’s Exact Test     | Fisher’s exact test for numericals               |
| **IR**        | [`ob_numerical_ir()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_ir.md)               | Isotonic Regression     | Isotonic regression for binning                  |
| **JEDI**      | [`ob_numerical_jedi()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_jedi.md)           | Joint Entropy-Driven    | Information maximization with Bayesian smoothing |
| **JEDI-MWoE** | [`ob_numerical_jedi_mwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_jedi_mwoe.md) | JEDI Multinomial        | Multinomial WoE for multi-class targets          |
| **KMB**       | [`ob_numerical_kmb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_kmb.md)             | K-Means Binning         | K-means clustering for binning                   |
| **LDB**       | [`ob_numerical_ldb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_ldb.md)             | Local Density           | Local density estimation binning                 |
| **LPDB**      | [`ob_numerical_lpdb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_lpdb.md)           | Local Polynomial        | Local polynomial density binning                 |
| **MBLP**      | [`ob_numerical_mblp()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mblp.md)           | Monotonic LP            | Monotonic binning via linear programming         |
| **MDLP**      | [`ob_numerical_mdlp()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mdlp.md)           | Min Description Length  | Minimum Description Length Principle             |
| **MOB**       | [`ob_numerical_mob()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mob.md)             | Monotonic Optimal       | Monotonic optimal binning                        |
| **MRBLP**     | [`ob_numerical_mrblp()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mrblp.md)         | Monotonic Regression LP | Monotonic regression with LP                     |
| **OSLP**      | [`ob_numerical_oslp()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_oslp.md)           | Optimal Supervised LP   | Optimal supervised learning path                 |
| **Sketch**    | [`ob_numerical_sketch()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_sketch.md)       | KLL Sketch              | Quantile approximation for large data            |
| **UBSD**      | [`ob_numerical_ubsd()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_ubsd.md)           | Unsupervised StdDev     | Standard deviation-based intervals               |
| **UDT**       | [`ob_numerical_udt()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_udt.md)             | Unsupervised DT         | Unsupervised decision tree binning               |

## Utility Functions

| Function                                                                                                          | Description                                                                       |
|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`ob_preprocess()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_preprocess.md)                   | Data preprocessing with missing value and outlier handling (IQR, Z-score, Grubbs) |
| [`ob_gains_table()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_gains_table.md)                 | Comprehensive gains table with KS, Gini, lift, KL/JS divergence                   |
| [`ob_gains_table_feature()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_gains_table_feature.md) | Per-feature gains table generation                                                |
| [`ob_apply_woe_num()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_num.md)             | Apply WoE transformation to numerical features                                    |
| [`ob_apply_woe_cat()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_apply_woe_cat.md)             | Apply WoE transformation to categorical features                                  |
| `ob_binning_cutpoints_num()`                                                                                      | Extract cutpoints from numerical binning results                                  |
| `ob_binning_cutpoints_cat()`                                                                                      | Extract cutpoints from categorical binning results                                |
| [`ob_check_distincts()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_check_distincts.md)         | Check distinct values in features                                                 |

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
