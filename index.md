# OptimalBinningWoE

## Overview

**OptimalBinningWoE** is a high-performance R package for **optimal
binning** and **Weight of Evidence (WoE)** transformation, designed for
credit scoring, risk assessment, and predictive modeling applications.

### Why OptimalBinningWoE?

| Feature                   | Benefit                                                    |
|---------------------------|------------------------------------------------------------|
| **36 Algorithms**         | Choose the best method for your data characteristics       |
| **C++ Performance**       | Process millions of records efficiently via Rcpp/RcppEigen |
| **tidymodels Ready**      | Seamless integration with modern ML pipelines              |
| **Regulatory Compliance** | Monotonic binning for Basel/IFRS 9 requirements            |
| **Production Quality**    | Comprehensive testing and documentation                    |

## Installation

``` r
# Install from CRAN (when available)
install.packages("OptimalBinningWoE")

# Or install the development version from GitHub
# install.packages("pak")
pak::pak("evandeilton/OptimalBinningWoE")
```

## Quick Start

### Basic Usage

``` r
library(OptimalBinningWoE)

# Create sample data
set.seed(123)
df <- data.frame(
  age = rnorm(1000, 45, 15),
  income = exp(rnorm(1000, 10, 0.5)),
  education = sample(c("HS", "BA", "MA", "PhD"), 1000, replace = TRUE),
  target = rbinom(1000, 1, 0.15)
)

# Automatic optimal binning with WoE calculation
result <- obwoe(
  data = df,
  target = "target",
  algorithm = "jedi", # Joint Entropy-Driven Information
  min_bins = 3,
  max_bins = 6
)

# View summary
print(result)

# Examine binning details
result$results$age
```

### Integration with tidymodels

``` r
library(tidymodels)
library(OptimalBinningWoE)

# Create a preprocessing recipe with WoE transformation
rec <- recipe(default ~ ., data = credit_data) %>%
  step_obwoe(
    all_predictors(),
    outcome = "default",
    algorithm = "mob", # Monotonic Optimal Binning
    min_bins = 3,
    max_bins = tune(), # Tune the number of bins
    output = "woe"
  )

# Works seamlessly in ML workflows
workflow() %>%
  add_recipe(rec) %>%
  add_model(logistic_reg()) %>%
  fit(data = training_data)
```

## Core Concepts

### Weight of Evidence (WoE)

WoE quantifies the predictive power of each bin by measuring the
log-odds ratio:

$$\text{WoE}_{i} = \ln\left( \frac{\text{Distribution of Goods}_{i}}{\text{Distribution of Bads}_{i}} \right)$$

**Interpretation:**

- **WoE \> 0**: Lower risk than average (more “goods” than expected)
- **WoE \< 0**: Higher risk than average (more “bads” than expected)
- **WoE ≈ 0**: Similar to population average

### Information Value (IV)

IV measures the overall predictive power of a feature:

$$\text{IV} = \sum\limits_{i = 1}^{n}\left( \text{Dist. Goods}_{i} - \text{Dist. Bads}_{i} \right) \times \text{WoE}_{i}$$

| IV Range    | Predictive Power | Recommendation         |
|-------------|------------------|------------------------|
| \< 0.02     | Unpredictive     | Exclude                |
| 0.02 – 0.10 | Weak             | Use cautiously         |
| 0.10 – 0.30 | Medium           | Good predictor         |
| 0.30 – 0.50 | Strong           | Excellent predictor    |
| \> 0.50     | Suspicious       | Check for data leakage |

## Algorithm Reference

OptimalBinningWoE provides **36 algorithms** optimized for different
scenarios:

### Universal Algorithms (Numerical & Categorical)

| Algorithm    | Function                                                                                                    | Best For                               |
|--------------|-------------------------------------------------------------------------------------------------------------|----------------------------------------|
| **JEDI**     | [`ob_numerical_jedi()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_jedi.md)     | General purpose, balanced performance  |
| **MOB**      | [`ob_numerical_mob()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mob.md)       | Regulatory compliance (monotonic)      |
| **ChiMerge** | [`ob_numerical_cm()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_cm.md)         | Statistical significance-based merging |
| **DP**       | [`ob_numerical_dp()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_dp.md)         | Optimal partitioning with constraints  |
| **Sketch**   | [`ob_numerical_sketch()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_sketch.md) | Large-scale / streaming data           |

### Numerical-Only Algorithms (20)

| Algorithm | Function                                                                                                | Specialty                                |
|-----------|---------------------------------------------------------------------------------------------------------|------------------------------------------|
| **MDLP**  | [`ob_numerical_mdlp()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mdlp.md) | Entropy-based discretization             |
| **MBLP**  | [`ob_numerical_mblp()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_mblp.md) | Monotonic binning via linear programming |
| **IR**    | [`ob_numerical_ir()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_ir.md)     | Isotonic regression binning              |
| **EWB**   | [`ob_numerical_ewb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_ewb.md)   | Fast equal-width binning                 |
| **KMB**   | [`ob_numerical_kmb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_numerical_kmb.md)   | K-means clustering approach              |

**View all 20 numerical algorithms**

| Acronym   | Full Name               | Description              |
|-----------|-------------------------|--------------------------|
| BB        | Branch and Bound        | Exact optimization       |
| CM        | ChiMerge                | Chi-square merging       |
| DMIV      | Decision Tree MIV       | Recursive partitioning   |
| DP        | Dynamic Programming     | Optimal partitioning     |
| EWB       | Equal Width             | Fixed-width bins         |
| Fast-MDLP | Fast MDLP               | Optimized entropy        |
| FETB      | Fisher’s Exact Test     | Statistical significance |
| IR        | Isotonic Regression     | Order-preserving         |
| JEDI      | Joint Entropy-Driven    | Information maximization |
| JEDI-MWoE | JEDI Multinomial        | Multi-class targets      |
| KMB       | K-Means Binning         | Clustering-based         |
| LDB       | Local Density           | Density estimation       |
| LPDB      | Local Polynomial        | Smooth density           |
| MBLP      | Monotonic LP            | LP optimization          |
| MDLP      | Min Description Length  | Entropy-based            |
| MOB       | Monotonic Optimal       | IV-optimal + monotonic   |
| MRBLP     | Monotonic Regression LP | Regression + LP          |
| OSLP      | Optimal Supervised LP   | Supervised learning      |
| Sketch    | KLL Sketch              | Streaming quantiles      |
| UBSD      | Unsupervised StdDev     | Standard deviation       |
| UDT       | Unsupervised DT         | Decision tree            |

### Categorical-Only Algorithms (16)

| Algorithm | Function                                                                                                    | Specialty                 |
|-----------|-------------------------------------------------------------------------------------------------------------|---------------------------|
| **SBLP**  | [`ob_categorical_sblp()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_sblp.md) | Similarity-based grouping |
| **IVB**   | [`ob_categorical_ivb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_ivb.md)   | IV maximization           |
| **GMB**   | [`ob_categorical_gmb()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_gmb.md)   | Greedy monotonic          |
| **SAB**   | [`ob_categorical_sab()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_categorical_sab.md)   | Simulated annealing       |

**View all 16 categorical algorithms**

| Acronym   | Full Name            | Description              |
|-----------|----------------------|--------------------------|
| CM        | ChiMerge             | Chi-square merging       |
| DMIV      | Decision Tree MIV    | Recursive partitioning   |
| DP        | Dynamic Programming  | Optimal partitioning     |
| FETB      | Fisher’s Exact Test  | Statistical significance |
| GMB       | Greedy Monotonic     | Greedy monotonic binning |
| IVB       | Information Value    | IV maximization          |
| JEDI      | Joint Entropy-Driven | Information maximization |
| JEDI-MWoE | JEDI Multinomial     | Multi-class targets      |
| MBA       | Modified Binning     | Modified approach        |
| MILP      | Mixed Integer LP     | LP optimization          |
| MOB       | Monotonic Optimal    | IV-optimal + monotonic   |
| SAB       | Simulated Annealing  | Stochastic optimization  |
| SBLP      | Similarity-Based LP  | Similarity grouping      |
| Sketch    | Count-Min Sketch     | Streaming counts         |
| SWB       | Sliding Window       | Window-based             |
| UDT       | Unsupervised DT      | Decision tree            |

## Algorithm Selection Guide

| Use Case                         | Recommended          | Rationale                                  |
|----------------------------------|----------------------|--------------------------------------------|
| **General Credit Scoring**       | `jedi`, `mob`        | Best balance of speed and predictive power |
| **Regulatory Compliance**        | `mob`, `mblp`, `ir`  | Guaranteed monotonic WoE patterns          |
| **Large Datasets (\>1M rows)**   | `sketch`, `ewb`      | Sublinear memory, single-pass              |
| **High Cardinality Categorical** | `sblp`, `gmb`, `ivb` | Intelligent category grouping              |
| **Interpretability Focus**       | `dp`, `mdlp`         | Clear, explainable bins                    |
| **Multi-class Targets**          | `jedi_mwoe`          | Multinomial WoE support                    |

## Key Functions

| Function                                                                                        | Purpose                                    |
|-------------------------------------------------------------------------------------------------|--------------------------------------------|
| [`obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)                 | Main interface for optimal binning and WoE |
| [`obwoe_apply()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)     | Apply learned binning to new data          |
| [`obwoe_gains()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_gains.md)     | Compute gains table with KS, Gini, lift    |
| [`step_obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md)       | tidymodels recipe step                     |
| [`ob_preprocess()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_preprocess.md) | Data preprocessing with outlier handling   |

## Example Workflow

``` r
library(OptimalBinningWoE)

# 1. Fit binning model on training data
model <- obwoe(
  data = train_data,
  target = "default",
  algorithm = "mob",
  min_bins = 3,
  max_bins = 5
)

# 2. View feature importance by IV
print(model$summary[order(-model$summary$total_iv), ])

# 3. Apply transformation
train_woe <- obwoe_apply(train_data, model)
test_woe <- obwoe_apply(test_data, model)

# 4. Compute performance metrics
gains <- obwoe_gains(model, feature = "income")
print(gains)
plot(gains, type = "ks")
```

## Performance

OptimalBinningWoE is optimized for speed through:

- **RcppEigen**: Vectorized linear algebra operations
- **Efficient algorithms**: O(n log n) or better complexity
- **Memory-conscious design**: Streaming algorithms for large data

Typical performance on a standard laptop:

| Data Size | Processing Time |
|-----------|-----------------|
| 100K rows | \< 1 second     |
| 1M rows   | 2-5 seconds     |
| 10M rows  | 20-60 seconds   |

## Documentation

- 📖 [Package
  Vignette](https://evandeilton.github.io/OptimalBinningWoE/articles/introduction.html):
  Comprehensive guide with examples
- 📚 [Function
  Reference](https://evandeilton.github.io/OptimalBinningWoE/reference/):
  Complete API documentation
- 🐛 [Issue
  Tracker](https://github.com/evandeilton/OptimalBinningWoE/issues):
  Report bugs or request features

## Contributing

Contributions are welcome! Please see our [Contributing
Guidelines](https://evandeilton.github.io/OptimalBinningWoE/CONTRIBUTING.md)
and [Code of
Conduct](https://evandeilton.github.io/OptimalBinningWoE/CODE_OF_CONDUCT.md).

## Citation

If you use OptimalBinningWoE in your research, please cite:

``` bibtex
@software{optimalbinningwoe,
  author = {Lopes, José Evandeilton},
  title = {OptimalBinningWoE: Optimal Binning for Weight of Evidence},
  year = {2025},
  url = {https://github.com/evandeilton/OptimalBinningWoE}
}
```

## References

- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and
  Implementing Intelligent Credit Scoring*. John Wiley & Sons.
- Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). *Credit Scoring
  and Its Applications*. SIAM.
- Navas-Palencia, G. (2020). Optimal Binning: Mathematical Programming
  Formulation. arXiv:2001.08025.

## License

MIT License © 2026 José Evandeilton Lopes
