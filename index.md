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
# Install from CRAN
install.packages("OptimalBinningWoE")

# Or install the development version from GitHub
# install.packages("pak")
pak::pak("evandeilton/OptimalBinningWoE")
```

## Quick Start

### Basic Usage with German Credit Data

``` r
library(OptimalBinningWoE)
library(scorecard)

# Load the German Credit dataset
data("germancredit", package = "scorecard")

# Create binary target variable
german <- germancredit
german$default <- factor(
  ifelse(german$creditability == "bad", 1, 0),
  levels = c(0, 1),
  labels = c("good", "bad")
)
german$creditability <- NULL

# Select key features for demonstration
german_model <- german[, c(
  "default",
  "duration.in.month",
  "credit.amount",
  "age.in.years",
  "status.of.existing.checking.account",
  "credit.history",
  "savings.account.and.bonds"
)]

# Run Optimal Binning with JEDI algorithm (general purpose)
binning_results <- obwoe(
  data = german_model,
  target = "default",
  algorithm = "jedi",
  min_bins = 3,
  max_bins = 5
)

# View summary
print(binning_results)

# Check Information Value (IV) summary to see feature importance
print(binning_results$summary)

# View detailed binning for a specific feature
binning_results$results$duration.in.month
```

### Single Feature Binning

``` r
library(OptimalBinningWoE)
library(scorecard)

# Load data
data("germancredit", package = "scorecard")
german <- germancredit
german$default <- factor(
  ifelse(german$creditability == "bad", 1, 0),
  levels = c(0, 1),
  labels = c("good", "bad")
)

# Bin a single feature with specific algorithm
result_single <- obwoe(
  data = german,
  target = "default",
  feature = "credit.amount",
  algorithm = "mob",
  min_bins = 3,
  max_bins = 6
)

# View results
print(result_single)

# Detailed binning table
bins <- result_single$results$credit.amount
data.frame(
  Bin = bins$bin,
  Count = bins$count,
  Event_Rate = round(bins$count_pos / bins$count * 100, 2),
  WoE = round(bins$woe, 4),
  IV = round(bins$iv, 4)
)
```

### Apply WoE Transformation to New Data

``` r
library(OptimalBinningWoE)
library(scorecard)

# Load and prepare data
data("germancredit", package = "scorecard")
german <- germancredit
german$default <- factor(
  ifelse(german$creditability == "bad", 1, 0),
  levels = c(0, 1),
  labels = c("good", "bad")
)

# Train/test split
set.seed(123)
train_idx <- sample(1:nrow(german), size = 0.7 * nrow(german))
train_data <- german[train_idx, ]
test_data <- german[-train_idx, ]

# Fit binning model on training data
model <- obwoe(
  data = train_data,
  target = "default",
  algorithm = "mob",
  min_bins = 2,
  max_bins = 5
)

# Apply learned bins to training and test data
train_woe <- obwoe_apply(train_data, model, keep_original = FALSE)
test_woe <- obwoe_apply(test_data, model, keep_original = FALSE)

# View transformed features
head(train_woe[, c("default", "duration.in.month_woe", "credit.amount_woe")])
```

### Gains Table Analysis

``` r
library(OptimalBinningWoE)
library(scorecard)

# Load and prepare data
data("germancredit", package = "scorecard")
german <- germancredit
german$default <- factor(
  ifelse(german$creditability == "bad", 1, 0),
  levels = c(0, 1),
  labels = c("good", "bad")
)

# Fit binning model
model <- obwoe(
  data = german,
  target = "default",
  algorithm = "cm",
  min_bins = 3,
  max_bins = 5
)

# Compute gains table for a feature
gains <- obwoe_gains(model, feature = "duration.in.month", sort_by = "id")

# View gains table with KS, Gini, and lift metrics
print(gains)

# Visualize gains curves
par(mfrow = c(2, 2))
plot(gains, type = "cumulative")
plot(gains, type = "ks")
plot(gains, type = "lift")
plot(gains, type = "woe_iv")
par(mfrow = c(1, 1))
```

## Integration with tidymodels

**OptimalBinningWoE** integrates seamlessly with `tidymodels` recipes.

``` r
library(tidymodels)
library(OptimalBinningWoE)
library(scorecard)

# Load and prepare data
data("germancredit", package = "scorecard")
german <- germancredit
german$default <- factor(
  ifelse(german$creditability == "bad", 1, 0),
  levels = c(0, 1),
  labels = c("good", "bad")
)
german$creditability <- NULL

# Select features
german_model <- german[, c(
  "default",
  "duration.in.month",
  "credit.amount",
  "age.in.years",
  "status.of.existing.checking.account",
  "credit.history"
)]

# Train/test split
set.seed(123)
german_split <- initial_split(german_model, prop = 0.7, strata = default)
train_data <- training(german_split)
test_data <- testing(german_split)

# Create recipe with WoE transformation
rec_woe <- recipe(default ~ ., data = train_data) %>%
  step_obwoe(
    all_predictors(),
    outcome = "default",
    algorithm = "jedi",
    min_bins = 2,
    max_bins = 5,
    bin_cutoff = 0.05,
    output = "woe"
  )

# Define model specification
lr_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Create workflow
wf_credit <- workflow() %>%
  add_recipe(rec_woe) %>%
  add_model(lr_spec)

# Fit the workflow
final_fit <- fit(wf_credit, data = train_data)

# Evaluate on test data
test_pred <- augment(final_fit, test_data)

# Performance metrics
metrics <- metric_set(roc_auc, accuracy)
metrics(test_pred,
  truth = default,
  estimate = .pred_class,
  .pred_bad,
  event_level = "second"
)

# ROC curve
roc_curve(test_pred,
  truth = default,
  .pred_bad,
  event_level = "second"
) %>%
  autoplot() +
  labs(title = "ROC Curve - German Credit Model")
```

### Hyperparameter Tuning

``` r
library(tidymodels)
library(OptimalBinningWoE)
library(scorecard)

# Load and prepare data
data("germancredit", package = "scorecard")
german <- germancredit
german$default <- factor(
  ifelse(german$creditability == "bad", 1, 0),
  levels = c(0, 1),
  labels = c("good", "bad")
)
german$creditability <- NULL

german_model <- german[, c(
  "default",
  "duration.in.month",
  "credit.amount",
  "status.of.existing.checking.account",
  "credit.history"
)]

# Split data
set.seed(123)
german_split <- initial_split(german_model, prop = 0.7, strata = default)
train_data <- training(german_split)

# Recipe with tunable max_bins
rec_woe <- recipe(default ~ ., data = train_data) %>%
  step_obwoe(
    all_predictors(),
    outcome = "default",
    algorithm = "jedi",
    min_bins = 2,
    max_bins = tune(),
    output = "woe"
  )

# Model specification
lr_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Workflow
wf_credit <- workflow() %>%
  add_recipe(rec_woe) %>%
  add_model(lr_spec)

# Cross-validation folds
set.seed(456)
cv_folds <- vfold_cv(train_data, v = 5, strata = default)

# Tuning grid
tune_grid <- tibble(max_bins = c(3, 4, 5, 6))

# Tune
tune_results <- tune_grid(
  wf_credit,
  resamples = cv_folds,
  grid = tune_grid,
  metrics = metric_set(roc_auc)
)

# Best parameters
best_params <- select_best(tune_results, metric = "roc_auc")
print(best_params)

# Visualize tuning results
autoplot(tune_results, metric = "roc_auc")
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

| Function                                                                                              | Purpose                                    |
|-------------------------------------------------------------------------------------------------------|--------------------------------------------|
| [`obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe.md)                       | Main interface for optimal binning and WoE |
| [`obwoe_apply()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_apply.md)           | Apply learned binning to new data          |
| [`obwoe_gains()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_gains.md)           | Compute gains table with KS, Gini, lift    |
| [`step_obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/step_obwoe.md)             | tidymodels recipe step                     |
| [`ob_preprocess()`](https://evandeilton.github.io/OptimalBinningWoE/reference/ob_preprocess.md)       | Data preprocessing with outlier handling   |
| [`obwoe_algorithms()`](https://evandeilton.github.io/OptimalBinningWoE/reference/obwoe_algorithms.md) | List all available algorithms              |
| [`control.obwoe()`](https://evandeilton.github.io/OptimalBinningWoE/reference/control.obwoe.md)       | Create control parameters                  |

## Complete Workflow Example

Here is a complete end-to-end credit scoring workflow:

``` r
library(OptimalBinningWoE)
library(scorecard)
library(pROC)

# ============================================
# 1. Data Preparation
# ============================================

# Load German Credit dataset
data("germancredit", package = "scorecard")

# Create binary target
german <- germancredit
german$default <- factor(
  ifelse(german$creditability == "bad", 1, 0),
  levels = c(0, 1),
  labels = c("good", "bad")
)
german$creditability <- NULL

# Select features for modeling
features_num <- c("duration.in.month", "credit.amount", "age.in.years")
features_cat <- c(
  "status.of.existing.checking.account",
  "credit.history",
  "savings.account.and.bonds",
  "purpose"
)

german_model <- german[, c("default", features_num, features_cat)]

# Train/test split
set.seed(123)
train_idx <- sample(1:nrow(german_model), size = 0.7 * nrow(german_model))
train_data <- german_model[train_idx, ]
test_data <- german_model[-train_idx, ]

cat("Training set:", nrow(train_data), "observations\n")
cat("Test set:", nrow(test_data), "observations\n")
cat(
  "Training default rate:",
  round(mean(train_data$default == "bad") * 100, 2), "%\n"
)

# ============================================
# 2. Fit Optimal Binning Model
# ============================================

# Use Monotonic Optimal Binning for regulatory compliance
sc_binning <- obwoe(
  data = train_data,
  target = "default",
  algorithm = "mob",
  min_bins = 2,
  max_bins = 5,
  control = control.obwoe(
    bin_cutoff = 0.05,
    convergence_threshold = 1e-6
  )
)

# View summary
summary(sc_binning)

# ============================================
# 3. Feature Selection by IV
# ============================================

# Extract IV summary and select predictive features
iv_summary <- sc_binning$summary[!sc_binning$summary$error, ]
iv_summary <- iv_summary[order(-iv_summary$total_iv), ]

cat("\nFeature Ranking by Information Value:\n")
print(iv_summary[, c("feature", "total_iv", "n_bins")])

# Select features with IV >= 0.02
selected_features <- iv_summary$feature[iv_summary$total_iv >= 0.02]
cat("\nSelected features (IV >= 0.02):", length(selected_features), "\n")
print(selected_features)

# ============================================
# 4. Apply WoE Transformation
# ============================================

# Transform training and test data
train_woe <- obwoe_apply(train_data, sc_binning, keep_original = FALSE)
test_woe <- obwoe_apply(test_data, sc_binning, keep_original = FALSE)

# Preview transformed features
cat("\nTransformed training data (first 5 rows):\n")
print(head(train_woe[, c(
  "default",
  paste0(selected_features[1:3], "_woe")
)], 5))

# ============================================
# 5. Build Logistic Regression Model
# ============================================

# Build formula with WoE-transformed features
woe_vars <- paste0(selected_features, "_woe")
formula_str <- paste("default ~", paste(woe_vars, collapse = " + "))

# Fit logistic regression
scorecard_glm <- glm(
  as.formula(formula_str),
  data = train_woe,
  family = binomial(link = "logit")
)

cat("\nModel Summary:\n")
summary(scorecard_glm)

# ============================================
# 6. Model Evaluation
# ============================================

# Predictions on test set
test_woe$score <- predict(scorecard_glm, newdata = test_woe, type = "response")

# ROC curve and AUC
roc_obj <- roc(test_woe$default, test_woe$score, quiet = TRUE)
auc_val <- auc(roc_obj)

# KS statistic
ks_stat <- max(abs(
  ecdf(test_woe$score[test_woe$default == "bad"])(seq(0, 1, 0.01)) -
    ecdf(test_woe$score[test_woe$default == "good"])(seq(0, 1, 0.01))
))

# Gini coefficient
gini <- 2 * auc_val - 1

cat("\n============================================\n")
cat("Scorecard Performance Metrics:\n")
cat("============================================\n")
cat("  AUC:  ", round(auc_val, 4), "\n")
cat("  Gini: ", round(gini, 4), "\n")
cat("  KS:   ", round(ks_stat * 100, 2), "%\n")

# Plot ROC curve
plot(roc_obj,
  main = "Scorecard ROC Curve",
  print.auc = TRUE,
  print.thres = "best"
)

# ============================================
# 7. Gains Analysis
# ============================================

# Compute gains for best numerical feature
best_num_feature <- iv_summary$feature[iv_summary$feature %in% features_num][1]

gains <- obwoe_gains(sc_binning, feature = best_num_feature, sort_by = "id")
print(gains)

# Plot WoE and IV
plot(gains, type = "woe_iv")
```

## Data Preprocessing

Handle missing values and outliers before binning:

``` r
library(OptimalBinningWoE)

# Simulate problematic feature
set.seed(2024)
problematic_feature <- c(
  rnorm(800, 5000, 2000), # Normal values
  rep(NA, 100), # Missing values
  runif(100, -10000, 50000) # Outliers
)
target_sim <- rbinom(1000, 1, 0.3)

# Preprocess with IQR method
preproc_result <- ob_preprocess(
  feature = problematic_feature,
  target = target_sim,
  outlier_method = "iqr",
  outlier_process = TRUE,
  preprocess = "both"
)

# View preprocessing report
print(preproc_result$report)

# Access cleaned feature
cleaned_feature <- preproc_result$preprocess$feature_preprocessed
```

## Algorithm Comparison

Compare different algorithms on the same feature:

``` r
library(OptimalBinningWoE)
library(scorecard)

# Load data
data("germancredit", package = "scorecard")
german <- germancredit
german$default <- factor(
  ifelse(german$creditability == "bad", 1, 0),
  levels = c(0, 1),
  labels = c("good", "bad")
)

# Test multiple algorithms
algorithms <- c("jedi", "mob", "mdlp", "ewb", "cm")

compare_results <- lapply(algorithms, function(algo) {
  tryCatch(
    {
      fit <- obwoe(
        data = german,
        target = "default",
        feature = "credit.amount",
        algorithm = algo,
        min_bins = 3,
        max_bins = 6
      )

      data.frame(
        Algorithm = algo,
        N_Bins = fit$summary$n_bins[1],
        IV = round(fit$summary$total_iv[1], 4),
        Converged = fit$summary$converged[1]
      )
    },
    error = function(e) {
      data.frame(
        Algorithm = algo,
        N_Bins = NA,
        IV = NA,
        Converged = FALSE
      )
    }
  )
})

# Combine and display results
comparison_df <- do.call(rbind, compare_results)
comparison_df <- comparison_df[order(-comparison_df$IV), ]

cat("Algorithm Comparison on 'credit.amount':\n\n")
print(comparison_df, row.names = FALSE)

# View available algorithms
algorithms_info <- obwoe_algorithms()
print(algorithms_info[, c("algorithm", "numerical", "categorical")])
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

## Best Practices

### Workflow Recommendations

1.  **Start Simple**: Use `algorithm = "jedi"` as default
2.  **Check IV**: Select features with IV ≥ 0.02
3.  **Validate Monotonicity**: Use `mob`, `mblp`, or `ir` for regulatory
    models
4.  **Cross-Validate**: Tune binning parameters with CV
5.  **Monitor Stability**: Track WoE distributions over time
6.  **Document Thoroughly**: Save metadata with models

### Common Pitfalls to Avoid

``` r
# RONG: Bin on full dataset before splitting (causes data leakage!)
bad_approach <- obwoe(full_data, target = "default")
train_woe <- obwoe_apply(train_data, bad_approach)

# ORRECT: Bin only on training data
good_approach <- obwoe(train_data, target = "default")
test_woe <- obwoe_apply(test_data, good_approach)

# RONG: Ignore IV thresholds (IV > 0.50 likely indicates target leakage)
suspicious_features <- result$summary$feature[result$summary$total_iv > 0.50]
# Investigate these features carefully!

# RONG: Over-bin (too many bins reduces interpretability)
# max_bins > 10 may cause overfitting
```

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
Guidelines](https://github.com/evandeilton/OptimalBinningWoE/blob/main/CONTRIBUTING.md)
and [Code of
Conduct](https://github.com/evandeilton/OptimalBinningWoE/blob/main/CODE_OF_CONDUCT.md).

## Citation

If you use OptimalBinningWoE in your research, please cite:

``` bibtex
@software{optimalbinningwoe,
  author = {José Evandeilton Lopes},
  title = {OptimalBinningWoE: Optimal Binning and Weight of Evidence Framework for Modeling},
  year = {2026},
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
- Anderson, R. (2007). *The Credit Scoring Toolkit: Theory and Practice
  for Retail Credit Risk Management*. Oxford University Press.

## License

MIT License © 2026 José Evandeilton Lopes
