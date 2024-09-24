
<!-- README.md is generated from README.Rmd. Please edit that file -->

# OptimalBinningWoE

<!-- badges: start -->

[![R-CMD-check](https://github.com/evandeilton/OptimalBinningWoE/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/evandeilton/OptimalBinningWoE/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

The OptimalBinningWoE package offers a robust and flexible
implementation of optimal binning and Weight of Evidence (WoE)
calculation for data analysis and predictive modeling. This package is
particularly useful for data preparation in credit scoring models but
can be applied in various statistical modeling contexts.

## Installation

You can install the development version of OptimalBinningWoE from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("your_username/OptimalBinningWoE")
```

## Overview

OptimalBinningWoE offers the following main functionalities:

1.  Optimal binning for categorical and numerical variables
2.  Weight of Evidence (WoE) calculation
3.  Automatic selection of the best binning method
4.  Data preprocessing, including handling of missing values and
    outliers

## Supported Algorithms

OptimalBinningWoE supports the following binning algorithms:

1.  CAIM (Class-Attribute Interdependence Maximization): Applicable to
    both categorical and numerical variables.
2.  ChiMerge: Applicable to both categorical and numerical variables.
3.  MDLP (Minimum Description Length Principle): Applicable to both
    categorical and numerical variables.
4.  MIP (Minimum Information Pure): Applicable to both categorical and
    numerical variables.
5.  MOB (Monotone Optimal Binning): Applicable to both categorical and
    numerical variables.
6.  IV (Information Value): Applicable only to categorical variables.
7.  PAVA (Pool Adjacent Violators Algorithm): Applicable only to
    numerical variables.
8.  Tree-based binning: Applicable only to numerical variables.

Each algorithm has its own strengths and may perform differently
depending on the nature of the data. The automatic method selection
option tests applicable algorithms and chooses the one that produces the
highest Information Value.

## Control Parameters

The package offers various control parameters to adjust the behavior of
binning and preprocessing:

- `min_bads`: Minimum proportion of “bad” cases in each bin (default:
  0.05)
- `pvalue_threshold`: P-value threshold for statistical tests (default:
  0.05)
- `max_n_prebins`: Maximum number of pre-bins before optimization
  (default: 20)
- `monotonicity_direction`: Direction of monotonicity (“increase” or
  “decrease”) (default: “increase”)
- `lambda`: Regularization parameter for tree-based methods (default:
  0.1)
- `min_bin_size`: Minimum proportion of cases in each bin (default:
  0.05)
- `min_iv_gain`: Minimum IV gain for creating a new split (default:
  0.01)
- `max_depth`: Maximum depth for tree-based methods (default: 10)
- `num_miss_value`: Value to represent missing numeric values (default:
  -999.0)
- `char_miss_value`: Value to represent missing categorical values
  (default: “N/A”)
- `outlier_method`: Method for outlier detection (“iqr”, “zscore”, or
  “grubbs”) (default: “iqr”)
- `outlier_process`: Whether to process outliers (default: FALSE)
- `iqr_k`: Factor for the IQR method (default: 1.5)
- `zscore_threshold`: Threshold for the Z-score method (default: 3)
- `grubbs_alpha`: Significance level for Grubbs’ test (default: 0.05)

## Usage Examples

``` r
library(OptimalBinningWoE)
library(data.table)
library(scorecard)
library(janitor)

# Load the data
data("germancredit")
da <- data.table::setDT(germancredit) %>%
  data.table::copy() %>% 
  janitor::clean_names()

# Define the target variable
da[, default := ifelse(creditability == "bad", 1, 0)]
da$creditability <- NULL
target <- "default"

# Copy data
dt <- data.table::copy(da)

# Run OptimalBinningWoE with automatic method selection
out <- OptimalBinningWoE(dt, target = "default", method = "auto")

# View results
head(out$woe_feature)
head(out$woe_woebins)
head(out$prep_report)
```

## Detailed Examples

### 1. Binning a numeric variable with a specific method

``` r
# Copy data
dt <- data.table::copy(da)
# Using the MDLP method for the 'age_in_years' variable
out <- OptimalBinningWoE(dt, target = "default", feature = "age_in_years", method = "mdlp")
print(out$woe_woebins)
```

### 2. Binning a categorical variable

``` r
# Copy data
dt <- data.table::copy(da)
# Using the ChiMerge method for the 'purpose' variable
out <- OptimalBinningWoE(dt, target = "default", feature = "purpose", method = "chimerge")
print(out$woe_woebins)
```

### 3. Handling missing values in preprocessing

For this example, we’ll artificially add some missing values:

``` r
# Copy data
dt <- data.table::copy(da)

# Add missing values
set.seed(123)
dt[sample(1:nrow(dt), 50), age_in_years := NA]
dt[sample(1:nrow(dt), 30), credit_amount := NA]

# Run OptimalBinningWoE with preprocessing
out <- OptimalBinningWoE(dt, target = "default", 
                         feature = c("age_in_years", "credit_amount"), 
                         preprocess = TRUE)
```

### 4. Handling outliers

``` r
# Copy data
dt <- data.table::copy(da)

# Add some outliers to the data
dt[sample(1:nrow(dt), 10), credit_amount := rnorm(10, mean = 100000, sd = 10000)]

# Run OptimalBinningWoE with outlier treatment
out <- OptimalBinningWoE(dt, target = "default", 
                         feature = "credit_amount", preprocess = TRUE,
                         control = list(outlier_method = "iqr", outlier_process = TRUE))
```

### 5. Comparison of different methods

``` r
methods <- c("caim", "chimerge", "mdlp", "mip", "mob")
out <- list()

for (method in methods) {
  out[[method]] <- OptimalBinningWoE(dt, target = "default", 
                                     feature = "duration_in_month", 
                                     method = method)
}

# Compare the number of bins and total IV for each method
comparison <- data.frame(
  Method = methods,
  Num_Bins = sapply(out, function(x) nrow(x$woe_woebins)),
  Total_IV = sapply(out, function(x) sum(x$woe_woebins$iv))
)
```

### 6. Processing multiple variables

``` r
# Select multiple features for processing
selected_features <- c("age_in_years", "credit_amount", "duration_in_month", 
                       "present_residence_since", "number_of_existing_credits_at_this_bank")

# Run OptimalBinningWoE for multiple features
out <- OptimalBinningWoE(dt, target = "default", feature = selected_features, method = "auto")

# View summarized results
summary_results <- out$woe_woebins
summary_results <- summary_results[, c("total_iv", "nclass") := list(sum(iv), .N), by = feature]
```

### 7. Analysis of an ordinal variable

``` r
# Analyze the 'present_residence_since' variable
out <- OptimalBinningWoE(dt, target = "default", 
                         feature = "present_residence_since", method = "mob")
```

These examples demonstrate the use of OptimalBinningWoE with the German
Credit dataset, including:

1.  Use of real variables from a credit scoring dataset.
2.  Handling of numeric, categorical, and ordinal variables.
3.  Dealing with missing values and outliers (artificially introduced
    for demonstration).
4.  Comparison of different binning methods.
5.  Simultaneous processing of multiple variables.
6.  Specific analysis of an ordinal variable.

`This version of the examples uses the German Credit dataset, which is a real dataset widely used in credit scoring studies. The examples cover a variety of scenarios and variable types present in this dataset, providing a more realistic and relevant demonstration of the OptimalBinningWoE package usage.`

## Final Considerations

The OptimalBinningWoE package offers a comprehensive solution for
binning and WoE calculation, with support for various algorithms and
preprocessing options. When using this package, consider the following
points:

1.  Automatic method selection (`method = "auto"`) can be useful when
    you’re unsure which algorithm to use, but it may be computationally
    intensive for large datasets.

2.  Data preprocessing, including handling missing values and outliers,
    can significantly impact binning results. Adjust the control
    parameters as necessary for your specific dataset.

3.  Different binning methods can produce significantly different
    results. It’s good practice to compare the results of several
    methods before making a final choice.

4.  For very large datasets, consider using a representative sample to
    determine the optimal bins and then apply these bins to the full
    dataset.

5.  The package provides flexibility to handle different types of data
    and modeling scenarios. Experiment with different settings to find
    the best approach for your specific data.

For more details on available options and interpretation of results,
please refer to the complete package documentation.

## Contributing

Contributions to OptimalBinningWoE are welcome! Please refer to the
CONTRIBUTING.md file for guidelines on how to contribute to this
project.

## License

This project is licensed under the MIT License - see the LICENSE.md file
for details.