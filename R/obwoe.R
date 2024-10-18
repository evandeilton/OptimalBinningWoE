#' Optimal Binning and Weight of Evidence Calculation
#'
#' @description
#' This function performs optimal binning and calculates Weight of Evidence (WoE)
#' for both numerical and categorical features. It implements a variety of
#' advanced binning algorithms to discretize continuous variables and optimize
#' categorical variables for predictive modeling, particularly in credit scoring
#' and risk assessment applications.
#'
#' The function supports automatic method selection, data preprocessing, and handles
#' both numerical and categorical features. It aims to maximize the predictive power
#' of features while maintaining interpretability through monotonic binning and
#' information value optimization.
#'
#' @details
#' Supported Algorithms:
#' The function implements the following binning algorithms:
#'
#' For Categorical Variables:
#' \itemize{
#'   \item FETB (Fisher's Exact Test Binning): Uses Fisher's exact test for binning
#'   \item CM (ChiMerge): Merges categories based on chi-square statistic
#'   \item UDT (Unsupervised Decision Trees): Uses decision tree algorithms for categorical binning
#'   \item IVB (Information Value Binning): Bins based on information value
#'   \item GMB (Greedy Monotonic Binning): Uses a greedy approach to create monotonic bins for categories
#'   \item SWB (Sliding Window Binning): Adapts the sliding window approach for categorical variables
#'   \item DPLC (Dynamic Programming with Local Constraints): Applies dynamic programming with local constraints
#'   \item MOB (Monotonic Optimal Binning): Ensures monotonicity in Weight of Evidence across categories
#'   \item MBA (Modified Binning Algorithm): A modified approach for categorical variable binning
#'   \item MILP (Mixed Integer Linear Programming): Applies mixed integer linear programming to categorical binning
#'   \item SAB (Simulated Annealing Binning): Uses simulated annealing for optimal binning
#' }
#'
#' For Numerical Variables:
#' \itemize{
#'   \item UDT (Unsupervised Decision Trees): Applies decision tree algorithms in an unsupervised manner for binning
#'   \item MDLP (Minimum Description Length Principle): Uses the MDLP criterion for binning
#'   \item MOB (Monotonic Optimal Binning): Ensures monotonicity in Weight of Evidence across bins
#'   \item MBLP (Monotonic Binning via Linear Programming): Uses linear programming for monotonic binning
#'   \item DPLC (Dynamic Programming with Local Constraints): Uses dynamic programming with local constraints
#'   \item LPDB (Local Polynomial Density Binning): Employs local polynomial density estimation
#'   \item UBSD (Unsupervised Binning with Standard Deviation): Uses standard deviation in unsupervised binning
#'   \item FETB (Fisher's Exact Test Binning): Applies Fisher's exact test to numerical variables
#'   \item EWB (Equal Width Binning): Creates bins of equal width across the range of the variable
#'   \item KMB (K-means Binning): Applies k-means clustering for binning
#'   \item OSLP (Optimal Supervised Learning Path): Uses a supervised learning path for optimal binning
#'   \item MRBLP (Monotonic Regression-Based Linear Programming): Combines monotonic regression with linear programming
#'   \item IR (Isotonic Regression): Uses isotonic regression for binning
#'   \item BB (Branch and Bound): Uses a branch and bound algorithm for optimal binning
#'   \item LDB (Local Density Binning): Uses local density estimation for binning
#' }
#'
#' Key Concepts:
#' \itemize{
#'   \item Weight of Evidence (WoE): \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
#'     where \eqn{P(X_i|Y=1)} is the proportion of positive cases in bin i, and
#'     \eqn{P(X_i|Y=0)} is the proportion of negative cases in bin i.
#'
#'   \item Information Value (IV): \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
#'     The total IV is the sum of IVs across all bins: \deqn{IV_{total} = \sum_{i=1}^{n} IV_i}
#' }
#'
#' Method Selection:
#' When method = "auto", the function tests multiple algorithms and selects the one
#' that produces the highest total Information Value while respecting the specified constraints.
#'
#' @param dt A data.table containing the dataset.
#' @param target The name of the target variable (must be binary).
#' @param features Vector of feature names to process. If NULL, all features except the target will be processed.
#' @param method The binning method to use. Can be "auto" or one of the methods listed in the details section.
#' @param preprocess Logical. Whether to preprocess the data before binning (default: TRUE).
#' @param outputall Logical. If TRUE, returns only the optimal binning gains table. If FALSE, returns a list with data, gains table, and reports (default: TRUE).
#' @param min_bins Minimum number of bins (default: 3).
#' @param max_bins Maximum number of bins (default: 4).
#' @param control A list of additional control parameters:
#'   \itemize{
#'     \item cat_cutoff: Minimum frequency for a category (default: 0.05)
#'     \item bin_cutoff: Minimum frequency for a bin (default: 0.05)
#'     \item min_bads: Minimum proportion of bad cases in a bin (default: 0.05)
#'     \item pvalue_threshold: P-value threshold for statistical tests (default: 0.05)
#'     \item max_n_prebins: Maximum number of pre-bins (default: 20)
#'     \item monotonicity_direction: Direction of monotonicity for some algorithms ("increase" or "decrease")
#'     \item lambda: Regularization parameter for some algorithms (default: 0.1)
#'     \item min_bin_size: Minimum bin size as a proportion of total observations (default: 0.05)
#'     \item min_iv_gain: Minimum IV gain for bin splitting for some algorithms (default: 0.01)
#'     \item max_depth: Maximum depth for tree-based algorithms (default: 10)
#'     \item num_miss_value: Value to replace missing numeric values (default: -999.0)
#'     \item char_miss_value: Value to replace missing categorical values (default: "N/A")
#'     \item outlier_method: Method for outlier detection ("iqr", "zscore", or "grubbs")
#'     \item outlier_process: Whether to process outliers (default: FALSE)
#'     \item iqr_k: IQR multiplier for outlier detection (default: 1.5)
#'     \item zscore_threshold: Z-score threshold for outlier detection (default: 3)
#'     \item grubbs_alpha: Significance level for Grubbs' test (default: 0.05)
#'     \item n_threads: Number of threads for parallel processing (default: 1)
#'     \item is_monotonic: Whether to enforce monotonicity in binning (default: TRUE)
#'     \item population_size: Population size for genetic algorithm (default: 50)
#'     \item max_generations: Maximum number of generations for genetic algorithm (default: 100)
#'     \item mutation_rate: Mutation rate for genetic algorithm (default: 0.1)
#'     \item initial_temperature: Initial temperature for simulated annealing (default: 1)
#'     \item cooling_rate: Cooling rate for simulated annealing (default: 0.995)
#'     \item max_iterations: Maximum number of iterations for iterative algorithms (default: 1000)
#'     \item include_upper_bound: Include upper bound for numeric bins (default is TRUE)
#'     \item bin_separator: Bin separator for optimal bins categorical variables (default = "%;%")
#'   }
#' @param positive Character string specifying which category should be considered as positive. Must be either "bad|1" or "good|1".
#' @param progress Logical. Whether to display a progress bar. Default is TRUE.
#' @param trace Logical. Whether to generate error logs when testing existing methods.
#'
#' @return Depending on the value of outputall:
#' If outputall = FALSE:
#'   A data.table containing the optimal binning gains table (woebin).
#' If outputall = TRUE:
#'   A list containing:
#'   \item{data}{The original dataset with added WoE columns}
#'   \item{woebin}{Information about the bins created, including:
#'     \itemize{
#'       \item feature: Name of the feature
#'       \item bin: Bin label or range
#'       \item count: Number of observations in the bin
#'       \item count_distr: Proportion of observations in the bin
#'       \item good: Number of good cases (target = 0) in the bin
#'       \item bad: Number of bad cases (target = 1) in the bin
#'       \item good_rate: Proportion of good cases in the bin
#'       \item bad_rate: Proportion of bad cases in the bin
#'       \item woe: Weight of Evidence for the bin
#'       \item iv: Information Value contribution of the bin
#'     }
#'   }
#'   \item{report_best_model}{Report on the best tested models, including:
#'     \itemize{
#'       \item feature: Name of the feature
#'       \item method: Best method selected for the feature
#'       \item iv_total: Total Information Value achieved
#'       \item n_bins: Number of bins created
#'       \item runtime: Execution time for binning the feature
#'     }
#'   }
#'   \item{report_preprocess}{Preprocessing report for each feature, including:
#'     \itemize{
#'       \item feature: Name of the feature
#'       \item type: Data type of the feature
#'       \item missing_count: Number of missing values
#'       \item outlier_count: Number of outliers detected
#'       \item unique_count: Number of unique values
#'       \item mean_before: Mean value before preprocessing
#'       \item mean_after: Mean value after preprocessing
#'       \item sd_before: Standard deviation before preprocessing
#'       \item sd_after: Standard deviation after preprocessing
#'     }
#'   }
#'
#' @import data.table
#' @importFrom stats rnorm pchisq median sd quantile
#' @importFrom utils modifyList setTxtProgressBar txtProgressBar
#'
#' @examples
#' \dontrun{
#' # Example 1: Using the German Credit Data
#' library(OptimalBinningWoE)
#' library(data.table)
#' library(scorecard)
#' data(germancredit, package = "scorecard")
#' dt <- as.data.table(germancredit)
#'
#' # Process all features with MBLP method
#' result <- obwoe(dt,
#'   target = "creditability", method = "mblp",
#'   min_bins = 3, max_bins = 5, positive = "bad|1"
#' )
#'
#' # View WoE binning information
#' print(result)
#'
#' # Process only numeric features with MBLP method and get detailed output
#' numeric_features <- names(dt)[sapply(dt, is.numeric)]
#' numeric_features <- setdiff(numeric_features, "creditability")
#'
#' result_detailed <- obwoe(dt,
#'   target = "creditability", features = numeric_features,
#'   method = "mblp", preprocess = TRUE, outputall = FALSE,
#'   min_bins = 3, max_bins = 5, positive = "bad|1"
#' )
#'
#' # View WoE-transformed data
#' head(result_detailed$data)
#'
#' # View preprocessing report
#' print(result_detailed$report_preprocess)
#'
#' # View best model report
#' print(result_detailed$report_best_model)
#'
#' # Process only categoric features with UDT method
#' categoric_features <- names(dt)[sapply(dt, function(i) !is.numeric(i))]
#' categoric_features <- setdiff(categoric_features, "creditability")
#' result_cat <- obwoe(dt,
#'   target = "creditability", features = categoric_features,
#'   method = "udt", preprocess = TRUE,
#'   min_bins = 3, max_bins = 4, positive = "bad|1"
#' )
#'
#' # View binning information for categorical features
#' print(result_cat)
#' }
#'
#' @export
obwoe <- function(dt, target, features = NULL, min_bins = 3, max_bins = 4, method = "fetb",
                  positive = "bad|1", preprocess = TRUE, progress = TRUE, trace = FALSE,
                  outputall = TRUE, control = list()) {
  # Default control
  defaultControl <- list(
    cat_cutoff = 0.05, bin_cutoff = 0.05, min_bads = 0.05,
    pvalue_threshold = 0.05, max_n_prebins = 20,
    monotonicity_direction = "increase", lambda = 0.1,
    min_bin_size = 0.05, min_iv_gain = 0.01, max_depth = 10,
    num_miss_value = -999.0, char_miss_value = "N/A",
    outlier_method = "iqr", outlier_process = FALSE, iqr_k = 1.5,
    zscore_threshold = 3, grubbs_alpha = 0.05, n_threads = 1L,
    is_monotonic = TRUE, population_size = 50L, max_generations = 100L,
    mutation_rate = 0.1, initial_temperature = 1, cooling_rate = 0.995,
    max_iterations = 1000L,
    include_upper_bound = TRUE,
    bin_separator = "%;%"
  )

  # Update default control
  control <- utils::modifyList(defaultControl, control)

  # Ensure dt is a data.table
  dt <- data.table::setDT(data.table::copy(dt))

  # Validate inputs
  OptimalBinningValidateInputs(dt, target, features, method, preprocess, min_bins, max_bins, control, positive)

  # Determine the features to process
  if (is.null(features)) {
    features <- setdiff(names(dt), c(target, "target"))
  }

  # Initialize results list
  results <- reports <- woebins <- bestmod <- list()
  failed_features <- character()
  nonprocessed_features <- character()
  singleclass_target_features <- character()

  # Map target based on 'positive' argument
  dt <- OptimalBinningMapTargetVariable(dt, target, positive)

  # Check for unsupported variable types and single-class target in categorical variables
  for (feat in features) {
    if (!is.numeric(dt[[feat]]) && !is.factor(dt[[feat]]) && !is.character(dt[[feat]])) {
      nonprocessed_features <- c(nonprocessed_features, feat)
    } else if (is.factor(dt[[feat]]) || is.character(dt[[feat]])) {
      target_values <- dt[[target]][!is.na(dt[[feat]])]
      feature_values <- dt[[feat]][!is.na(dt[[feat]])]
      if (length(unique(target_values)) == 1) {
        singleclass_target_features <- c(singleclass_target_features, feat)
      }
    }
  }

  # Remove nonprocessed and singleclass target features from the features list
  features <- setdiff(features, c(nonprocessed_features, singleclass_target_features))

  # Preprocess data if required
  if (preprocess) {
    preprocessed_data <- OptimalBinningPreprocessData(dt, target, features, control, preprocess = "both")
  } else {
    preprocessed_data <- list()
    for (feat in features) {
      preprocessed_data[[feat]] <- list(
        preprocess = data.table::data.table(feature_preprocessed = dt[[feat]]),
        report = OptimalBinningPreprocessData(dt, target, features, control, preprocess = "report")
      )
    }
  }

  if (length(features) > 0) {
    results <- OptimalBinningSelectBestModel(dt, target, features, method, min_bins, max_bins, control, progress, trace)
  }

  # Prepare woebin gains table stats
  woebin <- data.table::rbindlist(lapply(results, function(x) data.table::setDT(x$woebin)), idcol = "feature")

  if (!outputall) {
    return(woebin)
  } else {
    # Prepare data pos-processed
    data <- data.table::data.table()[, (target) := dt[[target]]]
    data <- cbind(data, do.call(cbind, lapply(results, function(x) x$woefeature)))

    # Best model selection report
    report_best_model <- data.table::rbindlist(lapply(results, function(x) data.table::setDT(x$report)), idcol = "feature")
    # data.table::setorder(report_best_model, feature, id)

    # Stats from pré-processed data
    report_preprocess <- data.table::rbindlist(lapply(preprocessed_data, function(x) data.table::setDT(x$report)), idcol = "feature")
    # data.table::setorder(report_preprocess, feature, variable_type)

    return(
      list(data = data, woebin = woebin, report_best_model = report_best_model, report_preprocess = report_preprocess)
    )
  }
}


#' Select Optimal Features Based on Weight of Evidence
#'
#' @description
#' This function selects optimal features from the result of an Optimal Binning and
#' Weight of Evidence (WoE) analysis. It filters features based on their Information
#' Value (IV), allowing for fine-tuned feature selection for predictive modeling.
#'
#' @param obresult A list containing the result of the Optimal Binning and WoE analysis.
#'   Must include elements 'woedt' (a data.table with WoE transformed data) and
#'   'bestsreport' (a data.table with feature performance metrics).
#' @param target Character. The name of the target variable in the dataset.
#' @param iv_threshold Numeric. The minimum Information Value threshold for feature selection.
#'   Features with IV below this threshold will be excluded. Default is 0.02.
#' @param min_features Integer. The minimum number of features to select, regardless of
#'   their IV. If fewer features meet the IV threshold, this ensures a minimum
#'   set is still selected. Default is 5.
#' @param max_features Integer or NULL. The maximum number of features to select.
#'   If NULL (default), no maximum limit is applied.
#'
#' @return A list containing:
#'   \item{data}{A data.table with the selected WoE features and the target variable.}
#'   \item{selected_features}{A character vector of the selected WoE feature names.}
#'   \item{feature_iv}{A data.table with all features and their total IV.}
#'   \item{report}{A data.table summarizing the feature selection process.}
#'
#' @details
#' The function performs the following steps:
#' 1. Validates input parameters.
#' 2. Extracts and sorts features by their Information Value.
#' 3. Selects features based on the provided IV threshold.
#' 4. Adjusts the selection to meet minimum and maximum feature count requirements.
#' 5. Prepares a final dataset with selected WoE features and the target variable.
#' 6. Generates a summary report of the selection process.
#'
#' Mathematical Background:
#'
#' Weight of Evidence (WoE) and Information Value (IV) are key concepts in predictive modeling,
#' especially in credit scoring. They are derived from information theory and provide a way to
#' measure the predictive power of an independent variable in relation to the dependent variable.
#'
#' Let \eqn{Y} be a binary target variable and \eqn{X} be a predictor variable.
#'
#' For a given bin \eqn{i} of \eqn{X}:
#'
#' \deqn{P(X_i|Y=1) = \frac{\text{Number of events in bin i}}{\text{Total number of events}}}
#'
#' \deqn{P(X_i|Y=0) = \frac{\text{Number of non-events in bin i}}{\text{Total number of non-events}}}
#'
#' The Weight of Evidence for bin \eqn{i} is defined as:
#'
#' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
#'
#' The Information Value for the entire variable \eqn{X} is:
#'
#' \deqn{IV = \sum_{i} (P(X_i|Y=1) - P(X_i|Y=0)) \cdot WoE_i}
#'
#' Interpretation of Information Value:
#'
#' | IV Range  | Predictive Power |
#' |-----------|-------------------|
#' | < 0.02    | Useless           |
#' | 0.02-0.1  | Weak              |
#' | 0.1-0.3   | Medium            |
#' | 0.3-0.5   | Strong            |
#' | > 0.5     | Suspicious        |
#'
#' Note: An IV > 0.5 might indicate overfitting or data leakage and should be investigated.
#'
#' @examples
#' \dontrun{
#' # Assuming 'obwoe_result' is the output from an Optimal Binning and WoE analysis
#' result <- OptimalBinningSelectOptimalFeatures(
#'   obresult = obwoe_result,
#'   target = "target_variable",
#'   iv_threshold = 0.05,
#'   min_features = 10,
#'   max_features = 30
#' )
#'
#' # Access the final dataset with selected WoE features
#' final_dataset <- result$data
#'
#' # View the selected WoE feature names
#' print(result$selected_features)
#'
#' # View the feature selection summary report
#' print(result$report)
#' }
#'
#' @importFrom data.table setDT setorder data.table
#' @export
OptimalBinningSelectOptimalFeatures <- function(obresult, target, iv_threshold = 0.02, min_features = 5, max_features = NULL) {
  # Input validation
  if (!is.list(obresult) || !all(c("woedt", "woebins") %in% names(obresult))) {
    stop("'obresult' must be a list containing 'woedt' and 'woebins' elements")
  }
  if (!is.character(target) || length(target) != 1) {
    stop("'target' must be a single character string")
  }
  if (!is.numeric(iv_threshold) || iv_threshold < 0) {
    stop("'iv_threshold' must be a non-negative numeric value")
  }
  if (!is.numeric(min_features) || min_features < 1) {
    stop("'min_features' must be a positive integer")
  }
  if (!is.null(max_features) && (!is.numeric(max_features) || max_features < min_features)) {
    stop("'max_features' must be NULL or a numeric value greater than or equal to 'min_features'")
  }

  # Ensure woedt is a data.table
  obdt <- data.table::setDT(obresult$woedt)

  # Validate target variable
  if (!target %in% names(obdt)) {
    stop(sprintf("Target variable '%s' not found in the dataset", target))
  }

  # Calculate total IV for each feature from woebins
  feature_iv <- obresult$woebins[, .(total_iv = sum(iv)), by = .(feature)]
  data.table::setorder(feature_iv, -total_iv)

  # Select features based on IV threshold
  selected_features <- unique(feature_iv[total_iv >= iv_threshold, feature])

  # Adjust for min_features if necessary
  if (length(selected_features) < min_features) {
    selected_features <- feature_iv[1:min(min_features, nrow(feature_iv)), feature]
  }

  # Adjust for max_features if specified
  if (!is.null(max_features) && length(selected_features) > max_features) {
    selected_features <- selected_features[1:max_features]
  }

  # Add "_woe" suffix to selected features
  selected_woe_features <- paste0(selected_features, "_woe")

  # Validate selected WoE features exist in the dataset
  missing_features <- selected_woe_features[!selected_woe_features %in% names(obdt)]
  if (length(missing_features) > 0) {
    warning(sprintf(
      "The following WoE features were not found in the dataset: %s",
      paste(missing_features, collapse = ", ")
    ))
    selected_woe_features <- selected_woe_features[selected_woe_features %in% names(obdt)]
  }

  # Prepare the final dataset with selected WoE features and target
  final_dt <- obdt[, c(target, selected_woe_features), with = FALSE]

  # Prepare the report
  report <- data.table::data.table(
    total_features = sum(grepl("_woe$", names(obdt))),
    selected_features = length(selected_woe_features),
    iv_threshold = iv_threshold,
    min_iv = feature_iv[feature %in% selected_features, min(total_iv)],
    max_iv = feature_iv[feature %in% selected_features, max(total_iv)],
    mean_iv = feature_iv[feature %in% selected_features, mean(total_iv)]
  )

  # Return the results
  return(list(
    data = final_dt,
    selected_features = selected_woe_features,
    feature_iv = feature_iv,
    report = report
  ))
}


#' Preprocess Data for Optimal Binning
#'
#' @param dt A data.table containing the dataset.
#' @param target Target name
#' @param features Vector of feature names to process.
#' @param control A list of control parameters.
#' @param preprocess Preprocess feature. 'both' feature and report. Can also be 'both' or 'feature'
#'
#' @return A list of preprocessed data for each feature.
#'
#' @export
OptimalBinningPreprocessData <- function(dt, target, features, control, preprocess = "both") {
  preprocessed_data <- list()
  for (feat in features) {
    preprocessed_data[[feat]] <- OptimalBinningDataPreprocessor(
      target = dt[[target]],
      feature = dt[[feat]],
      num_miss_value = control$num_miss_value,
      char_miss_value = control$char_miss_value,
      outlier_method = control$outlier_method,
      outlier_process = control$outlier_process,
      preprocess = preprocess,
      iqr_k = control$iqr_k,
      zscore_threshold = control$zscore_threshold,
      grubbs_alpha = control$grubbs_alpha
    )
  }
  return(preprocessed_data)
}

#' Check if WoE values are monotonic
#'
#' @param woe_values Vector of WoE values
#'
#' @return Logical indicating if WoE values are monotonic
#'
#' @keywords internal
#' @export
OptimalBinningIsWoEMonotonic <- function(woe_values) {
  diff_woe <- diff(woe_values)
  all(diff_woe >= 0) || all(diff_woe <= 0)
}

#' Map Target Variable
#'
#' @param dt Data table
#' @param target Target variable name
#' @param positive Positive class indicator
#'
#' @return Updated data table with mapped target variable
#'
#' @keywords internal
#' @export
OptimalBinningMapTargetVariable <- function(dt, target, positive) {
  if (is.character(dt[[target]]) || is.factor(dt[[target]])) {
    if (length(unique(dt[[target]])) > 2) {
      stop("Target variable must have exactly two categories")
    }
    positive_value <- strsplit(positive, "\\|")[[1]][1]
    target_col <- dt[[target]]
    dt[, (target) := ifelse(target_col == positive_value, 1, 0)]
  } else if (!all(dt[[target]] %in% c(0, 1))) {
    stop("Target variable must be binary (0 or 1) or a string with two categories")
  }
  return(dt)
}

#' Calculate Special WoE
#'
#' @param target Target values for special cases
#'
#' @return WoE value for special cases
#'
#' @keywords internal
#' @export
OptimalBinningCalculateSpecialWoE <- function(target) {
  counts <- table(target)
  log(counts[2] / sum(counts) / (counts[1] / sum(counts)))
}

#' Create Special Bin
#'
#' @param dt_special Data for special cases
#' @param woebin Existing WoE bins
#' @param special_woe WoE value for special cases
#'
#' @return Special bin information
#'
#' @keywords internal
#' @export
OptimalBinningCreateSpecialBin <- function(dt_special, woebin, special_woe) {
  data.table::data.table(
    bin = "Special",
    count = nrow(dt_special),
    count_neg = sum(dt_special$target == 0),
    count_pos = sum(dt_special$target == 1),
    woe = special_woe,
    iv = (sum(dt_special$target == 1) / (sum(woebin$count_pos) + sum(dt_special$target == 1)) -
      sum(dt_special$target == 0) / (sum(woebin$count_neg) + sum(dt_special$target == 0))) * special_woe
  )
}


#' Validate Inputs for Optimal Binning
#'
#' @param dt A data.table containing the dataset.
#' @param target The name of the target variable.
#' @param features Vector of feature names to process.
#' @param method The binning method to use.
#' @param preprocess Logical. Whether to preprocess the data before binning.
#' @param min_bins Minimum number of bins.
#' @param max_bins Maximum number of bins.
#' @param control A list of additional control parameters.
#' @param positive Character string specifying which category should be considered as positive.
#'
#' @return None. Throws an error if any input is invalid.
#'
#' @keywords internal
#' @export
OptimalBinningValidateInputs <- function(dt, target, features, method, preprocess, min_bins, max_bins, control, positive) {
  # Check if dt is a data.table
  if (!data.table::is.data.table(dt)) {
    stop("The 'dt' argument must be a data.table.")
  }

  # Check if target exists in dt
  if (!target %in% names(dt)) {
    stop("The 'target' variable does not exist in the provided data.table.")
  }

  # Check if target is binary or has two categories
  if (is.numeric(dt[[target]])) {
    if (!all(dt[[target]] %in% c(0, 1))) {
      stop("The 'target' variable must be binary (0 or 1) when numeric.")
    }
  } else if (is.character(dt[[target]]) || is.factor(dt[[target]])) {
    if (length(unique(dt[[target]])) != 2) {
      stop("The 'target' variable must have exactly two categories when categorical.")
    }
  } else {
    stop("The 'target' variable must be either numeric (0/1) or categorical (two categories).")
  }

  # Check features (if provided)
  if (!is.null(features)) {
    if (!all(features %in% names(dt))) {
      stop("One or more specified 'features' do not exist in the provided data.table.")
    }
  }

  # Define all possible methods
  # all_methods <- c(
  #   "auto", "cm", "dplc", "gmb", "ldb", "mba", "mblp", "milp", "mob", "obnp", "swb", "udt",
  #   "bb", "bs", "dpb", "eb", "eblc", "efb", "ewb", "ir", "jnbo", "kmb", "mdlp", "mrblp", "plaob", "qb", "sbb", "ubsd"
  # )

  all_methods <- c("auto", sort(unique(unlist(unname(sapply(OptimalBinningGetAlgoName(), names))))))

  # Check binning method
  if (!method %in% all_methods) {
    stop(paste("Invalid binning method. Choose one of the following:", paste(all_methods, collapse = ", ")))
  }

  # Check preprocess
  if (!is.logical(preprocess)) {
    stop("'preprocess' must be a logical value (TRUE or FALSE).")
  }

  # Check min_bins and max_bins
  # if (!is.numeric(min_bins) || min_bins < 2) {
  #   stop("min_bins must be an integer greater than or equal to 2.")
  # }
  # if (!is.numeric(max_bins) | max_bins < min_bins) {
  #   stop("max_bins must be an integer greater than or equal to min_bins.")
  # }

  # Check control
  if (!is.list(control)) {
    stop("'control' must be a list.")
  }

  # Check specific control parameters
  if (!is.numeric(control$cat_cutoff) || control$cat_cutoff <= 0 || control$cat_cutoff >= 1) {
    stop("control$cat_cutoff must be a number between 0 and 1.")
  }
  if (!is.numeric(control$bin_cutoff) || control$bin_cutoff <= 0 || control$bin_cutoff >= 1) {
    stop("control$bin_cutoff must be a number between 0 and 1.")
  }

  # Check positive argument
  if (!is.character(positive) || !grepl("^(bad|good)\\|1$", positive)) {
    stop("'positive' must be either 'bad|1' or 'good|1'")
  }

  # If all checks pass, the function will return silently
}




#' Select Optimal Binning Algorithm
#'
#' @description
#' This function selects the appropriate binning algorithm based on the method and variable type.
#'
#' @param feature The name of the feature to bin.
#' @param method The binning method to use.
#' @param dt A data.table containing the dataset.
#' @param min_bin Minimum number of bins.
#' @param max_bin Maximum number of bins.
#' @param control A list of additional control parameters.
#'
#' @return A list containing the selected algorithm, its parameters, and the method name.
#'
#' @keywords internal
#' @export
OptimalBinningSelectAlgorithm <- function(feature, method, dt, min_bin, max_bin, control) {
  # Determine if the feature is categorical or numeric
  is_categorical <- is.factor(dt[[feature]]) || is.character(dt[[feature]])

  # Get available algorithms using OptimalBinningGetAlgoName()
  available_algorithms <- OptimalBinningGetAlgoName()

  # Select the appropriate algorithm based on the method and variable type
  data_type <- if (is_categorical) "char" else "num"

  # Find the algorithm that matches the method
  selected_algorithm <- NULL
  for (algo in names(available_algorithms[[data_type]])) {
    selected_algorithm[[algo]] <- available_algorithms[[data_type]][[algo]]
  }

  # Check if a valid algorithm was found
  if (is.null(selected_algorithm)) {
    stop(paste("The", method, "method is not applicable for", if (data_type == "char") "categorical" else "numeric", "variables."))
  }

  # Define default parameters for all algorithms
  default_params <- list(
    min_bins = min_bin,
    max_bins = max_bin,
    bin_cutoff = control$bin_cutoff,
    max_n_prebins = control$max_n_prebins
  )

  # Define specific parameters for certain algorithms
  specific_params <- list(
    optimal_binning_categorical_gab = list(
      population_size = control$population_size,
      num_generations = control$num_generations,
      mutation_rate = control$mutation_rate,
      crossover_rate = control$crossover_rate,
      time_limit_seconds = control$time_limit_seconds
    ),
    optimal_binning_categorical_oslp = list(monotonic = control$monotonic),
    optimal_binning_categorical_sab = list(
      initial_temperature = control$initial_temperature,
      cooling_rate = control$cooling_rate,
      max_iterations = control$max_iterations
    ),
    optimal_binning_numerical_bb = list(is_monotonic = control$is_monotonic),
    optimal_binning_numerical_cart = list(is_monotonic = control$is_monotonic),
    optimal_binning_numerical_dpb = list(n_threads = control$n_threads),
    optimal_binning_numerical_eb = list(n_threads = control$n_threads),
    optimal_binning_numerical_mba = list(n_threads = control$n_threads),
    optimal_binning_numerical_milp = list(n_threads = control$n_threads),
    optimal_binning_numerical_mrblp = list(n_threads = control$n_threads)
  )

  # Merge default parameters with specific parameters for the selected algorithm
  algorithm_params <- c(
    default_params,
    specific_params[[selected_algorithm]] %||% list()
  )

  # Merge with user-provided control parameters for the specific algorithm
  if (!is.null(control[[selected_algorithm]])) {
    algorithm_params <- utils::modifyList(algorithm_params, control[[selected_algorithm]])
  }

  # Return the selected algorithm and its parameters
  list(
    algorithm = selected_algorithm,
    params = algorithm_params,
    method = method
  )
}


#' Get Available Optimal Binning Algorithms
#'
#' @description
#' This function retrieves all available optimal binning algorithms from the OptimalBinningWoE package,
#' separating them into categorical and numerical types.
#'
#' @return A list containing two elements:
#'   \item{char}{A named list of categorical binning algorithms}
#'   \item{num}{A named list of numerical binning algorithms}
#'
#' @details
#' The function searches for all exported functions in the OptimalBinningWoE package that start with
#' "optimal_binning_categorical_" or "optimal_binning_numerical_". It then creates two separate lists
#' for categorical and numerical algorithms, using the last part of the function name (after the last
#' underscore) as the list item name.
#'
#' @examples
#' \dontrun{
#' algorithms <- OptimalBinningGetAlgoName()
#' print(algorithms$char) # List of categorical algorithms
#' print(algorithms$num) # List of numerical algorithms
#' }
#'
#' @export
OptimalBinningGetAlgoName <- function() {
  # Get all exported functions from OptimalBinningWoE package
  pk <- getNamespaceExports("OptimalBinningWoE")

  # Helper function to extract the last part of the function name
  get_last_part <- function(x) {
    sapply(strsplit(as.character(x), "_"), function(parts) parts[length(parts)])
  }

  # Filter and process categorical algorithms
  obj <- pk[grepl("optimal_binning_categorical_", pk)]

  categorical <- lapply(obj, function(f) {
    o <- formals(f)
    o <- o[setdiff(names(o), c("target", "feature"))]
    list(algorithm = f, params = o, method = get_last_part(f))
  })
  names(categorical) <- get_last_part(obj)


  # Filter and process numerical algorithms
  obj <- pk[grepl("optimal_binning_numerical_", pk)]
  numerical <- lapply(obj, function(f) {
    o <- formals(f)
    o <- o[setdiff(names(o), c("target", "feature"))]
    list(algorithm = f, params = o, method = get_last_part(f))
  })
  names(numerical) <- get_last_part(obj)

  # Return a list with both types of algorithms
  return(list(char = categorical, num = numerical))
}


#' Select the Best Model for Optimal Binning
#'
#' This function selects the best model for optimal binning across multiple features
#' using various binning algorithms for both numerical and categorical variables.
#'
#' @param dt A data.table containing the target variable and features to be binned.
#' @param target The name of the target variable in the data.table.
#' @param features A character vector of feature names to be binned.
#' @param method A method for use. If not available, test all.
#' @param min_bins The minimum number of bins to use in the binning process.
#' @param max_bins The maximum number of bins to use in the binning process.
#' @param control A list of control parameters for the binning algorithms (not used directly in this function).
#' @param progress Logical; if TRUE, display a progress bar during processing (default is TRUE).
#' @param trace Logical; if TRUE, provide more detailed output for debugging (default is FALSE).
#'
#' @return A list containing the results for each feature:
#'   \item{woebin}{The Weight of Evidence (WoE) binning result for the best model.}
#'   \item{woefeature}{The WoE-transformed feature for the best model.}
#'   \item{bestmethod}{The name of the algorithm that produced the best model.}
#'   \item{report}{A data.table summarizing the performance of all tried models.}
#'
#' @details
#' The function iterates through each feature, applying various binning algorithms
#' suitable for either numerical or categorical data. It then selects the best model
#' based on monotonicity, number of zero-count bins, total number of bins, and
#' Information Value (IV).
#'
#' For features with 2 or fewer distinct values, the function forces them to be
#' treated as factors and applies categorical binning methods.
#'
#' If a binning algorithm fails, the function attempts to relax the binning parameters
#' and try again. If it still fails, that method is skipped for that feature.
#'
#' @importFrom data.table data.table rbindlist setorder
#' @importFrom utils modifyList
#' @importFrom progress progress_bar
#'
#' @examples
#' \dontrun{
#' library(data.table)
#' dt <- data.table(
#'   target = sample(0:1, 1000, replace = TRUE),
#'   feature1 = rnorm(1000),
#'   feature2 = sample(letters[1:5], 1000, replace = TRUE)
#' )
#' results <- OptimalBinningSelectBestModel(
#'   dt = dt,
#'   target = "target",
#'   features = c("feature1", "feature2"),
#'   min_bins = 3,
#'   max_bins = 10
#' )
#' }
#'
#' @export
OptimalBinningSelectBestModel <- function(dt, target, features, method = NULL, min_bins, max_bins, control, progress = TRUE, trace = FALSE) {
  results <- list()

  algoes <- OptimalBinningGetAlgoName()

  numerical_methods <- lapply(algoes$num, function(a) {
    a$params <- utils::modifyList(a$params, list(min_bins = min_bins, max_bins = max_bins))
    return(a)
  })

  categorical_methods <- lapply(algoes$char, function(a) {
    a$params <- utils::modifyList(a$params, list(min_bins = min_bins, max_bins = max_bins, bin_separator = control$bin_separator))
    return(a)
  })

  if (progress) {
    pb <- progress::progress_bar$new(
      format = "Processing :what [:bar] :percent | ETA: :eta | Elapsed: :elapsed",
      total = length(features) * 3, # Multiplicamos por 3 para representar as principais etapas
      clear = FALSE,
      width = 80
    )
  }

  for (feat in features) {
    if (progress) {
      pb$tick(tokens = list(what = sprintf("%-5s", paste0("Feature: ", feat))))
    }

    featdim <- OptimalBinningCheckDistinctsLength(dt[[feat]], dt[[target]])

    dt_feature <- if (featdim[1] <= 2) {
      data.table::copy(
        data.table::data.table(
          target = dt[[target]],
          feature = as.factor(dt[[feat]])
        )
      )
    } else {
      data.table::copy(
        data.table::data.table(
          target = dt[[target]],
          feature = dt[[feat]]
        )
      )
    }

    is_numeric <- is.numeric(dt_feature$feature)
    methods_to_try <- if (is_numeric) numerical_methods else categorical_methods

    if (!is.null(method) && method != "auto") {
      if (method %in% names(methods_to_try)) {
        methods_to_try <- methods_to_try[method]
      } else {
        warning(sprintf(
          "The provided method '%s' is not available for this variable type (%s). Using automatic method selection.",
          method,
          if (is_numeric) "numeric" else "categorical"
        ))
      }
    }

    if (progress) {
      pb$tick(tokens = list(what = sprintf("%-5s", "Trying methods")))
    }

    OO <- lapply(methods_to_try, function(m) {
      tryCatch(
        {
          binning_result <- suppressWarnings(suppressMessages(
            do.call(m$algorithm, c(list(target = dt_feature$target, feature = dt_feature$feature), m$params))
          ))

          if (is.null(binning_result) || featdim[1] <= 2) {
            # Relaxed parameters for binary or unary classes
            relaxed_params <- utils::modifyList(m$params, list(
              min_bins = 2,
              max_bins = 3,
              bin_cutoff = 0.001,
              max_n_prebins = 5
            ))

            binning_result <- suppressWarnings(suppressMessages(
              do.call(m$algorithm, c(list(target = dt_feature$target, feature = dt_feature$feature), relaxed_params))
            ))
          }

          if (!is.null(binning_result)) {
            binning_result$algorithm <- m$algorithm
            binning_result$method <- m$method
            return(binning_result)
          } else {
            return(NULL)
          }
        },
        error = function(e) {
          if (trace) {
            message(sprintf("Error in method %s for feature %s: %s", m$method, feat, e$message))
          }
          return(NULL)
        }
      )
    })

    # Remove NULL results
    OO <- Filter(Negate(is.null), OO)

    # OO <- lapply(methods_to_try, function(m) {
    #
    #   binning_result <- try({
    #     suppressWarnings(suppressMessages(
    #       do.call(m$algorithm, c(list(target = dt_feature$target, feature = dt_feature$feature), m$params))
    #     ))
    #   })
    #
    #   if (inherits(binning_result, "try-error") | featdim[1] <= 2) {
    #     m$params$min_bins <- 2
    #     m$params$max_bins <- 3
    #     m$params$bin_cutoff <- 0.001
    #     m$params$max_n_prebins <- 5
    #
    #     binning_result <- try({
    #       do.call(
    #         m$algorithm,
    #         c(list(target = dt_feature$target, feature = dt_feature$feature), m$params)
    #       )
    #     })
    #   }
    #
    #   if (!inherits(binning_result, "try-error")) {
    #     binning_result$algorithm <- m$algorithm
    #     binning_result$method <- m$method
    #     return(binning_result)
    #   } else {
    #     NULL
    #   }
    # })

    mm <- data.table::rbindlist(
      lapply(names(OO), function(m) {
        tryCatch(
          {
            data.table::data.table(
              model_method = m,
              model_algorithm = OO[[m]]$algorithm,
              total_iv = sum(OO[[m]]$iv, na.rm = TRUE),
              total_bins = length(OO[[m]]$bin),
              total_zero_pos = sum(OO[[m]]$count_pos == 0, na.rm = TRUE),
              total_zero_neg = sum(OO[[m]]$count_neg == 0, na.rm = TRUE),
              is_monotonic = as.numeric(OptimalBinningIsWoEMonotonic(OO[[m]]$woe))
            )
          },
          error = function(e) {
            data.table::data.table(
              model_method = m,
              model_algorithm = NA_character_,
              total_iv = NA_real_,
              total_bins = NA_integer_,
              total_zero_pos = NA_integer_,
              total_zero_neg = NA_integer_,
              is_monotonic = NA_real_
            )
          }
        )
      }),
      fill = TRUE,
      use.names = TRUE
    )

    fn_rank <- function(m) {
      m$rk0 <- rank(-m$is_monotonic)
      m$rk1 <- rank(-m$total_iv)
      m$rk2 <- rank(m$total_zero_pos)
      m$rk3 <- rank(m$total_zero_neg)
      m$id <- apply(m[, c("rk0", "rk0", "rk1", "rk2", "rk3")], 1, mean)
      m <- m[order(m$id)]
      m$rk0 <- m$rk1 <- m$rk2 <- m$rk3 <- NULL
      return(m)
    }

    mm <- fn_rank(mm)
    best_model <- OO[[unique(head(mm, 1)$model_method)]]

    woefeature <- if (is_numeric) {
      OptimalBinningApplyWoENum(best_model, dt_feature$feature, include_upper_bound = control$include_upper_bound)
    } else {
      OptimalBinningApplyWoECat(best_model, dt_feature$feature, bin_separator = control$bin_separator)
    }

    woebin <- OptimalBinningGainsTableFeature(binned_feature = woefeature$bin, dt_feature$target)

    bestmethod <- best_model$algorithm

    report <- mm

    if (progress) {
      pb$tick(tokens = list(what = sprintf("%-5s", "Finalizing")))
    }

    results[[feat]] <- list(
      woebin = data.table::setDT(woebin),
      woefeature = data.table::setDT(woefeature),
      report = data.table::setDT(report),
      bestmethod = bestmethod
    )
  }

  return(results)
}



# OptimalBinningSelectBestModel <- function(dt, target, features, method = NULL, min_bins, max_bins, control, progress = TRUE, trace = FALSE) {
#   results <- list()
#
#   # numerical_methods <- sort(unique(names(OptimalBinningGetAlgoName()$num)))
#   # categorical_methods <- sort(unique(names(OptimalBinningGetAlgoName()$char)))
#
#   # Update agrs list for all methods
#   algoes <- OptimalBinningGetAlgoName()
#
#   numerical_methods <- lapply(algoes$num, function(a) {
#     a$params <- utils::modifyList(a$params, list(min_bins = min_bins, max_bins = max_bins))
#     return(a)
#   })
#
#   categorical_methods <- lapply(algoes$char, function(a) {
#     a$params <- utils::modifyList(a$params, list(min_bins = min_bins, max_bins = max_bins, bin_separator = control$bin_separator))
#     return(a)
#   })
#
#   if (progress) {
#     pb <- progress::progress_bar$new(
#       format = "Processing features [:bar] :percent eta: :eta",
#       total = length(features),
#       clear = FALSE,
#       width = 80
#     )
#   }
#
#   for (feat in features) {
#     if (progress) pb$tick()
#
#     featdim <- OptimalBinningCheckDistinctsLength(dt[[feat]], dt[[target]])
#
#     # If categ has only one class force factor to pass on char methods
#     dt_feature <- if (featdim[1] <= 2) {
#       data.table::copy(
#         data.table::data.table(
#           target = dt[[target]],
#           feature = as.factor(dt[[feat]])
#         )
#       )
#     } else {
#       data.table::copy(
#         data.table::data.table(
#           target = dt[[target]],
#           feature = dt[[feat]]
#         )
#       )
#     }
#
#     is_numeric <- is.numeric(dt_feature$feature)
#     methods_to_try <- if (is_numeric) numerical_methods else categorical_methods
#
#     if (!is.null(method) && method != "auto") {
#       if (method %in% names(methods_to_try)) {
#         # The method provided by the user is in the list of available methods
#         methods_to_try <- methods_to_try[method]
#         # message(sprintf("Using the user-provided method '%s'.", method))
#       } else {
#         # The method provided by the user is not in the list of available methods
#         warning(sprintf(
#           "The provided method '%s' is not available for this variable type (%s). Using automatic method selection.",
#           method,
#           if (is_numeric) "numeric" else "categorical"
#         ))
#         # Keep all methods for automatic selection
#       }
#     }
#
#     OO <- lapply(methods_to_try, function(m) {
#       # cat("log: rodando ", m$algorithm, "para ", feat, "\n")
#       binning_result <- try({
#         suppressWarnings(suppressMessages(
#           do.call(m$algorithm, c(list(target = dt_feature$target, feature = dt_feature$feature), m$params))
#         ))
#       })
#
#       if (inherits(binning_result, "try-error") | featdim[1] <= 2) {
#         # relax args for binary or unary ckasses and try again
#         m$params$min_bins <- 2
#         m$params$max_bins <- 3
#         m$params$bin_cutoff <- 0.001
#         m$params$max_n_prebins <- 5
#
#         binning_result <- try({
#           do.call(
#             m$algorithm,
#             c(list(target = dt_feature$target, feature = dt_feature$feature), m$params)
#           )
#         })
#       }
#
#       if (!inherits(binning_result, "try-error")) {
#         binning_result$algorithm <- m$algorithm
#         binning_result$method <- m$method
#         return(binning_result)
#       } else {
#         NULL
#       }
#     })
#
#     mm <- data.table::rbindlist(
#       lapply(names(OO), function(m) {
#         tryCatch(
#           {
#             data.table::data.table(
#               model_method = m,
#               model_algorithm = OO[[m]]$algorithm,
#               total_iv = sum(OO[[m]]$iv, na.rm = TRUE),
#               total_bins = length(OO[[m]]$bin),
#               total_zero_pos = sum(OO[[m]]$count_pos == 0, na.rm = TRUE),
#               total_zero_neg = sum(OO[[m]]$count_neg == 0, na.rm = TRUE),
#               is_monotonic = as.numeric(OptimalBinningIsWoEMonotonic(OO[[m]]$woe))
#             )
#           },
#           error = function(e) {
#             # Se houver um erro, retorna uma linha com NAs, mantendo a estrutura
#             data.table::data.table(
#               model_method = m,
#               model_algorithm = NA_character_,
#               total_iv = NA_real_,
#               total_bins = NA_integer_,
#               total_zero_pos = NA_integer_,
#               total_zero_neg = NA_integer_,
#               is_monotonic = NA_real_
#             )
#           }
#         )
#       }),
#       fill = TRUE,
#       use.names = TRUE
#     )
#
#     fn_rank <- function(m) {
#       m$rk0 <- rank(-m$is_monotonic)
#       m$rk1 <- rank(-m$total_iv)
#       m$rk2 <- rank(m$total_zero_pos)
#       m$rk3 <- rank(m$total_zero_neg)
#       m$id <- apply(m[, c("rk0", "rk0", "rk1", "rk2", "rk3")], 1, mean)
#       m <- m[order(m$id)]
#       m$rk0 <- m$rk1 <- m$rk2 <- m$rk3 <- NULL
#       return(m)
#     }
#     # Sort models to select the best
#
#     mm <- fn_rank(mm)
#     best_model <- OO[[unique(head(mm, 1)$model_method)]]
#
#     # Apply WOE
#     woefeature <- if (is_numeric) {
#       OptimalBinningApplyWoENum(best_model, dt_feature$feature, include_upper_bound = control$include_upper_bound)
#     } else {
#       OptimalBinningApplyWoECat(best_model, dt_feature$feature, bin_separator = control$bin_separator)
#     }
#
#     # Prepare gains table after applying optimal woe
#     woebin <- OptimalBinningGainsTableFeature(binned_feature = woefeature$bin, dt_feature$target)
#
#     # Get mest method algo
#     bestmethod <- best_model$algorithm
#
#     # Get report from model selection
#     report <- mm
#
#     results[[feat]] <- list(
#       woebin = data.table::setDT(woebin),
#       woefeature = data.table::setDT(woefeature),
#       report = data.table::setDT(report),
#       bestmethod = bestmethod
#     )
#   }
#
#   return(results)
# }



#'
#'
#' #' Optimal Binning and Weight of Evidence Calculation for Numerical Variables
#' #'
#' #' This function performs optimal binning and calculates Weight of Evidence (WoE)
#' #' for numerical variables. It supports various binning methods, handles special cases,
#' #' and provides detailed results including binning information and WoE calculations.
#' #'
#' #' @param dt A data.table containing the dataset.
#' #' @param target Character. The name of the target variable column in `dt`.
#' #' @param features Character vector. Names of numeric feature columns to process.
#' #' @param method Character. The binning method to use. Use "auto" for automatic selection.
#' #' @param min_bins Integer. Minimum number of bins to create.
#' #' @param max_bins Integer. Maximum number of bins to create.
#' #' @param control List. Additional control parameters for binning algorithms.
#' #' @param positive Character. Specifies which category of the target should be considered positive.
#' #' @param preprocessed_data List. Preprocessed data for each feature.
#' #' @param progress Logical. Whether to display a progress bar. Default is TRUE.
#' #' @param trace Logical, whether to generate error logs when testing existing methods.
#' #'
#' #' @return A list containing:
#' #'   \item{results}{List of WoE values for each processed feature.}
#' #'   \item{reports}{List of preprocessing reports for each feature.}
#' #'   \item{woebins}{List of binning and WoE information for each feature.}
#' #'   \item{bestmod}{List of best model reports for each feature when method is "auto".}
#' #'   \item{failed_features}{Character vector of features that failed processing.}
#' #'
#' #' @details
#' #' The function processes each feature in `features`, applying the specified binning method
#' #' and calculating WoE. It handles special cases (e.g., missing values) separately.
#' #' If `method` is "auto", it selects the best binning method for each feature.
#' #'
#' #' @examples
#' #' \dontrun{
#' #' result <- OptimalBinningNumericalWoE(
#' #'   dt = my_data,
#' #'   target = "target_column",
#' #'   features = c("feature1", "feature2"),
#' #'   method = "auto",
#' #'   min_bins = 3,
#' #'   max_bins = 10,
#' #'   control = list(num_miss_value = -999),
#' #'   positive = "1",
#' #'   preprocessed_data = my_preprocessed_data,
#' #'   progress = TRUE
#' #' )
#' #' }
#' #'
#' #' @importFrom data.table data.table :=
#' #' @importFrom utils modifyList
#' #' @importFrom progress progress_bar
#' #'
#' #' @export
#' OptimalBinningNumericalWoE <- function(dt, target, features, method, min_bins, max_bins, control, positive, preprocessed_data, progress = TRUE, trace = TRUE) {
#'
#'   results <- reports <- woebins <- bestmod <- list()
#'   failed_features <- character()
#'   numerical_methods <- sort(unique(names(OptimalBinningGetAlgoName()$num)))
#'
#'   # Initialize progress bar if progress is TRUE
#'   if (progress) {
#'     pb <- progress::progress_bar$new(
#'       format = "Processing WoE for num. [:bar] :percent eta: :eta",
#'       total = length(features),
#'       clear = FALSE,
#'       width = 80
#'     )
#'   }
#'
#'   for (feat in features) {
#'     tryCatch(
#'       {
#'         if (progress) pb$tick()
#'
#'         dt_proc <- data.table::data.table(
#'           target = dt[[target]],
#'           feature = preprocessed_data[[feat]]$preprocess$feature_preprocessed,
#'           original_index = seq_len(nrow(dt))
#'         )
#'         # Identify special cases
#'         is_special <- dt_proc$feature == control$num_miss_value
#'         # Perform binning on non-special cases
#'         dt_binning <- dt_proc[!is_special]
#'         # Select the best method if method is "auto"
#'         if (method == "auto") {
#'           binning_result <- OptimalBinningSelectBestModel(data.table::copy(dt_binning), "target", "feature", min_bins, max_bins, control, numerical_methods, trace)
#'         } else {
#'           algo_info <- OptimalBinningSelectAlgorithm("feature", method, dt_binning, min_bins, max_bins, control)
#'           algo_params <- utils::modifyList(list(min_bins = min_bins, max_bins = max_bins), algo_info$params)
#'           binning_result <- do.call(algo_info$algorithm, c(list(target = dt_binning$target, feature = dt_binning$feature), algo_params))
#'           binning_result$best_model_report <- NULL
#'           binning_result$best_method <- method
#'         }
#'         # Add WoE values to non-special cases
#'         dt_proc[!is_special, woe := binning_result$woefeature]
#'         # Handle special cases
#'         if (any(is_special)) {
#'           special_woe <- OptimalBinningCalculateSpecialWoE(dt_proc[is_special, target])
#'           dt_proc[is_special, woe := special_woe]
#'           # Add special bin to woebin
#'           special_bin <- OptimalBinningCreateSpecialBin(dt_proc[is_special], binning_result$woebin, special_woe)
#'           special_bin[, bin := paste0(special_bin$bin, "(", control$num_miss_value, ")")]
#'           binning_result$woebin <- rbind(binning_result$woebin, special_bin, fill = TRUE)
#'         }
#'         # Sort by original index
#'         dt_proc <- dt_proc[order(original_index)]
#'         # Update results
#'         results[[feat]] <- dt_proc$woe
#'         reports[[feat]] <- preprocessed_data[[feat]]$report
#'         woebins[[feat]] <- OptimalBinningGainsTable(binning_result)
#'         bestmod[[feat]] <- binning_result$best_model_report
#'       },
#'       error = function(e) {
#'         warning(paste("Error processing feature:", feat, "-", e$message))
#'         failed_features <- c(failed_features, feat)
#'       }
#'     )
#'   }
#'   list(results = results, reports = reports, woebins = woebins, bestmod = bestmod, failed_features = failed_features)
#' }
#'
#'
#' #' Optimal Binning and Weight of Evidence Calculation for Categorical Variables
#' #'
#' #' This function performs optimal binning and calculates Weight of Evidence (WoE)
#' #' for categorical variables. It supports various binning methods, handles special cases,
#' #' and provides detailed results including binning information and WoE calculations.
#' #'
#' #' @param dt A data.table containing the dataset.
#' #' @param target Character. The name of the target variable column in `dt`.
#' #' @param features Character vector. Names of categorical feature columns to process.
#' #' @param method Character. The binning method to use. Use "auto" for automatic selection.
#' #' @param min_bins Integer. Minimum number of bins to create.
#' #' @param max_bins Integer. Maximum number of bins to create.
#' #' @param control List. Additional control parameters for binning algorithms.
#' #' @param positive Character. Specifies which category of the target should be considered positive.
#' #' @param preprocessed_data List. Preprocessed data for each feature.
#' #' @param progress Logical. Whether to display a progress bar. Default is TRUE.
#' #' @param trace Logical, whether to generate error logs when testing existing methods
#' #'
#' #' @return A list containing:
#' #'   \item{results}{List of WoE values for each processed feature.}
#' #'   \item{reports}{List of preprocessing reports for each feature.}
#' #'   \item{woebins}{List of binning and WoE information for each feature.}
#' #'   \item{bestmod}{List of best model reports for each feature when method is "auto".}
#' #'   \item{failed_features}{Character vector of features that failed processing.}
#' #'
#' #' @details
#' #' The function processes each feature in `features`, applying the specified binning method
#' #' and calculating WoE. It handles special cases (e.g., missing values) separately.
#' #' If `method` is "auto", it selects the best binning method for each feature from a set of
#' #' categorical binning algorithms.
#' #'
#' #' @examples
#' #' \dontrun{
#' #' result <- OptimalBinningCategoricalWoE(
#' #'   dt = my_data,
#' #'   target = "target_column",
#' #'   features = c("category1", "category2"),
#' #'   method = "auto",
#' #'   min_bins = 2,
#' #'   max_bins = 5,
#' #'   control = list(char_miss_value = "NA"),
#' #'   positive = "1",
#' #'   preprocessed_data = my_preprocessed_data,
#' #'   progress = TRUE
#' #' )
#' #' }
#' #'
#' #' @importFrom data.table data.table :=
#' #' @importFrom utils modifyList
#' #' @importFrom progress progress_bar
#' #'
#' #' @export
#' OptimalBinningCategoricalWoE <- function(dt, target, features, method, min_bins, max_bins, control, positive, preprocessed_data, progress = TRUE, trace = TRUE) {
#'   results <- reports <- woebins <- bestmod <- list()
#'   failed_features <- character()
#'   categorical_methods <- sort(unique(names(OptimalBinningGetAlgoName()$char)))
#'
#'   # Initialize progress bar if progress is TRUE
#'   if (progress) {
#'     pb <- progress::progress_bar$new(
#'       format = "Processing WoE for cat. [:bar] :percent eta: :eta",
#'       total = length(features),
#'       clear = FALSE,
#'       width = 80
#'     )
#'   }
#'
#'   for (feat in features) {
#'     tryCatch(
#'       {
#'         if (progress) pb$tick()
#'
#'         dt_proc <- data.table::data.table(
#'           target = dt[[target]],
#'           feature = preprocessed_data[[feat]]$preprocess$feature_preprocessed,
#'           original_index = seq_len(nrow(dt))
#'         )
#'         # Identify special cases
#'         is_special <- dt_proc$feature == control$char_miss_value
#'         # Perform binning on non-special cases
#'         dt_binning <- dt_proc[!is_special]
#'         # Select the best method if method is "auto"
#'         if (method == "auto") {
#'           binning_result <- OptimalBinningSelectBestModel(dt_binning, "target", "feature", min_bins, max_bins, control, categorical_methods, trace)
#'         } else {
#'           algo_info <- OptimalBinningSelectAlgorithm("feature", method, dt_binning, min_bins, max_bins, control)
#'           algo_params <- utils::modifyList(list(min_bins = min_bins, max_bins = max_bins), algo_info$params)
#'           binning_result <- do.call(algo_info$algorithm, c(list(target = dt_binning$target, feature = dt_binning$feature), algo_params))
#'           binning_result$best_model_report <- NULL
#'           binning_result$best_method <- method
#'         }
#'         # Add WoE values to non-special cases
#'         dt_proc[!is_special, woe := binning_result$woefeature]
#'         # Handle special cases
#'         if (any(is_special)) {
#'           special_woe <- OptimalBinningCalculateSpecialWoE(dt_proc[is_special, target])
#'           dt_proc[is_special, woe := special_woe]
#'           # Add special bin to woebin
#'           special_bin <- OptimalBinningCreateSpecialBin(dt_proc[is_special], binning_result$woebin, special_woe)
#'           special_bin[, bin := paste0(special_bin$bin, "(", control$char_miss_value, ")")]
#'           binning_result$woebin <- rbind(binning_result$woebin, special_bin, fill = TRUE)
#'         }
#'         # Sort by original index
#'         dt_proc <- dt_proc[order(original_index)]
#'         # Update results
#'         results[[feat]] <- dt_proc$woe
#'         reports[[feat]] <- preprocessed_data[[feat]]$report
#'         woebins[[feat]] <- OptimalBinningGainsTable(binning_result)
#'         bestmod[[feat]] <- binning_result$best_model_report
#'       },
#'       error = function(e) {
#'         warning(paste("Error processing feature:", feat, "-", e$message))
#'         failed_features <- c(failed_features, feat)
#'       }
#'     )
#'   }
#'   list(results = results, reports = reports, woebins = woebins, bestmod = bestmod, failed_features = failed_features)
#' }
#'
#' #' Select Best Binning Model
#' #'
#' #' @param dt_binning Data for binning
#' #' @param target Target variable name
#' #' @param feature Feature variable name
#' #' @param min_bins Minimum number of bins
#' #' @param max_bins Maximum number of bins
#' #' @param control Control parameters
#' #' @param allowed_methods Vector of allowed binning methods
#' #' @param trace Logical, whether to generate error logs when testing existing methods
#' #'
#' #' @return Best binning result
#' #'
#' #' @keywords internal
#' OptimalBinningSelectBestModel <- function(dt_binning, target, feature, min_bins, max_bins, control, allowed_methods, trace = FALSE) {
#'
#'   best_method <- NULL
#'   best_iv <- -Inf
#'   best_result <- NULL
#'   best_monotonic <- FALSE
#'   best_bins <- Inf
#'
#'   best_model_report <- data.table::data.table(
#'     model_name = character(),
#'     model_method = character(),
#'     total_iv = numeric(),
#'     bin_counts = integer()
#'   )
#'
#'   error_log <- character()
#'   failed_methods <- character()
#'
#'   for (method in allowed_methods) {
#'     tryCatch(
#'       {
#'         algo_info <- OptimalBinningSelectAlgorithm(feature, method, dt_binning, min_bins, max_bins, control)
#'
#'         binning_result <- do.call(algo_info$algorithm,
#'                                   c(list(target = dt_binning[[target]],
#'                                          feature = dt_binning[[feature]]),
#'                                     algo_info$params)
#'                                   )
#'
#'         if (length(binning_result$woebin$bin) >= 2) {
#'           model_name <- algo_info$algorithm
#'           current_iv <- sum(binning_result$woebin$iv, na.rm = TRUE)
#'           is_monotonic <- OptimalBinningIsWoEMonotonic(binning_result$woebin$woe)
#'           current_bins <- length(binning_result$woebin$bin)
#'           best_model_report <- data.table::rbindlist(
#'             list(
#'               best_model_report,
#'               data.table::data.table(
#'                 model_name = model_name,
#'                 model_method = method,
#'                 total_iv = current_iv,
#'                 bin_counts = current_bins
#'               )
#'             )
#'           )
#'           if (current_iv > best_iv ||
#'             (current_iv == best_iv && is_monotonic && !best_monotonic) ||
#'             (current_iv == best_iv && is_monotonic == best_monotonic && current_bins < best_bins)) {
#'             best_iv <- current_iv
#'             best_method <- method
#'             best_result <- binning_result
#'             best_monotonic <- is_monotonic
#'             best_bins <- current_bins
#'           }
#'         }
#'       },
#'       error = function(e) {
#'         error_message <- paste("Method (", method, ") not suitable for feature. Trying another one - ", e$message)
#'         if (trace) {
#'           warning(error_message)
#'         }
#'         error_log <- c(error_log, error_message)
#'         failed_methods <- c(failed_methods, method)
#'       }
#'     )
#'   }
#'
#'   if (is.null(best_method)) {
#'     if (length(allowed_methods) > 0) {
#'       warning("All methods failed. Returning the first tested method.")
#'       best_method <- allowed_methods[1]
#'       best_result <- tryCatch(
#'         {
#'           algo_info <- OptimalBinningSelectAlgorithm(feature, best_method, dt_binning, min_bins, max_bins, control)
#'           do.call(
#'             algo_info$algorithm,
#'             c(
#'               list(target = dt_binning[[target]], feature = dt_binning[[feature]]),
#'               algo_info$params
#'             )
#'           )
#'         },
#'         error = function(e) {
#'           list(error = paste("Error in first method:", e$message), feature = feature)
#'         }
#'       )
#'     } else {
#'       error_message <- "No methods provided or all methods failed."
#'       warning(error_message)
#'       return(list(error = error_message, feature = feature, error_log = error_log, failed_methods = failed_methods))
#'     }
#'   }
#'
#'   data.table::setorder(best_model_report, -total_iv)
#'   best_result$best_method <- best_method
#'   best_result$best_model_report <- best_model_report
#'   best_result$error_log <- error_log
#'   best_result$failed_methods <- failed_methods
#'   return(best_result)
#' }
