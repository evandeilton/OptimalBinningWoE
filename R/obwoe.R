#' Optimal Binning and Weight of Evidence Calculation
#'
#' @description
#' This function performs optimal binning and calculates Weight of Evidence (WoE)
#' for both numerical and categorical features. It implements a wide variety of
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
#'   \item LDB (Local Density Binning): Applies local density estimation to categorical variables
#'   \item CM (ChiMerge): Merges categories based on chi-square statistic
#'   \item IVB (Information Value Binning): Bins based on information value
#'   \item UDT (Unsupervised Decision Trees): Uses decision tree algorithms for categorical binning
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
#'   \item EB (Entropy-Based): Uses entropy-based criteria for binning
#'   \item CART (Classification and Regression Trees): Uses decision tree algorithm for binning
#'   \item UDT (Unsupervised Decision Trees): Applies decision tree algorithms in an unsupervised manner for binning
#'   \item DPLC (Dynamic Programming with Local Constraints): Uses dynamic programming with local constraints
#'   \item GAB (Genetic Algorithm Binning): Uses genetic algorithms for optimal binning
#'   \item LPDB (Local Polynomial Density Binning): Employs local polynomial density estimation
#'   \item UBSD (Unsupervised Binning with Standard Deviation): Uses standard deviation in unsupervised binning
#'   \item SBLP (Supervised Binning via Linear Programming): Uses linear programming for supervised binning
#'   \item FETB (Fisher's Exact Test Binning): Applies Fisher's exact test to numerical variables
#'   \item EWB (Equal Width Binning): Creates bins of equal width across the range of the variable
#'   \item KMB (K-means Binning): Applies k-means clustering for binning
#'   \item OSLP (Optimal Supervised Learning Path): Uses a supervised learning path for optimal binning
#'   \item IR (Isotonic Regression): Uses isotonic regression for binning
#'   \item SAB (Simulated Annealing Binning): Applies simulated annealing to numerical variables
#'   \item BB (Branch and Bound): Uses a branch and bound algorithm for optimal binning
#'   \item QB (Quantile-based Binning): Creates bins based on quantiles of the feature distribution
#'   \item DPB (Dynamic Programming Binning): Applies dynamic programming for optimal binning
#'   \item SBB (Supervised Boundary Binning): Uses supervised learning to determine bin boundaries
#'   \item LDB (Local Density Binning): Uses local density estimation for binning
#'   \item JNBO (Joint Neighborhood-based Optimization): Optimizes bins based on joint neighborhoods
#'   \item MILP (Mixed Integer Linear Programming): Applies mixed integer linear programming for binning
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
#' @param min_bins Minimum number of bins (default: 3).
#' @param max_bins Maximum number of bins (default: 4).
#' @param control A list of additional control parameters:
#'   \itemize{
#'     \item cat_cutoff: Minimum frequency for a category (default: 0.05)
#'     \item bin_cutoff: Minimum frequency for a bin (default: 0.05)
#'     \item min_bads: Minimum proportion of bad cases in a bin (default: 0.05)
#'     \item pvalue_threshold: P-value threshold for statistical tests (default: 0.05)
#'     \item max_n_prebins: Maximum number of pre-bins (default: 20)
#'     \item monotonicity_direction: Direction of monotonicity ("increase" or "decrease")
#'     \item lambda: Regularization parameter for some algorithms (default: 0.1)
#'     \item min_bin_size: Minimum bin size as a proportion of total observations (default: 0.05)
#'     \item min_iv_gain: Minimum IV gain for bin splitting (default: 0.01)
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
#'   }
#' @param positive Character string specifying which category should be considered as positive. Must be either "bad|1" or "good|1".
#'
#' @return A list containing:
#'   \item{woedt}{The original dataset with added WoE columns}
#'   \item{woebins}{Information about the bins created, including:
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
#'   \item{prepreport}{Preprocessing report for each feature, including:
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
#'   \item{bestsreport}{Report on the best models used, including:
#'     \itemize{
#'       \item feature: Name of the feature
#'       \item method: Best method selected for the feature
#'       \item iv_total: Total Information Value achieved
#'       \item n_bins: Number of bins created
#'       \item runtime: Execution time for binning the feature
#'     }
#'   }
#'   \item{failedfeatures}{List of features that failed processing}
#'   \item{bestmethod}{Best method used for binning across all features}
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
#' result <- obwoe(dt,
#'   target = "creditability", method = "mblp",
#'   min_bins = 3, max_bins = 5, positive = "bad|1"
#' )
#'
#' # View WoE-transformed data
#' head(result$woedt)
#' # View binning information
#' print(result$woebins)
#'
#' # Process only numeric features
#' numeric_features <- names(dt)[sapply(dt, is.numeric)]
#' numeric_features <- setdiff(numeric_features, "creditability")
#'
#' result <- obwoe(dt,
#'   target = "creditability", features = numeric_features,
#'   method = "mblp", preprocess = TRUE,
#'   min_bins = 3, max_bins = 5, positive = "bad|1"
#' )
#'
#' # View preprocessing report
#' print(result$prepreport)
#'
#' # View best model report
#' print(result$bestsreport)
#'
#' # Process only categoric features
#' categoric_features <- names(dt)[sapply(dt, function(i) !is.numeric(i))]
#' categoric_features <- setdiff(categoric_features, "creditability")
#' result <- obwoe(dt,
#'   target = "creditability", features = categoric_features,
#'   method = "udt", preprocess = TRUE,
#'   min_bins = 3, max_bins = 4, positive = "bad|1"
#' )
#'
#' # View binning information for categorical features
#' print(result$woebins)
#' }
#'
#' @export
obwoe <- function(dt, target, features = NULL, method = "auto", preprocess = TRUE,
                  min_bins = 3, max_bins = 4, control = list(), positive = "bad|1") {
  # Ensure dt is a data.table
  data.table::setDT(dt)

  # Default parameters
  default_control <- list(
    cat_cutoff = 0.05,
    bin_cutoff = 0.05,
    min_bads = 0.05,
    pvalue_threshold = 0.05,
    max_n_prebins = 20,
    monotonicity_direction = "increase",
    lambda = 0.1,
    min_bin_size = 0.05,
    min_iv_gain = 0.01,
    max_depth = 10,
    num_miss_value = -999.0,
    char_miss_value = "N/A",
    outlier_method = "iqr",
    outlier_process = FALSE,
    iqr_k = 1.5,
    zscore_threshold = 3,
    grubbs_alpha = 0.05,
    n_threads = 1L,
    is_monotonic = TRUE,
    population_size = 50L,
    max_generations = 100L,
    mutation_rate = 0.1,
    initial_temperature = 1,
    cooling_rate = 0.995,
    max_iterations = 1000L
  )

  # Update control, if needed
  control <- utils::modifyList(default_control, control)

  # Validate inputs
  OptimalBinningValidateInputs(dt, target, features, method, preprocess, min_bins, max_bins, control, positive)

  # Determine the features to process
  if (is.null(features)) {
    features <- setdiff(names(dt), target)
  }

  # Initialize results list
  results <- reports <- woebins <- bestmod <- list()
  failed_features <- character()

  # Map target based on 'positive' argument
  dt <- mapTargetVariable(dt, target, positive)

  # Preprocess data if required
  if (preprocess) {
    preprocessed_data <- OptimalBinningPreprocessData(dt, features, control)
  } else {
    preprocessed_data <- list()
    for (feat in features) {
      preprocessed_data[[feat]] <- list(
        preprocess = list(feature_preprocessed = dt[[feat]]),
        report = data.table::data.table(
          feature = feat,
          type = class(dt[[feat]]),
          missing_count = sum(is.na(dt[[feat]])),
          outlier_count = 0,
          unique_count = length(unique(dt[[feat]])),
          mean_before = mean(dt[[feat]], na.rm = TRUE),
          mean_after = mean(dt[[feat]], na.rm = TRUE),
          sd_before = sd(dt[[feat]], na.rm = TRUE),
          sd_after = sd(dt[[feat]], na.rm = TRUE)
        )
      )
    }
  }

  # Separate numerical and categorical features
  numeric_features <- features[sapply(dt[, ..features], is.numeric)]
  categorical_features <- setdiff(features, numeric_features)

  # Process numerical features
  if (length(numeric_features) > 0) {
    numeric_results <- OptimalBinningNumericalWoE(
      dt, target, numeric_features, method,
      min_bins, max_bins, control, positive,
      preprocessed_data[numeric_features]
    )
    results <- c(results, numeric_results$results)
    reports <- c(reports, numeric_results$reports)
    woebins <- c(woebins, numeric_results$woebins)
    bestmod <- c(bestmod, numeric_results$bestmod)
    failed_features <- c(failed_features, numeric_results$failed_features)
  }

  # Process categorical features
  if (length(categorical_features) > 0) {
    categorical_results <- OptimalBinningCategoricalWoE(
      dt, target, categorical_features, method,
      min_bins, max_bins, control, positive,
      preprocessed_data[categorical_features]
    )
    results <- c(results, categorical_results$results)
    reports <- c(reports, categorical_results$reports)
    woebins <- c(woebins, categorical_results$woebins)
    bestmod <- c(bestmod, categorical_results$bestmod)
    failed_features <- c(failed_features, categorical_results$failed_features)
  }

  # Combine results
  woedt <- data.table::copy(dt)
  for (feat in names(results)) {
    woedt[[paste0(feat, "_woe")]] <- results[[feat]]
  }

  return(
    list(
      woedt = woedt,
      woebins = data.table::rbindlist(woebins, fill = TRUE, idcol = "feature"),
      prepreport = data.table::rbindlist(reports, fill = TRUE, idcol = "feature"),
      bestsreport = data.table::rbindlist(bestmod, fill = TRUE, idcol = "feature"),
      failedfeatures = failed_features,
      bestmethod = method
    )
  )
}


#' Preprocess Data for Optimal Binning
#'
#' @param dt A data.table containing the dataset.
#' @param features Vector of feature names to process.
#' @param control A list of control parameters.
#'
#' @return A list of preprocessed data for each feature.
#'
#' @export
OptimalBinningPreprocessData <- function(dt, features, control) {
  preprocessed_data <- list()
  for (feat in features) {
    preprocessed_data[[feat]] <- OptimalBinningDataPreprocessor(
      target = dt[[target]],
      feature = dt[[feat]],
      num_miss_value = control$num_miss_value,
      char_miss_value = control$char_miss_value,
      outlier_method = control$outlier_method,
      outlier_process = control$outlier_process,
      preprocess = as.character(c("both")),
      iqr_k = control$iqr_k,
      zscore_threshold = control$zscore_threshold,
      grubbs_alpha = control$grubbs_alpha
    )
  }
  return(preprocessed_data)
}

#' Optimal Binning and Weight of Evidence Calculation for Numerical Variables
#'
#' @param dt A data.table containing the dataset.
#' @param target The name of the target variable.
#' @param features Vector of numeric feature names to process.
#' @param method The binning method to use.
#' @param min_bins Minimum number of bins.
#' @param max_bins Maximum number of bins.
#' @param control A list of additional control parameters.
#' @param positive Character string specifying which category should be considered as positive.
#' @param preprocessed_data List of preprocessed data for each feature.
#'
#' @return A list containing results, reports, woebins, bestmod, and failed_features.
#'
#' @export
OptimalBinningNumericalWoE <- function(dt, target, features, method, min_bins, max_bins, control, positive, preprocessed_data) {
  results <- reports <- woebins <- bestmod <- list()
  failed_features <- character()

  numerical_methods <- sort(unique(names(OptimalBinningGetAlgoName()$num)))

  # preprocessed_data <- preprocessed_data[numeric_features]

  for (feat in features) {
    tryCatch(
      {
        dt_proc <- data.table::data.table(
          target = dt[[target]],
          feature = preprocessed_data[[feat]]$preprocess$feature_preprocessed,
          original_index = seq_len(nrow(dt))
        )

        # Identify special cases
        is_special <- dt_proc$feature == control$num_miss_value

        # Perform binning on non-special cases
        dt_binning <- dt_proc[!is_special]

        # Select the best method if method is "auto"
        if (method == "auto") {
          binning_result <- OptimalBinningSelectBestModel(dt_binning, "target", "feature", min_bins, max_bins, control, numerical_methods)
        } else {
          algo_info <- OptimalBinningSelectAlgorithm("feature", method, dt_binning, min_bins, max_bins, control)
          algo_params <- utils::modifyList(list(min_bins = min_bins, max_bins = max_bins), algo_info$params)
          binning_result <- do.call(algo_info$algorithm, c(list(target = dt_binning$target, feature = dt_binning$feature), algo_params))
          binning_result$best_model_report <- NULL
          binning_result$best_method <- method
        }

        # Add WoE values to non-special cases
        dt_proc[!is_special, woe := binning_result$woefeature]

        # Handle special cases
        if (any(is_special)) {
          special_woe <- CalculateSpecialWoE(dt_proc[is_special, target])
          dt_proc[is_special, woe := special_woe]

          # Add special bin to woebin
          special_bin <- CreateSpecialBin(dt_proc[is_special], binning_result$woebin, special_woe)
          special_bin[, bin := paste0(special_bin$bin, "(", control$num_miss_value, ")")]

          binning_result$woebin <- rbind(binning_result$woebin, special_bin, fill = TRUE)
        }

        # Sort by original index
        dt_proc <- dt_proc[order(original_index)]

        # Update results
        results[[feat]] <- dt_proc$woe
        reports[[feat]] <- preprocessed_data[[feat]]$report
        woebins[[feat]] <- OptimalBinningGainsTable(binning_result)
        bestmod[[feat]] <- binning_result$best_model_report
      },
      error = function(e) {
        warning(paste("Error processing feature:", feat, "-", e$message))
        failed_features <- c(failed_features, feat)
      }
    )
  }

  list(results = results, reports = reports, woebins = woebins, bestmod = bestmod, failed_features = failed_features)
}


#' Optimal Binning and Weight of Evidence Calculation for Categorical Variables
#'
#' @param dt A data.table containing the dataset.
#' @param target The name of the target variable.
#' @param features Vector of categorical feature names to process.
#' @param method The binning method to use.
#' @param min_bins Minimum number of bins.
#' @param max_bins Maximum number of bins.
#' @param control A list of additional control parameters.
#' @param positive Character string specifying which category should be considered as positive.
#' @param preprocessed_data List of preprocessed data for each feature.
#'
#' @return A list containing results, reports, woebins, bestmod, and failed_features.
#'
#' @export
OptimalBinningCategoricalWoE <- function(dt, target, features, method, min_bins, max_bins, control, positive, preprocessed_data) {
  results <- reports <- woebins <- bestmod <- list()
  failed_features <- character()

  # categorical_methods <- c("cm", "dplc", "gmb", "ldb", "mba", "mblp", "milp", "mob", "obnp", "swb", "udt")
  categorical_methods <- sort(unique(names(OptimalBinningGetAlgoName()$char)))

  for (feat in features) {
    tryCatch(
      {
        dt_proc <- data.table::data.table(
          target = dt[[target]],
          feature = preprocessed_data[[feat]]$preprocess$feature_preprocessed,
          original_index = seq_len(nrow(dt))
        )

        # Identify special cases
        is_special <- dt_proc$feature == control$char_miss_value

        # Perform binning on non-special cases
        dt_binning <- dt_proc[!is_special]

        # Select the best method if method is "auto"
        if (method == "auto") {
          binning_result <- OptimalBinningSelectBestModel(dt_binning, "target", "feature", min_bins, max_bins, control, categorical_methods)
        } else {
          algo_info <- OptimalBinningSelectAlgorithm("feature", method, dt_binning, min_bins, max_bins, control)
          algo_params <- utils::modifyList(list(min_bins = min_bins, max_bins = max_bins), algo_info$params)
          binning_result <- do.call(algo_info$algorithm, c(list(target = dt_binning$target, feature = dt_binning$feature), algo_params))
          binning_result$best_model_report <- NULL
          binning_result$best_method <- method
        }

        # Add WoE values to non-special cases
        dt_proc[!is_special, woe := binning_result$woefeature]

        # Handle special cases
        if (any(is_special)) {
          special_woe <- CalculateSpecialWoE(dt_proc[is_special, target])
          dt_proc[is_special, woe := special_woe]

          # Add special bin to woebin
          special_bin <- CreateSpecialBin(dt_proc[is_special], binning_result$woebin, special_woe)
          special_bin[, bin := paste0(special_bin$bin, "(", control$char_miss_value, ")")]

          binning_result$woebin <- rbind(binning_result$woebin, special_bin, fill = TRUE)
        }

        # Sort by original index
        dt_proc <- dt_proc[order(original_index)]

        # Update results
        results[[feat]] <- dt_proc$woe
        reports[[feat]] <- preprocessed_data[[feat]]$report
        woebins[[feat]] <- OptimalBinningGainsTable(binning_result)
        bestmod[[feat]] <- binning_result$best_model_report
      },
      error = function(e) {
        warning(paste("Error processing feature:", feat, "-", e$message))
        failed_features <- c(failed_features, feat)
      }
    )
  }

  list(results = results, reports = reports, woebins = woebins, bestmod = bestmod, failed_features = failed_features)
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
mapTargetVariable <- function(dt, target, positive) {
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
CalculateSpecialWoE <- function(target) {
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
CreateSpecialBin <- function(dt_special, woebin, special_woe) {
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

#' Select Best Binning Model
#'
#' @param dt_binning Data for binning
#' @param target Target variable name
#' @param feature Feature variable name
#' @param min_bins Minimum number of bins
#' @param max_bins Maximum number of bins
#' @param control Control parameters
#' @param allowed_methods Vector of allowed binning methods
#'
#' @return Best binning result
#'
#' @keywords internal
OptimalBinningSelectBestModel <- function(dt_binning, target, feature, min_bins, max_bins, control, allowed_methods) {
  best_method <- NULL
  best_iv <- -Inf
  best_result <- NULL
  best_monotonic <- FALSE
  best_bins <- Inf

  best_model_report <- data.table::data.table(
    model_name = character(),
    model_method = character(),
    total_iv = numeric(),
    bin_counts = integer()
  )

  for (method in allowed_methods) {
    tryCatch(
      {
        algo_info <- OptimalBinningSelectAlgorithm(feature, method, dt_binning, min_bins, max_bins, control)

        binning_result <- do.call(
          algo_info$algorithm,
          c(
            list(target = dt_binning[[target]], feature = dt_binning[[feature]]),
            algo_info$params
          )
        )

        if (length(binning_result$woebin$bin) >= 2) {
          model_name <- algo_info$algorithm
          current_iv <- sum(binning_result$woebin$iv, na.rm = TRUE)
          is_monotonic <- is_woe_monotonic(binning_result$woebin$woe)
          current_bins <- length(binning_result$woebin$bin)

          best_model_report <- data.table::rbindlist(
            list(
              best_model_report,
              data.table::data.table(
                model_name = model_name,
                model_method = method,
                total_iv = current_iv,
                bin_counts = current_bins
              )
            )
          )

          if (current_iv > best_iv ||
            (current_iv == best_iv && is_monotonic && !best_monotonic) ||
            (current_iv == best_iv && is_monotonic == best_monotonic && current_bins < best_bins)) {
            best_iv <- current_iv
            best_method <- method
            best_result <- binning_result
            best_monotonic <- is_monotonic
            best_bins <- current_bins
          }
        }
      },
      error = function(e) {
        warning(paste("Error processing method:", method, "for feature:", feature, "-", e$message))
      }
    )
  }

  if (is.null(best_method)) {
    stop(paste("No suitable method found for feature:", feature))
  }

  data.table::setorder(best_model_report, -total_iv)
  best_result$best_method <- best_method
  best_result$best_model_report <- best_model_report
  return(best_result)
}

#' Check if WoE values are monotonic
#'
#' @param woe_values Vector of WoE values
#'
#' @return Logical indicating if WoE values are monotonic
#'
#' @keywords internal
is_woe_monotonic <- function(woe_values) {
  diff_woe <- diff(woe_values)
  all(diff_woe >= 0) || all(diff_woe <= 0)
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
  if (!is.numeric(min_bins) || min_bins < 2) {
    stop("min_bins must be an integer greater than or equal to 2.")
  }
  if (!is.numeric(max_bins) || max_bins <= min_bins) {
    stop("max_bins must be an integer greater than min_bins.")
  }

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
    selected_algorithm <- available_algorithms[[data_type]][[algo]]
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
  categorical <- pk[grepl("optimal_binning_categorical_", pk)]
  categorical <- as.list(categorical)
  names(categorical) <- get_last_part(categorical)

  # Filter and process numerical algorithms
  numerical <- pk[grepl("optimal_binning_numerical_", pk)]
  numerical <- as.list(numerical)
  names(numerical) <- get_last_part(numerical)

  # Return a list with both types of algorithms
  return(list(char = categorical, num = numerical))
}
