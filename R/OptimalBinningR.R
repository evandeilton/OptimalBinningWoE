## --------------------------------------------------------------------------------------------- ##
## Main
## --------------------------------------------------------------------------------------------- ##

#' Optimal Binning and Weight of Evidence Calculation
#'
#' @description
#' This function performs optimal binning and calculates Weight of Evidence (WoE) for multiple features.
#' It supports automatic method selection and data preprocessing.
#'
#' @param dt A data.table containing the dataset.
#' @param target The name of the target variable (must be binary).
#' @param feature Optional. Name of a specific feature to process. If NULL, all features except the target will be processed.
#' @param method The binning method to use. Can be "auto" or a specific method name. See Details for more information.
#' @param preprocess Logical. Whether to preprocess the data before binning.
#' @param min_bins Minimum number of bins.
#' @param max_bins Maximum number of bins.
#' @param cat_cutoff Cutoff for categorical variables.
#' @param bin_cutoff Cutoff for numeric variables.
#' @param control A list of additional control parameters. See Details for more information.
#'
#' @return A list containing:
#' \itemize{
#'   \item woe_feature: The original dataset with added WoE columns
#'   \item woe_woebins: Information about the bins created
#'   \item prep_report: Preprocessing report for each feature
#' }
#'
#' @details
#' Supported Algorithms:
#' The function supports the following binning algorithms:
#' \itemize{
#'   \item CAIM (Class-Attribute Interdependence Maximization): 
#'         Applicable to both categorical and numeric variables. 
#'         It maximizes the class-attribute interdependence to find optimal bins.
#'         - Categorical: OptimalBinningCategoricalCAIM
#'         - Numeric: OptimalBinningNumericCAIM
#'   
#'   \item ChiMerge: 
#'         Applicable to both categorical and numeric variables. 
#'         It uses chi-square statistic to iteratively merge adjacent intervals until a stopping criterion is met.
#'         - Categorical: OptimalBinningCategoricalChiMerge
#'         - Numeric: OptimalBinningNumericChiMerge
#'   
#'   \item MDLP (Minimum Description Length Principle): 
#'         Applicable to both categorical and numeric variables. 
#'         It uses the principle of minimum description length to find the optimal splitting points.
#'         - Categorical: OptimalBinningCategoricalMDLP
#'         - Numeric: OptimalBinningNumericMDLP
#'   
#'   \item MIP (Minimum Information Pure): 
#'         Applicable to both categorical and numeric variables. 
#'         It aims to minimize the impurity of information within each bin.
#'         - Categorical: OptimalBinningCategoricalMIP
#'         - Numeric: OptimalBinningNumericMIP
#'   
#'   \item MOB (Monotone Optimal Binning): 
#'         Applicable to both categorical and numeric variables. 
#'         It ensures monotonicity in the binning process while optimizing a chosen metric.
#'         - Categorical: OptimalBinningCategoricalMOB
#'         - Numeric: OptimalBinningNumericMOB
#'   
#'   \item IV (Information Value): 
#'         Applicable only to categorical variables. 
#'         It bins categories based on their information value with respect to the target variable.
#'         - Categorical: OptimalBinningCategoricalIV
#'   
#'   \item PAVA (Pool Adjacent Violators Algorithm): 
#'         Applicable only to numeric variables. 
#'         It ensures monotonicity in the binning process by pooling adjacent bins that violate the monotonicity constraint.
#'         - Numeric: OptimalBinningNumericPAVA
#'   
#'   \item Tree-based binning: 
#'         Applicable only to numeric variables. 
#'         It uses decision tree algorithms to create bins, allowing for more flexible and potentially non-linear binning.
#'         - Numeric: OptimalBinningNumericTree
#' }
#'
#' Each algorithm has its own strengths and may perform differently depending on the nature of the data. 
#' The automatic method selection option tests applicable algorithms and chooses the one that produces the highest Information Value.
#'
#' When specifying a method, use the short name (e.g., "caim", "chimerge") rather than the full algorithm name.
#' Control Parameters:
#' The control list can include the following parameters:
#' \itemize{
#'   \item min_bads: Minimum proportion of "bad" cases in each bin (default: 0.05)
#'   \item pvalue_threshold: P-value threshold for statistical tests (default: 0.05)
#'   \item max_n_prebins: Maximum number of pre-bins before optimization (default: 20)
#'   \item monotonicity_direction: Direction of monotonicity ("increase" or "decrease") (default: "increase")
#'   \item lambda: Regularization parameter for tree-based methods (default: 0.1)
#'   \item min_bin_size: Minimum proportion of cases in each bin (default: 0.05)
#'   \item min_iv_gain: Minimum IV gain for creating a new split (default: 0.01)
#'   \item max_depth: Maximum depth for tree-based methods (default: 10)
#'   \item num_miss_value: Value to represent missing numeric values (default: -999.0)
#'   \item char_miss_value: Value to represent missing categorical values (default: "N/A")
#'   \item outlier_method: Method for outlier detection ("iqr", "zscore", or "grubbs") (default: "iqr")
#'   \item outlier_process: Whether to process outliers (default: FALSE)
#'   \item iqr_k: Factor for IQR method (default: 1.5)
#'   \item zscore_threshold: Threshold for Z-score method (default: 3)
#'   \item grubbs_alpha: Significance level for Grubbs' test (default: 0.05)
#' }
#'
#' @examples
#' \dontrun{
#' # Load necessary libraries
#' library(data.table)
#'
#' # Create a sample dataset
#' dt <- data.table(
#'   target = sample(0:1, 1000, replace = TRUE),
#'   num_feat = rnorm(1000),
#'   cat_feat = sample(letters[1:5], 1000, replace = TRUE)
#' )
#'
#' # Run OptimalBinningWoE with automatic method selection
#' result <- OptimalBinningWoE(dt, target = "target", method = "auto")
#'
#' # Check the results
#' head(result$woe_feature)
#' head(result$woe_woebins)
#' head(result$prep_report)
#' }
#'
#' @importFrom data.table setDT copy setalloccol rbindlist
#' @importFrom stats rnorm
#' @importFrom utils modifyList
#'
#' @export
OptimalBinningWoE <- function(dt, target, feature = NULL, method = "auto", preprocess = TRUE,
                               min_bins = 2, max_bins = 4, cat_cutoff = 0.05, bin_cutoff = 0.05,
                               control = list()) {
  
  dt <- data.table::copy(data.table::setDT(dt))
  
  # Default pars
  default_control <- list(min_bads = 0.05, pvalue_threshold = 0.05, max_n_prebins = 20, 
                          monotonicity_direction = "increase", lambda = 0.1, min_bin_size = 0.05, 
                          min_iv_gain = 0.01, max_depth = 10, num_miss_value = -999.0,
                          char_miss_value = "N/A", outlier_method = "iqr", outlier_process = FALSE, 
                          iqr_k = 1.5, zscore_threshold = 3, grubbs_alpha = 0.05)
  
  # Update control, if needed
  control <- modifyList(default_control, control)
  
  # Determine the features to process
  features_to_process <- if(is.null(feature)) setdiff(names(dt), target) else feature
  
  # Initialize results list
  results <- reports <- woebins <- list()
  
  # Process each feature
  for (feat in features_to_process) {
    tryCatch({
      # Preprocess data if required
      if (preprocess) {
        preprocessed_data <- OptimalBinningDataPreprocessor(
          target = dt[[target]],
          feature = dt[[feat]],
          num_miss_value = control$num_miss_value,
          char_miss_value = control$char_miss_value,
          outlier_method = control$outlier_method,
          outlier_process = control$outlier_process,
          preprocess = "both",
          iqr_k = control$iqr_k,
          zscore_threshold = control$zscore_threshold,
          grubbs_alpha = control$grubbs_alpha
        )
        
        preprocessed_dt <- data.table::setalloccol(preprocessed_data$preprocess) # data.table::copy(preprocessed_data$preprocess)
        preprocess_report <- data.table::setalloccol(preprocessed_data$report) #data.table::copy(preprocessed_data$report)
        
        # Identify special values
        is_special <- is.na(preprocessed_dt$feature_preprocessed) | 
          preprocessed_dt$feature_preprocessed == control$num_miss_value |
          preprocessed_dt$feature_preprocessed == control$char_miss_value
      } else {
        preprocessed_dt <- data.table::data.table(feature_preprocessed = dt[[feat]]) #data.table::data.table(feature_preprocessed = dt[[feat]])
        preprocess_report <- data.table::data.table()
        is_special <- is.na(preprocessed_dt$feature_preprocessed)
      }
      
      # Separate normal and special data
      normal_data <- preprocessed_dt[!is_special]
      special_data <- preprocessed_dt[is_special]
      
      # Select the best method if method is "auto"
      if (method == "auto") {
        normal_data[, (target) := dt[[get(target)]][!is_special]]
        binning_result <- OptimalBinningselectBestModel(normal_data, target = target, "feature_preprocessed", control, min_bins, max_bins)
      } else {
        normal_data[, (target) := dt[[get(target)]][!is_special]]
        # Select the appropriate algorithm and get its parameters
        algo_info <- OptimalBinningSelectAlgorithm(feature = "feature_preprocessed", method, normal_data, control)
        
        if(algo_info$method %in% c("pava","tree")){
          algo_params <- modifyList(
            list(max_bins = max_bins),
            algo_info$params
          )
        } else {
          algo_params <- modifyList(
            list(min_bins = min_bins, max_bins = max_bins),
            algo_info$params
          )
        }
        # Apply binning
        binning_result <- do.call(algo_info$algorithm,
                                  c(list(target = normal_data[[target]], feature = normal_data$feature_preprocessed), algo_params))
      }
      
      # Create special bin for missing/special values if any
      if (nrow(special_data) > 0) {
        special_bin <- data.table::data.table(
          bin = paste0("Special(", ifelse(binning_result$is_categorical, control$char_miss_value, control$num_miss_value),")"),
          count = nrow(special_data),
          count_pos = sum(dt[[target]][is_special] == 1), #bad
          count_neg = sum(dt[[target]][is_special] == 0)  #good
        )
        special_bin[, `:=`(
          woe = log((count_pos / sum(binning_result$woebin$count_pos)) / (count_neg / sum(binning_result$woebin$count_neg))))
        ][,`:=` (iv = (count_pos / sum(binning_result$woebin$count_pos) - count_neg / sum(binning_result$woebin$count_neg)) * woe)]
        
        # Add special bin to binning result
        binning_result$woebin <- rbind(binning_result$woebin, special_bin, fill = TRUE)
      }
      
      # Assign WOE to original data
      woe_vector <- numeric(nrow(dt))
      woe_vector[!is_special] <- binning_result$woefeature$woefeature
      if (nrow(special_data) > 0) {
        woe_vector[is_special] <- special_bin$woe
      }
      
      # Create new column name with "_woe" suffix
      woe_col_name <- paste0(feat, "_woe")
      
      # Update dt efficiently using data.table syntax
      dt[, (woe_col_name) := woe_vector]
      
      # Update report list
      reports[[woe_col_name]] <- preprocess_report
      woebins[[woe_col_name]] <- OptimalBinningGainsTable(binning_result)
    }, error = function(e) {
      warning(paste("Error processing feature:", feat, "-", e$message))
    })
  }
  
  results$woe_feature <- data.table::copy(dt)
  results$woe_woebins <- data.table::rbindlist(woebins, fill = TRUE, idcol = "feature")
  results$prep_report <- data.table::rbindlist(reports, fill = TRUE, idcol = "feature")
  
  return(results)
}

# 
# OptimalBinningWoE <- function(dt, target, feature = NULL, method = "auto", preprocess = TRUE,
#                               min_bins = 2, max_bins = 4, cat_cutoff = 0.05, bin_cutoff = 0.05,
#                               control = list()) {
#   
#   # Default pars
#   default_control <- list(min_bads = 0.05, pvalue_threshold = 0.05, max_n_prebins = 20, 
#                           monotonicity_direction = "increase", lambda = 0.1, min_bin_size = 0.05, 
#                           min_iv_gain = 0.01, max_depth = 10, num_miss_value = -999.0,
#                           char_miss_value = "N/A", outlier_method = "iqr", outlier_process = FALSE, 
#                           iqr_k = 1.5, zscore_threshold = 3, grubbs_alpha = 0.05)
#   
#   # Update control, if needed
#   control <- modifyList(default_control, control)
#   
#   # Determine the features to process
#   features_to_process <- if(is.null(feature)) setdiff(names(dt), target) else feature
#   
#   # Initialize results list
#   results <- reports <- woebins <- list()
#   
#   # Process each feature
#   for (feat in features_to_process) {
#     tryCatch({
#       # Preprocess data if required
#       if (preprocess) {
#         preprocessed_data <- OptimalBinningDataPreprocessor(
#           target = dt[[target]],
#           feature = dt[[feat]],
#           num_miss_value = control$num_miss_value,
#           char_miss_value = control$char_miss_value,
#           outlier_method = control$outlier_method,
#           outlier_process = control$outlier_process,
#           preprocess = "both",
#           iqr_k = control$iqr_k,
#           zscore_threshold = control$zscore_threshold,
#           grubbs_alpha = control$grubbs_alpha
#         )
#         
#         preprocessed_dt <- data.table::copy(preprocessed_data$preprocess)
#         preprocess_report <- data.table::copy(preprocessed_data$report)
#       } else {
#         preprocessed_dt <- data.table::data.table(feature_preprocessed = dt[[feat]])
#         preprocess_report <- data.table::data.table()
#       }
#       
#       # Select the best method if method is "auto"
#       if (method == "auto") {
#         preprocessed_dt[,(target) := dt[[target]]]
#         binning_result <- OptimalBinningselectBestModel(preprocessed_dt, target, "feature_preprocessed", control, min_bins, max_bins)
#       } else {
#         best_method <- method
#         # Select the appropriate algorithm and get its parameters
#         algo_info <- OptimalBinningSelectAlgorithm(feature = "feature_preprocessed", best_method, preprocessed_dt, control)
#         
#         if(algo_info$method %in% c("pava","tree")){
#           algo_params <- modifyList(
#             list(max_bins = max_bins),
#             algo_info$params
#           )
#         } else {
#           algo_params <- modifyList(
#             list(min_bins = min_bins, max_bins = max_bins),
#             algo_info$params
#           )
#         }
#         # Apply binning
#         binning_result <- do.call(algo_info$algorithm,
#                                   c(list(target = dt[[target]], feature = preprocessed_dt$feature_preprocessed), algo_params))
#       }
#       
#       # Create new column name with "_woe" suffix
#       woe_col_name <- paste0(feat, "_woe")
#       # Update dt efficiently using data.table syntax
#       dt[, (woe_col_name) := binning_result$woefeature$woefeature]
#       
#       # Update report list
#       reports[[woe_col_name]] <- preprocess_report
#       woebins[[woe_col_name]] <- OptimalBinningGainsTable(binning_result) # binning_result$woebin
#     }, error = function(e) {
#       warning(paste("Error processing feature:", feat, "-", e$message))
#     })
#   }
#   
#   results$woe_feature <- data.table::copy(dt)
#   results$woe_woebins <- data.table::rbindlist(woebins, fill = TRUE, idcol = "feature")
#   results$prep_report <- data.table::rbindlist(reports, fill = TRUE, idcol = "feature")
#   
#   return(results)
# }


## --------------------------------------------------------------------------------------------- ##
## Validade imputs
## --------------------------------------------------------------------------------------------- ##

#' Validate Inputs for Optimal Binning
#'
#' @description
#' This function validates all inputs for the OptimalBinningWoE function.
#'
#' @param dt A data.table containing the dataset.
#' @param target The name of the target variable.
#' @param feature Optional. Name of a specific feature to process.
#' @param method The binning method to use.
#' @param min_bins Minimum number of bins.
#' @param max_bins Maximum number of bins.
#' @param cat_cutoff Cutoff for categorical variables.
#' @param bin_cutoff Cutoff for numeric variables.
#' @param control A list of additional control parameters.
#' @return No return value, called for side effects
#'
#' @keywords internal
OptimalBinningValidateInputs <- function(dt, target, feature, method, min_bins, max_bins, cat_cutoff, bin_cutoff, control) {
  
  # Check if target exists in dt
  if (!target %in% names(dt)) {
    stop("The 'target' variable does not exist in the provided data.table.")
  }
  
  # Check if target is binary
  if (!all(dt[[target]] %in% c(0, 1))) {
    stop("The 'target' variable must be binary (0 or 1).")
  }
  
  # Check feature (if provided)
  if (!is.null(feature)) {
    if (!feature %in% names(dt)) {
      stop("The specified 'feature' variable does not exist in the provided data.table.")
    }
  }
  
  # Check binning method
  valid_methods <- c("caim", "chimerge", "mdlp", "mip", "mob", "iv", "pava", "tree")
  if (!method %in% valid_methods) {
    stop(paste("Invalid binning method. Choose one of the following:", paste(valid_methods, collapse = ", ")))
  }
  
  # Check min_bins and max_bins
  if (!is.numeric(min_bins) || min_bins < 2) {
    stop("min_bins must be an integer greater than or equal to 2.")
  }
  if (!is.numeric(max_bins) || max_bins <= min_bins) {
    stop("max_bins must be an integer greater than min_bins.")
  }
  
  # Check cat_cutoff and bin_cutoff
  if (!is.numeric(cat_cutoff) || cat_cutoff <= 0 || cat_cutoff >= 1) {
    stop("cat_cutoff must be a number between 0 and 1.")
  }
  if (!is.numeric(bin_cutoff) || bin_cutoff <= 0 || bin_cutoff >= 1) {
    stop("bin_cutoff must be a number between 0 and 1.")
  }
  
  # Check control
  required_control <- c("min_bads", "pvalue_threshold", "max_n_prebins", "monotonicity_direction",
                        "lambda", "min_bin_size", "min_iv_gain", "max_depth", "num_miss_value",
                        "char_miss_value", "outlier_method", "outlier_process")
  missing_control <- setdiff(required_control, names(control))
  if (length(missing_control) > 0) {
    stop(paste("The following control parameters are missing:", paste(missing_control, collapse = ", ")))
  }
  
  # Specific checks for control parameters
  if (!control$monotonicity_direction %in% c("increase", "decrease")) {
    stop("monotonicity_direction must be 'increase' or 'decrease'.")
  }
  if (!control$outlier_method %in% c("iqr", "zscore", "grubbs")) {
    stop("outlier_method must be 'iqr', 'zscore', or 'grubbs'.")
  }
  if (!is.logical(control$outlier_process)) {
    stop("outlier_process must be TRUE or FALSE.")
  }
  
  # If all checks pass, the function will return silently
}

## --------------------------------------------------------------------------------------------- ##
## Select algorithms
## --------------------------------------------------------------------------------------------- ##
#' Select Optimal Binning Algorithm
#'
#' @description
#' This function selects the appropriate binning algorithm based on the method and variable type.
#'
#' @param feature The name of the feature to bin.
#' @param method The binning method to use.
#' @param dt A data.table containing the dataset.
#' @param control A list of additional control parameters.
#'
#' @return A list containing the selected algorithm, its parameters, and the method name.
#'
#' @keywords internal
OptimalBinningSelectAlgorithm <- function(feature, method, dt, control) {
  # Determine if the feature is categorical or numeric
  is_categorical <- is.factor(dt[[feature]]) || is.character(dt[[feature]])
  
  # Define the mapping of method names to algorithm names
  method_mapping <- list(
    caim = list(categorical = "OptimalBinningCategoricalCAIM", numeric = "OptimalBinningNumericCAIM"),
    chimerge = list(categorical = "OptimalBinningCategoricalChiMerge", numeric = "OptimalBinningNumericChiMerge"),
    mdlp = list(categorical = "OptimalBinningCategoricalMDLP", numeric = "OptimalBinningNumericMDLP"),
    mip = list(categorical = "OptimalBinningCategoricalMIP", numeric = "OptimalBinningNumericMIP"),
    mob = list(categorical = "OptimalBinningCategoricalMOB", numeric = "OptimalBinningNumericMOB"),
    iv = list(categorical = "OptimalBinningCategoricalIV", numeric = NULL),
    pava = list(categorical = NULL, numeric = "OptimalBinningNumericPAVA"),
    tree = list(categorical = NULL, numeric = "OptimalBinningNumericTree")
  )
  
  # Select the appropriate algorithm based on the method and variable type
  data_type <- if(is_categorical) "categorical" else "numeric"
  selected_algorithm <- method_mapping[[method]][[data_type]]
  
  # Check if the selected algorithm is valid for the variable type
  if (is.null(selected_algorithm)) {
    stop(paste("The", method, "method is not applicable for", data_type, "variables."))
  }
  
  # Get the default parameters for the selected algorithm
  default_params <- switch(selected_algorithm,
                           OptimalBinningCategoricalCAIM = list(target, feature, min_bins = 2L, max_bins = 7L, cat_cutoff = 0.05, min_bads = 0.05),
                           OptimalBinningCategoricalChiMerge = list(target, feature, min_bins = 2L, max_bins = 7L, pvalue_threshold = 0.05, cat_cutoff = 0.05, min_bads = 0.05, max_n_prebins = 20L),
                           OptimalBinningCategoricalIV = list(target, feature, min_bins = 2L, max_bins = 7L, cat_cutoff = 0.05, min_bads = 0.05),
                           OptimalBinningCategoricalMDLP = list(target, feature, min_bins = 2L, max_bins = 7L, cat_cutoff = 0.05, min_bads = 0.05),
                           OptimalBinningCategoricalMIP = list(target, feature, cat_cutoff = 0.05, min_bins = 2L, max_bins = 5L),
                           OptimalBinningCategoricalMOB = list(target, feature, min_bins = 2L, max_bins = 7L, cat_cutoff = 0.05, min_bads = 0.05, max_n_prebins = 20L),
                           OptimalBinningNumericCAIM = list(target, feature, min_bins = 2L, max_bins = 7L, bin_cutoff = 0.05, min_bads = 0.05, max_n_prebins = 20L),
                           OptimalBinningNumericChiMerge = list(target, feature, min_bins = 2L, max_bins = 7L, pvalue_threshold = 0.05, bin_cutoff = 0.05, min_bads = 0.05, max_n_prebins = 20L),
                           OptimalBinningNumericMDLP = list(target, feature, min_bins = 2L, max_bins = 7L, bin_cutoff = 0.05, min_bads = 0.05, max_n_prebins = 20L),
                           OptimalBinningNumericMIP = list(target, feature, min_bins = 2L, max_bins = 7L, bin_cutoff = 0.05, max_n_prebins = 20L),
                           OptimalBinningNumericMOB = list(target, feature, min_bins = 2L, max_bins = 7L, bin_cutoff = 0.05, min_bads = 0.05, max_n_prebins = 20L),
                           OptimalBinningNumericPAVA = list(target, feature, max_bins = 7L, bin_cutoff = 0.05, min_bads = 0.05, max_n_prebins = 20L, monotonicity_direction = "increase"),
                           OptimalBinningNumericTree = list(target, feature, max_bins = 7L, lambda = 0.1, min_bin_size = 0.05,  min_iv_gain = 0.01, max_depth = 10L, monotonicity_direction = "increase"),
                           stop(paste("Unknown algorithm:", selected_algorithm)) 
  )
  
  # Merge default parameters with user-provided control parameters
  algorithm_params <- modifyList(default_params, control[[selected_algorithm]] %||% list())
  
  # Return the selected algorithm and its parameters
  list(
    algorithm = selected_algorithm,
    params = algorithm_params,
    method = method
  )
}

## --------------------------------------------------------------------------------------------- ##
## OptimalBinningselectBestModel
## --------------------------------------------------------------------------------------------- ##
#' Select Best Binning Model
#'
#' @description
#' This function tests multiple binning methods and selects the best one based on Information Value (IV).
#'
#' @param dt A data.table containing the dataset.
#' @param target The name of the target variable.
#' @param feature The name of the feature to bin.
#' @param control A list of additional control parameters.
#' @param min_bins Minimum number of bins.
#' @param max_bins Maximum number of bins.
#'
#' @return The binning result of the best method.
#'
#' @keywords internal
OptimalBinningselectBestModel <- function(dt, target, feature, control, min_bins, max_bins) {
  methods <- c("caim", "chimerge", "mdlp", "mob") #"mip"
  is_categorical <- is.factor(dt[[feature]]) || is.character(dt[[feature]])
  
  if (!is_categorical) {
    methods <- c(methods, "pava", "tree")
  }
  
  best_method <- NULL
  best_iv <- -Inf
  
  for (method in methods) {
    tryCatch({
      # cat(method, "\n")
      algo_info <- OptimalBinningSelectAlgorithm(feature = feature, method, dt, control)
      
      if(algo_info$method %in% c("pava","tree")){
        algo_params <- modifyList(
          list(max_bins = max_bins),
          algo_info$params
        )
      } else {
        algo_params <- modifyList(
          list(min_bins = min_bins, max_bins = max_bins),
          algo_info$params
        )
      }
      
      binning_result <- do.call(algo_info$algorithm, c(list(target = dt[[target]], feature = dt[[feature]]), algo_params))
      
      if (length(binning_result$woebin$bin) >= 2) {
        iv <- sum(binning_result$woebin$iv)
        if (iv > best_iv) {
          best_iv <- iv
          best_method <- method
        }
      }
    }, error = function(e) {
      warning(paste("Error processing method:", method, "for feature:", feature, "-", e$message))
    })
  }
  # Get variable type
  binning_result$is_categorical <- is_categorical
  
  if (is.null(best_method)) {
    stop(paste("No suitable method found for feature:", feature))
  } else {
    return(binning_result)  
  }
}

