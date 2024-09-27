## --------------------------------------------------------------------------------------------- ##
## Main
## --------------------------------------------------------------------------------------------- ##

#' #' Optimal Binning and Weight of Evidence Calculation
#' #'
#' #' @description
#' #' This function performs optimal binning and calculates Weight of Evidence (WoE) for multiple features.
#' #' It supports automatic method selection and data preprocessing.
#' #'
#' #' @param dt A data.table containing the dataset.
#' #' @param target The name of the target variable (must be binary).
#' #' @param feature Optional. Name of a specific feature to process. If NULL, all features except the target will be processed.
#' #' @param method The binning method to use. Can be "auto" or a specific method name. See Details for more information.
#' #' @param preprocess Logical. Whether to preprocess the data before binning.
#' #' @param min_bins Minimum number of bins.
#' #' @param max_bins Maximum number of bins.
#' #' @param control A list of additional control parameters. See Details for more information.
#' #' @param positive Character string specifying which category should be considered as positive. Must be either "bad|1" or "good|1".
#' #'
#' #' @return A list containing:
#' #' \itemize{
#' #'   \item woe_feature: The original dataset with added WoE columns
#' #'   \item woe_woebins: Information about the bins created
#' #'   \item prep_report: Preprocessing report for each feature
#' #' }
#' #'
#' #' @details
#' #' Supported Algorithms:
#' #' The function supports various binning algorithms including CAIM, ChiMerge, MDLP, MIP, MOB, IV, PAVA, and Tree-based binning.
#' #' Each algorithm has its own strengths and may perform differently depending on the nature of the data. 
#' #' The automatic method selection option tests applicable algorithms and chooses the one that produces the highest Information Value.
#' #'
#' #' When specifying a method, use the short name (e.g., "caim", "chimerge") rather than the full algorithm name.
#' #'
#' #' Control Parameters:
#' #' The control list can include parameters such as min_bads, pvalue_threshold, max_n_prebins, monotonicity_direction, lambda, min_bin_size, min_iv_gain, max_depth, num_miss_value, char_miss_value, outlier_method, outlier_process, iqr_k, zscore_threshold, and grubbs_alpha.
#' #'
#' #' @examples
#' #' \dontrun{
#' #' # Load necessary libraries
#' #' library(data.table)
#' #'
#' #' # Create a sample dataset
#' #' dt <- data.table(
#' #'   target = sample(0:1, 1000, replace = TRUE),
#' #'   num_feat = rnorm(1000),
#' #'   cat_feat = sample(letters[1:5], 1000, replace = TRUE)
#' #' )
#' #'
#' #' # Run OptimalBinningWoE with automatic method selection
#' #' result <- OptimalBinningWoE(dt, target = "target", method = "auto")
#' #'
#' #' # Check the results
#' #' head(result$woe_feature)
#' #' head(result$woe_woebins)
#' #' head(result$prep_report)
#' #' }
#' #'
#' #' @importFrom data.table setDT copy setalloccol rbindlist
#' #' @importFrom stats rnorm
#' #' @importFrom utils modifyList
#' #'
#' #' @export
#' OptimalBinningWoE <- function(dt, target, feature = NULL, method = "auto", preprocess = TRUE,
#'                               min_bins = 3, max_bins = 4, control = list(), positive = "bad|1") {
#'   
#'   # Default parameters
#'   default_control <- list(
#'     cat_cutoff = 0.05,
#'     bin_cutoff = 0.05,
#'     min_bads = 0.05,
#'     pvalue_threshold = 0.05,
#'     max_n_prebins = 20,
#'     monotonicity_direction = "increase",
#'     lambda = 0.1,
#'     min_bin_size = 0.05,
#'     min_iv_gain = 0.01,
#'     max_depth = 10,
#'     num_miss_value = -999.0,
#'     char_miss_value = "N/A",
#'     outlier_method = "iqr",
#'     outlier_process = FALSE,
#'     iqr_k = 1.5,
#'     zscore_threshold = 3,
#'     grubbs_alpha = 0.05
#'   )
#'   
#'   # Update control, if needed
#'   control <- utils::modifyList(default_control, control)
#'   
#'   # Validate inputs
#'   OptimalBinningValidateInputs(dt, target, feature, method, preprocess, min_bins, max_bins, control, positive)
#'   
#'   # Determine the features to process
#'   features_to_process <- if(is.null(feature)) setdiff(names(dt), target) else feature
#'   
#'   # Initialize results list
#'   results <- reports <- woebins <- bestmod <- list()
#'   failed_features <- character()
#'   
#'   # Map target based on 'positive' argument
#'   if (is.character(dt[[target]]) || is.factor(dt[[target]])) {
#'     if (length(unique(dt[[target]])) > 2) {
#'       stop("Target variable must have exactly two categories")
#'     }
#'     positive_value <- strsplit(positive, "\\|")[[1]][1]
#'     target_col <- dt[[target]]
#'     dt[, (target) := ifelse(target_col == positive_value, 1, 0)]
#'   } else if (!all(dt[[target]] %in% c(0, 1))) {
#'     stop("Target variable must be binary (0 or 1) or a string with two categories")
#'   }
#'   
#'   # Process each feature
#'   for (feat in features_to_process) {
#'     tryCatch({
#'       # Skip date variables
#'       if (is_date_time_dt(dt[[feat]])) {
#'         next
#'       }
#'       
#'       # Create a copy of the feature and target for processing
#'       va <- c(target, feat)
#'       dt_proc <- dt[, ..va][, ("original_index") := seq_len(nrow(dt))]
#'       
#'       # Preprocess data if required
#'       if (preprocess) {
#'         preprocessed_data <- OptimalBinningDataPreprocessor(
#'           target = dt_proc[[target]],
#'           feature = dt_proc[[feat]],
#'           num_miss_value = control$num_miss_value,
#'           char_miss_value = control$char_miss_value, 
#'           outlier_method = control$outlier_method,
#'           outlier_process = control$outlier_process,
#'           preprocess = as.character(c("both")), 
#'           iqr_k = control$iqr_k,
#'           zscore_threshold = control$zscore_threshold,
#'           grubbs_alpha = control$grubbs_alpha
#'         )
#'         dt_proc$feature_preprocessed <- preprocessed_data$preprocess$feature_preprocessed
#'         preprocess_report <- preprocessed_data$report
#'       } else {
#'         dt_proc$feature_preprocessed <- dt_proc$feature
#'         preprocess_report <- data.table::data.table()
#'       }
#'       
#'       # Identify special cases
#'       is_special <- if (is.numeric(dt_proc$feature_preprocessed)) {
#'         dt_proc$feature_preprocessed == control$num_miss_value
#'       } else {
#'         dt_proc$feature_preprocessed == control$char_miss_value
#'       }
#'       
#'       # Perform binning on non-special cases
#'       dt_binning <- dt_proc[!is_special]
#'       
#'       # Select the best method if method is "auto"
#'       if (method == "auto") {
#'         binning_result <- OptimalBinningSelectBestModel(dt_binning, target, "feature_preprocessed", min_bins, max_bins, control)
#'       } else {
#'         # Select the appropriate algorithm and get its parameters
#'         algo_info <- OptimalBinningSelectAlgorithm(feature = "feature_preprocessed", method, dt_binning, min_bins, max_bins, control)
#'         
#'         algo_params <- utils::modifyList(
#'           list(min_bins = min_bins, max_bins = max_bins),
#'           algo_info$params
#'         )
#'         
#'         # Apply binning
#'         binning_result <- do.call(algo_info$algorithm,
#'                                   c(list(target = dt_binning[[target]], feature = dt_binning$feature_preprocessed), algo_params))
#'         binning_result$best_model_report <- NULL
#'         binning_result$best_method <- method
#'       }
#'       
#'       # Add WoE values to non-special cases
#'       dt_proc[!is_special, woe := binning_result$woefeature]
#'       
#'       # Handle special cases
#'       if (any(is_special)) {
#'         special_woe <- CalculateSpecialWoE(dt_proc[is_special, target])
#'         dt_proc[is_special, woe := special_woe]
#'         
#'         # Add special bin to woebin
#'         special_bin <- CreateSpecialBin(dt_proc[is_special], binning_result$woebin, special_woe)
#'         special_bin[, bin := (
#'           paste0(special_bin$bin, "(",
#'                  ifelse(is.character(dt_proc[[feat]]) || is.factor(dt_proc[[feat]]),
#'                         control$char_miss_value, control$num_miss_value), ")")
#'         )]
#'         
#'         binning_result$woebin <- rbind(binning_result$woebin, special_bin, fill = TRUE)
#'       }
#'       
#'       # Sort by original index
#'       dt_proc <- dt_proc[order(original_index)]
#'       
#'       # Create new column name with "_woe" suffix
#'       woe_col_name <- paste0(feat, "_woe")
#'       
#'       # Update dt efficiently using data.table syntax
#'       dt[, (woe_col_name) := dt_proc$woe]
#'       
#'       # Update report list
#'       reports[[woe_col_name]] <- preprocess_report
#'       woebins[[woe_col_name]] <- OptimalBinningGainsTable(binning_result)
#'       bestmod[[woe_col_name]] <- binning_result$best_model_report
#'     }, error = function(e) {
#'       warning(paste("Error processing feature:", feat, "-", e$message))
#'       failed_features <- c(failed_features, feat)
#'     })
#'   }
#'   
#'   return(
#'     list(woedt = data.table::copy(dt),
#'          woebins = data.table::rbindlist(woebins, fill = TRUE, idcol = "feature"),
#'          prepreport = data.table::rbindlist(reports, fill = TRUE, idcol = "feature"),
#'          bestsreport = data.table::rbindlist(bestmod, fill = TRUE, idcol = "feature"),
#'          failedfeatures = failed_features,
#'          bestmethod = binning_result$best_method)
#'     
#'   )
#' }

# # Helper function to calculate WoE for special cases
# CalculateSpecialWoE <- function(target) {
#   count_neg <- sum(target == 0)
#   count_pos <- sum(target == 1)
#   good_rate <- count_neg / length(target)
#   bad_rate <- count_pos / length(target)
#   log(bad_rate / good_rate)
# }

# # Helper function to create a special bin
# CreateSpecialBin <- function(dt_special, woebin, special_woe) {
#   data.table::data.table(
#     bin = "Special",
#     count = nrow(dt_special),
#     count_neg = sum(dt_special$target == 0),
#     count_pos = sum(dt_special$target == 1),
#     woe = special_woe,
#     iv = (sum(dt_special$target == 1) / (sum(woebin$count_pos) + sum(dt_special$target == 1)) -
#             sum(dt_special$target == 0) / (sum(woebin$count_neg) + sum(dt_special$target == 0))) * special_woe
#   )
# }
#' 
#' #' Validate Inputs for Optimal Binning
#' #'
#' #' @description
#' #' This function validates the input parameters for the OptimalBinningWoE function.
#' #'
#' #' @param dt A data.table containing the dataset.
#' #' @param target The name of the target variable.
#' #' @param feature Optional. Name of a specific feature to process.
#' #' @param method The binning method to use.
#' #' @param preprocess Logical. Whether to preprocess the data before binning.
#' #' @param min_bins Minimum number of bins.
#' #' @param max_bins Maximum number of bins.
#' #' @param control A list of additional control parameters.
#' #' @param positive Character string specifying which category should be considered as positive.
#' #'
#' #' @return None. Throws an error if any input is invalid.
#' #'
#' #' @keywords internal
#' OptimalBinningValidateInputs <- function(dt, target, feature, method, preprocess, min_bins, max_bins, control, positive) {
#'   
#'   # Check if dt is a data.table
#'   if (!data.table::is.data.table(dt)) {
#'     stop("The 'dt' argument must be a data.table.")
#'   }
#'   
#'   # Check if target exists in dt
#'   if (!target %in% names(dt)) {
#'     stop("The 'target' variable does not exist in the provided data.table.")
#'   }
#'   
#'   # Check if target is binary or has two categories
#'   if (is.numeric(dt[[target]])) {
#'     if (!all(dt[[target]] %in% c(0, 1))) {
#'       stop("The 'target' variable must be binary (0 or 1) when numeric.")
#'     }
#'   } else if (is.character(dt[[target]]) || is.factor(dt[[target]])) {
#'     if (length(unique(dt[[target]])) != 2) {
#'       stop("The 'target' variable must have exactly two categories when categorical.")
#'     }
#'   } else {
#'     stop("The 'target' variable must be either numeric (0/1) or categorical (two categories).")
#'   }
#'   
#'   # Check feature (if provided)
#'   if (!is.null(feature)) {
#'     if (!all(feature %in% names(dt))) {
#'       stop("One or more specified 'feature' variables do not exist in the provided data.table.")
#'     }
#'   }
#'   
#'   # Define methods for categorical and numerical data
#'   all_method <- "auto" 
#'   categorical_methods <- c("cm", "dplc", "gmb", "ldb", "mba", "mblp", 
#'                            "milp", "mob", "obnp", "swb", "udt")
#'   
#'   numerical_methods <- c("cm", "dplc", "gmb", "ldb", "lpdb", "mba", 
#'                          "mblp", "milp", "mob", "obnp", "swb", "udt", "bb",
#'                          "bs", "dpb", "eb", "eblc", "efb", "ewb", "ir", 
#'                          "jnbo", "kmb", "mdlp", "mrblp", "plaob", "qb", "sbb", "ubsd")
#'   
#'   # Check binning method
#'   valid_methods <- sort(unique(c(all_method, categorical_methods, numerical_methods)))
#' 
#'   if (!method %in% valid_methods) {
#'     stop(paste("Invalid binning method. Choose one of the following:", paste(valid_methods, collapse = ", ")))
#'   }
#'   
#'   # Check preprocess
#'   if (!is.logical(preprocess)) {
#'     stop("'preprocess' must be a logical value (TRUE or FALSE).")
#'   }
#'   
#'   # Check min_bins and max_bins
#'   if (!is.numeric(min_bins) || min_bins < 2) {
#'     stop("min_bins must be an integer greater than or equal to 2.")
#'   }
#'   if (!is.numeric(max_bins) || max_bins <= min_bins) {
#'     stop("max_bins must be an integer greater than min_bins.")
#'   }
#'   
#'   # Check control
#'   if (!is.list(control)) {
#'     stop("'control' must be a list.")
#'   }
#'   
#'   # Check specific control parameters
#'   if (!is.numeric(control$cat_cutoff) || control$cat_cutoff <= 0 || control$cat_cutoff >= 1) {
#'     stop("control$cat_cutoff must be a number between 0 and 1.")
#'   }
#'   if (!is.numeric(control$bin_cutoff) || control$bin_cutoff <= 0 || control$bin_cutoff >= 1) {
#'     stop("control$bin_cutoff must be a number between 0 and 1.")
#'   }
#'   
#'   # Check positive argument
#'   if (!is.character(positive) || !grepl("^(bad|good)\\|1$", positive)) {
#'     stop("'positive' must be either 'bad|1' or 'good|1'")
#'   }
#'   
#'   # If all checks pass, the function will return silently
#' }

#' #' Select Best Binning Model
#' #'
#' #' @description
#' #' This function selects the best binning model by testing multiple methods and comparing their performance.
#' #'
#' #' @param dt A data.table containing the dataset.
#' #' @param target The name of the target variable.
#' #' @param feature The name of the feature to bin.
#' #' @param min_bin Minimum number of bins.
#' #' @param max_bin Maximum number of bins.
#' #' @param control A list of additional control parameters.
#' #'
#' #' @return A list containing the best binning result and additional information.
#' #'
#' #' @keywords internal
#' OptimalBinningSelectBestModel <- function(dt, target, feature, min_bin, max_bin, control) {
#'   # Define methods for categorical and numerical data
#'   categorical_methods <- c("cm", "dplc", "gmb", "ldb", "mba", "mblp", 
#'                            "milp", "mob", "obnp", "swb", "udt")
#'   
#'   numerical_methods <- c("cm", "dplc", "gmb", "ldb", "lpdb", "mba", 
#'                          "mblp", "milp", "mob", "obnp", "swb", "udt", "bb",
#'                          "bs", "dpb", "eb", "eblc", "efb", "ewb", "ir", 
#'                          "jnbo", "kmb", "mdlp", "mrblp", "plaob", "qb", "sbb", "ubsd")
#'   
#'   is_categorical <- is.factor(dt[[feature]]) || is.character(dt[[feature]])
#'   
#'   # Select appropriate methods based on data type
#'   methods <- if(is_categorical) categorical_methods else numerical_methods
#'   
#'   best_method <- NULL
#'   best_iv <- -Inf
#'   best_result <- NULL
#'   best_monotonic <- FALSE
#'   best_bins <- Inf
#'   
#'   # Create data.table to store reports
#'   best_model_report <- data.table::data.table(
#'     model_name = character(),
#'     model_method = character(),
#'     total_iv = numeric(),
#'     bin_counts = integer()
#'   )
#'   
#'   for (method in methods) {
#'     tryCatch({
#'       algo_info <- OptimalBinningSelectAlgorithm(feature = feature, method, dt, min_bin, max_bin, control)
#'       binning_result <- do.call(algo_info$algorithm, c(list(target = dt[[target]], feature = dt[[feature]]), algo_info$params))
#'       
#'       if (length(binning_result$woebin$bin) >= 2) {
#'         model_name <- algo_info$algorithm
#'         current_iv <- sum(binning_result$woebin$iv, na.rm = TRUE)
#'         is_monotonic <- is_woe_monotonic(binning_result$woebin$woe)
#'         current_bins <- length(binning_result$woebin$bin)
#'         
#'         # Add result to report
#'         best_model_report <- data.table::rbindlist(
#'           list(best_model_report,
#'                      data.table::data.table(model_name = model_name,
#'                                             model_method = method,
#'                                             total_iv = current_iv,
#'                                             bin_counts = current_bins
#'                                             )))
#'         
#'         # Check if current result is better based on criteria
#'         if (current_iv > best_iv ||
#'             (current_iv == best_iv && is_monotonic && !best_monotonic) ||
#'             (current_iv == best_iv && is_monotonic == best_monotonic && current_bins < best_bins)) {
#'           best_iv <- current_iv
#'           best_method <- method
#'           best_result <- binning_result
#'           best_monotonic <- is_monotonic
#'           best_bins <- current_bins
#'         }
#'       }
#'     }, error = function(e) {
#'       warning(paste("Error processing method:", method, "for feature:", feature, "-", e$message))
#'     })
#'   }
#'   
#'   if (is.null(best_method)) {
#'     stop(paste("No suitable method found for feature:", feature))
#'   }
#'   
#'   data.table::setorder(best_model_report, -total_iv)
#'   best_result$best_method <- best_method
#'   best_result$is_categorical <- is_categorical
#'   best_result$is_monotonic <- best_monotonic
#'   best_result$num_bins <- best_bins
#'   best_result$iv <- best_iv
#'   best_result$best_model_report <- best_model_report
#'   return(best_result)
#' }

# # Helper function to check WoE monotonicity
# is_woe_monotonic <- function(woes) {
#   diffs <- diff(woes)
#   all(diffs >= 0) || all(diffs <= 0)
# }
# 
# # Function to check date and datetime
# Rcpp::cppFunction('
#   bool is_date_format_cpp(const std::string& date_string) {
#     if (date_string.length() < 8 || date_string.length() > 19) return false;
#     
#     int separators = 0;
#     bool has_time = false;
#     
#     for (char c : date_string) {
#       if (c == \'-\' || c == \'/\' || c == \':\') {
#         separators++;
#       } else if (c == \' \' || c == \'T\') {
#         has_time = true;
#       } else if (c < \'0\' || c > \'9\') {
#         return false;
#       }
#     }
#     return (separators == 2 || (separators >= 4 && has_time));
#   }
# ')
# 
# # Wrapper function for use in data.table
# is_date_time_dt <- function(x) {
#   if (is.character(x) || is.factor(x)) {
#     return(is_date_format_cpp(as.character(x[1])))
#   }
#   return(inherits(x, "Date") || inherits(x, "POSIXct") || inherits(x, "POSIXlt"))
# }
