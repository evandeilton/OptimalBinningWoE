#' Optimal Binning Data Preprocessor Interface
#'
#' This function serves as an interface to the Rcpp function `OptimalBinningDataPreprocessor`.
#' It preprocesses the data for optimal binning by handling missing values, outliers, and other preprocessing steps.
#'
#' @param target A numeric binary vector (0 and 1) representing the target variable.
#' @param feature A numeric or character vector representing the feature to be preprocessed.
#' @param num_miss_value A numeric value representing the missing value for numeric features. Default is -999.
#' @param char_miss_value A character value representing the missing value for character features. Default is "N/A".
#' @param outlier_method A character string specifying the method for outlier detection. Default is "grubbs".
#' @param outlier_process A logical value indicating whether to process outliers. Default is TRUE.
#' @param preprocess A character vector specifying the preprocessing steps to be performed. Default is "both".
#' @param iqr_k A numeric value specifying the interquartile range multiplier for outlier detection. Default is 1.5.
#' @param zscore_threshold A numeric value specifying the z-score threshold for outlier detection. Default is 3.
#' @param grubbs_alpha A numeric value specifying the significance level for Grubbs' test. Default is 0.05.
#'
#' @return A data.table containing the preprocessed data.
#'
#' @examples
#' \dontrun{
#' # Example usage
#' target <- c(0, 1, 0, 1, 0)
#' feature <- c(1, 2, 3, 4, 5)
#' result <- OptimalBinningDataPreprocessorInterface(target, feature)
#' print(result)
#' }
#' @importFrom data.table data.table
#' @export
OptimalBinningDataPreprocessorInterface <- function(
    target, feature, num_miss_value = -999, char_miss_value = "N/A",
    outlier_method = "grubbs", outlier_process = TRUE, preprocess = as.character(c("both")),
    iqr_k = 1.5, zscore_threshold = 3, grubbs_alpha = 0.05) {
  # Convert input vectors to data.table
  data <- data.table::data.table(target = target, feature = feature)

  # Call the Rcpp function
  result <- OptimalBinningDataPreprocessor(
    target = data$target, feature = data$feature,
    num_miss_value = num_miss_value, char_miss_value = char_miss_value,
    outlier_method = outlier_method, outlier_process = outlier_process,
    preprocess = preprocess, iqr_k = iqr_k, zscore_threshold = zscore_threshold,
    grubbs_alpha = grubbs_alpha
  )

  # Convert the result to data.table
  result_dt <- data.table::as.data.table(feature = data[, get(target)], feature_woe = result$feature_woe)

  return(result_dt)
}
