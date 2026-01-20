#' @title Data Preprocessor for Optimal Binning
#'
#' @description
#' Prepares features for optimal binning by handling missing values and optionally
#' detecting/treating outliers. Supports both numerical and categorical variables
#' with configurable preprocessing strategies.
#'
#' @param feature Vector (numeric, character, or factor) to be preprocessed.
#'   Type is automatically detected.
#' @param target Numeric or integer vector of binary target values (0/1). Must
#'   have the same length as \code{feature}. Used for validation but not directly
#'   in preprocessing.
#' @param num_miss_value Numeric value to replace missing (\code{NA}) values in
#'   numerical features (default: \code{-999.0}). Choose a value outside the
#'   expected range of the feature.
#' @param char_miss_value Character string to replace missing (\code{NA}) values
#'   in categorical features (default: \code{"N/A"}).
#' @param outlier_method Character string specifying the outlier detection method
#'   for numerical features (default: \code{"iqr"}). Options:
#'   \itemize{
#'     \item \code{"iqr"}: Interquartile Range method. Outliers are values
#'       \eqn{< Q_1 - k \times IQR} or \eqn{> Q_3 + k \times IQR}.
#'     \item \code{"zscore"}: Z-score method. Outliers are values with
#'       \eqn{|z| > \text{threshold}} where \eqn{z = (x - \mu) / \sigma}.
#'     \item \code{"grubbs"}: Grubbs' test for outliers (iterative). Removes
#'       the most extreme value if it exceeds the critical G-statistic at
#'       significance level \code{grubbs_alpha}.
#'   }
#' @param outlier_process Logical flag to enable outlier detection and treatment
#'   (default: \code{FALSE}). Only applies to numerical features.
#' @param preprocess Character vector specifying output components (default: \code{"both"}):
#'   \itemize{
#'     \item \code{"feature"}: Return preprocessed feature data only.
#'     \item \code{"report"}: Return preprocessing report only (summary statistics,
#'       counts).
#'     \item \code{"both"}: Return both preprocessed data and report.
#'   }
#' @param iqr_k Multiplier for the IQR method (default: 1.5). Larger values are
#'   more conservative (fewer outliers). Common values: 1.5 (standard), 3.0 (extreme).
#' @param zscore_threshold Z-score threshold for outlier detection (default: 3.0).
#'   Values with \eqn{|z| > \text{threshold}} are considered outliers.
#' @param grubbs_alpha Significance level for Grubbs' test (default: 0.05).
#'   Lower values are more conservative (fewer outliers detected).
#'
#' @return A list with up to two elements (depending on \code{preprocess}):
#' \describe{
#'   \item{preprocess}{Data frame with columns:
#'     \itemize{
#'       \item \code{feature}: Original feature values.
#'       \item \code{feature_preprocessed}: Preprocessed feature values (NAs replaced,
#'         outliers capped or removed).
#'     }
#'   }
#'   \item{report}{Data frame with one row containing:
#'     \itemize{
#'       \item \code{variable_type}: \code{"numeric"} or \code{"categorical"}.
#'       \item \code{missing_count}: Number of \code{NA} values replaced.
#'       \item \code{outlier_count}: Number of outliers detected (numeric only,
#'         \code{NA} for categorical).
#'       \item \code{original_stats}: String representation of summary statistics
#'         before preprocessing (min, Q1, median, mean, Q3, max for numeric).
#'       \item \code{preprocessed_stats}: Summary statistics after preprocessing.
#'     }
#'   }
#' }
#'
#' @details
#' \strong{Preprocessing Pipeline}:
#'
#' \enumerate{
#'   \item \strong{Type Detection}: Automatically classifies \code{feature} as
#'     numeric or categorical based on R type.
#'   \item \strong{Missing Value Handling}: Replaces \code{NA} with
#'     \code{num_miss_value} (numeric) or \code{char_miss_value} (categorical).
#'   \item \strong{Outlier Detection} (if \code{outlier_process = TRUE} for numeric):
#'     \itemize{
#'       \item \strong{IQR Method}: Caps outliers at boundaries
#'         \eqn{[Q_1 - k \times IQR, Q_3 + k \times IQR]}.
#'       \item \strong{Z-score Method}: Caps outliers at
#'         \eqn{[\mu - t \times \sigma, \mu + t \times \sigma]}.
#'       \item \strong{Grubbs' Test}: Iteratively removes the most extreme value
#'         if \eqn{G = \frac{\max|x_i - \bar{x}|}{s} > G_{\text{critical}}}.
#'     }
#'   \item \strong{Summary Calculation}: Computes statistics before and after
#'     preprocessing for validation.
#' }
#'
#' \strong{Outlier Treatment Strategies}:
#' \itemize{
#'   \item IQR and Z-score: \strong{Winsorization} (capping at boundaries).
#'   \item Grubbs: \strong{Removal} (replaced with \code{num_miss_value}).
#' }
#'
#' \strong{Use Cases}:
#' \itemize{
#'   \item \strong{Before binning}: Stabilize binning algorithms by removing
#'     extreme values that could create singleton bins.
#'   \item \strong{Data quality audit}: Identify features with excessive missingness
#'     or outliers.
#'   \item \strong{Model deployment}: Ensure test data undergoes identical
#'     preprocessing as training data.
#' }
#'
#' @examples
#' \donttest{
#' # Numerical feature with outliers
#' set.seed(123)
#' feature_num <- c(rnorm(95, 50, 10), NA, NA, 200, -100, 250)
#' target <- sample(0:1, 100, replace = TRUE)
#'
#' # Preprocess with IQR outlier detection
#' result_iqr <- ob_preprocess(
#'   feature = feature_num,
#'   target = target,
#'   outlier_process = TRUE,
#'   outlier_method = "iqr",
#'   iqr_k = 1.5
#' )
#'
#' print(result_iqr$report)
#' # Shows: missing_count = 2, outlier_count = 3
#'
#' # Categorical feature
#' feature_cat <- c(rep("A", 30), rep("B", 40), rep("C", 28), NA, NA)
#' target_cat <- sample(0:1, 100, replace = TRUE)
#'
#' result_cat <- ob_preprocess(
#'   feature = feature_cat,
#'   target = target_cat,
#'   char_miss_value = "Missing"
#' )
#'
#' # Compare original vs preprocessed
#' head(result_cat$preprocess)
#' # Shows NA replaced with "Missing"
#'
#' # Return only report (no data)
#' result_report <- ob_preprocess(
#'   feature = feature_num,
#'   target = target,
#'   preprocess = "report",
#'   outlier_process = TRUE
#' )
#'
#' # Grubbs' test (most conservative)
#' result_grubbs <- ob_preprocess(
#'   feature = feature_num,
#'   target = target,
#'   outlier_process = TRUE,
#'   outlier_method = "grubbs",
#'   grubbs_alpha = 0.01 # Very strict
#' )
#' }
#'
#' @references
#' \itemize{
#'   \item Grubbs, F. E. (1950). "Sample Criteria for Testing Outlying Observations".
#'     \emph{Annals of Mathematical Statistics}, 21(1), 27-58.
#'   \item Tukey, J. W. (1977). \emph{Exploratory Data Analysis}. Addison-Wesley.
#'     [IQR method]
#' }
#'
#' @export
ob_preprocess <- function(feature,
                          target,
                          num_miss_value = -999.0,
                          char_miss_value = "N/A",
                          outlier_method = "iqr",
                          outlier_process = FALSE,
                          preprocess = "both",
                          iqr_k = 1.5,
                          zscore_threshold = 3.0,
                          grubbs_alpha = 0.05) {
  if (length(feature) == 0) {
    stop("feature cannot be empty.")
  }

  if (!is.vector(target) || !(is.integer(target) || is.numeric(target))) {
    stop("target must be an integer or numeric vector.")
  }

  if (length(feature) != length(target)) {
    stop("feature and target must have the same length.")
  }

  target <- as.numeric(target)

  unique_target <- unique(target[!is.na(target)])
  if (length(unique_target) != 2 || !all(unique_target %in% c(0, 1))) {
    stop("target must be binary (contain only 0 and 1, excluding NAs).")
  }

  if (!is.numeric(num_miss_value)) {
    stop("num_miss_value must be numeric.")
  }

  if (!is.character(char_miss_value) || length(char_miss_value) != 1) {
    stop("char_miss_value must be a single character string.")
  }

  valid_methods <- c("iqr", "zscore", "grubbs")
  if (!outlier_method %in% valid_methods) {
    stop(paste("outlier_method must be one of:", paste(valid_methods, collapse = ", ")))
  }

  if (!is.logical(outlier_process) || length(outlier_process) != 1) {
    stop("outlier_process must be a single logical value (TRUE or FALSE).")
  }

  valid_preprocess <- c("feature", "report", "both")
  if (length(preprocess) != 1 || !preprocess %in% valid_preprocess) {
    stop(paste("preprocess must be one of:", paste(valid_preprocess, collapse = ", ")))
  }

  if (!is.numeric(iqr_k) || iqr_k <= 0) {
    stop("iqr_k must be a positive numeric value.")
  }

  if (!is.numeric(zscore_threshold) || zscore_threshold <= 0) {
    stop("zscore_threshold must be a positive numeric value.")
  }

  if (!is.numeric(grubbs_alpha) || grubbs_alpha <= 0 || grubbs_alpha >= 1) {
    stop("grubbs_alpha must be in the range (0, 1).")
  }

  preprocess_char <- as.character(preprocess)

  .Call(
    "_OptimalBinningWoE_OBDataPreprocessor",
    target,
    feature,
    num_miss_value,
    char_miss_value,
    outlier_method,
    outlier_process,
    preprocess_char,
    iqr_k,
    zscore_threshold,
    grubbs_alpha,
    PACKAGE = "OptimalBinningWoE"
  )
}
