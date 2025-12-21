#' Binning Categorical Variables using Custom Cutpoints
#'
#' This function applies user-defined binning to a categorical variable by grouping
#' specified categories into bins and calculating Weight of Evidence (WoE) and
#' Information Value (IV) for each bin.
#'
#' The function takes a character vector defining how categories should be grouped.
#' Each element in the \code{cutpoints} vector defines one bin by listing the
#' original categories that should be merged, separated by "+" signs.
#'
#' For example, if you want to create two bins from categories "A", "B", "C", "D":
#' \itemize{
#'   \item Bin 1: "A+B"
#'   \item Bin 2: "C+D"
#' }
#'
#' @param feature A character vector or factor representing the categorical
#'   predictor variable.
#' @param target An integer vector containing binary outcome values (0 or 1).
#'   Must be the same length as \code{feature}.
#' @param cutpoints A character vector where each element defines a bin by
#'   concatenating the original category names with "+" as separator.
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{woefeature}}{Numeric vector of WoE values corresponding to each
#'         observation in the input \code{feature}}
#'   \item{\code{woebin}}{Data frame with one row per bin containing:
#'     \itemize{
#'       \item \code{bin}: The bin definition (original categories joined by "+")
#'       \item \code{count}: Total number of observations in the bin
#'       \item \code{count_pos}: Number of positive outcomes (target=1) in the bin
#'       \item \code{count_neg}: Number of negative outcomes (target=0) in the bin
#'       \item \code{woe}: Weight of Evidence for the bin
#'       \item \code{iv}: Information Value contribution of the bin
#'     }}
#' }
#'
#' @note
#' \itemize{
#'   \item Target variable must contain only 0 and 1 values.
#'   \item Every unique category in \code{feature} must be included in exactly
#'         one bin definition in \code{cutpoints}.
#'   \item Categories not mentioned in \code{cutpoints} will be assigned to bin 0
#'         (which may lead to unexpected results).
#' }
#'
#' @examples
#' # Sample data
#' feature <- c("A", "B", "C", "D", "A", "B", "C", "D")
#' target <- c(1, 0, 1, 0, 1, 1, 0, 0)
#'
#' # Define custom bins: (A,B) and (C,D)
#' cutpoints <- c("A+B", "C+D")
#'
#' # Apply binning
#' result <- ob_cutpoints_cat(feature, target, cutpoints)
#'
#' # View bin statistics
#' print(result$woebin)
#'
#' # View WoE-transformed feature
#' print(result$woefeature)
#'
#' @export
ob_cutpoints_cat <- function(feature, target, cutpoints) {
  .Call("_OptimalBinningWoE_binning_categorical_cutpoints",
    as.character(feature),
    as.integer(target),
    as.character(cutpoints),
    PACKAGE = "OptimalBinningWoE"
  )
}
