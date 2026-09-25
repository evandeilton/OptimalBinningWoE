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
#'       \item \code{id}: Sequential bin identifier
#'       \item \code{bin}: The bin definition -- original categories joined by
#'         \code{"\%;\%"} (\strong{not} the "+" used in the \code{cutpoints}
#'         input; see Details), matching the separator
#'         \code{\link{ob_apply_woe_cat}} defaults to and every
#'         \code{ob_categorical_*()} algorithm in the main pipeline emits
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
#'   \item Target variable must contain only 0 and 1 values (no \code{NA}),
#'         with both classes present, and have the same length as
#'         \code{feature}; otherwise an error is raised.
#'   \item Every unique category in \code{feature} must be included in exactly
#'         one bin definition in \code{cutpoints}. A category of
#'         \code{feature} that no bin lists, or a category listed in two bins,
#'         is an error (unlisted categories used to be counted silently in the
#'         first bin). \code{NA} values of \code{feature} are matched as the
#'         category \code{"NA"}, the token the \code{ob_categorical_*()}
#'         wrappers use for missing values.
#' }
#'
#' @details
#' \code{cutpoints} still uses \code{"+"} as the input separator (simple to type,
#' e.g. \code{"A+B"}), but \code{result$woebin} is built so it can be handed
#' straight back to \code{\link{ob_apply_woe_cat}} with its defaults --
#' \code{ob_apply_woe_cat(result$woebin, new_feature)} -- and get the same WoE
#' this function itself assigned. Before 1.13.1 the emitted \code{bin} labels
#' echoed the "+"-joined input verbatim and carried no \code{id} column, so a
#' round trip through \code{\link{ob_apply_woe_cat}}'s default
#' \code{bin_separator = "\%;\%"} matched no category and silently fell back to
#' a \code{"Special"}/\code{NA} bin for every observation.
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
#' # Round-trip through ob_apply_woe_cat() with its defaults
#' woe_new <- ob_apply_woe_cat(result$woebin, feature)
#' stopifnot(isTRUE(all.equal(woe_new$woe, result$woefeature)))
#'
#' @export
ob_cutpoints_cat <- function(feature, target, cutpoints) {
  if (length(feature) != length(target)) {
    stop("'feature' and 'target' must have the same length.")
  }
  if (!(is.numeric(target) || is.logical(target)) || anyNA(target) ||
    !all(target %in% c(0, 1))) {
    stop("'target' must contain only 0 and 1 (no missing values).")
  }
  .Call("_OptimalBinningWoE_binning_categorical_cutpoints",
    as.character(feature),
    as.integer(target),
    as.character(cutpoints),
    PACKAGE = "OptimalBinningWoE"
  )
}
