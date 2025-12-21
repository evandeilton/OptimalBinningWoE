#' Binning Numerical Variables using Custom Cutpoints
#'
#' This function applies user-defined binning to a numerical variable by using
#' specified cutpoints to create intervals and calculates Weight of Evidence (WoE)
#' and Information Value (IV) for each interval bin.
#'
#' The function takes a numeric vector of cutpoints that define the boundaries
#' between bins. For \code{n} cutpoints, \code{n+1} bins are created:
#' \itemize{
#'   \item Bin 1: \eqn{(-\infty, cutpoint_1)}
#'   \item Bin 2: \eqn{[cutpoint_1, cutpoint_2)}
#'   \item ...
#'   \item Bin n+1: \eqn{[cutpoint_n, +\infty)}
#' }
#'
#' @param feature A numeric vector representing the continuous predictor variable.
#' @param target An integer vector containing binary outcome values (0 or 1).
#'   Must be the same length as \code{feature}.
#' @param cutpoints A numeric vector of cutpoints that define bin boundaries.
#'   These will be automatically sorted in ascending order.
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{woefeature}}{Numeric vector of WoE values corresponding to each
#'         observation in the input \code{feature}}
#'   \item{\code{woebin}}{Data frame with one row per bin containing:
#'     \itemize{
#'       \item \code{bin}: The bin interval notation (e.g., "[10.00;20.00)")
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
#'   \item Cutpoints are sorted automatically in ascending order.
#'   \item Interval notation uses "[" for inclusive and ")" for exclusive bounds.
#'   \item Infinite values in feature are handled appropriately.
#' }
#'
#' @examples
#' # Sample data
#' feature <- c(5, 15, 25, 35, 45, 55, 65, 75)
#' target <- c(0, 0, 1, 1, 1, 1, 0, 0)
#'
#' # Define custom cutpoints
#' cutpoints <- c(30, 60)
#'
#' # Apply binning
#' result <- ob_cutpoints_num(feature, target, cutpoints)
#'
#' # View bin statistics
#' print(result$woebin)
#'
#' # View WoE-transformed feature
#' print(result$woefeature)
#'
#' @export
ob_cutpoints_num <- function(feature, target, cutpoints) {
  .Call("_OptimalBinningWoE_binning_numerical_cutpoints",
    as.numeric(feature),
    as.integer(target),
    as.numeric(cutpoints),
    PACKAGE = "OptimalBinningWoE"
  )
}
