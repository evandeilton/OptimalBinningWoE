#' Compute Multiple Robust Correlations Between Numeric Variables
#'
#' This function computes various correlation coefficients between all pairs of
#' numeric variables in a data frame. It implements several classical and robust
#' correlation measures, including Pearson, Spearman, Kendall, Hoeffding's D,
#' Distance Correlation, Biweight Midcorrelation, and Percentage Bend correlation.
#'
#' The function supports multiple correlation methods simultaneously and utilizes
#' OpenMP for parallel computation when available.
#'
#' Available correlation methods:
#' \itemize{
#'   \item \strong{Pearson}: Standard linear correlation coefficient.
#'   \item \strong{Spearman}: Rank-based correlation coefficient.
#'   \item \strong{Kendall}: Kendall's tau-b correlation coefficient.
#'   \item \strong{Hoeffding}: Hoeffding's D statistic (scaled by 30).
#'   \item \strong{Distance}: Distance correlation (Székely et al., 2007).
#'   \item \strong{Biweight}: Biweight midcorrelation (robust alternative).
#'   \item \strong{Pbend}: Percentage bend correlation (robust alternative).
#' }
#'
#' @param df A data frame containing numeric variables. Non-numeric columns will
#'   be automatically excluded. At least two numeric variables are required.
#' @param method A character string specifying which correlation method(s) to compute.
#'   Possible values are:
#'   \itemize{
#'     \item \code{"all"}: Compute all available correlation methods (default).
#'     \item \code{"pearson"}: Compute only Pearson correlation.
#'     \item \code{"spearman"}: Compute only Spearman correlation.
#'     \item \code{"kendall"}: Compute only Kendall correlation.
#'     \item \code{"hoeffding"}: Compute only Hoeffding's D.
#'     \item \code{"distance"}: Compute only distance correlation.
#'     \item \code{"biweight"}: Compute only biweight midcorrelation.
#'     \item \code{"pbend"}: Compute only percentage bend correlation.
#'     \item \code{"robust"}: Compute robust correlations (biweight and pbend).
#'     \item \code{"alternative"}: Compute alternative correlations (hoeffding and distance).
#'   }
#' @param threads An integer specifying the number of threads to use for parallel
#'   computation. If 0 (default), uses all available cores. Ignored if OpenMP
#'   is not available.
#'
#' @return A data frame with the following columns:
#' \describe{
#'   \item{\code{x}, \code{y}}{Names of the variable pairs being correlated.}
#'   \item{\code{pearson}}{Pearson correlation coefficient.}
#'   \item{\code{spearman}}{Spearman rank correlation coefficient.}
#'   \item{\code{kendall}}{Kendall's tau-b correlation coefficient.}
#'   \item{\code{hoeffding}}{Hoeffding's D statistic (scaled).}
#'   \item{\code{distance}}{Distance correlation coefficient.}
#'   \item{\code{biweight}}{Biweight midcorrelation coefficient.}
#'   \item{\code{pbend}}{Percentage bend correlation coefficient.}
#' }
#' The exact columns returned depend on the \code{method} parameter.
#'
#' @note
#' \itemize{
#'   \item Missing values (NA) are handled appropriately for each correlation method.
#'   \item For robust methods (biweight, pbend), fallback to Pearson correlation
#'     occurs when there are insufficient data points or numerical instability.
#'   \item Hoeffding's D requires at least 5 complete pairs.
#'   \item Distance correlation is computed without forming NxN distance matrices
#'     for memory efficiency.
#'   \item When OpenMP is available, computations are automatically parallelized
#'     across variable pairs.
#' }
#'
#' @references
#' Székely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007). Measuring and testing
#' dependence by correlation of distances. The Annals of Statistics, 35(6), 2769-2794.
#'
#' Wilcox, R.R. (1994). The percentage bend correlation coefficient.
#' Psychometrika, 59(4), 601-616.
#'
#' @examples
#' # Create sample data
#' set.seed(123)
#' n <- 100
#' df <- data.frame(
#'   x1 = rnorm(n),
#'   x2 = rnorm(n),
#'   x3 = rt(n, df = 3), # Heavy-tailed distribution
#'   x4 = sample(c(0, 1), n, replace = TRUE), # Binary variable
#'   category = sample(letters[1:3], n, replace = TRUE) # Non-numeric column
#' )
#'
#' # Add some relationships
#' df$x2 <- df$x1 + rnorm(n, 0, 0.5)
#' df$x3 <- df$x1^2 + rnorm(n, 0, 0.5)
#'
#' # Compute all correlations
#' result_all <- obcorr(df)
#' head(result_all)
#'
#' # Compute only robust correlations
#' result_robust <- obcorr(df, method = "robust")
#'
#' # Compute only Pearson correlation with 2 threads
#' result_pearson <- obcorr(df, method = "pearson", threads = 2)
#'
#' @export
obcorr <- function(df, method = "all", threads = 0L) {
  .Call("_OptimalBinningWoE_obcorr", df, method, as.integer(threads), PACKAGE = "OptimalBinningWoE")
}
