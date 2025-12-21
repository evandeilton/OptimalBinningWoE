#' Optimal Binning for Multiclass Targets using JEDI M-WOE
#'
#' Performs supervised discretization of continuous numerical variables for
#' \strong{multiclass} target variables (e.g., 0, 1, 2). It extends the Joint
#' Entropy-Driven Interval (JEDI) discretization framework to calculate and optimize
#' the Multinomial Weight of Evidence (M-WOE) for each class simultaneously.
#'
#' @param feature A numeric vector representing the continuous predictor variable.
#'   Missing values (NA) should be excluded prior to execution.
#' @param target An integer vector of multiclass outcomes (0, 1, ..., K-1)
#'   corresponding to each observation in \code{feature}. Must have at least 2 distinct classes.
#' @param min_bins Integer. The minimum number of bins to produce. Must be \eqn{\ge} 2.
#'   Defaults to 3.
#' @param max_bins Integer. The maximum number of bins to produce. Must be \eqn{\ge}
#'   \code{min_bins}. Defaults to 5.
#' @param bin_cutoff Numeric. The minimum fraction of total observations required
#'   for a bin to be considered valid. Bins smaller than this threshold are merged.
#'   Defaults to 0.05.
#' @param max_n_prebins Integer. The number of initial quantiles to generate
#'   during the pre-binning phase. Defaults to 20.
#' @param convergence_threshold Numeric. The threshold for the change in total
#'   Multinomial IV to determine convergence. Defaults to 1e-6.
#' @param max_iterations Integer. Safety limit for the maximum number of iterations.
#'   Defaults to 1000.
#'
#' @return A list containing the binning results:
#'   \itemize{
#'     \item \code{id}: Integer vector of bin identifiers.
#'     \item \code{bin}: Character vector of bin labels in interval notation.
#'     \item \code{woe}: A numeric matrix where each column represents the WoE
#'           for a specific class (One-vs-Rest).
#'     \item \code{iv}: A numeric matrix where each column represents the IV contribution
#'           for a specific class.
#'     \item \code{count}: Integer vector of total observations per bin.
#'     \item \code{class_counts}: A matrix of observation counts per class per bin.
#'     \item \code{cutpoints}: Numeric vector of upper boundaries (excluding Inf).
#'     \item \code{n_classes}: The number of distinct target classes found.
#'   }
#'
#' @details
#' \strong{Multinomial Weight of Evidence (M-WOE):}
#' For a target with \eqn{K} classes, the WoE for class \eqn{k} in bin \eqn{i} is defined
#' using a "One-vs-Rest" approach:
#' \deqn{WOE_{i,k} = \ln\left(\frac{P(X \in bin_i | Y=k)}{P(X \in bin_i | Y \neq k)}\right)}
#'
#' \strong{Algorithm Workflow:}
#' \enumerate{
#'   \item \strong{Multiclass Initialization:} The algorithm starts with quantile-based bins
#'         and computes the initial event rates for all \eqn{K} classes.
#'   \item \strong{Joint Monotonicity:} The algorithm attempts to enforce monotonicity for
#'         \emph{all} classes. If bin \eqn{i} violates the trend for Class 1 OR Class 2,
#'         it may be merged. This ensures the variable is predictive across the entire
#'         spectrum of outcomes.
#'   \item \strong{Global IV Optimization:} When reducing the number of bins to \code{max_bins},
#'         the algorithm merges the pair of bins that minimizes the loss of the
#'         \emph{Sum of IVs} across all classes:
#'         \deqn{Loss = \sum_{k=0}^{K-1} \Delta IV_k}
#' }
#'
#' This method is ideal for use cases like:
#' \itemize{
#'   \item predicting loan status (Current, Late, Default)
#'   \item customer churn levels (Active, Dormant, Churned)
#'   \item ordinal survey responses.
#' }
#'
#' @seealso \code{\link{ob_numerical_jedi}} for the binary version.
#'
#' @examples
#' # Example: Multiclass target (0, 1, 2)
#' set.seed(123)
#' feature <- rnorm(1000)
#' # Class 0: low feature, Class 1: medium, Class 2: high
#' target <- cut(feature + rnorm(1000, 0, 0.5),
#'   breaks = c(-Inf, -0.5, 0.5, Inf),
#'   labels = FALSE
#' ) - 1
#'
#' result <- ob_numerical_jedi_mwoe(feature, target,
#'   min_bins = 3,
#'   max_bins = 5
#' )
#'
#' # Check WoE for Class 2 (High values)
#' print(result$woe[, 3]) # Column 3 corresponds to Class 2
#'
#' @export
ob_numerical_jedi_mwoe <- function(feature, target, min_bins = 3, max_bins = 5,
                                   bin_cutoff = 0.05, max_n_prebins = 20,
                                   convergence_threshold = 1e-6, max_iterations = 1000) {
  # Type Validation
  if (!is.numeric(feature)) {
    warning("Feature converted to numeric for processing.")
    feature <- as.numeric(feature)
  }

  if (!is.integer(target)) {
    target <- as.integer(target)
  }

  # Dimension Check
  if (length(feature) != length(target)) {
    stop("Length of 'feature' and 'target' must match.")
  }

  # Class Check
  if (length(unique(target)) < 2) {
    stop("Target must contain at least 2 distinct classes.")
  }

  # .Call Interface
  .Call("_OptimalBinningWoE_optimal_binning_numerical_jedi_mwoe",
    target,
    feature,
    as.integer(min_bins),
    as.integer(max_bins),
    as.numeric(bin_cutoff),
    as.integer(max_n_prebins),
    as.numeric(convergence_threshold),
    as.integer(max_iterations),
    PACKAGE = "OptimalBinningWoE"
  )
}
