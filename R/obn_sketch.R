#' @title Optimal Binning for Numerical Variables using Sketch-based Algorithm
#'
#' @description
#' Implements optimal binning using the **KLL Sketch** (Karnin, Lang, Liberty, 2016),
#' a probabilistic data structure for quantile approximation in data streams. This is
#' the \strong{only method in the package} that uses a fundamentally different algorithmic
#' approach (streaming algorithms) compared to batch processing methods (MOB, MDLP, etc.).
#'
#' The sketch-based approach enables:
#' \itemize{
#'   \item \strong{Sublinear space complexity}: O(k log N) vs O(N) for batch methods
#'   \item \strong{Single-pass processing}: Suitable for streaming data
#'   \item \strong{Provable approximation guarantees}: Quantile error \eqn{\epsilon \approx O(1/k)}
#' }
#'
#' The method combines KLL Sketch for candidate generation with either Dynamic Programming
#' (for small N <= 50) or greedy IV-based selection (for larger datasets), followed by
#' monotonicity enforcement via the Pool Adjacent Violators Algorithm (PAVA).
#'
#' @param feature Numeric vector of feature values. Missing values (NA) are \strong{not
#'   permitted} and will trigger an error. Infinite values (Inf, -Inf) and NaN are also
#'   not allowed.
#' @param target Integer vector of binary target values (must contain only 0 and 1).
#'   Must have the same length as \code{feature}. Missing values are not permitted.
#' @param min_bins Minimum number of bins (default: 3). Must be at least 2.
#' @param max_bins Maximum number of bins (default: 5). Must be >= \code{min_bins}.
#' @param bin_cutoff Minimum fraction of total observations per bin (default: 0.05).
#'   Must be in (0, 1). Bins with fewer observations will be merged with neighbors.
#' @param max_n_prebins Maximum number of pre-bins to generate from quantiles (default: 20).
#'   This parameter controls the initial granularity of binning candidates. Higher values
#'   provide more flexibility but increase computational cost.
#' @param monotonic Logical flag to enforce WoE monotonicity (default: TRUE). Uses
#'   PAVA (Pool Adjacent Violators Algorithm) for enforcement. Direction (increasing/
#'   decreasing) is automatically detected from the data.
#' @param convergence_threshold Convergence threshold for IV change (default: 1e-6).
#'   Optimization stops when the change in total IV between iterations falls below this value.
#' @param max_iterations Maximum iterations for bin optimization (default: 1000).
#'   Prevents infinite loops in the optimization process.
#' @param sketch_k Integer parameter controlling sketch accuracy (default: 200).
#'   Larger values improve quantile precision but increase memory usage.
#'   \strong{Approximation error}: \eqn{\epsilon \approx 1/k} (200 → 0.5\% error).
#'   \strong{Valid range}: [10, 1000]. Typical values: 50 (fast), 200 (balanced), 500 (precise).
#'
#' @return A list of class \code{c("OptimalBinningSketch", "OptimalBinning")} containing:
#' \describe{
#'   \item{id}{Numeric vector of bin identifiers (1-based indexing).}
#'   \item{bin_lower}{Numeric vector of lower bin boundaries (inclusive).}
#'   \item{bin_upper}{Numeric vector of upper bin boundaries (inclusive for last bin,
#'     exclusive for others).}
#'   \item{woe}{Numeric vector of Weight of Evidence values. Monotonic if
#'     \code{monotonic = TRUE}.}
#'   \item{iv}{Numeric vector of Information Value contributions per bin.}
#'   \item{count}{Integer vector of total observations per bin.}
#'   \item{count_pos}{Integer vector of positive class (target = 1) counts per bin.}
#'   \item{count_neg}{Integer vector of negative class (target = 0) counts per bin.}
#'   \item{cutpoints}{Numeric vector of bin split points (length = number of bins - 1).
#'     These are the internal boundaries between bins.}
#'   \item{converged}{Logical flag indicating whether optimization converged.}
#'   \item{iterations}{Integer number of optimization iterations performed.}
#' }
#'
#' @details
#' \strong{Algorithm Overview}
#'
#' The sketch-based binning algorithm executes in four phases:
#'
#' \strong{Phase 1: KLL Sketch Construction}
#'
#' The KLL Sketch maintains a compressed, multi-level representation of the data distribution:
#'
#' \deqn{\text{Sketch} = \{\text{Compactor}_0, \text{Compactor}_1, \ldots, \text{Compactor}_L\}}
#'
#' where each \eqn{\text{Compactor}_\ell} stores items with weight \eqn{2^\ell}. When a
#' compactor exceeds capacity \eqn{k} (controlled by \code{sketch_k}), it is compacted.
#'
#' \strong{Theoretical Guarantees} (Karnin et al., 2016):
#'
#' For a quantile \eqn{q} with estimated value \eqn{\hat{q}}:
#'
#' \deqn{|\text{rank}(\hat{q}) - q \cdot N| \le \epsilon \cdot N}
#'
#' where \eqn{\epsilon \approx O(1/k)} and space complexity is \eqn{O(k \log(N/k))}.
#'
#' \strong{Phase 2: Candidate Extraction}
#'
#' Approximately 40 quantiles are extracted from the sketch using a non-uniform grid
#' with higher resolution in distribution tails.
#'
#' \strong{Phase 3: Optimal Cutpoint Selection}
#'
#' For small datasets (N <= 50), Dynamic Programming maximizes total IV. For larger
#' datasets, a greedy IV-based selection is used.
#'
#' \strong{Phase 4: Bin Refinement}
#'
#' Bins are refined through frequency constraint enforcement, monotonicity enforcement
#' (if requested), and bin count optimization to minimize IV loss.
#'
#' \strong{Computational Complexity}
#'
#' \itemize{
#'   \item \strong{Time}: \eqn{O(N \log k + N \times C + k^2 \times I)}
#'   \item \strong{Space}: \eqn{O(k \log N)} for large N
#' }
#'
#' \strong{When to Use Sketch-based Binning}
#'
#' \itemize{
#'   \item \strong{Use}: Large datasets (N > 10^6) with memory constraints or streaming data
#'   \item \strong{Avoid}: Small datasets (N < 1000) where approximation error may dominate
#' }
#'
#' @references
#' \itemize{
#'   \item Karnin, Z., Lang, K., & Liberty, E. (2016). "Optimal Quantile Approximation in
#'     Streams". \emph{Proceedings of the 57th Annual IEEE Symposium on Foundations of
#'     Computer Science (FOCS)}, 71-78. \doi{10.1109/FOCS.2016.20}
#'   \item Greenwald, M., & Khanna, S. (2001). "Space-efficient online computation of
#'     quantile summaries". \emph{ACM SIGMOD Record}, 30(2), 58-66.
#'     \doi{10.1145/376284.375670}
#'   \item Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D. (1972).
#'     \emph{Statistical Inference Under Order Restrictions}. Wiley.
#'   \item Siddiqi, N. (2006). \emph{Credit Risk Scorecards: Developing and Implementing
#'     Intelligent Credit Scoring}. Wiley. \doi{10.1002/9781119201731}
#' }
#'
#' @examples
#' \donttest{
#' # Example 1: Basic usage with simulated data
#' set.seed(123)
#' feature <- rnorm(500, mean = 100, sd = 20)
#' target <- rbinom(500, 1, prob = plogis((feature - 100) / 20))
#'
#' result <- ob_numerical_sketch(
#'   feature = feature,
#'   target = target,
#'   min_bins = 3,
#'   max_bins = 5
#' )
#'
#' # Display results
#' print(data.frame(
#'   Bin = result$id,
#'   Count = result$count,
#'   WoE = round(result$woe, 4),
#'   IV = round(result$iv, 4)
#' ))
#'
#' # Example 2: Comparing different sketch_k values
#' set.seed(456)
#' x <- rnorm(1000, 50, 15)
#' y <- rbinom(1000, 1, prob = 0.3)
#'
#' result_k50 <- ob_numerical_sketch(x, y, sketch_k = 50)
#' result_k200 <- ob_numerical_sketch(x, y, sketch_k = 200)
#'
#' cat("K=50 IV:", sum(result_k50$iv), "\n")
#' cat("K=200 IV:", sum(result_k200$iv), "\n")
#' }
#'
#' @author
#' Lopes, J. E.
#'
#' @seealso
#' \code{\link{ob_numerical_mdlp}}, \code{\link{ob_numerical_mblp}}
#'
#' @export
ob_numerical_sketch <- function(feature,
                                target,
                                min_bins = 3,
                                max_bins = 5,
                                bin_cutoff = 0.05,
                                max_n_prebins = 20,
                                monotonic = TRUE,
                                convergence_threshold = 1e-6,
                                max_iterations = 1000,
                                sketch_k = 200) {
  # Input Validation


  # Validate feature type
  if (!is.numeric(feature) && !is.integer(feature)) {
    stop("Argument 'feature' must be a numeric or integer vector.",
      call. = FALSE
    )
  }

  # Validate target type
  if (!is.numeric(target) && !is.integer(target)) {
    stop("Argument 'target' must be a numeric or integer vector.",
      call. = FALSE
    )
  }

  # Check for length mismatch
  if (length(feature) != length(target)) {
    stop(sprintf(
      "Length mismatch: 'feature' has %d elements, 'target' has %d elements.",
      length(feature), length(target)
    ), call. = FALSE)
  }

  # Check for empty inputs
  if (length(feature) == 0) {
    stop("'feature' and 'target' cannot be empty vectors.",
      call. = FALSE
    )
  }

  # Convert to appropriate types
  feature <- as.numeric(feature)
  target <- as.integer(target)

  # Check for missing values in feature
  if (any(is.na(feature))) {
    stop(paste(
      "Missing values (NA) detected in 'feature'.",
      "Please remove or impute missing values before binning."
    ), call. = FALSE)
  }

  # Check for infinite values in feature
  if (any(is.infinite(feature))) {
    stop(paste(
      "Infinite values (Inf/-Inf) detected in 'feature'.",
      "Please remove or cap infinite values before binning."
    ), call. = FALSE)
  }

  # Check for NaN values in feature
  if (any(is.nan(feature))) {
    stop("NaN values detected in 'feature'. Please clean the data before binning.",
      call. = FALSE
    )
  }

  # Check for missing values in target
  if (any(is.na(target))) {
    stop("Missing values (NA) detected in 'target'.", call. = FALSE)
  }

  # Validate target values
  unique_target <- unique(target)
  if (!all(unique_target %in% c(0L, 1L))) {
    stop(sprintf(
      "Target must contain only 0 and 1. Found values: %s",
      paste(setdiff(unique_target, c(0L, 1L)), collapse = ", ")
    ), call. = FALSE)
  }

  if (length(unique_target) != 2L) {
    stop(sprintf(
      "Target must contain both classes (0 and 1). Found only: %s",
      paste(unique_target, collapse = ", ")
    ), call. = FALSE)
  }

  # Validate min_bins
  if (!is.numeric(min_bins) || length(min_bins) != 1 || is.na(min_bins)) {
    stop("Argument 'min_bins' must be a single numeric value.",
      call. = FALSE
    )
  }
  min_bins <- as.integer(min_bins)
  if (min_bins < 2L) {
    stop(sprintf("'min_bins' must be at least 2 (got %d).", min_bins),
      call. = FALSE
    )
  }

  # Validate max_bins
  if (!is.numeric(max_bins) || length(max_bins) != 1 || is.na(max_bins)) {
    stop("Argument 'max_bins' must be a single numeric value.",
      call. = FALSE
    )
  }
  max_bins <- as.integer(max_bins)
  if (max_bins < min_bins) {
    stop(sprintf(
      "'max_bins' (%d) must be greater than or equal to 'min_bins' (%d).",
      max_bins, min_bins
    ), call. = FALSE)
  }

  # Validate bin_cutoff
  if (!is.numeric(bin_cutoff) || length(bin_cutoff) != 1 || is.na(bin_cutoff)) {
    stop("Argument 'bin_cutoff' must be a single numeric value.",
      call. = FALSE
    )
  }
  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    stop(sprintf(
      "'bin_cutoff' must be in the range (0, 1), got %.4f.",
      bin_cutoff
    ), call. = FALSE)
  }

  # Validate max_n_prebins
  if (!is.numeric(max_n_prebins) || length(max_n_prebins) != 1 || is.na(max_n_prebins)) {
    stop("Argument 'max_n_prebins' must be a single numeric value.",
      call. = FALSE
    )
  }
  max_n_prebins <- as.integer(max_n_prebins)
  if (max_n_prebins < 2L || max_n_prebins > 1000L) {
    stop(sprintf(
      "'max_n_prebins' must be in the range [2, 1000], got %d.",
      max_n_prebins
    ), call. = FALSE)
  }

  # Validate monotonic
  if (!is.logical(monotonic) || length(monotonic) != 1 || is.na(monotonic)) {
    stop("Argument 'monotonic' must be a single logical value (TRUE/FALSE).",
      call. = FALSE
    )
  }

  # Validate convergence_threshold
  if (!is.numeric(convergence_threshold) || length(convergence_threshold) != 1 ||
    is.na(convergence_threshold)) {
    stop("Argument 'convergence_threshold' must be a single numeric value.",
      call. = FALSE
    )
  }
  if (convergence_threshold <= 0) {
    stop(sprintf(
      "'convergence_threshold' must be positive, got %.2e.",
      convergence_threshold
    ), call. = FALSE)
  }

  # Validate max_iterations
  if (!is.numeric(max_iterations) || length(max_iterations) != 1 ||
    is.na(max_iterations)) {
    stop("Argument 'max_iterations' must be a single numeric value.",
      call. = FALSE
    )
  }
  max_iterations <- as.integer(max_iterations)
  if (max_iterations < 1L) {
    stop(sprintf("'max_iterations' must be at least 1, got %d.", max_iterations),
      call. = FALSE
    )
  }

  # Validate sketch_k
  if (!is.numeric(sketch_k) || length(sketch_k) != 1 || is.na(sketch_k)) {
    stop("Argument 'sketch_k' must be a single numeric value.",
      call. = FALSE
    )
  }
  sketch_k <- as.integer(sketch_k)
  if (sketch_k < 10L || sketch_k > 1000L) {
    stop(sprintf(
      "'sketch_k' must be in the range [10, 1000], got %d.",
      sketch_k
    ), call. = FALSE)
  }

  # Check for constant feature values
  if (length(unique(feature)) == 1) {
    warning(
      "Feature has constant value. Returning single bin.",
      call. = FALSE,
      immediate. = TRUE
    )
  }


  # Call C++ Implementation


  result <- tryCatch(
    {
      .Call(
        "_OptimalBinningWoE_optimal_binning_numerical_sketch",
        target, # IntegerVector
        feature, # NumericVector
        min_bins, # int
        max_bins, # int
        bin_cutoff, # double
        max_n_prebins, # int
        monotonic, # bool
        convergence_threshold, # double
        max_iterations, # int
        sketch_k, # int
        PACKAGE = "OptimalBinningWoE"
      )
    },
    error = function(e) {
      stop(sprintf(
        "C++ execution failed: %s",
        e$message
      ), call. = FALSE)
    }
  )
  return(result)
}
