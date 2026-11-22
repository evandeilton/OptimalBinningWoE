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
#' The method combines KLL Sketch for candidate generation (every distinct value for
#' N <= 50) with an exact Dynamic Programming selection of the cutpoints, followed by
#' monotonicity enforcement via the Pool Adjacent Violators Algorithm (PAVA).
#'
#' @param feature Numeric vector of feature values. Missing values (NA/NaN) are
#'   dropped silently: those rows are counted in no bin and the counts add up to the
#'   number of non-missing rows. \code{-Inf} and \code{+Inf} are kept as extreme values
#'   in the first and last bin and never become cutpoints. A feature whose values are all
#'   missing is an error. The sketch
#'   summarises the finite values; with fewer than two distinct finite values a single
#'   bin is returned.
#' @param target Integer vector of binary target values (must contain only 0 and 1).
#'   Must have the same length as \code{feature}. Missing values are not permitted.
#' @param min_bins Minimum number of bins (default: 3). Must be at least 2.
#' @param max_bins Maximum number of bins (default: 5). Must be >= \code{min_bins}. It is a
#'   ceiling, not a target: the number of bins is chosen by BIC in
#'   \code{[min_bins, max_bins]} (see Details, Phase 3).
#' @param bin_cutoff Minimum fraction of total observations per bin (default: 0.05).
#'   Must be in (0, 1). Bins with fewer observations will be merged with neighbors.
#' @param max_n_prebins Accepted for compatibility with the other binning functions
#'   (default: 20; must be in [2, 1000]). The candidate grid is a fixed set of about 40
#'   sketch quantiles.
#' @param monotonic Logical flag to enforce WoE monotonicity (default: TRUE). Uses
#'   a monotone constraint inside the cutpoint optimization, then pools adjacent
#'   violators. Direction (increasing/decreasing) is the one with the larger BIC.
#' @param convergence_threshold Convergence threshold (default: 1e-6). Accepted for
#'   compatibility: the selected cutpoints never exceed \code{max_bins - 1}, so no
#'   iterative bin-count reduction is needed.
#' @param max_iterations Maximum iterations for bin optimization (default: 1000).
#'   Prevents infinite loops in the optimization process.
#' @param sketch_k Integer parameter controlling sketch accuracy (default: 200).
#'   Larger values improve quantile precision but increase memory usage.
#'   \strong{Approximation error}: observed rank error about \eqn{1/k} of the sample
#'   size (200 → about 0.5\%).
#'   \strong{Valid range}: [10, 1000]. Typical values: 50 (fast), 200 (balanced), 500 (precise).
#'
#' @return A list of class \code{c("OptimalBinningSketch", "OptimalBinning", "list")} containing:
#' \describe{
#'   \item{id}{Numeric vector of bin identifiers (1-based indexing).}
#'   \item{bin}{Character vector of right-closed bin labels \code{"(lower;upper]"};
#'     the first bin is labelled from \code{-Inf} and the last one up to \code{+Inf}.}
#'   \item{bin_lower}{Numeric vector of lower bin boundaries: the observed minimum
#'     (included) for the first bin, the previous cutpoint (excluded) otherwise.}
#'   \item{bin_upper}{Numeric vector of upper bin boundaries (included); the observed
#'     maximum for the last bin.}
#'   \item{woe}{Numeric vector of Weight of Evidence values. Monotonic if
#'     \code{monotonic = TRUE}.}
#'   \item{iv}{Numeric vector of Information Value contributions per bin.}
#'   \item{count}{Integer vector of total observations per bin.}
#'   \item{count_pos}{Integer vector of positive class (target = 1) counts per bin.}
#'   \item{count_neg}{Integer vector of negative class (target = 0) counts per bin.}
#'   \item{total_iv}{Total Information Value (sum of bin IVs).}
#'   \item{cutpoints}{Numeric vector of bin split points (length = number of bins - 1).
#'     These are the internal boundaries between bins.}
#'   \item{converged}{Logical flag indicating whether optimization converged.}
#'   \item{monotonic}{Logical. TRUE when the final WoE of the regular bins is monotone
#'     (in the reported direction when one is fixed), checked on the reported WoE and on
#'     the reference WoE \code{log((count_pos + 0.5) / (count_neg + 0.5))}. When
#'     monotonicity was requested and this is FALSE (the merges stop at \code{min_bins}),
#'     a warning is raised. Up to version 1.14.1 nothing signalled it.}
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
#' This implementation compacts deterministically (the kept element of each pair
#' alternates between levels) instead of using KLL's random coin, so results are
#' reproducible without a seed. Its worst-case rank error is \eqn{O(\log(N/k)/k)};
#' the observed error is about \eqn{1/k}. The quantiles only propose candidate
#' cutpoints: every bin statistic is then computed exactly from the data.
#'
#' \strong{Phase 2: Candidate Extraction}
#'
#' Approximately 40 quantiles are extracted from the sketch using a non-uniform grid
#' with higher resolution in distribution tails.
#'
#' \strong{Phase 3: Optimal Cutpoint Selection}
#'
#' An exact Dynamic Programming over the intervals between consecutive candidates
#' (about 40 sketch quantiles; for N <= 50, the boundaries between consecutive distinct
#' values) returns the partition with the largest total IV for each number of bins
#' \eqn{K} between \code{min_bins} and \code{max_bins}. Among them, the one with the
#' largest Schwarz (1978) BIC of the binned event-rate model,
#' \deqn{\mathrm{BIC} = \sum_j \left[ n^+_j \log \hat p_j + n^-_j \log (1 - \hat p_j) \right] - \frac{K - 1}{2} \log N,}
#' is kept (ties: fewer bins): a bin is added only when it raises the log-likelihood
#' by more than \eqn{\log(N) / 2}, so \code{max_bins} is a ceiling, not a target.
#' Picking the \eqn{K} with the largest training IV over-split small samples and rare
#' events. With \code{monotonic = TRUE} the optimization is restricted to partitions
#' with a monotone (smoothed) WoE, in both directions, and the BIC decides between
#' them. The objective is
#' the IV with normalized Jeffreys smoothing (0.5 per bin and class), and every bin must
#' hold at least \code{ceiling(bin_cutoff * N)} observations inside the optimization.
#' (Up to version 1.14.1 the larger datasets used the \code{max_bins - 1} candidates
#' with the largest single-cut IV, which clustered next to each other.) When no number
#' of bins in that range can give every bin that many observations, \code{min_bins} bins
#' as large as possible are used.
#'
#' \strong{Phase 4: Bin Refinement}
#'
#' Bins are refined through frequency constraint enforcement, monotonicity enforcement
#' (if requested). Bins that hold no observation are always merged.
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
  target <- .ob_integer_target(target)

  # Missing feature values (NA/NaN) are dropped silently and +/-Inf are kept
  # as extreme values; both are handled in C++ so obwoe() agrees.

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

  class(result) <- c("OptimalBinningSketch", "OptimalBinning", "list")

  return(result)
}
