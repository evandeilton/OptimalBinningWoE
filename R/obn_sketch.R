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
#'   permitted} and will trigger an error. For streaming applications, pre-filter NAs.
#' @param target Integer vector of binary target values (must contain only 0 and 1).
#'   Must have the same length as \code{feature}.
#' @param min_bins Minimum number of bins (default: 3). Must be at least 2.
#' @param max_bins Maximum number of bins (default: 5). Must be >= \code{min_bins}.
#' @param bin_cutoff Minimum fraction of total observations per bin (default: 0.05).
#'   Must be in (0, 1).
#' @param special_codes Character string for special value handling (default: "").
#'   Currently unused; reserved for future extensions.
#' @param monotonic Logical flag to enforce WoE monotonicity (default: TRUE). Uses
#'   PAVA for enforcement.
#' @param convergence_threshold Convergence threshold for IV change (default: 1e-6).
#' @param max_iterations Maximum iterations for bin optimization (default: 1000).
#' @param sketch_k Integer parameter controlling sketch accuracy (default: 200).
#'   Larger values improve quantile precision but increase memory. Typical range: 50-500.
#'   \strong{Approximation error}: \eqn{\epsilon \approx 1/k} (200 → 0.5\% error).
#'
#' @return A list containing:
#' \describe{
#'   \item{id}{Integer bin identifiers (1-based).}
#'   \item{bin_lower}{Numeric lower bounds of bins.}
#'   \item{bin_upper}{Numeric upper bounds of bins.}
#'   \item{woe}{Numeric WoE values (monotonic if \code{monotonic = TRUE}).}
#'   \item{iv}{Numeric IV contributions per bin.}
#'   \item{count}{Integer total observations per bin.}
#'   \item{count_pos}{Integer positive class counts.}
#'   \item{count_neg}{Integer negative class counts.}
#'   \item{cutpoints}{Numeric vector of bin boundaries (internal splits only).}
#'   \item{converged}{Logical convergence flag.}
#'   \item{iterations}{Integer iteration count.}
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
#' compactor exceeds capacity \eqn{k} (controlled by \code{sketch_k}), it is compacted:
#'
#' \enumerate{
#'   \item Sort items in \eqn{\text{Compactor}_\ell}
#'   \item Merge adjacent pairs based on level parity:
#'     \itemize{
#'       \item Even levels (\eqn{\ell \bmod 2 = 0}): Merge pairs at even indices
#'       \item Odd levels (\eqn{\ell \bmod 2 = 1}): Merge pairs at odd indices
#'     }
#'   \item Promote merged items to \eqn{\text{Compactor}_{\ell+1}}
#' }
#'
#' \strong{Theoretical Guarantees} (Karnin et al., 2016):
#'
#' For a quantile \eqn{q} with estimated value \eqn{\hat{q}}:
#'
#' \deqn{|\text{rank}(\hat{q}) - q \cdot N| \le \epsilon \cdot N \quad \text{w.p. } \ge 1 - \delta}
#'
#' where \eqn{\epsilon \approx O(1/k)} and space complexity is \eqn{O(k \log(N/k))}.
#'
#' \strong{Phase 2: Candidate Extraction}
#'
#' Approximately 40 quantiles are extracted from the sketch using a non-uniform grid:
#'
#' \itemize{
#'   \item \strong{Tail regions} (0.01-0.1, 0.9-0.99): 10 quantiles per tail (step 0.01)
#'   \item \strong{Central region} (0.1-0.9): 17 quantiles (step 0.05)
#' }
#'
#' This adaptive grid ensures higher resolution in distribution tails where extreme values
#' may significantly impact WoE.
#'
#' \strong{Phase 3: Optimal Cutpoint Selection}
#'
#' Two strategies are employed based on dataset size:
#'
#' \strong{3a. Dynamic Programming (N <= 50)}:
#'
#' For small datasets, an exact DP solution maximizes total IV:
#'
#' \deqn{\text{dp}[i][j] = \max_{l < i} \left\{ \text{dp}[l][j-1] + \text{IV}(\text{bin from } l+1 \text{ to } i) \right\}}
#'
#' where \eqn{\text{dp}[i][j]} is the maximum IV using \eqn{i} observations in \eqn{j} bins.
#'
#' \strong{Note}: The DP implementation has known bugs for \eqn{N > 50} (conservative limit
#' to prevent crashes). For larger datasets, fallback uses uniform quantiles.
#'
#' \strong{3b. Greedy IV-based Selection (N > 50)}:
#'
#' For each candidate cutpoint \eqn{c}, compute the split IV:
#'
#' \deqn{\text{IV}_{\text{split}}(c) = \text{IV}_{\text{left}} + \text{IV}_{\text{right}}}
#'
#' where left/right refer to observations \eqn{\le c} and \eqn{> c}, respectively.
#'
#' Candidates are ranked by \eqn{\text{IV}_{\text{split}}} (descending), and the top
#' \code{max_bins - 1} are selected.
#'
#' \strong{Phase 4: Bin Refinement}
#'
#' \strong{4a. Frequency Constraint Enforcement}:
#'
#' Bins with count \eqn{< \text{bin\_cutoff} \times N} are merged with the adjacent bin
#' having the most similar event rate:
#'
#' \deqn{\text{merge\_with} = \arg\min_{j \in \{i-1, i+1\}} |\text{event\_rate}_i - \text{event\_rate}_j|}
#'
#' \strong{4b. Monotonicity Enforcement (PAVA)}:
#'
#' If \code{monotonic = TRUE}, the Pool Adjacent Violators Algorithm ensures:
#'
#' \deqn{\text{WoE}_1 \le \text{WoE}_2 \le \cdots \le \text{WoE}_k \quad \text{(increasing)}}
#'
#' or the reverse for decreasing patterns. Direction is auto-detected via:
#'
#' \deqn{\text{increasing} = \begin{cases} \text{TRUE} & \text{if } \text{WoE}_{\text{last}} \ge \text{WoE}_{\text{first}} \\ \text{FALSE} & \text{otherwise} \end{cases}}
#'
#' Violations are resolved by merging adjacent bins iteratively.
#'
#' \strong{4c. Bin Count Optimization}:
#'
#' If the number of bins exceeds \code{max_bins}, bins are merged to minimize IV loss:
#'
#' \deqn{\Delta \text{IV}_{i,i+1} = \text{IV}_i + \text{IV}_{i+1} - \text{IV}_{\text{merged}}}
#'
#' The pair with smallest \eqn{\Delta \text{IV}} is merged iteratively until
#' \eqn{k \le \text{max\_bins}}.
#'
#' \strong{Computational Complexity}
#'
#' \itemize{
#'   \item \strong{Time}:
#'     \itemize{
#'       \item Sketch construction: \eqn{O(N \log k)} for \eqn{N} updates
#'       \item Candidate evaluation: \eqn{O(N \times C)} where \eqn{C \approx 40}
#'       \item DP (if applicable): \eqn{O(N^2 \times k)}
#'       \item PAVA: \eqn{O(k^2)} worst case
#'       \item \strong{Total}: \eqn{O(N \log k + N \times C + k^2 \times I)} where \eqn{I} is iterations
#'     }
#'   \item \strong{Space}:
#'     \itemize{
#'       \item Sketch: \eqn{O(k \log N)} vs \eqn{O(N)} for batch methods
#'       \item DP table (if N <= 50): \eqn{O(N \times k)}
#'       \item \strong{Total}: \eqn{O(k \log N)} for large N
#'     }
#' }
#'
#' \strong{Comparison with Batch Methods}
#'
#' \tabular{lllll}{
#'   \strong{Method} \tab \strong{Space} \tab \strong{Passes} \tab \strong{Guarantees} \tab \strong{Scalability} \cr
#'   Sketch \tab O(k log N) \tab 1 \tab Probabilistic (\eqn{\epsilon}) \tab Streaming-ready \cr
#'   MDLP \tab O(N) \tab 1 \tab Deterministic (MDL) \tab Batch only \cr
#'   MOB/MBLP \tab O(N) \tab Multiple \tab Heuristic \tab Batch only \cr
#' }
#'
#' \strong{When to Use Sketch-based Binning}
#'
#' \itemize{
#'   \item \strong{Use Sketch}: For very large datasets (N > 10^6) where memory is constrained,
#'     or for streaming data where single-pass processing is required.
#'   \item \strong{Use MDLP}: For moderate datasets (N < 10^5) where exact quantiles and
#'     deterministic results are preferred.
#'   \item \strong{Avoid Sketch}: For small datasets (N < 1000) where approximation error
#'     may dominate, or when reproducibility with exact quantiles is critical.
#' }
#'
#' \strong{Tuning sketch_k}
#'
#' The \code{sketch_k} parameter controls the accuracy-memory tradeoff:
#'
#' \itemize{
#'   \item \strong{k = 50}: Fast, low memory, \eqn{\epsilon \approx 2\%} (suitable for exploration)
#'   \item \strong{k = 200} (default): Balanced, \eqn{\epsilon \approx 0.5\%} (production)
#'   \item \strong{k = 500}: High precision, \eqn{\epsilon \approx 0.2\%} (critical applications)
#' }
#'
#' @references
#' \itemize{
#'   \item Karnin, Z., Lang, K., & Liberty, E. (2016). "Optimal Quantile Approximation in
#'     Streams". \emph{Proceedings of the 57th Annual IEEE Symposium on Foundations of
#'     Computer Science (FOCS)}, pp. 71-78.
#'   \item Greenwald, M., & Khanna, S. (2001). "Space-efficient online computation of
#'     quantile summaries". \emph{ACM SIGMOD Record}, 30(2), 58-66.
#'   \item Cormode, G., & Duffield, N. (2014). "Sampling for Big Data: A Tutorial".
#'     \emph{Proceedings of the 20th ACM SIGKDD}, pp. 1975-1975.
#'   \item Munro, J. I., & Paterson, M. S. (1980). "Selection and sorting with limited
#'     storage". \emph{Theoretical Computer Science}, 12(3), 315-323.
#'   \item Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D. (1972).
#'     \emph{Statistical Inference Under Order Restrictions}. Wiley.
#'   \item Siddiqi, N. (2006). \emph{Credit Risk Scorecards}. Wiley.
#' }
#'
#' @examples
#' \dontrun{
#' # Simulate large-scale credit scoring data
#' set.seed(2024)
#' n <- 100000 # Large dataset where sketch shines
#'
#' feature <- c(
#'   rnorm(40000, mean = 580, sd = 60),
#'   rnorm(40000, mean = 680, sd = 50),
#'   rnorm(20000, mean = 750, sd = 40)
#' )
#'
#' target <- c(
#'   rbinom(40000, 1, 0.30),
#'   rbinom(40000, 1, 0.12),
#'   rbinom(20000, 1, 0.04)
#' )
#'
#' # Apply sketch-based binning
#' result <- ob_numerical_sketch(
#'   feature = feature,
#'   target = target,
#'   min_bins = 3,
#'   max_bins = 5,
#'   sketch_k = 200, # Standard accuracy
#'   monotonic = TRUE
#' )
#'
#' # Inspect results
#' print(result$woe)
#' print(result$cutpoints)
#' cat(sprintf(
#'   "Converged: %s (iterations: %d)\n",
#'   result$converged, result$iterations
#' ))
#'
#' # Compare sketch_k values
#' result_k50 <- ob_numerical_sketch(feature, target, sketch_k = 50)
#' result_k500 <- ob_numerical_sketch(feature, target, sketch_k = 500)
#'
#' # Check cutpoint stability (higher k → more stable)
#' data.frame(
#'   k = c(50, 200, 500),
#'   N_Bins = c(
#'     length(result_k50$woe),
#'     length(result$woe),
#'     length(result_k500$woe)
#'   ),
#'   First_Cutpoint = c(
#'     result_k50$cutpoints[1],
#'     result$cutpoints[1],
#'     result_k500$cutpoints[1]
#'   )
#' )
#'
#' # Memory comparison (conceptual - not executed)
#' # Sketch: ~200 items x 4 levels x 16 bytes = approx 12 KB
#' # Batch:  100,000 items x 16 bytes = approx 1.6 MB
#' # Ratio:  ~0.75% of batch memory
#' }
#'
#' @author
#' Lopes, J. E. (KLL Sketch implementation based on Karnin et al., 2016)
#'
#' @seealso
#' \code{\link{ob_numerical_mdlp}} for deterministic binning with exact quantiles,
#' \code{\link{ob_numerical_mblp}} for batch processing with monotonicity.
#'
#' @export
ob_numerical_sketch <- function(feature,
                                target,
                                min_bins = 3,
                                max_bins = 5,
                                bin_cutoff = 0.05,
                                special_codes = "",
                                monotonic = TRUE,
                                convergence_threshold = 1e-6,
                                max_iterations = 1000,
                                sketch_k = 200) {
  if (!is.numeric(feature)) {
    stop("Feature must be a numeric vector.")
  }

  if (!is.vector(target) || !(is.integer(target) || is.numeric(target))) {
    stop("Target must be an integer or numeric vector.")
  }

  if (length(feature) != length(target)) {
    stop("Feature and target must have the same length.")
  }

  feature <- as.numeric(feature)
  target <- as.integer(target)

  unique_target <- unique(target[!is.na(target)])
  if (!all(unique_target %in% c(0L, 1L)) || length(unique_target) != 2L) {
    stop("Target must contain exactly two classes: 0 and 1.")
  }

  if (min_bins < 2L) {
    stop("min_bins must be at least 2.")
  }

  if (max_bins < min_bins) {
    stop("max_bins must be greater than or equal to min_bins.")
  }

  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    stop("bin_cutoff must be in the range (0, 1).")
  }

  if (convergence_threshold <= 0) {
    stop("convergence_threshold must be positive.")
  }

  if (max_iterations < 1L) {
    stop("max_iterations must be at least 1.")
  }

  if (sketch_k < 10L || sketch_k > 1000L) {
    stop("sketch_k must be in the range [10, 1000]. Typical values: 50-500.")
  }

  result <- .Call(
    "_OptimalBinningWoE_optimal_binning_numerical_sketch",
    target,
    feature,
    as.integer(min_bins),
    as.integer(max_bins),
    as.numeric(bin_cutoff),
    as.character(special_codes),
    as.logical(monotonic),
    as.numeric(convergence_threshold),
    as.integer(max_iterations),
    as.integer(sketch_k),
    PACKAGE = "OptimalBinningWoE"
  )

  class(result) <- c("OptimalBinningSketch", "OptimalBinning", "list")

  return(result)
}
